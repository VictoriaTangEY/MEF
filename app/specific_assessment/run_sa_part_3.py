import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from util.loggers import createLogHandler
from input_handler.data_preprocessor import data_preprocessor
from input_handler.load_parameters import load_configuration_file, load_parameters
from input_handler.env_setting import run_setting
from ecl_engine.output_file_handler import convert_to_LCY
from specific_assessment import sa_collateral_lgd as col_lgd_ts
from specific_assessment import sa_cashflow_lgd as cfl_lgd_ts
from specific_assessment import run_sa_cfl_lgd_part_2 as sa_2
import warnings


warnings.filterwarnings("ignore")


# show all columns and rows of a dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def add_lgd_approach(sa_df, cfl_lgd, col_lgd, param):
    """
    Simplified version of LGD source tagging
    Priority: CFL > COL > ASSIGNED (100% LGD)
    """
    # Copy original data
    sa_ecl_df = sa_df.copy()

    # Special treatment for no sa_fs_table
    if not cfl_lgd.empty:
        # Step 1: Consolidation of cust-level LGDs (CFL priority)
        # Generate cust-level LGDs with source tagging
        cust_lgd = pd.concat([
            cfl_lgd[['CUST_ID', 'LGD']].assign(LGD_SOURCE='CFL'),
            col_lgd[['CUST_ID', 'LGD']].assign(LGD_SOURCE='COL')
        ]).drop_duplicates('CUST_ID', keep='first')  # keep the 1st sources（CFL priority）

    else:
        cust_lgd = col_lgd[['CUST_ID', 'LGD']].assign(LGD_SOURCE='COL')

    # Merge to master table
    sa_ecl_df = pd.merge(
        sa_ecl_df,
        cust_lgd,
        on='CUST_ID',
        how='left'
    )

    # # Step 2: Consolidation of pool-level LGDs
    # pool_lgd = param['AutoLGDParam'][['LGD_POOL_ID', 'LGD_0']]\
    #     .rename(columns={'LGD_0': 'LGD_pool'})

    # sa_ecl_df = pd.merge(
    #     sa_ecl_df,
    #     pool_lgd,
    #     on='LGD_POOL_ID',
    #     how='left'
    # )

    # # Step 3: Determine final LGD values and sources
    # sa_ecl_df['LGD'] = sa_ecl_df['LGD'].combine_first(
    #     sa_ecl_df['LGD_pool'])

    # # Override LGD_APPROACH
    # conditions = [
    #     sa_ecl_df['LGD_SOURCE'] == 'CFL',
    #     sa_ecl_df['LGD_SOURCE'] == 'COL',
    #     sa_ecl_df['LGD_pool'].notnull()
    # ]
    # #TODO: change proxy into 100%
    # choices = ['CFL', 'COL', 'ASSIGNED']

    # sa_ecl_df['LGD_APPROACH'] = np.select(
    #     conditions, choices, default='UNDEFINED')

    # drop unnecessary columns

    # Step 2: Determine final LGD values and sources
    conditions = [
        sa_ecl_df['LGD_SOURCE'] == 'CFL',
        sa_ecl_df['LGD_SOURCE'] == 'COL',
        True
    ]
    choices = [
        sa_ecl_df['LGD'],
        sa_ecl_df['LGD'],
        1.0  # Assign 100% LGD for ASSIGNED cases
    ]
    sa_ecl_df['LGD'] = np.select(conditions, choices, default=np.nan)

    # Step 3: Set LGD_APPROACH
    approach_conditions = [
        sa_ecl_df['LGD_SOURCE'] == 'CFL',
        sa_ecl_df['LGD_SOURCE'] == 'COL',
        True
    ]
    approach_choices = ['CFL', 'COL', 'ASSIGNED']
    sa_ecl_df['LGD_APPROACH'] = np.select(
        approach_conditions, approach_choices, default='UNDEFINED')
    
    return sa_ecl_df.drop(columns=['LGD_SOURCE'])


def run(rc, parent_logger=None):
    if parent_logger:
        logger = parent_logger
    else:
        logger = createLogHandler(
            'sa_part3', rc.logPath/'Log_file_specific_assessment.log')

    logger.info('-' * 50)
    logger.info('Starting Part 3: Specific Assessment Final ECL Calculation')
    logger.info('-' * 50)

    try:
        # Initialize data preprocessor
        dp = data_preprocessor(context=rc)

        logger.info('Loading parameters')
        param = load_parameters(parmPath=rc.parmPath)
        logger.info('Parameters loaded successfully')

        logger.info('Reading ECL input files')
        instr_df, fx_df, repay_df = dp.load_input_data(param=param)

        _, _, sa_df, _, _ = dp.run(instr_df_raw=instr_df,
                                   run_scope=rc.run_scope,
                                   param=param)

        fs_df, _ = dp.standardize_data(param_dict=param,
                                              inDataPath=rc.inDataPath,
                                              rawDataName=rc.sa_fs_table_name,
                                              inputDataExt=rc.inputDataExtECL,
                                              dtype_tbl=rc.dtype_tbl,
                                              )
        logger.info('Input data loaded successfully')

        # Special treatment if no sa_fs_table
        if not fs_df.empty:
            # Load previous calculated cashflow LGD result
            cfl_custs = fs_df['CUST_ID'].astype(str).unique()
            cfl_lgd = pd.read_csv(
                rc.resultPath/'interim/cfl_lgd.csv', dtype={'CUST_ID': str})
            logger.info('Previous cashflow LGD results loaded successfully')

        else:
            cfl_custs = []
            cfl_lgd = pd.DataFrame()

        logger.info('-' * 30)
        logger.info('Calculating SA collateral LGD')
        logger.info('-' * 30)
        # Get list of all collateral types from parameters
        coll_list = param['SA_COL_param'].collateral_type.to_list()

        # Calculate total collateral value per customer:
        cust_coll_sum = sa_df.groupby('CUST_ID')[coll_list].sum().sum(axis=1)

        # Filter customers with positive collateral value (>0)
        custs_with_collateral = cust_coll_sum[cust_coll_sum > 0].index.tolist()

        # Exclude customers already processed in cash flow analysis
        col_custs = [c for c in custs_with_collateral if c not in cfl_custs]

        # calc col lgd
        col_lgd_p = col_lgd_ts.SACollateralLGD(context=rc)
        col_lgd = col_lgd_p.get_collateral_lgd(sa_df=sa_df,
                                               param=param,
                                               custs=col_custs)
        logger.info('Collateral LGD calculation completed successfully')

        logger.info('-' * 30)
        logger.info('Assigning SA LGD approaches')
        logger.info('-' * 30)
        sa_ecl_df = add_lgd_approach(sa_df, cfl_lgd, col_lgd, param)
        logger.info('LGD approaches assigned successfully')

        logger.info('-' * 30)
        logger.info('Calculating final SA ECL')
        logger.info('-' * 30)
        # adjus off-balance
        sa_ecl_df_adj, sa_ecl_df_on, sa_ecl_df_off_yes, sa_ecl_df_off_no = sa_2.off_bal(sa_ecl_df)
        
        # set LGD to None for all sa_ecl_df_off_no rows (as you already have)
        sa_ecl_df_adj.loc[sa_ecl_df_off_no.index, 'LGD'] = None
        
        # for each CUST_ID that has both on-balance and off-balance (yes) loans
        for cust_id in sa_ecl_df_adj['CUST_ID'].unique():
            on_rows = sa_ecl_df_on[sa_ecl_df_on['CUST_ID'] == cust_id]
            off_yes_rows = sa_ecl_df_off_yes[sa_ecl_df_off_yes['CUST_ID'] == cust_id]
            
            # If customer has both on-balance and off-balance (yes) loans
            if not on_rows.empty and not off_yes_rows.empty:
                # Apply this LGD to all off-balance (yes) rows
                on_lgd = on_rows['LGD'].iloc[0]
                sa_ecl_df_adj.loc[off_yes_rows.index, 'LGD'] = on_lgd

        # Calculate ECL_OCY = EAD_OCY * LGD
        sa_ecl_df_adj['ECL_FINAL_OCY'] = sa_ecl_df_adj['EAD_OCY'] * sa_ecl_df_adj['LGD']

        # Transfer OCY to LCY
        sa_ecl_df_adj = convert_to_LCY(df=sa_ecl_df_adj, fx_tbl=fx_df, param=param)
        logger.info('Final ECL calculation completed successfully')

        logger.info('Exporting SA ECL results')
        sa_ecl_df_adj.to_csv(
            rc.resultPath/"SAECL_calculation_result_files_deal.csv", index=False, encoding='utf-8-sig')

        logger.info('-' * 50)
        logger.info('Part 3 completed: SA ECL results exported successfully')
        logger.info('-' * 50)
        return True

    except Exception as e:
        logger.exception(
            "Error in Part 3: Specific Assessment Final ECL Calculation")
        return False
