import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import csv
from util.loggers import createLogHandler
from input_handler.data_preprocessor import data_preprocessor
from input_handler.load_parameters import load_configuration_file, load_parameters
from input_handler.env_setting import run_setting
from ecl_engine.output_file_handler import convert_to_LCY
from specific_assessment import sa_collateral_lgd as col_lgd_ts
from specific_assessment import sa_cashflow_lgd as cfl_lgd_ts
import warnings


warnings.filterwarnings("ignore")


# show all columns and rows of a dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# configPath = Path(
#     r'C:\Users\SA814XM\Engagement\03_KhanBank\kb_ecl_engine\khb_engine\run_config_file.json')
# c = load_configuration_file(configPath=configPath)
# rc = run_setting(run_config=c)


def run(rc):
    print('============ Generating specific assessment ECL ============')

    logger = createLogHandler(
        'sa_cal', rc.logPath/'Log_file_specific_assessment.log')

    logger.info('********** Initiate specific assessment calculator **********')

    # Loading necessary parameters and input data
    print('------------ Loading parameters and input data ------------')

    print('------- Loading parameters -------')
    logger.info('Loading parameters ...')

    dp = data_preprocessor(context=rc)

    try:
        param = load_parameters(parmPath=rc.parmPath)
    except Exception as e:
        logger.exception("message")
    else:
        logger.info('Parameters loading complete.')

    logger.info('Loading scenario data ...')
    print('------- Loading scenario data -------')

    try:
        npl_data = dp.load_scenario_data(
            data_path=rc.dataPathScen, file_pattern='NPL')
    except Exception as e:
        logger.exception("message")
    else:
        logger.info('Scenario data loading complete.')

    logger.info('Reading ECL input files ...')
    print('------- Reading ECL input files -------')

    try:
        instr_df, fx_df, repay_df = dp.load_input_data(param=param)

        _, _, sa_df, _, _ = dp.run(instr_df_raw=instr_df,
                                   run_scope=rc.run_scope,
                                   param=param)

        fs_df, is_error = dp.standardize_data(param_dict=param,
                                              inDataPath=rc.inDataPath,
                                              rawDataName=rc.sa_fs_table_name,
                                              inputDataExt=rc.inputDataExtECL,
                                              dtype_tbl=rc.dtype_tbl,
                                              )

        other_df, is_error = dp.standardize_data(param_dict=param,
                                                 inDataPath=rc.inDataPath,
                                                 rawDataName=rc.sa_other_debt_table_name,
                                                 inputDataExt=rc.inputDataExtECL,
                                                 dtype_tbl=rc.dtype_tbl,
                                                 )
        
        # chk
        # import os
        # output_dir = r"C:\Users\UV665AR\OneDrive - EY\99_Data_Server_Out\production\20241231"
        # os.makedirs(output_dir, exist_ok=True)
        # output_file = os.path.join(output_dir, "sa_test.xlsx")
        # sa_df.to_excel(output_file, index=False)

    except Exception as e:
        logger.exception("message")
    else:
        logger.info('ECL input files read successful.')

    print('------------ Parameters and input data load successful ------------')

    # **************************
    # calculate SA cashflow LGD
    # **************************
    print('------------ Calculating SA LGD ------------')
    print('------- Cashflow LGD -------')
    print('--- Exporting LGD Param ---')

    logger.info('Exporting SA cashflow LGD parameter ...')

    try:

        # get SA CFL customers: custs in the fs_df
        cfl_custs = fs_df['CUST_ID'].unique()

        lgd_param_p = cfl_lgd_ts.SACashflowLGDParam(context=rc)
        lgd_param, cash_profit_df, ciwc_df, noc_df = lgd_param_p.run(
            fs_df=fs_df,
            sa_df=sa_df,
            other_df=other_df,
            custs=cfl_custs)

        # export sa cfl lgd params
        # TODO uncomment after testing
        lgd_param.to_excel(rc.parmPath/'SAFSParam.xlsx',
                           header=False, index=False, sheet_name='sa_fs_param')

    except Exception as e:
        logger.exception("message")
    else:
        logger.info('SA cashflow LGD parameter export complete.')

    # chk
    # print('exported param:\n', lgd_param)
    # print('noc_df:\n', noc_df)

    ##### stop point #####

    ##### reload point #####
    print('--- Reloading LGD Param ---')
    logger.info('Reloading parameters ...')

    try:
        param = load_parameters(parmPath=rc.parmPath)
    except Exception as e:
        logger.exception("message")
    else:
        logger.info('Parameters loading complete.')

    ##### calculate LGD #####
    logger.info('Calculating SA cashflow LGD ...')

    try:

        sa_fs_param_1 = param['sa_fs_param']

        # Converting a DataFrame's datetime column name to a date type
        sa_fs_param_1.columns = [
            col.date() if isinstance(col, datetime.datetime) else col
            for col in sa_fs_param_1.columns
        ]

        # chk
        # print('reloaded param:\n', sa_fs_param_1)

        # calculate sa cfl lgd
        # get SA CFL customers: custs in the fs_df
        cfl_custs = fs_df['CUST_ID'].unique()

        cfl_lgd_p = cfl_lgd_ts.SACashflowLGD(context=rc)
        cfl_lgd, noc_df_1, pwa_noc_df = cfl_lgd_p.run(sa_df=sa_df,
                                                      fs_df=fs_df,
                                                      other_df=other_df,
                                                      repay_df=repay_df,
                                                      param=param,
                                                      npl_data=npl_data,
                                                      custs=cfl_custs,
                                                      sa_fs_param_1=sa_fs_param_1)

    except Exception as e:
        logger.exception("message")
    else:
        logger.info('SA cashflow LGD calculation complete.')

    # chk
    # print('final cfl lgd:\n', cfl_lgd.head())
    # print('final noc:\n', noc_df_1)

    # ***************************
    # calculate SA collteral LGD
    # ***************************
    print('------- Collateral LGD -------')
    logger.info('Calculating SA collateral LGD ...')

    try:

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

    except Exception as e:
        logger.exception("message")
    else:
        logger.info('SA collateral LGD calculation complete.')

    # chk
    # print('final col lgd:\n', col_lgd.head())

    # **************
    # Assign SA LGD
    # **************
    print('------- LGD Assignment -------')
    logger.info('Assign SA LGD ...')

    def add_lgd_approach(sa_df, cfl_lgd, col_lgd, param):
        """
        Simplified version of LGD source tagging
        Priority: CFL > COL > Pool PROXY
        """
        # Copy original data
        sa_ecl_df = sa_df.copy()

        # Step 1: Consolidation of cust-level LGDs (CFL priority)
        # Generate cust-level LGDs with source tagging
        cust_lgd = pd.concat([
            cfl_lgd[['CUST_ID', 'LGD']].assign(LGD_SOURCE='CFL'),
            col_lgd[['CUST_ID', 'LGD']].assign(LGD_SOURCE='COL')
        ]).drop_duplicates('CUST_ID', keep='first')  # keep the 1st sources（CFL priority）

        # Merge to master table
        sa_ecl_df = pd.merge(
            sa_ecl_df,
            cust_lgd,
            on='CUST_ID',
            how='left'
        )

        # Step 2: Consolidation of pool-level LGDs
        pool_lgd = param['AutoLGDParam'][['LGD_POOL_ID', 'LGD_0']]\
            .rename(columns={'LGD_0': 'LGD_pool'})

        sa_ecl_df = pd.merge(
            sa_ecl_df,
            pool_lgd,
            on='LGD_POOL_ID',
            how='left'
        )

        # Step 3: Determine final LGD values and sources
        sa_ecl_df['LGD'] = sa_ecl_df['LGD'].combine_first(
            sa_ecl_df['LGD_pool'])

        # Override LGD_APPROACH
        conditions = [
            sa_ecl_df['LGD_SOURCE'] == 'CFL',
            sa_ecl_df['LGD_SOURCE'] == 'COL',
            sa_ecl_df['LGD_pool'].notnull()
        ]
        choices = ['CFL', 'COL', 'PROXY']

        sa_ecl_df['LGD_APPROACH'] = np.select(
            conditions, choices, default='UNDEFINED')

        # drop unnecessary columns
        return sa_ecl_df.drop(columns=['LGD_SOURCE', 'LGD_pool'])

    try:

        sa_ecl_df = add_lgd_approach(sa_df, cfl_lgd, col_lgd, param)

    except Exception as e:
        logger.exception("message")
    else:
        logger.info('SA LGD assignment complete.')

    # chk
    # print(sa_ecl_df.head())
    # sa_ecl_df.to_csv("C:\\Users\\SA814XM\\Downloads\\sa_ecl_df_chk.csv")

    # *****************
    # Calculate SA ECL
    # *****************
    print('------------ Calculating SA ECL ------------')
    logger.info('Calculating SA ECL ...')

    try:

        # Calculate ECL_OCY = EAD_OCY * LGD
        sa_ecl_df['ECL_FINAL_OCY'] = sa_ecl_df['EAD_OCY'] * sa_ecl_df['LGD']

        # Transfero OCY to LCY
        sa_ecl_df = convert_to_LCY(df=sa_ecl_df, fx_tbl=fx_df, param=param)

    except Exception as e:
        logger.exception("message")
    else:
        logger.info('SA collateral LGD calculation complete.')

    # *********************
    # Export SA ECL result
    # *********************
    print('------------ Exporting SA ECL result ------------')
    logger.info('Exporting SA ECL result...')

    try:

        sa_ecl_df.to_csv(
            rc.resultPath/"SAECL_calculation_result_files_deal.csv", index=False, encoding='utf-8-sig',quoting=csv.QUOTE_NONNUMERIC)

        # noc_df_1.to_csv(
        #     rc.resultPath/"SA_noc_df.csv", index=False, encoding='utf-8-sig')

        pwa_noc_df.to_csv(
            rc.resultPath/"SA_pwa_noc_df.csv", index=False, encoding='utf-8-sig')

    except Exception as e:
        logger.exception("message")
    else:
        logger.info('SA ECL result export complete.')

    return
