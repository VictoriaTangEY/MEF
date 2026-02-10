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
import warnings


warnings.filterwarnings("ignore")


# show all columns and rows of a dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def off_bal(sa_df):
    """
    Function to identify off-balance loans that should be marked as SA_OUTSCOPE.
    Returns:
        sa_df: Original DataFrame with ECL_APPROACH updated for SA_OUTSCOPE loans.
        sa_df_off_no: DataFrame containing only off-balance loans marked as SA_OUTSCOPE.
    """
    excluded_instr_types = [
        'credit card',
        'credit line',
        'Partially allotted term loan',
        'Trade credit facility',
        'Domestic guarantee facility',
        'Corporate/business card facility',
        'Multi purpose credit facility committed',
        'Overdraft credit facility',
        'Multi purpose credit facility non committed',
        'factoring'
    ]
    
    # Create a copy of the input DataFrame to avoid modifying the original
    sa_df_adj = sa_df.copy()
    
    # Split into on-balance and off-balance loans
    sa_df_on = sa_df_adj[sa_df_adj['ON_OFF_BAL_IND'] == 'ON'].copy()
    sa_df_off = sa_df_adj[sa_df_adj['ON_OFF_BAL_IND'] == 'OFF'].copy()
    
    # Identify off-balance loans to mark as SA_OUTSCOPE
    sa_df_off_no = sa_df_off[
        (sa_df_off['STAGE_FINAL'] == 3) & 
        (sa_df_off['INSTR_TYPE'].isin(excluded_instr_types))
    ].copy()
    sa_df_off_no['ECL_APPROACH'] = 'SA_OUTSCOPE'
    sa_df_off_yes = sa_df_off[~sa_df_off.index.isin(sa_df_off_no.index)].copy()

    # Update ECL_APPROACH in the adjusted DataFrame
    sa_df_adj.loc[sa_df_off_no.index, 'ECL_APPROACH'] = 'SA_OUTSCOPE'

    # Validation check
    try:
        if len(sa_df_on) + len(sa_df_off_yes) + len(sa_df_off_no) != len(sa_df_adj):
            print("Error: SA off-balance segmentation counts don't match total SA count")
    except Exception as e:
        print(f"Validation error: {str(e)}")

    return sa_df_adj, sa_df_on, sa_df_off_yes, sa_df_off_no


def run(rc, parent_logger=None):
    if parent_logger:
        logger = parent_logger
    else:
        logger = createLogHandler(
            'sa_part2', rc.logPath/'Log_file_specific_assessment.log')

    logger.info('-' * 50)
    logger.info('Starting Part 2: Specific Assessment LGD Calculation')
    logger.info('-' * 50)

    try:
        # Initialize data preprocessor
        dp = data_preprocessor(context=rc)

        logger.info('Loading parameters')
        param = load_parameters(parmPath=rc.parmPath)
        logger.info('Parameters loaded successfully')

        logger.info('Loading scenario data')
        npl_data = dp.load_scenario_data(
            data_path=rc.dataPathScen, file_pattern='NPL')
        logger.info('Scenario data loaded successfully')

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

        other_df, _ = dp.standardize_data(param_dict=param,
                                          inDataPath=rc.inDataPath,
                                          rawDataName=rc.sa_other_debt_table_name,
                                          inputDataExt=rc.inputDataExtECL,
                                          dtype_tbl=rc.dtype_tbl,
                                          )
        logger.info('Input data loaded successfully')
        

        logger.info('-' * 30)
        logger.info('Calculating SA cashflow LGD')
        logger.info('-' * 30)

        # Special treatment for no sa_fs_table
        if not fs_df.empty:
            # Get customers that exist in both fs_df and sa_df
            cfl_custs = fs_df['CUST_ID'].astype(str).unique()

            # adjust sa_df for off-balance loans
            _, sa_df_on, _, _ = off_bal(sa_df)

            # Check if manual NOC file exists
            manual_noc_path = Path(rc.inDataPath) / 'sa_manual_noc.csv'
            if manual_noc_path.exists():
                logger.info('Using precomputed NOC values from manual file')
                cfl_lgd_p = cfl_lgd_ts.SACashflowLGDPrecomputed(context=rc)
                cfl_lgd, sa_interim_output_df, noc_interim_output_df = cfl_lgd_p.run(
                    sa_df=sa_df_on,
                    repay_df=repay_df,
                    param=param,
                    npl_data=npl_data,
                    custs=cfl_custs
                )

            else:
                logger.info('Calculating NOC from financial statements')
                sa_fs_param = param['sa_fs_param']
                sa_fs_param.columns = [
                    col.date() if isinstance(col, datetime.datetime) else col
                    for col in sa_fs_param.columns
                ]

                cfl_lgd_p = cfl_lgd_ts.SACashflowLGD(context=rc)
                cfl_lgd, sa_interim_output_df, noc_interim_output_df = cfl_lgd_p.run(
                    sa_df=sa_df_on,
                    fs_df=fs_df,
                    other_df=other_df,
                    repay_df=repay_df,
                    param=param,
                    npl_data=npl_data,
                    custs=cfl_custs,
                    sa_fs_param_1=sa_fs_param
                )

            # export to result interim folder
            sa_interim_output_df.to_csv(
                rc.resultPath/'interim/sa_cfl_lgd_interim_output.csv', index=False)
            noc_interim_output_df.to_csv(
                rc.resultPath/'interim/noc_interim_output.csv', index=False)
            cfl_lgd.to_csv(
                rc.resultPath/'interim/cfl_lgd.csv', index=False)
        
        else:
            pass

        logger.info('-' * 50)
        logger.info('Part 2 completed: SA cashflow LGD calculated and saved')
        logger.info('-' * 50)
        return True

    except Exception as e:
        logger.exception(
            "Error in Part 2: Specific Assessment LGD Calculation")
        return False
