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


# configPath = Path(
#     r'C:\Users\SA814XM\Engagement\03_KhanBank\kb_ecl_engine\khb_engine\run_config_file.json')
# c = load_configuration_file(configPath=configPath)
# rc = run_setting(run_config=c)


def run(rc, parent_logger=None):
    if parent_logger:
        logger = parent_logger
    else:
        logger = createLogHandler(
            'sa_part1', rc.logPath/'Log_file_specific_assessment.log')

    logger.info('-' * 50)
    logger.info('Starting Part 1: Specific Assessment Parameter Generation')
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
        logger.info('Calculating SA cashflow LGD parameters')
        logger.info('-' * 30)
        # get SA CFL customers: custs in the fs_df
        
        # Special treatment for no sa_fs_table
        if not fs_df.empty:
            cfl_custs = fs_df['CUST_ID'].unique()

            lgd_param_p = cfl_lgd_ts.SACashflowLGDParam(context=rc)
            lgd_param, _, _, _ = lgd_param_p.run(
                fs_df=fs_df,
                sa_df=sa_df,
                other_df=other_df,
                custs=cfl_custs)

            # export sa cfl lgd params
            lgd_param.to_excel(rc.parmPath/'SAFSParam.xlsx',
                            header=False, index=False, sheet_name='sa_fs_param')

            # save interim output
            fx_df.to_csv(rc.resultPath/'interim/fx_df.csv', index=False)
            sa_df.to_csv(rc.resultPath/'interim/sa_df.csv', index=False)
            fs_df.to_csv(rc.resultPath/'interim/fs_df.csv', index=False)
            other_df.to_csv(rc.resultPath/'interim/other_df.csv', index=False)
            npl_data.to_excel(rc.resultPath/'interim/npl_data.xlsx', index=False)

            # only save the repay_df for cust in sa scope
            repay_df_sa = repay_df[repay_df['CONTRACT_ID'].isin(
                sa_df['CONTRACT_ID'].unique())]
            repay_df_sa.to_csv(rc.resultPath/'interim/repay_df.csv', index=False)

        else:
            logger.warning('No sa_fs_table available')
            sa_df.to_csv(rc.resultPath/'interim/sa_df.csv', index=False)
        
        logger.info('-' * 50)
        logger.info(
            'Part 1 completed: SA cashflow LGD parameters generated and saved')
        logger.info('-' * 50)
        return True

    except Exception as e:
        logger.exception(
            "Error in Part 1: Specific Assessment Parameter Generation")
        return False
