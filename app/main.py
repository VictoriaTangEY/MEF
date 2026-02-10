##############################################################################
'''
Khan Bank ECL calculation engine
Version number: 0.1
Date: 2024-10-02
Remark: Developing version
'''
##############################################################################

# %%
##############################################################################
# Load Packages
##############################################################################
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import argparse as ap
import json

from input_handler.env_setting import run_setting
from input_handler.load_parameters import load_configuration_file, load_parameters
from input_handler.data_preprocessor import data_preprocessor


from data_validation.validator import data_health_check_report
from data_validation.validator_post import post_run_validation_report
from scenario_engine import run_scenario_engine as senario_engine
from specific_assessment import run_sa_cfl_lgd_part_1 as sa_1
from specific_assessment import run_sa_cfl_lgd_part_2 as sa_2
from specific_assessment import run_sa_part_3 as sa_3
from specific_assessment import run_sa_part_4 as sa_4

from result_merge import run_output_final as opt

from ecl_engine import (collective_assessment as ca,
                        management_overlay as mo,
                        output_file_handler as ofh,
                        proxy_model as pm)


from reporting import report_run

from util.loggers import createLogHandler

##############################################################################
# ECL engine main function
##############################################################################
# configPath = Path(r'C:\Users\WH947CH\Engagement\Khan Bank\03_ECL_engine\02_Development\khb_engine\run_config_file.json')
# c = load_configuration_file(configPath=configPath)
# rc = run_setting(run_config=c)


def main(run_config):

    c = run_config.copy()
    rc = run_setting(run_config=c)

##############################################################################
# Risk parameter generation module
##############################################################################
    if (rc.RUN_MODE in [1, 3, 15]):
        senario_engine.run(rc=rc)

##############################################################################
# Data validation
# To be implemented
##############################################################################
    if (rc.RUN_MODE in [2, 3, 15]):
        print('------------Generating pre-run data health check report------------')
        logger = createLogHandler(
            'pre_validate', rc.logPath/'Log_file_pre_run_validation.log')
        try:
            logger.info(
                '********** Initiate Pre Run Data Validator **********')

            dvp = data_health_check_report(context=rc)
            out_dict = dvp.run()
            logger.info('Generating pre-run data health check report...')

            writer = pd.ExcelWriter(
                rc.reportPath / "Data_Health_Check_Report.xlsx", engine='openpyxl')
            for key in out_dict.keys():
                out_dict[key].to_excel(writer, index=False, sheet_name=key)
            writer.close()

        except Exception as e:
            logger.exception("message")
        else:
            logger.info('Export successful ...')
            logger.info(
                '********** Pre-run data health check report generation complete **********')
            logger.handlers.clear()
        print('-------Pre-run data health check report generation complete-------')

##############################################################################
# Collective Assessment Module
##############################################################################
    if (rc.RUN_MODE in [4, 12, 15]):
        logger = createLogHandler(
            'ecl_calc', rc.logPath/'Log_file_ecl_calculation.log')
        logger.info('********** Initiate ECL Engine Calculator **********')
        dp = data_preprocessor(context=rc)
        # Load all parameters

        logger.info('Loading all necessary parameters ...')
        print('------- Loading parameter files -------')

        try:
            param = load_parameters(parmPath=rc.parmPath)
        except Exception as e:
            logger.exception("message")
        else:
            logger.info('Parameter loading complete.')

        logger.info('Reading ECL input files ...')
        print('------- Reading ECL input files -------')

        try:
            instr_df, fx_df, repay_df = dp.load_input_data(param=param)
        except Exception as e:
            logger.exception("message")
        else:
            logger.info('Input data read successful.')

        logger.info('Performing data preprocessing ...')
        print('------- Performing data preprocessing -------')
        # Filter out required run scope by users
        proc_dfs = dp.run(instr_df_raw=instr_df,
                          run_scope=rc.run_scope,
                          param=param)

        logger.info('Data preprocessing successful.')

        # Perform Stage allocation
        logger.info('Calculate collective assessment ECL ...')
        print('------- Calculating collective assessment ECL -------')

        ce = ca.ecl_engine(context=rc)
        ecl_df, ecl_interim_df = ce.run(instr_df=proc_dfs[1],
                                        repay_df=repay_df,
                                        param=param)

        logger.info('Collective assessment complete.')

        logger.info('Performing management overlay ...')
        print('------- Performing management overlay -------')
        me = mo.overlay_engine(context=rc)
        ecl_df_final = me.run(ecl_df=ecl_df, param=param)
        logger.info('Management overlay completed.')

        logger.info('Exporting the ECL results ...')
        print('------- Exporting the ECL results -------')

        ecl_df_fmt, ecl_interim_df_fmt = ofh.run(df=ecl_df_final,
                                                 calc_proc_df=ecl_interim_df,
                                                 fx_df=fx_df,
                                                 param=param,
                                                 dtype_tbl=rc.dtype_tbl,
                                                 calc_file_partition=20,
                                                 resultPath=rc.resultPath,
                                                 is_export=True)

        logger.info('Exporting successful.')
        logger.info('********** ECL calculation process complete **********.')
        print('------- ECL calculation complete -------')
        logger.handlers.clear()

##############################################################################
# Proxy module
##############################################################################
    if (rc.RUN_MODE in [5, 12, 15]):

        logger = createLogHandler(
            'proxy_cal', rc.logPath/'Log_file_ecl_calculation.log')
        try:
            logger.info(
                '********** Calculating Proxy ECL **********')
            print('------- Calculating proxy ECL -------')
            dp = data_preprocessor(context=rc)

            logger.info('Loading all necessary parameters ...')
            print('------- Loading parameter files -------')

            try:
                param = load_parameters(parmPath=rc.parmPath)
            except Exception as e:
                logger.exception("message")
            else:
                logger.info('Parameter loading complete.')

            logger.info('Reading ECL input files ...')
            print('------- Reading ECL input files -------')

            try:
                instr_df, fx_df, repay_df = dp.load_input_data(param=param)
            except Exception as e:
                logger.exception("message")
            else:
                logger.info('Input data read successful.')
                
            proc_dfs = dp.run(instr_df_raw=instr_df,
                    run_scope=rc.run_scope,
                    param=param)
            prm = pm.ProxyModel(context=rc)
            prm.run(px_df_=proc_dfs[3], fx_df=fx_df, param=param)

        except Exception as e:
            logger.exception("message")
        else:
            logger.info('Proxy successful ...')
            logger.info(
                '********** Proxy ECL calculation complete **********')
            logger.handlers.clear()
        print('-------Proxy ECL calculation complete-------')


##############################################################################
# Specific Assessment Module
# Modes:
# 601 - Generate SA Financial Statement Parameters (SAFSParam.xlsx)
# 602 - Calculate Cashflow LGD (requires SAFSParam.xlsx)
# 603 - Calculate Final SA ECL (requires cfl_lgd.csv)
# 604 - Overall SA run
##############################################################################
    if (rc.RUN_MODE in [6, 12, 15]):
        # Create main logger for specific assessment
        logger = createLogHandler(
            'sa_module', rc.logPath/'Log_file_specific_assessment.log')
        logger.info('=' * 80)
        print('------- Starting Specific Assessment Module -------')
        logger.info('Starting Specific Assessment Module')
        logger.info('=' * 80)

        try:
            # Mode 601: Generate SA Financial Statement Parameters
            if rc.RUN_MODE_SA in [601]:
                logger.info('*' * 50)
                print('------- Mode 601: Generating SA Financial Statement Parameters -------')
                logger.info('Mode 601: Generating SA Financial Statement Parameters')
                logger.info('*' * 50)
                sa_1.run(rc=rc, parent_logger=logger)
                logger.info('SA Financial Statement Parameters generated successfully')
                print('-------SA Financial Statement Parameters generation completed-------')

            # Mode 602: Calculate Cashflow LGD
            if rc.RUN_MODE_SA in [602]:
                logger.info('*' * 50)
                print('------- Mode 602: Calculating Cashflow LGD -------')
                logger.info('Mode 602: Calculating Cashflow LGD')
                logger.info('*' * 50)

                # Check for required SAFSParam.xlsx (only if Mode 601 wasn't run)
                safs_param_path = rc.parmPath / 'SAFSParam.xlsx'
                if not safs_param_path.exists():
                    logger.info('SAFSParam.xlsx not found, generating parameters first')
                    sa_1.run(rc=rc, parent_logger=logger)

                # Calculate Cashflow LGD
                sa_2.run(rc=rc, parent_logger=logger)
                logger.info('Cashflow LGD calculation completed successfully')
                print('-------SA Cashflow LGD calculation completed-------')

            # Mode 603: Calculate Final SA ECL
            if rc.RUN_MODE_SA in [603]:
                logger.info('*' * 50)
                print('------- Mode 603: Calculating Final SA ECL -------')
                logger.info('Mode 603: Calculating Final SA ECL')
                logger.info('*' * 50)

                # Calculate Final SA ECL
                sa_3.run(rc=rc, parent_logger=logger)
                logger.info('Final SA ECL calculation completed successfully')

            # Mode 604: Overall SA run
            if rc.RUN_MODE_SA in [604]:
                logger.info('*' * 50)
                logger.info('Mode 604: execute Modes 601, 602, and 603 sequentially')
                logger.info('*' * 50)
                sa_4.run(rc=rc, parent_logger=logger)
                logger.info('Overall SA run completed successfully')
                print('-------Overall SA run completed-------')

            logger.info('=' * 80)
            print('------- Specific Assessment Module completed successfully -------')
            logger.info('Specific Assessment Module completed successfully')
            logger.info('=' * 80)
            logger.handlers.clear()

        except Exception as e:
            logger.exception("Error in Specific Assessment Module")
            logger.handlers.clear()
            raise


##############################################################################
# Output module
# combine ca,sa,proxy and outscoped as one result file.
##############################################################################
    if (rc.RUN_MODE in [7, 12, 15]):
        opt.run(rc=rc)

##############################################################################
# Post run data assessment module
##############################################################################
    if (rc.RUN_MODE in [8, 13, 15]):
        print('------------Generating post-run data health check report------------')
        logger = createLogHandler(
            'post_validate', rc.logPath/'Log_file_post_run_validation.log')
        try:
            logger.info(
                '********** Initiate Post Run Data Validator **********')

            prd = post_run_validation_report(context=rc)
            out_dict = prd.run()
            logger.info('Generating post-run data health check report...')

            writer = pd.ExcelWriter(
                rc.reportPath / "PostRun_Result_Validation_Report.xlsx", engine='openpyxl')
            for key in out_dict.keys():
                out_dict[key].to_excel(writer, index=False, sheet_name=key)
            writer.close()
            # print(out_dict['validReport'])

        except Exception as e:
            logger.exception("message")
        else:
            logger.info('Export successful ...')
            logger.info(
                '********** Post-run data health check report generation complete **********')
            logger.handlers.clear()
        print('-------Post-run data health check report generation complete-------')


##############################################################################
# Reporting module
##############################################################################
    if (rc.RUN_MODE in [9, 13, 15]):
        print('------------Generating report------------')
        logger = createLogHandler(
            'reporting', rc.logPath/'Log_file_reporting.log')
        try:
            logger.info(
                '********** Initiate Reporting **********')
            # rpm = report_run(context=rc)
            wb = report_run.run(rc=rc)
            # out_dict_rp = rpm.run_reporting_gen()
            logger.info('Generating reporting...')

            wb.save(rc.reportPath / "ECL_Result_Reports.xlsx")

        except Exception as e:
            logger.exception("message")
        else:
            logger.info('Export successful ...')
            logger.info(
                '********** Reporting generation complete **********')
            logger.handlers.clear()
        print('-------Reporting generation complete-------')

    return


###########################################################
# Main Function
###########################################################
if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('--configPath', type=str)
    args = parser.parse_args()

    with open(Path(args.configPath), 'r') as fp:
        c = json.load(fp)

    main(run_config=c)
