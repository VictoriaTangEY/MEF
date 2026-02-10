import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from input_handler.load_parameters import load_parameters
from input_handler.data_preprocessor import data_preprocessor

from scenario_engine import forward_looking_model as flm
from scenario_engine import cashflow_lgd_ts as cfl_lgd_ts
from scenario_engine import collateral_lgd_ts as col_lgd_ts
from scenario_engine import pd_term_structure as pd_ts

from util.loggers import createLogHandler


def run(rc):

    print("\nchk path: ", rc.parmPath)

    ##############################################################################
    # Risk parameter generation module
    ##############################################################################

    # Initiate scenario engine
    print('============ Generating risk parameters ============')

    logger = createLogHandler(
        'ts_gen', rc.logPath/'Log_file_risk_parameter_generation.log')

    logger.info('********** Generating risk parameters **********')

    # Loading necessary parameters and input data
    print('------------ Loading parameters and input data ------------')

    print('------- Loading parameters -------')
    logger.info('Loading parameters ...')

    dp = data_preprocessor(context=rc)

    # load params
    try:
        param = load_parameters(parmPath=rc.parmPath)
    except Exception as e:
        logger.exception("message")
    else:
        logger.info('Parameters loading complete.')

    # load scenario data
    logger.info('Loading scenario data ...')
    print('------- Loading scenario data -------')

    try:
        snr_df = dp.load_scenario_data(
            data_path=rc.dataPathScen, file_pattern='MEF')

    except Exception as e:
        logger.exception("message")
    else:
        logger.info('Scenario data loading complete.')

    # load input data
    logger.info('Loading ECL input files ...')
    print('------- Loading ECL input files -------')

    try:
        instr_df, _, _ = dp.load_input_data(param=param)

        _, ca_df, _, _, _ = dp.run(instr_df_raw=instr_df,
                                   run_scope=rc.run_scope,
                                   param=param)
    except Exception as e:
        logger.exception("message")
    else:
        logger.info('ECL input files read successful.')

    print('------------ Parameters and input data load successful ------------')

    # # Calculating MEF multipliers
    # print('------------ Calculating MEF multipliers ------------')
    # logger.info('Calculating MEF multipliers ...')

    # try:
    #     flmodel = flm.ForwardLookingModel(context=rc)
    #     # transformation end date
    #     t_zero = rc.T_ZERO
    #     date_obj = datetime.strptime(str(t_zero), "%Y%m%d")
    #     end = date_obj.strftime("%Y/%m/%d")
    #     mef_df = flmodel.multiplier(end=end)

    # except Exception as e:
    #     logger.exception("message")
    # else:
    #     logger.info('MEF multipliers calculation complete.')

    # try:
    #     logger.info('Exporting MEF multipliers ...')
    #     mef_df.to_excel(rc.parmPath / 'AutoMEFParam.xlsx',
    #                     sheet_name='MEFmultipliers', index=False, header=False)

    # except Exception as e:
    #     logger.exception("message")
    # else:
    #     logger.info('Export successful ...')
    #     print('------------ MEF multipliers calculation complete ------------')

    # Gernerating risk parameters - LGD and PD term structure
    # calculate cashflow LGD parameter
    print('------------ Generating auto LGD parameter ------------')
    print('------- Cashflow LGD -------')

    logger.info('Generating cashflow LGD parameter...')

    try:
        cfl_lgd_model = cfl_lgd_ts.cashflow_lgd(context=rc)
        cfl_lgd = cfl_lgd_model.get_cashflow_lgd(param=param)
    except Exception as e:
        logger.exception("message")
    else:
        logger.info('Cashflow LGD parameter generation complete.')

    # calculate collateral LGD parameter
    print('------- Collateral LGD -------')

    logger.info('Calculating collateral LGD ...')

    try:
        col_lgd_model = col_lgd_ts.collateral_lgd(context=rc)
        col_lgd = col_lgd_model.get_collateral_lgd(
            instr_df=instr_df, snr_df=snr_df, ca_df=ca_df, param=param)

    except Exception as e:
        logger.exception("message")
    else:
        logger.info('Collateral LGD parameter generation complete.')

    # concat cashflow lgd and collateral lgd
    print('------- Concat cashflow LGD & collateral LGD -------')

    logger.info('Concating LGD ...')

    try:
        col_lgd_selected = col_lgd.iloc[4:]
        col_lgd_selected.columns = cfl_lgd.columns
        clgd_ts = pd.concat([cfl_lgd, col_lgd_selected],
                            axis=0, ignore_index=True)

        clgd_ts.iloc[4:, 0] = range(
            1, len(clgd_ts) - 4 + 1)  # reset the index

    except Exception as e:
        logger.exception("message")
    else:
        logger.info('Concat LGD successful.')

    print('------------ Auto LGD parameter generation complete------------')

    # calculate PD parameter
    print('------------ Generating auto PD parameter ------------')
    try:
        logger.info('Generating auto PD parameter ...')

        cpd_test = pd_ts.scenario_engine(context=rc)
        cpd_ts = cpd_test.ts_calculation(param=param)
    except Exception as e:
        logger.exception("message")
    else:
        logger.info('Auto PD parameter generation complete.')

    print('------------Auto PD parameter generation complete------------')

    # Export risk parameters
    try:
        logger.info('Exporting risk parameters...')

        clgd_ts.to_excel(rc.parmPath / 'AutoLGDParam.xlsx',
                         sheet_name='AutoLGDParam', index=False, header=False)

        cpd_ts.to_excel(rc.parmPath / 'AutoPDParam.xlsx',
                        sheet_name='AutoPDParam', index=False, header=False)

    except Exception as e:
        logger.exception("message")
    else:
        logger.info('Export successful.')
        logger.info(
            '********** Risk parameter generation complete **********')
        logger.handlers.clear()

    print('============ Risk parameter generation complete ============')

    return
