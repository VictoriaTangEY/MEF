import itertools
# Load packages
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import csv

from util.loggers import createLogHandler
from input_handler.load_parameters import (load_configuration_file,
                                           load_parameters)
from input_handler.env_setting import run_setting
from input_handler.data_preprocessor import data_preprocessor

import warnings
warnings.filterwarnings("ignore")

def standarize_ecl_df(ecldata: pd.DataFrame,
                   param: Dict[str, pd.DataFrame],
                   formatter_sheetname: str,
                   dtype_tbl: Dict[str, str]):
    """ This function is based on output_file_handler.perpare_output to cast data type.
    As formatter_sheetname is designed for collective assessement's output table,
    here the only cast the type of columns which exist in ca's output, others are kept as assigned format by python.
    """

    df = ecldata.copy()

    # Data Type casting and renaming
    controlParam = (param[formatter_sheetname]
                    .sort_values(by=['order']))

    rawKeepList = (controlParam.query("keep_ind == 'Y'")['colname']
                   .to_list())
    # renameList = (controlParam.query("keep_ind == 'Y'")['colname_std']
    #               .to_list())
    dtypeList = (controlParam.query("keep_ind == 'Y'")['data_type']
                 .to_list())

    colList = (df.columns.to_list())
    dtype_mapping = {}
    #rename_mapping = {}

    for var, dtype in zip(rawKeepList, dtypeList):
        dtype_mapping[var] = dtype_tbl[dtype]

    # Type Casting # 250109 control output
    for key, val in dtype_mapping.items():
        if key in colList:
            #print(key,val)
            df[key] = df[key].astype(val)

    # Rename columns
    return df


def run(rc):
    print('============ Generating aggerated ECL result============')

    logger = createLogHandler(
        'output_final', rc.logPath/'Log_file_output_final_result.log')

    logger.info('********** Reading ECL result files **********')
    results_ = dict()
    if len(list(rc.resultPath.glob('*_deal.csv'))) == 3:
         ## Check if all ECL result files are ready and read in
        for _wbFile in itertools.chain(rc.resultPath.glob('*_deal.csv')):
            df_ = pd.read_csv(_wbFile,dtype={'CONTRACT_ID': str,
                                        'CUST_ID': str})
            _key = df_.ECL_APPROACH[1]
            results_[_key] = df_
    else:
        print("Please check the completeness of ECL result files.")
        logger.info("Failed on loading ECL results")
    # Loading parameters and necessary data
    print('------------ Loading parameters and full set instrument table ------------')

    print('------- Loading parameters -------')
    logger.info('Loading parameters ...')
    param = load_parameters(parmPath=rc.parmPath)
    logger.info('Loading instrument table ...')
    dp = data_preprocessor(context=rc)
    instr_df_raw, _, _ = dp.load_input_data(param=param)

    print('------- Standarizing results ------')
    logger.info('Standarizing ecldata format ...')
    ecl_df_ca_fmt = standarize_ecl_df(ecldata=results_['COLLECTIVE_ASSESSMENT'],
                            param=param,
                            formatter_sheetname='ecl_deal_output',
                            dtype_tbl=rc.dtype_tbl)
    ecl_df_proxy_fmt = standarize_ecl_df(ecldata=results_['PROXY'],
                            param=param,
                            formatter_sheetname='ecl_deal_output',
                            dtype_tbl=rc.dtype_tbl)
    ecl_df_sa_fmt = standarize_ecl_df(ecldata=results_['SPECIFIC_ASSESSMENT'],
                            param=param,
                            formatter_sheetname='ecl_deal_output',
                            dtype_tbl=rc.dtype_tbl)
    col_ca = ecl_df_ca_fmt.columns.tolist()
    col_proxy = ecl_df_proxy_fmt.columns.tolist()
    col_sa = ecl_df_sa_fmt.columns.tolist()

    print('------- Merging results ------')
    logger.info('Merging CA with PROXY ...')
    #CA: keep all ca cols
    # PROXY: keep common_cols
    # PROXY: keep selected proxy cols, including ecl results and collateral information
    common_cols = list(set(col_ca) & set(col_proxy)) 
    keep_uni_cols_proxy = ['SA_IND','PROXY_PORTFOLIO_ID', 'PROXY_PORTFOLIO_CODE', 
                           'PROXY_ECL_KEY', 'PROXY_ECL_PCT', 'PROXY_ECL_OCY', 'PROXY_ECL_LCY']
    ecl_df_proxy_fmt_ = ecl_df_proxy_fmt[common_cols+                             
                             keep_uni_cols_proxy]

    df_ca_ = ecl_df_ca_fmt.assign(ECL_ULTIMATE_OCY =
                            ecl_df_ca_fmt.ECL_FINAL_OCY,
                            ECL_ULTIMATE_LCY =
                            ecl_df_ca_fmt.ECL_FINAL_LCY,
                            )
    df_proxy_ = ecl_df_proxy_fmt_.assign(ECL_ULTIMATE_OCY =
                            ecl_df_proxy_fmt_.PROXY_ECL_OCY,
                            ECL_ULTIMATE_LCY =
                            ecl_df_proxy_fmt_.PROXY_ECL_LCY)
    
    df_merge = []
    try:
        df_merge = pd.merge(df_ca_, df_proxy_, how='outer')
    except Exception as e:
        print("Encountering {e} when merging ca and proxy ecl")
        logger.exception("message")
    else:
        logger.info('Merged CA with PROXY.')
    #len(df_ca_)+len(df_proxy_) == len(df_merge)
    
    logger.info('Merging with SA ...')
    # Merge with SA result
    # keep all cols
    df_sa_ = ecl_df_sa_fmt.assign(ECL_ULTIMATE_OCY =
                            ecl_df_sa_fmt.ECL_FINAL_OCY,
                            ECL_ULTIMATE_LCY =
                            ecl_df_sa_fmt.ECL_FINAL_LCY)

    df_merge_3 = []
    try:
        df_merge_3 = pd.merge(df_merge, df_sa_, how='outer')
    except Exception as e:
        print("Encountering {e} when merging sa with others")
        logger.exception("message")
    else:
        logger.info('Merged SA with CA and PROXY.')
    #len(df_sa_)+len(df_merge) == len(df_merge_all)

    logger.info("Merging with outscoped records...")
    # outscope
    # the first few lines of dp.run
    instr_df = (instr_df_raw.assign(
        DRAWN_BAL_OCY = instr_df_raw.PRIN_BAL_OCY
                        +instr_df_raw.ACRU_INT_OCY
                        +instr_df_raw.PENALTY_OCY 
                        +instr_df_raw.OTHER_FEE_AND_CHARGES_OCY))
    # 20241227: Update the excluded rules
    # Exclude MIK but as fallback, also exclude those
    # On balance with 0 drawn balance oct or off balance with 0 undrawn ocy
    out_scope_logic = ("SUB_CATEGORY.str.upper()=='MIK'"
                        + "or (ON_OFF_BAL_IND == 'ON' and DRAWN_BAL_OCY == 0)"
                        + "or (ON_OFF_BAL_IND == 'OFF' and UNDRAWN_BAL_OCY == 0)"
                        )

    out_scope_df_ = instr_df.query(out_scope_logic)
    out_scope_df_['ECL_APPROACH'] = 'OUTSCOPE'
    
    df_merge_all = []
    try:
        df_merge_all = pd.merge(df_merge_3, out_scope_df_, how='outer')
    except Exception as e:
        print("Encountering {e} when merging outscoped with others")
        logger.exception("message")
    else:
        logger.info('Merged outscoped with CA, PROXY, and SA.')

    # Adjust a bit the results presenting.
    col_at_end = ['ECL_ULTIMATE_OCY','ECL_ULTIMATE_LCY']
    df_merge_all = df_merge_all[[c for c in df_merge_all if c not in col_at_end] 
            + [c for c in col_at_end if c in df_merge_all]]
    
    if len(df_merge_all) == len(instr_df_raw):
        print('------- Generated full set of ECL results------')
        logger.info('Completed merging.')
        print('------- Exporting ECL result ------')
        logger.info('Exporting ECL result...')
        try:
            df_merge_all.to_csv(
            rc.resultPath/'ECL_calculation_result_files_deal_all.csv', index=False, encoding='utf-8-sig',quoting=csv.QUOTE_NONNUMERIC)
        except Exception as e:
            logger.exception("message")
        else:
            print('------- Exporting ECL final result complete------')
            logger.info("The process of output final ECL complete.")
    else:
        try:
            df_merge_all.to_csv(
            rc.resultPath/'ECL_calculation_result_files_deal_imcomplete.csv', index=False, encoding='utf-8-sig',quoting=csv.QUOTE_NONNUMERIC)
        except Exception as e:
            logger.exception("message")


if __name__ == '__main__':
    configPath = Path(
        r'C:\Users\WH947CH\Engagement\Khan Bank\03_ECL_engine\02_Development\khb_engine\run_config_file.json')
    c = load_configuration_file(configPath=configPath)
    rc = run_setting(run_config=c)

    result = run(rc)

    print(result.keys())