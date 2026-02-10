import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import csv

from input_handler.data_preprocessor import data_preprocessor
from input_handler.load_parameters import (load_configuration_file,
                                           load_parameters)
from input_handler.env_setting import run_setting
# TBD: Just for analysis purpose
from memory_profiler import profile


def convert_to_LCY(df: pd.DataFrame,
                   fx_tbl: pd.DataFrame,
                   param: Dict[str, pd.DataFrame]
                   ) -> pd.DataFrame:
    """
    ### convert_to_LCY

    To convert designated OCY columns into
    LCY values

    ### Parameters


    ### Return
    Pandas dataframe with converted columns added
    """
    # Exchange rate & Reporting Currency Field

    df_ocy = (df.merge(fx_tbl[['XTRT_DATE',
                               'CCY_CD_OCY',
                               'CCY_CD_RPT',
                               'FX_RT']],
                       how='left',
                       left_on=['CCY_CD', 'XTRT_DATE'],
                       right_on=['CCY_CD_OCY', 'XTRT_DATE']
                       )
              )

    field_prefix_list = param['ecl_fx_convert']['colname_prefix'].to_list()
    field_suffix_list = param['ecl_fx_convert']['converted_suffix'].to_list()

    for field_prefix, field_suffix in zip(field_prefix_list,
                                          field_suffix_list):

        if f'{field_prefix}_OCY' in df_ocy.columns:

            df_ocy[f'{field_prefix}_{field_suffix}'] = (df_ocy[f'{field_prefix}_OCY']
                                                        * df_ocy.FX_RT)

    return df_ocy


def prepare_output(df: pd.DataFrame,
                   param: Dict[str, pd.DataFrame],
                   formatter_sheetname: str,
                   dtype_tbl: Dict[str, str]):

   # df = outdata.copy() 250205 to save memory

    # Data Type casting and renaming
    controlParam = (param[formatter_sheetname]
                    .sort_values(by=['order']))

    rawKeepList = (controlParam.query("keep_ind == 'Y'")['colname']
                   .to_list())
    renameList = (controlParam.query("keep_ind == 'Y'")['colname_std']
                  .to_list())
    dtypeList = (controlParam.query("keep_ind == 'Y'")['data_type']
                 .to_list())

    dtype_mapping = {}
    rename_mapping = {}

    for var, dtype in zip(rawKeepList, dtypeList):
        dtype_mapping[var] = dtype_tbl[dtype]

    for rawVarName, stdVarName in zip(rawKeepList, renameList):
        rename_mapping[rawVarName] = stdVarName

    # Type Casting #250109 control output
    for key, val in dtype_mapping.items():
        df[key] = df[key].astype(val)
        if df[key].dtype == float:
            df[key] = df[key].round(8)
    # Rename columns
    return (df[rawKeepList].rename(columns=rename_mapping))


def partition_df(df, partition_key, n_partition=10):
    _df = df.copy()

    df_key = pd.DataFrame(df[partition_key].unique(), columns=[partition_key])

    # Case: If n_partition > number of records in the data
    n_partition_cap = min(df_key.shape[0], n_partition)

    df_idx = (df_key
              .assign(
                  df_part=pd.qcut(df_key.index, n_partition_cap, labels=[
                                  x + 1 for x in range(n_partition_cap)])
              )
              )

    _df_part = (_df.merge(df_idx, how='left',
                          left_on=[partition_key], right_on=[partition_key]
                          )
                .set_index(['df_part'])
                )

    df_parts = []

    for i in range(1, n_partition_cap + 1):
        df_parts.append(_df_part.loc[i])

    return (df_parts, df_idx)


@profile
def run(df: pd.DataFrame,
        calc_proc_df: pd.DataFrame,
        fx_df: pd.DataFrame,
        param: Dict[str, pd.DataFrame],
        dtype_tbl: Dict[str, str],
        calc_file_partition: int,
        resultPath: str,
        is_export: bool = False,
        ) -> pd.DataFrame:

    ecl_df_lcy = convert_to_LCY(df=df,
                                fx_tbl=fx_df,
                                param=param)

    ecl_df_fmt = prepare_output(df=ecl_df_lcy,
                                param=param,
                                formatter_sheetname='ecl_deal_output',
                                dtype_tbl=dtype_tbl)

    ecl_interim_df_fmt = prepare_output(df=calc_proc_df,
                                        param=param,
                                        formatter_sheetname='ecl_interim_output',
                                        dtype_tbl=dtype_tbl)

    ecl_interim_df_part, ecl_interim_indx = partition_df(df=ecl_interim_df_fmt,
                                                         partition_key='CONTRACT_ID',
                                                         n_partition=calc_file_partition)

    if is_export:
        ecl_df_fmt.to_csv(resultPath/'ECL_calculation_result_files_deal.csv',
                          index=False,
                          encoding='utf-8-sig',
                          quoting=csv.QUOTE_NONNUMERIC)

        for i, df_part in enumerate(ecl_interim_df_part):
            df_part.to_csv(resultPath /
                           f'ECL_risk_metrics_deal_part{i+1}.csv', index=False,
                           quoting=csv.QUOTE_NONNUMERIC)

        ecl_interim_indx.to_csv(
            resultPath / 'ECL_risk_metrics_deal_index.csv', index=False,
            quoting=csv.QUOTE_NONNUMERIC)

    return ecl_df_fmt, ecl_interim_df_fmt
