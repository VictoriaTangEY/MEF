from input_handler.load_parameters import load_configuration_file
from input_handler.env_setting import run_setting
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from reporting import report_basic
import re
import os
import csv
from util.loggers import createLogHandler
from input_handler.data_preprocessor import data_preprocessor
from scenario_engine import forward_looking_model as flm
from scenario_engine import pd_term_structure as pd_ts
from ecl_engine import collective_assessment as ca
from ecl_engine import output_file_handler as ofh
from ecl_engine import management_overlay as mo

# Assumed change rates
PD_change_rate = 0.1 # 10% increase
LGD_change_rate = 0.1 # 10% increase
FL_change_rate = 0.1 # 10% increase in SEVE

def calculate_ecl(ecl_result_filtered):
    # applied to STAGE_FINAL = 1, value = max(ECL_ENGINE_LT_LCY, ECL_ENGINE_LCY) - ECL_ENGINE_LCY
    stage_1_data = ecl_result_filtered[ecl_result_filtered['STAGE_FINAL'] == 1].copy()
    required_cols = ['ECL_ENGINE_LT_LCY', 'ECL_ENGINE_LCY']
    stage_1_data['MAX_ECL'] = stage_1_data[['ECL_ENGINE_LT_LCY', 'ECL_ENGINE_LCY']].max(axis=1)
    stage_1_data['DIFF'] = stage_1_data['MAX_ECL'] - stage_1_data['ECL_ENGINE_LCY']
    
    total_diff = stage_1_data['DIFF'].sum()
    value_thousands = total_diff / 1000.0
    
    return value_thousands


def calculate_pd(ecl_result_filtered):
    # New scenario PD = min(1.1 * original PD, 100%)
    # ECL_new = ECL_ULTIMATE_LCY * (PD_new / PD_orig)
    # value = ECL_new - ECL_ULTIMATE_LCY
    required_cols = ['STAGE_FINAL', 'ECL_ULTIMATE_LCY']
    df = ecl_result_filtered.copy()
    
    pd_col_12m = 'IFRS9_PD_12M_MADJ'
    pd_col_lt = 'IFRS9_PD_LT'
    
    for col in [pd_col_12m, pd_col_lt]:
        if col not in df.columns:
            df[col] = np.nan
    
    df['ECL_NEW'] = df['ECL_ULTIMATE_LCY'].copy()
    
    # stage 1：using IFRS9_PD_12M_MADJ
    stage1_mask = (df['STAGE_FINAL'] == 1) & df[pd_col_12m].notna() & (df[pd_col_12m] > 0)
    if stage1_mask.any():
        pd_12m_orig = df.loc[stage1_mask, pd_col_12m]
        pd_12m_new = np.minimum((1 + PD_change_rate) * pd_12m_orig, 1.0)
        df.loc[stage1_mask, 'ECL_NEW'] = (
            df.loc[stage1_mask, 'ECL_ULTIMATE_LCY'] * (pd_12m_new / pd_12m_orig)
        )
    
    # stage 2/3: using IFRS9_PD_LT
    stage23_mask = (df['STAGE_FINAL'].isin([2, 3])) & df[pd_col_lt].notna() & (df[pd_col_lt] > 0)
    if stage23_mask.any():
        pd_lt_orig = df.loc[stage23_mask, pd_col_lt]
        pd_lt_new = np.minimum((1 + PD_change_rate) * pd_lt_orig, 1.0)
        df.loc[stage23_mask, 'ECL_NEW'] = (
            df.loc[stage23_mask, 'ECL_ULTIMATE_LCY'] * (pd_lt_new / pd_lt_orig)
        )
    
    df['ECL_DIFF'] = df['ECL_NEW'] - df['ECL_ULTIMATE_LCY']
    total_diff = df['ECL_DIFF'].sum()
    value_thousands = total_diff / 1000.0
    
    return value_thousands


def calculate_lgd(ecl_result_filtered):
    # New scenario LGD = min(1.1 * original LGD, 100%)
    # ECL_new = ECL_ULTIMATE_LCY * (LGD_new / LGD_orig)
    # value = ECL_new - ECL_ULTIMATE_LCY
    required_cols = ['STAGE_FINAL', 'ECL_ULTIMATE_LCY']
    df = ecl_result_filtered.copy()
    
    lgd_col_12m = 'IFRS9_LGD_12M'
    lgd_col_lt = 'IFRS9_LGD_LT'
    lgd_col_sa = 'LGD'
    
    for col in [lgd_col_12m, lgd_col_lt, lgd_col_sa]:
        if col not in df.columns:
            df[col] = np.nan
    
    df['ECL_NEW'] = df['ECL_ULTIMATE_LCY'].copy()
    
    # stage 1：using IFRS9_LGD_12M_MADJ or LGD from SA    
    stage1_mask = (df['STAGE_FINAL'] == 1) & (
        (df[lgd_col_12m].notna() & (df[lgd_col_12m] > 0)) | 
        (df[lgd_col_sa].notna() & (df[lgd_col_sa] > 0))
    )
    if stage1_mask.any():
        lgd_12m_orig = df.loc[stage1_mask, lgd_col_12m].fillna(df.loc[stage1_mask, lgd_col_sa])
        lgd_12m_new = np.minimum((1 + LGD_change_rate) * lgd_12m_orig, 1.0)
        df.loc[stage1_mask, 'ECL_NEW'] = (
            df.loc[stage1_mask, 'ECL_ULTIMATE_LCY'] * (lgd_12m_new / lgd_12m_orig)
        )
    
    # stage 2/3：using IFRS9_LGD_LT or LGD from SA
    stage23_mask = (df['STAGE_FINAL'].isin([2, 3])) & (
        (df[lgd_col_lt].notna() & (df[lgd_col_lt] > 0)) | 
        (df[lgd_col_sa].notna() & (df[lgd_col_sa] > 0))
    )
    if stage23_mask.any():
        lgd_lt_orig = df.loc[stage23_mask, lgd_col_lt].fillna(df.loc[stage23_mask, lgd_col_sa])
        lgd_lt_new = np.minimum((1 + LGD_change_rate) * lgd_lt_orig, 1.0)
        df.loc[stage23_mask, 'ECL_NEW'] = (
            df.loc[stage23_mask, 'ECL_ULTIMATE_LCY'] * (lgd_lt_new / lgd_lt_orig)
        )
    
    df['ECL_DIFF'] = df['ECL_NEW'] - df['ECL_ULTIMATE_LCY']
    total_diff = df['ECL_DIFF'].sum()
    value_thousands = total_diff / 1000.0
    
    return value_thousands


def run_commentary(context, param, report_temp, ecl_result, ecl_result_prev, wb, parent_logger=None):
    """
    context: environment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv (current period)
    ecl_result_prev: ECL_calculation_result_files_deal.csv (previous period)
    
    return: wb: openpyxl.workbook
    """
    if parent_logger:
        logger = parent_logger
    else:
        logger = createLogHandler('reporting', context.logPath / 'Log_file_reporting.log')
    
    # load reporting template
    tab_name = 'Commentary'
    sheet = wb[tab_name]

    rp_f = report_basic.report_cond_basic(context=context)
    RUN_YYMM = rp_f.format_date_slash(context.run_yymm)
    PREV_YYMM = rp_f.format_date_slash(context.prev_yymm)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")

    # load ecl results
    ecl_result_filter_prev = rp_f.overall_filter(df_conditions, ecl_result_prev)
    ecl_result_filter = rp_f.overall_filter(df_conditions, ecl_result)
    
    # row 1: ECL
    value_thousands = calculate_ecl(ecl_result_filter)
    last_reporting_value_thousands = calculate_ecl(ecl_result_filter_prev)
    
    direction_curr = 'higher' if value_thousands >= 0 else 'lower'
    direction_prev = 'higher' if last_reporting_value_thousands >= 0 else 'lower'
    
    value_thousands_abs = abs(value_thousands)
    last_reporting_value_thousands_abs = abs(last_reporting_value_thousands)
    
    sentence = (f"Should ECL on all loans and advances to customers be measured at lifetime ECL "
                f"(that is, including those that are currently in Stage 1 measured at 12- months ECL), "
                f"the expected credit loss allowance would be {direction_curr} by MNT {value_thousands_abs:,.0f} "
                f"as of {RUN_YYMM} ({PREV_YYMM}: {direction_prev} by MNT {last_reporting_value_thousands_abs:,.0f}).")
    sheet['B1'] = sentence
    
    # row 2: PD
    pd_value_thousands = calculate_pd(ecl_result_filter)
    pd_last_reporting_value_thousands = calculate_pd(ecl_result_filter_prev)
    
    pd_direction_curr = 'increase' if pd_value_thousands >= 0 else 'decrease'
    pd_direction_prev = 'increase' if pd_last_reporting_value_thousands >= 0 else 'decrease'
    
    pd_value_thousands_abs = abs(pd_value_thousands)
    pd_last_reporting_value_thousands_abs = abs(pd_last_reporting_value_thousands)
    
    pd_change_pct = abs(PD_change_rate) * 100
    pd_change_text = f"{pd_change_pct:.0f}% {'increase' if PD_change_rate > 0 else 'decrease'}"
    
    sentence_pd = (f"A {pd_change_text} in PD estimates would result in an {pd_direction_curr} "
                   f"in total expected credit loss allowances of MNT {pd_value_thousands_abs:,.0f} "
                   f"at {RUN_YYMM} ({PREV_YYMM}: {pd_direction_prev} of MNT {pd_last_reporting_value_thousands_abs:,.0f}).")
    sheet['B2'] = sentence_pd
    
    # row 3: LGD
    lgd_value_thousands = calculate_lgd(ecl_result_filter)
    lgd_last_reporting_value_thousands = calculate_lgd(ecl_result_filter_prev)
    
    lgd_direction_curr = 'increase' if lgd_value_thousands >= 0 else 'decrease'
    lgd_direction_prev = 'increase' if lgd_last_reporting_value_thousands >= 0 else 'decrease'
    
    lgd_value_thousands_abs = abs(lgd_value_thousands)
    lgd_last_reporting_value_thousands_abs = abs(lgd_last_reporting_value_thousands)
    
    lgd_change_pct = abs(LGD_change_rate) * 100
    lgd_change_text = f"{lgd_change_pct:.0f}% {'increase' if LGD_change_rate > 0 else 'decrease'}"
    
    sentence_lgd = (f"A {lgd_change_text} in LGD estimates would result in an {lgd_direction_curr} "
                    f"in total expected credit loss allowances of MNT {lgd_value_thousands_abs:,.0f} "
                    f"at {RUN_YYMM} ({PREV_YYMM}: {lgd_direction_prev} of MNT {lgd_last_reporting_value_thousands_abs:,.0f}).")
    sheet['B3'] = sentence_lgd
    
    return wb
