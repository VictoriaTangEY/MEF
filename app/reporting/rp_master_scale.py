from input_handler.load_parameters import load_configuration_file
from input_handler.env_setting import run_setting
from pathlib import Path
import pandas as pd
from reporting import report_basic

import re


def run_master_scale(context, param, report_temp, ecl_result, wb):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return: wb: openpyxl.workbook
    """
    # load parameters
    map = param['RatingGroup']
    # load report template
    tab_name = 'Master_scale'
    pop_cols = 'Corresponding PD interval'
    df_report_ = report_temp[tab_name].copy()
    # set condition
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    ecl_result_filter = rp_f.overall_filter(df_conditions, ecl_result)

    # match rating range
    ecl_filtered = ecl_result_filter.filter(
        items=['IFRS9_PD_12M_MADJ', 'CREDIT_RATING_CURRENT'])
    combined_ecl_df = pd.merge(
        ecl_filtered, map, on='CREDIT_RATING_CURRENT', how='left')

    # as there is empty row, to find the first and last valid row need to pop in
    start_idx = df_report_[df_report_.columns[0]].first_valid_index()
    end_idx = df_report_[df_report_.columns[0]].last_valid_index()
    # loop for all cell
    sheet = wb[tab_name]

    i = 1
    for idx in range(start_idx, end_idx+1):
        filtered_rows = combined_ecl_df[combined_ecl_df['RATING_GROUP']
                                        == str(i)]
        i += 1
        selected_values = filtered_rows['IFRS9_PD_12M_MADJ']
        if len(selected_values) > 1:
            min_value = min(selected_values)
            max_value = max(selected_values)
            min_percent = rp_f.unit_represent_num(min_value, df_conditions)
            max_percent = rp_f.unit_represent_num(max_value, df_conditions)
            sheet[f'C{idx + 2}'] = f"{min_percent} - {max_percent}"
        else:
            value_percent = rp_f.unit_represent_num(
                selected_values.values[0], df_conditions)
            sheet[f'C{idx + 2}'] = value_percent

    return wb
