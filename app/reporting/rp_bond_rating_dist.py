from input_handler.load_parameters import load_configuration_file
from input_handler.env_setting import run_setting
from pathlib import Path
import pandas as pd
from datetime import datetime
from reporting import report_basic

import re


def calculate_EAD(context, param, ecl_result):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return:
    pivot: calculated EAD dataframe
    """
    # load parameters
    map_rating = param['RatingGroup']

    # load report template
    tab_name = 'Bond_rating_dist'
    # set condition
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    ecl_result_filter = rp_f.overall_filter(df_conditions, ecl_result)

    # instr_sub_type selection
    ecl_filtered = ecl_result_filter.filter(items=['IFRS_AMOUNT_LCY', 'SUB_CATEGORY', 'INSTR_SUB_TYPE', 'CREDIT_RATING_CURRENT'])
    target_instr_types = [
    'Current account with BoM',
    'BoM teasury bills',
    'Due from banks_less than three months',
    'Government bonds']
    ecl_filtered = ecl_filtered[ecl_filtered['INSTR_SUB_TYPE'].isin(target_instr_types)].copy()

    # credit_rating mapping
    ecl_filtered = pd.merge(ecl_filtered, map_rating, on=['CREDIT_RATING_CURRENT'], how='left')
    ecl_filtered['RATING_GROUP'] = pd.to_numeric(ecl_filtered['RATING_GROUP'], errors='coerce').astype('Int64')
    ecl_filtered['IFRS_AMOUNT_LCY'] = pd.to_numeric(ecl_filtered['IFRS_AMOUNT_LCY'], errors='coerce')

    # combine rating 1 & 2
    ecl_filtered['RATING_combined'] = ecl_filtered['RATING_GROUP'].apply(lambda r: '1_2' if r in [1, 2] else str(int(r)) if pd.notna(r) else None)

    # pivot
    pivot_df = ecl_filtered.pivot_table(
        index='INSTR_SUB_TYPE',
        columns='RATING_combined',
        values='IFRS_AMOUNT_LCY',
        aggfunc='sum',
        fill_value=0
    )

    rating_rows = ['1_2', '3', '4', '5', '6', '7', '8']
    for col in rating_rows:
        if col not in pivot_df.columns:
            pivot_df[col] = 0
    pivot_df = pivot_df[rating_rows]

    current_with_bom = pivot_df.loc['Current account with BoM'].tolist()
    bom_bills = pivot_df.loc['BoM teasury bills'].tolist()
    due_from_banks = pivot_df.loc['Due from banks_less than three months'].tolist()
    gov_bonds = pivot_df.loc['Government bonds'].tolist()

    return current_with_bom, bom_bills, due_from_banks, gov_bonds


def run_bond_rating_dist(context, param, report_temp, ecl_result, wb):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return: wb: openpyxl.workbook
    """
    # load report template
    tab_name = 'Bond_rating_dist'
    current_with_bom, bom_bills, due_from_banks, gov_bonds = calculate_EAD(context, param, ecl_result)
    
    # write Excel
    sheet = wb[tab_name]
    start_row = 3

    for i in range(len(current_with_bom)):
        sheet[f'B{start_row + i}'] = current_with_bom[i]
        sheet[f'C{start_row + i}'] = bom_bills[i]
        sheet[f'D{start_row + i}'] = due_from_banks[i]
        sheet[f'E{start_row + i}'] = gov_bonds[i]

    # write sum
    for row in range(start_row, start_row+ len(current_with_bom)):
        sheet[f'F{row}'] = f'=SUM(B{row}:E{row})'

    for col_idx in range(2, 7): #from column B to F
        col_letter = chr(64 + col_idx)
        sheet[f'{col_letter}10'] = f'=SUM({col_letter}3:{col_letter}9)'
    
    return wb
