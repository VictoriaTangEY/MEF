from pathlib import Path
import pandas as pd
import numpy as np
from reporting import report_basic

import re


def sum_value(context, param, result_df):
    # load parameters
    map = param['OffBalance']
    # set condition
    tab_name = 'Asset_movement_disclosure'
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    df_filter = rp_f.overall_filter(df_conditions, result_df)

    # select category
    ecl_filtered = df_filter.filter(
        items=['IFRS_AMOUNT_LCY', 'ECL_ULTIMATE_LCY', 'INSTR_TYPE'])
    combined_ecl_df = pd.merge(
        ecl_filtered, map, on='INSTR_TYPE', how='left')

    guarantees_issued_df = combined_ecl_df[combined_ecl_df['OFF_BAL_GROUP']
                                           == 'Guarantees issued']
    letter_credit_df = combined_ecl_df[combined_ecl_df['OFF_BAL_GROUP']
                                       == 'Import letters of credit']
    credit_line_df = combined_ecl_df[combined_ecl_df['OFF_BAL_GROUP']
                                     == 'Undrawn credit lines']
    credit_card_df = combined_ecl_df[combined_ecl_df['OFF_BAL_GROUP']
                                     == 'Undrawn credit card']
    factoring_df = combined_ecl_df[combined_ecl_df['OFF_BAL_GROUP']
                                   == 'Factoring receivable']

    # Gross carrying amount
    GCA_guarantees_sum = guarantees_issued_df['IFRS_AMOUNT_LCY'].sum()
    GCA_letter_sum = letter_credit_df['IFRS_AMOUNT_LCY'].sum()
    GCA_line_sum = credit_line_df['IFRS_AMOUNT_LCY'].sum()
    GCA_card_sum = credit_card_df['IFRS_AMOUNT_LCY'].sum()
    GCA_factoring_sum = factoring_df['IFRS_AMOUNT_LCY'].sum()
    GCA_total = GCA_guarantees_sum + GCA_letter_sum + \
        GCA_line_sum + GCA_card_sum + GCA_factoring_sum

    # Credit loss  allowance
    CLA_guarantees_sum = -1 * guarantees_issued_df['ECL_ULTIMATE_LCY'].sum()
    CLA_letter_sum = -1 * letter_credit_df['ECL_ULTIMATE_LCY'].sum()
    CLA_line_sum = -1 * credit_line_df['ECL_ULTIMATE_LCY'].sum()
    CLA_card_sum = -1 * credit_card_df['ECL_ULTIMATE_LCY'].sum()
    CLA_factoring_sum = -1 * factoring_df['ECL_ULTIMATE_LCY'].sum()
    CLA_total = CLA_guarantees_sum + CLA_letter_sum + \
        CLA_line_sum + CLA_card_sum + CLA_factoring_sum

    # Carrying amount
    CA_guarantees_sum = GCA_guarantees_sum + CLA_guarantees_sum
    CA_letter_sum = GCA_letter_sum + CLA_letter_sum
    CA_line_sum = GCA_line_sum + CLA_line_sum
    CA_card_sum = GCA_card_sum + CLA_card_sum
    CA_factoring_sum = GCA_factoring_sum + CLA_factoring_sum
    CA_total = GCA_total + CLA_total

    GCA_list = [GCA_guarantees_sum, GCA_letter_sum,
                GCA_line_sum, GCA_card_sum, GCA_factoring_sum, GCA_total]
    CLA_list = [CLA_guarantees_sum, CLA_letter_sum,
                CLA_line_sum, CLA_card_sum, CLA_factoring_sum, CLA_total]
    CA_list = [CA_guarantees_sum, CA_letter_sum,
               CA_line_sum, CA_card_sum, CA_factoring_sum, CA_total]
    
    for value in GCA_list:
        value = rp_f.scaling_num(value, df_conditions)
    for value in CLA_list:
        value = rp_f.scaling_num(value, df_conditions)
    for value in CA_list:
        value = rp_f.scaling_num(value, df_conditions)

    return GCA_list, CLA_list, CA_list

def sum_onbal_value(context, param, result_df):
    # load parameters
    map = param['LoanGroupFirst']
    # set condition
    tab_name = 'Asset_movement_disclosure'
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    df_filter = result_df.query("ON_OFF_BAL_IND == 'ON'")

    # select category
    ecl_filtered = df_filter.filter(
        items=['IFRS_AMOUNT_LCY', 'ECL_ULTIMATE_LCY', 'SUB_CATEGORY'])
    combined_ecl_df = pd.merge(
        ecl_filtered, map, on='SUB_CATEGORY', how='left')

    business_df = combined_ecl_df[combined_ecl_df['LOAN_FIRST_LAYER']
                                  == 'Business loans']
    consumer_df = combined_ecl_df[combined_ecl_df['LOAN_FIRST_LAYER']
                                  == 'Consumer loans']
    agricultural_df = combined_ecl_df[combined_ecl_df['LOAN_FIRST_LAYER']
                                      == 'Agricultural loans']

    # Gross carrying amount
    GCA_business_sum = business_df['IFRS_AMOUNT_LCY'].sum()
    GCA_consumer_sum = consumer_df['IFRS_AMOUNT_LCY'].sum()
    GCA_agricultural_sum = agricultural_df['IFRS_AMOUNT_LCY'].sum()
    GCA_total = GCA_business_sum + GCA_consumer_sum + GCA_agricultural_sum

    # Credit loss  allowance
    CLA_business_sum = -1 * business_df['ECL_ULTIMATE_LCY'].sum()
    CLA_consumer_sum = -1 * consumer_df['ECL_ULTIMATE_LCY'].sum()
    CLA_agricultural_sum = -1 * agricultural_df['ECL_ULTIMATE_LCY'].sum()
    CLA_total = CLA_business_sum + CLA_consumer_sum + CLA_agricultural_sum

    # Carrying amount
    CA_business_sum = GCA_business_sum + CLA_business_sum
    CA_consumer_sum = GCA_consumer_sum + CLA_consumer_sum
    CA_agricultural_sum = GCA_agricultural_sum + CLA_agricultural_sum
    CA_total = GCA_total + CLA_total

    GCA_list = [GCA_business_sum, GCA_consumer_sum, GCA_agricultural_sum, GCA_total]
    CLA_list = [CLA_business_sum, CLA_consumer_sum, CLA_agricultural_sum, CLA_total]
    CA_list = [CA_business_sum, CA_consumer_sum, CA_agricultural_sum, CA_total]
    
    for value in GCA_list:
        value = rp_f.scaling_num(value, df_conditions)
    for value in CLA_list:
        value = rp_f.scaling_num(value, df_conditions)
    for value in CA_list:
        value = rp_f.scaling_num(value, df_conditions)

    return GCA_list, CLA_list, CA_list


def sum_mik_value(context, param, result_df):
    tab_name = 'Asset_movement_disclosure'
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    
    df_filter = result_df[
        (result_df['DATA_SOURCE_CD'] == 'LOAN') &
        (result_df['ON_OFF_BAL_IND'] == 'ON') &
        (result_df['SUB_CATEGORY'] == 'mik')
    ]
    
    GCA_mik_sum = df_filter['IFRS_AMOUNT_LCY'].sum()
    GCA_mik_scaled = rp_f.scaling_num(GCA_mik_sum, df_conditions)
    
    return GCA_mik_scaled


def run_asset_movement_disclosure(context, param, report_temp, ecl_result, wb):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return: wb: openpyxl.workbook
    """
    # load report template
    tab_name = 'Asset_movement_disclosure'

    # create offbal value list
    GCA_list, CLA_list, CA_list = sum_value(context, param, ecl_result[0])
    pre_GCA_list, pre_CLA_list, pre_CA_list = sum_value(
        context, param, ecl_result[1])
    
    # create onbal value list
    GCA_onbal_list, CLA_onbal_list, CA_onbal_list = sum_onbal_value(context, param, ecl_result[0])
    pre_GCA_onbal_list, pre_CLA_onbal_list, pre_CA_onbal_list = sum_onbal_value(
        context, param, ecl_result[1])
    
    mik_gca = sum_mik_value(context, param, ecl_result[0])
    pre_mik_gca = sum_mik_value(context, param, ecl_result[1])

    # fill in the offbal value
    sheet = wb[tab_name]
    rp_f = report_basic.report_cond_basic(context=context)
    RUN_YYMM = rp_f.format_date_slash(context.run_yymm)
    PREV_YYMM = rp_f.format_date_slash(context.prev_yymm)
    sheet.merge_cells('B2:D2')
    sheet['B2'] = RUN_YYMM
    sheet.merge_cells('E2:G2')
    sheet['E2'] = PREV_YYMM
    sheet['E16'] = RUN_YYMM
    sheet['F16'] = PREV_YYMM
    sheet.merge_cells('B24:E24')
    sheet['B24'] = RUN_YYMM
    sheet.merge_cells('F24:H24')
    sheet['F24'] = PREV_YYMM
    
    if 'Collateral_info' in wb.sheetnames:
        collateral_sheet = wb['Collateral_info']
        collateral_sheet['A1'] = f'Description of collateral held for loans to customers carried at amortised cost is as follows at {RUN_YYMM}:'
        collateral_sheet['A18'] = f'Description of collateral held for loans to customers carried at FVTPL is as follows at {RUN_YYMM}:'
    
    for i in range(len(GCA_list)):
        sheet[f'B{i + 5}'] = GCA_list[i]
        sheet[f'C{i + 5}'] = CLA_list[i]
        sheet[f'D{i + 5}'] = CA_list[i]
        sheet[f'E{i + 5}'] = pre_GCA_list[i]
        sheet[f'F{i + 5}'] = pre_CLA_list[i]
        sheet[f'G{i + 5}'] = pre_CA_list[i]
    
    # fill in the onbal value
    for i in range(len(GCA_onbal_list)):
        sheet[f'B{i + 26}'] = GCA_onbal_list[i]
        sheet[f'C{i + 26}'] = CLA_onbal_list[i]
        sheet[f'D{i + 26}'] = CA_onbal_list[i]
        sheet[f'E{i + 26}'] = pre_GCA_onbal_list[i]
        sheet[f'F{i + 26}'] = pre_CLA_onbal_list[i]
        sheet[f'G{i + 26}'] = pre_CA_onbal_list[i]
    sheet['E18'] = GCA_onbal_list[-1]
    sheet['F18'] = pre_GCA_onbal_list[-1]
    sheet['E19'] = CLA_onbal_list[-1]
    sheet['F19'] = pre_CLA_onbal_list[-1]
    sheet['E20'] = CA_onbal_list[-1]
    sheet['F20'] = pre_CA_onbal_list[-1]
    
    sheet['E21'] = mik_gca
    sheet['F21'] = pre_mik_gca
    sheet['E22'] = '=E21+E20'
    sheet['F22'] = '=F21+F20'

    return wb
