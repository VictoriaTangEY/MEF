from input_handler.load_parameters import load_configuration_file
from input_handler.env_setting import run_setting
from pathlib import Path
import pandas as pd
from datetime import datetime
from reporting import report_basic

import re


def calculate_amount_and_percent(context, param, report_temp, ecl_result):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return: 
    value_list
    percent_list
    """
    # load parameters
    map = param['EconSectorGroup']
    map_loan = param['LoanGroupFirst']
    # load report template
    tab_name = 'Concentration_risk_sector'
    # set condition
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    ecl_result_filter = rp_f.overall_filter(df_conditions, ecl_result)

    # calculate amount
    ecl_filtered = ecl_result_filter.filter(
        items=['IFRS_AMOUNT_LCY', 'ECON_SECTOR', 'SUB_CATEGORY'])
    combined_ecl_df_1 = pd.merge(
        ecl_filtered, map, on='ECON_SECTOR', how='left')
    combined_ecl_df_2 = pd.merge(
        combined_ecl_df_1, map_loan, on='SUB_CATEGORY', how='left')

    # Individual = Consumer loans + Agricultural loans
    individual_df = combined_ecl_df_2[combined_ecl_df_2['LOAN_FIRST_LAYER'].isin(
        ['Business loans', 'Agricultural loans'])]
    mik_df = combined_ecl_df_2[combined_ecl_df_2['SUB_CATEGORY']
                               == 'mik']
    combined_ecl_df = combined_ecl_df_2[combined_ecl_df_2['LOAN_FIRST_LAYER']
                                    == 'Business loans']

    Individuals_value = individual_df['IFRS_AMOUNT_LCY'].sum() + mik_df['IFRS_AMOUNT_LCY'].sum()
    Trade_and_commerce_value = combined_ecl_df[combined_ecl_df['SECTOR_GROUP']
                                               == 'Trade and commerce']['IFRS_AMOUNT_LCY'].sum()
    Construction_value = combined_ecl_df[combined_ecl_df['SECTOR_GROUP']
                                         == 'Construction']['IFRS_AMOUNT_LCY'].sum()
    Processing_value = combined_ecl_df[combined_ecl_df['SECTOR_GROUP']
                                       == 'Processing']['IFRS_AMOUNT_LCY'].sum()
    #Agriculture_value = combined_ecl_df[combined_ecl_df['SECTOR_GROUP']
    #                                    == 'Agriculture']['IFRS_AMOUNT_LCY'].sum()
    Small_private_enterprises_value = combined_ecl_df[combined_ecl_df['SECTOR_GROUP']
                                                      == 'Small private enterprises']['IFRS_AMOUNT_LCY'].sum()
    Transportation_value = combined_ecl_df[combined_ecl_df['SECTOR_GROUP']
                                           == 'Transportation']['IFRS_AMOUNT_LCY'].sum()
    Real_estate_value = combined_ecl_df[combined_ecl_df['SECTOR_GROUP']
                                        == 'Real estate']['IFRS_AMOUNT_LCY'].sum()
    Mining_value = combined_ecl_df[combined_ecl_df['SECTOR_GROUP']
                                   == 'Mining']['IFRS_AMOUNT_LCY'].sum()
    Health_and_social_organizations_value = combined_ecl_df[combined_ecl_df['SECTOR_GROUP']
                                                            == 'Health and social organizations']['IFRS_AMOUNT_LCY'].sum()
    Other_value = combined_ecl_df[combined_ecl_df['SECTOR_GROUP']
                                  == 'Other']['IFRS_AMOUNT_LCY'].sum()

    value_list = [Individuals_value, Trade_and_commerce_value, Construction_value, Processing_value,
                  Small_private_enterprises_value, Transportation_value, Real_estate_value, Mining_value, Health_and_social_organizations_value, Other_value]
    total_value = 0
    for i in value_list:
        total_value += i
    value_list.append(total_value)

    # calculate percentage
    percent_list = []
    for i in value_list:
        percent = i / total_value
        percent = rp_f.unit_represent_num(percent, df_conditions)
        percent_list.append(percent)

    return value_list, percent_list


def run_concentration_risk_sector(context, param, report_temp, ecl_result, wb):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return: wb: openpyxl.workbook
    """
    # load report template
    tab_name = 'Concentration_risk_sector'

    value_list, percent_list = calculate_amount_and_percent(
        context, param, report_temp, ecl_result[0])
    pre_value_list, pre_percent_list = calculate_amount_and_percent(
        context, param, report_temp, ecl_result[1])

    # write
    sheet = wb[tab_name]
    rp_f = report_basic.report_cond_basic(context=context)
    RUN_YYMM = rp_f.format_date_slash(context.run_yymm)
    PREV_YYMM = rp_f.format_date_slash(context.prev_yymm)
    sheet.merge_cells('B2:C2')
    sheet['B2'] = RUN_YYMM
    sheet.merge_cells('D2:E2')
    sheet['D2'] = PREV_YYMM
    
    for i in range(len(value_list)):
        sheet[f'B{i + 4}'] = value_list[i]
        sheet[f'C{i + 4}'] = percent_list[i]
        sheet[f'D{i + 4}'] = pre_value_list[i]
        sheet[f'E{i + 4}'] = pre_percent_list[i]

    return wb
