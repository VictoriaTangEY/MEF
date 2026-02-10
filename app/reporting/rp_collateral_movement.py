from pathlib import Path
import pandas as pd
import numpy as np
from reporting import report_basic

import re


def value_calculate(context, param, report_temp, ecl_result):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return:
    over_carring_value_list
    over_collateral_value_list
    under_carring_value_list
    under_carring_collateral_list
    """
    # load parameters
    map = param['LoanGroupFirst']
    instr = param['instrument_table']
    # load report template
    tab_name = 'Collateral_movement'
    # set condition
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    ecl_result_filter = rp_f.overall_filter(df_conditions, ecl_result)

    # TODO
    # select collateral col
    filtered_col_name = instr[instr['remark'].str.contains(
        'collateral value', na=False)]['colname_std'].tolist()
    # sum of collateral for each loan
    ecl_result_filter['COLLATERAL_SUM'] = ecl_result_filter[filtered_col_name].sum(
        axis=1)

    # merge
    ecl_filtered = ecl_result_filter.filter(
        items=['IFRS_AMOUNT_LCY', 'COLLATERAL_SUM', 'SUB_CATEGORY'])
    combined_ecl_df = pd.merge(
        ecl_filtered, map, on='SUB_CATEGORY', how='left')

    # segment
    over_collateralised_df = combined_ecl_df[combined_ecl_df['IFRS_AMOUNT_LCY']
                                             <= combined_ecl_df['COLLATERAL_SUM']]
    under_collateralised_df = combined_ecl_df[combined_ecl_df['IFRS_AMOUNT_LCY']
                                              >= combined_ecl_df['COLLATERAL_SUM']]

    over_business_df = over_collateralised_df[over_collateralised_df['LOAN_FIRST_LAYER']
                                              == 'Business loans']
    over_consumer_df = over_collateralised_df[over_collateralised_df['LOAN_FIRST_LAYER']
                                              == 'Consumer loans']
    over_agricultural_df = over_collateralised_df[over_collateralised_df['LOAN_FIRST_LAYER']
                                                  == 'Agricultural loans']
    under_business_df = under_collateralised_df[under_collateralised_df['LOAN_FIRST_LAYER']
                                                == 'Business loans']
    under_consumer_df = under_collateralised_df[under_collateralised_df['LOAN_FIRST_LAYER']
                                                == 'Consumer loans']
    under_agricultural_df = under_collateralised_df[under_collateralised_df['LOAN_FIRST_LAYER']
                                                    == 'Agricultural loans']

    # Carrying value of the assets
    over_business_carring_value = over_business_df['IFRS_AMOUNT_LCY'].sum()
    over_consumer_carring_value = over_consumer_df['IFRS_AMOUNT_LCY'].sum()
    over_agricultural_carring_value = over_agricultural_df['IFRS_AMOUNT_LCY'].sum()
    over_carring_value_list = [over_business_carring_value,
                               over_consumer_carring_value, over_agricultural_carring_value]

    under_business_carring_value = under_business_df['IFRS_AMOUNT_LCY'].sum()
    under_consumer_carring_value = under_consumer_df['IFRS_AMOUNT_LCY'].sum()
    under_agricultural_carring_value = under_agricultural_df['IFRS_AMOUNT_LCY'].sum()
    under_carring_value_list = [under_business_carring_value,
                                under_consumer_carring_value, under_agricultural_carring_value]

    # Value of collateral
    over_business_collateral_value = over_business_df['COLLATERAL_SUM'].sum()
    over_consumer_collateral_value = over_consumer_df['COLLATERAL_SUM'].sum()
    over_agricultural_collateral_value = over_agricultural_df['COLLATERAL_SUM'].sum(
    )
    over_collateral_value_list = [over_business_collateral_value,
                                  over_consumer_collateral_value, over_agricultural_collateral_value]

    under_business_collateral_value = under_business_df['COLLATERAL_SUM'].sum()
    under_consumer_collateral_value = under_consumer_df['COLLATERAL_SUM'].sum()
    under_agricultural_collateral_value = under_agricultural_df['COLLATERAL_SUM'].sum(
    )
    under_carring_collateral_list = [under_business_collateral_value,
                                     under_consumer_collateral_value, under_agricultural_collateral_value]

    return over_carring_value_list, over_collateral_value_list, under_carring_value_list, under_carring_collateral_list


def run_collateral_movement(context, param, report_temp, ecl_result, wb):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return: wb: openpyxl.workbook
    """
    # load report template
    tab_name = 'Collateral_movement'
    # set condition
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")

    # calculate
    result = value_calculate(context, param, report_temp, ecl_result[0])
    pre_result = value_calculate(context, param, report_temp, ecl_result[1])

    # write
    sheet = wb[tab_name]
    RUN_YYMM = rp_f.format_date_slash(context.run_yymm)
    PREV_YYMM = rp_f.format_date_slash(context.prev_yymm)
    sheet['A5'] = RUN_YYMM
    sheet['A10'] = PREV_YYMM
    
    for i in range(len(result[0])):
        for j in range(4):
            # number / 1000
            sheet.cell(row=i + 6, column=j +
                       2).value = rp_f.scaling_num(result[j][i], df_conditions)
            sheet.cell(row=i + 11, column=j +
                       2).value = rp_f.scaling_num(pre_result[j][i], df_conditions)

    return wb
