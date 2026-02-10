from input_handler.load_parameters import load_configuration_file
from input_handler.env_setting import run_setting
from pathlib import Path
import pandas as pd
from datetime import datetime
from reporting import report_basic
import re
import numpy as np


def create_last_pd_used(df):
    """
    Create LAST_PD_USED column based on STAGE_FINAL:
    - Stage 1: use IFRS9_PD_12M_MADJ
    - Stage 2, 3: use IFRS9_PD_LT
    """
    df = df.copy()
    df['LAST_PD_USED'] = np.nan
    stage1_mask = df['STAGE_FINAL'] == 1
    stage23_mask = df['STAGE_FINAL'].isin([2, 3])
    
    df.loc[stage1_mask, 'LAST_PD_USED'] = df.loc[stage1_mask, 'IFRS9_PD_12M_MADJ']
    df.loc[stage23_mask, 'LAST_PD_USED'] = df.loc[stage23_mask, 'IFRS9_PD_LT']
    
    return df


def assign_pd_by_approach(df):
    if df is None or df.empty:
        return df
    required_columns = {'LAST_PD_USED', 'ECL_APPROACH', 'STAGE_FINAL'}
    if not required_columns.issubset(df.columns):
        return df
    df = df.copy()
    pd_col = 'LAST_PD_USED'
    approach_col = 'ECL_APPROACH'
    stage_col = 'STAGE_FINAL'
    lgd_col = 'IFRS9_LGD_12M'
    ecl_pct_col = 'PROXY_ECL_PCT'

    # SA
    scope_mask = df[approach_col].isin(['SA_OUTSCOPE', 'SPECIFIC_ASSESSMENT'])
    df.loc[scope_mask, pd_col] = 1.0

    # Proxy
    if lgd_col in df.columns and ecl_pct_col in df.columns:
        stage1_mask = (df[stage_col] == 1) & df[pd_col].notna() & (df[pd_col] > 0)
        stage2_mask = (df[stage_col] == 2) & df[pd_col].notna() & (df[pd_col] > 0)
        ratio_stage1 = np.nan
        ratio_stage2 = np.nan
        if stage1_mask.any():
            ratio_stage1 = (df.loc[stage1_mask, lgd_col] / df.loc[stage1_mask, pd_col]).mean()
        if stage2_mask.any():
            ratio_stage2 = (df.loc[stage2_mask, lgd_col] / df.loc[stage2_mask, pd_col]).mean()

        # Proxy - Stage 1: PD = sqrt(ECL% / ratio between LGD & PD)
        proxy_stage1_mask = (
            (df[approach_col] == 'PROXY') &
            (df[stage_col] == 1) &
            ((df[pd_col].isna()) | (df[pd_col] == 0))
        )
        if proxy_stage1_mask.any() and np.isfinite(ratio_stage1) and ratio_stage1 > 0:
            df.loc[proxy_stage1_mask, pd_col] = np.sqrt(
                df.loc[proxy_stage1_mask, ecl_pct_col].clip(lower=0) / ratio_stage1
            )

        # Proxy - Stage 2: PD = sqrt(ECL% / ratio between LGD & PD)
        proxy_stage2_mask = (
            (df[approach_col] == 'PROXY') &
            (df[stage_col] == 2) &
            ((df[pd_col].isna()) | (df[pd_col] == 0))
        )
        if proxy_stage2_mask.any() and np.isfinite(ratio_stage2) and ratio_stage2 > 0:
            df.loc[proxy_stage2_mask, pd_col] = np.sqrt(
                df.loc[proxy_stage2_mask, ecl_pct_col].clip(lower=0) / ratio_stage2
            )

        # Proxy - Stage 3: PD = 100%
        proxy_stage3_mask = (df[approach_col] == 'PROXY') & (df[stage_col] == 3)
        df.loc[proxy_stage3_mask, pd_col] = 1.0

    return df


def calculate_EAD(context, param, report_temp, ecl_result, bins, labels, name, right=False):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv
    bins: range bins for grouping
    labels: range labels for grouping
    name: column name for grouping

    return: 
    business_value_list
    consumer_value_list
    agricultural_value_list
    total_value_list

    """
    # load parameters
    map_loan = param['LoanGroupFirst']
    # load report template
    tab_name = 'PD_Range_disclosure'
    # set condition
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    ecl_result_filter = rp_f.overall_filter(df_conditions, ecl_result)

    # calculate amount
    ecl_filtered = ecl_result_filter.filter(
        items=['IFRS_AMOUNT_LCY', name, 'SUB_CATEGORY'])
    combined_ecl_df = pd.merge(
        ecl_filtered, map_loan, on='SUB_CATEGORY', how='left')

    # group
    combined_ecl_df['RANGE_GROUP'] = pd.cut(
        combined_ecl_df[name], bins=bins, labels=labels, right=right)

    business_df = combined_ecl_df[combined_ecl_df['LOAN_FIRST_LAYER']
                                  == 'Business loans']
    consumer_df = combined_ecl_df[combined_ecl_df['LOAN_FIRST_LAYER']
                                  == 'Consumer loans']
    agricultural_df = combined_ecl_df[combined_ecl_df['LOAN_FIRST_LAYER']
                                      == 'Agricultural loans']

    # loop for all range
    business_value_list = []
    consumer_value_list = []
    agricultural_value_list = []
    total_value_list = []
    business_total_value = 0
    consumer_total_value = 0
    agricultural_total_value = 0
    total_total_value = 0
    for i in labels:
        business_value = business_df[business_df['RANGE_GROUP'] == i]['IFRS_AMOUNT_LCY'].sum(
        )
        business_value = rp_f.scaling_num(business_value, df_conditions)
        business_value_list.append(business_value)
        business_total_value += business_value

        consumer_value = consumer_df[consumer_df['RANGE_GROUP'] == i]['IFRS_AMOUNT_LCY'].sum(
        )
        consumer_value = rp_f.scaling_num(consumer_value, df_conditions)
        consumer_value_list.append(consumer_value)
        consumer_total_value += consumer_value

        agricultural_value = agricultural_df[agricultural_df['RANGE_GROUP'] == i]['IFRS_AMOUNT_LCY'].sum(
        )
        agricultural_value = rp_f.scaling_num(
            agricultural_value, df_conditions)
        agricultural_value_list.append(agricultural_value)
        agricultural_total_value += agricultural_value

        total_value = business_value + consumer_value + agricultural_value
        total_value_list.append(total_value)
        total_total_value += total_value
    business_value_list.append(business_total_value)
    consumer_value_list.append(consumer_total_value)
    agricultural_value_list.append(agricultural_total_value)
    total_value_list.append(total_total_value)

    return business_value_list, consumer_value_list, agricultural_value_list, total_value_list


def calculate_offbal_value(context, param, report_temp, ecl_result, bins, labels):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv
    bins: range bins for grouping
    labels: range labels for grouping

    return: 
    credit_line_list
    credit_card_list
    guarantees_issued_list
    letter_credit_list
    factoring_list
    total_value_list
    """
    # load parameters
    map = param['OffBalance']
    # load report template
    tab_name = 'PD_Range_disclosure'
    # set condition
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    ecl_result_filter = rp_f.overall_filter(df_conditions, ecl_result)

    # calculate amount
    ecl_filtered = ecl_result_filter.filter(
        items=['IFRS_AMOUNT_LCY', 'LAST_PD_USED', 'INSTR_TYPE'])
    combined_ecl_df = pd.merge(
        ecl_filtered, map, on='INSTR_TYPE', how='left')

    # group by PD range
    combined_ecl_df['RANGE_GROUP'] = pd.cut(
        combined_ecl_df['LAST_PD_USED'], bins=bins, labels=labels, right=False)

    credit_line_df = combined_ecl_df[combined_ecl_df['OFF_BAL_GROUP']
                                     == 'Undrawn credit lines']
    credit_card_df = combined_ecl_df[combined_ecl_df['OFF_BAL_GROUP']
                                     == 'Undrawn credit card']
    guarantees_issued_df = combined_ecl_df[combined_ecl_df['OFF_BAL_GROUP']
                                           == 'Guarantees issued']
    letter_credit_df = combined_ecl_df[combined_ecl_df['OFF_BAL_GROUP']
                                       == 'Import letters of credit']
    factoring_df = combined_ecl_df[combined_ecl_df['OFF_BAL_GROUP']
                                   == 'Factoring receivable']

    # loop for all range
    credit_line_list = []
    credit_card_list = []
    guarantees_issued_list = []
    letter_credit_list = []
    factoring_list = []
    total_value_list = []
    credit_line_total_value = 0
    credit_card_total_value = 0
    guarantees_issued_total_value = 0
    letter_credit_total_value = 0
    factoring_total_value = 0
    total_total_value = 0
    for i in labels:
        credit_line_value = credit_line_df[credit_line_df['RANGE_GROUP'] == i]['IFRS_AMOUNT_LCY'].sum(
        )
        credit_line_value = rp_f.scaling_num(credit_line_value, df_conditions)
        credit_line_list.append(credit_line_value)
        credit_line_total_value += credit_line_value

        credit_card_value = credit_card_df[credit_card_df['RANGE_GROUP'] == i]['IFRS_AMOUNT_LCY'].sum(
        )
        credit_card_value = rp_f.scaling_num(credit_card_value, df_conditions)
        credit_card_list.append(credit_card_value)
        credit_card_total_value += credit_card_value

        guarantees_issued_value = guarantees_issued_df[guarantees_issued_df['RANGE_GROUP'] == i]['IFRS_AMOUNT_LCY'].sum(
        )
        guarantees_issued_value = rp_f.scaling_num(
            guarantees_issued_value, df_conditions)
        guarantees_issued_list.append(guarantees_issued_value)
        guarantees_issued_total_value += guarantees_issued_value

        letter_credit_value = letter_credit_df[letter_credit_df['RANGE_GROUP'] == i]['IFRS_AMOUNT_LCY'].sum(
        )
        letter_credit_value = rp_f.scaling_num(
            letter_credit_value, df_conditions)
        letter_credit_list.append(letter_credit_value)
        letter_credit_total_value += letter_credit_value

        factoring_value = factoring_df[factoring_df['RANGE_GROUP'] == i]['IFRS_AMOUNT_LCY'].sum(
        )
        factoring_value = rp_f.scaling_num(
            factoring_value, df_conditions)
        factoring_list.append(factoring_value)
        factoring_total_value += factoring_value

        total_value = credit_line_value + credit_card_value + \
            guarantees_issued_value + letter_credit_value + factoring_value
        total_value_list.append(total_value)
        total_total_value += total_value
    credit_line_list.append(credit_line_total_value)
    credit_card_list.append(credit_card_total_value)
    guarantees_issued_list.append(guarantees_issued_total_value)
    letter_credit_list.append(letter_credit_total_value)
    factoring_list.append(factoring_total_value)
    total_value_list.append(total_total_value)

    return credit_line_list, credit_card_list, guarantees_issued_list, letter_credit_list, factoring_list, total_value_list


def run_pd_range_disclosure(context, param, report_temp, ecl_result, wb):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return: wb: openpyxl.workbook
    """
    # load report template
    tab_name = 'PD_Range_disclosure'
    df_report_ = report_temp[tab_name].copy()

    onbal_df = assign_pd_by_approach(
        create_last_pd_used(ecl_result[0][ecl_result[0]['ON_OFF_BAL_IND'] == 'ON']))
    pre_onbal_df = assign_pd_by_approach(
        create_last_pd_used(ecl_result[1][ecl_result[1]['ON_OFF_BAL_IND'] == 'ON']))
    offbal_df = assign_pd_by_approach(
        create_last_pd_used(ecl_result[0][ecl_result[0]['ON_OFF_BAL_IND'] == 'OFF']))
    pre_offbal_df = assign_pd_by_approach(
        create_last_pd_used(ecl_result[1][ecl_result[1]['ON_OFF_BAL_IND'] == 'OFF']))

    onbal_stage1_df = onbal_df[onbal_df['STAGE_FINAL'] == 1]
    pre_onbal_stage1_df = pre_onbal_df[pre_onbal_df['STAGE_FINAL'] == 1]
    onbal_stage2_df = onbal_df[onbal_df['STAGE_FINAL'] == 2]
    pre_onbal_stage2_df = pre_onbal_df[pre_onbal_df['STAGE_FINAL'] == 2]
    onbal_stage3_df = onbal_df[onbal_df['STAGE_FINAL'] == 3]
    pre_onbal_stage3_df = pre_onbal_df[pre_onbal_df['STAGE_FINAL'] == 3]
    offbal_stage1_df = offbal_df[offbal_df['STAGE_FINAL'] == 1]
    pre_offbal_stage1_df = pre_offbal_df[pre_offbal_df['STAGE_FINAL'] == 1]
    offbal_stage2_df = offbal_df[offbal_df['STAGE_FINAL'] == 2]
    pre_offbal_stage2_df = pre_offbal_df[pre_offbal_df['STAGE_FINAL'] == 2]

    # TODO: use map to get the range
    # onbal stage 1 & 2
    bins = [0, 0.0015, 0.0025, 0.005, 0.0075, 0.025, 0.1, 0.45, float('inf')]
    labels = [1, 2, 3, 4, 5, 6, 7, 8]
    value1_list = calculate_EAD(
        context, param, report_temp, onbal_stage1_df, bins, labels, 'LAST_PD_USED')
    pre_value1_list = calculate_EAD(
        context, param, report_temp, pre_onbal_stage1_df, bins, labels, 'LAST_PD_USED')
    value2_list = calculate_EAD(
        context, param, report_temp, onbal_stage2_df, bins, labels, 'LAST_PD_USED')
    pre_value2_list = calculate_EAD(
        context, param, report_temp, pre_onbal_stage2_df, bins, labels, 'LAST_PD_USED')

    # onbal stage 3
    bins_3 = [-0.1, 12, 24, 36, 48, 60, 84, float('inf')]
    labels_3 = [1, 2, 3, 4, 5, 6, 7]
    value3_list = calculate_EAD(
        context, param, report_temp, onbal_stage3_df, bins_3, labels_3, 'MONTH_IN_DEFT', right=True)
    pre_value3_list = calculate_EAD(
        context, param, report_temp, pre_onbal_stage3_df, bins_3, labels_3, 'MONTH_IN_DEFT', right=True)

    # offbal
    offbal_value1_list = calculate_offbal_value(
        context, param, report_temp, offbal_stage1_df, bins, labels)
    pre_offbal_value1_list = calculate_offbal_value(
        context, param, report_temp, pre_offbal_stage1_df, bins, labels)
    offbal_value2_list = calculate_offbal_value(
        context, param, report_temp, offbal_stage2_df, bins, labels)
    pre_offbal_value2_list = calculate_offbal_value(
        context, param, report_temp, pre_offbal_stage2_df, bins, labels)

    # write
    start_idx = df_report_[df_report_.columns[0]].first_valid_index()

    sheet = wb[tab_name]
    rp_f = report_basic.report_cond_basic(context=context)
    RUN_YYMM = rp_f.format_date_slash(context.run_yymm)
    PREV_YYMM = rp_f.format_date_slash(context.prev_yymm)
    sheet.merge_cells('C1:F1')
    sheet['C1'] = RUN_YYMM
    sheet.merge_cells('G1:J1')
    sheet['G1'] = PREV_YYMM
    sheet.merge_cells('C38:H38')
    sheet['C38'] = RUN_YYMM
    sheet.merge_cells('I38:N38')
    sheet['I38'] = PREV_YYMM

    for i in range(len(value1_list[0]) - 1):
        for j in range(4):
            sheet.cell(row=start_idx + i + 2, column=j +
                       3, value=value1_list[j][i])
            sheet.cell(row=start_idx + i + 2, column=j +
                       7, value=pre_value1_list[j][i])
            sheet.cell(row=start_idx + 13 + i, column=j +
                       3, value=value2_list[j][i])
            sheet.cell(row=start_idx + 13 + i, column=j +
                       7, value=pre_value2_list[j][i])

    for j in range(4):
        sheet.cell(
            row=start_idx + len(value1_list[0]) + 2, column=j + 3, value=value1_list[j][-1])
        sheet.cell(
            row=start_idx + len(value1_list[0]) + 2, column=j + 7, value=pre_value1_list[j][-1])
        sheet.cell(row=start_idx + 13 +
                   len(value1_list[0]), column=j + 3, value=value2_list[j][-1])
        sheet.cell(row=start_idx + 13 +
                   len(value1_list[0]), column=j + 7, value=pre_value2_list[j][-1])

    for i in range(len(value3_list[0])):
        for j in range(4):
            sheet.cell(row=start_idx + 25 + i, column=j +
                       3, value=value3_list[j][i])
            sheet.cell(row=start_idx + 25 + i, column=j +
                       7, value=pre_value3_list[j][i])

    for i in range(len(offbal_value1_list[0]) - 1):
        for j in range(6):
            sheet.cell(row=start_idx + 39 + i, column=j +
                       3, value=offbal_value1_list[j][i])
            sheet.cell(row=start_idx + 39 + i, column=j + 9,
                       value=pre_offbal_value1_list[j][i])
            sheet.cell(row=start_idx + 50 + i, column=j +
                       3, value=offbal_value2_list[j][i])
            sheet.cell(row=start_idx + 50 + i, column=j + 9,
                       value=pre_offbal_value2_list[j][i])

    for j in range(6):
        sheet.cell(row=start_idx + 39 +
                   len(offbal_value1_list[0]), column=j + 3, value=offbal_value1_list[j][-1])
        sheet.cell(row=start_idx + 39 +
                   len(offbal_value1_list[0]), column=j + 9, value=pre_offbal_value1_list[j][-1])
        sheet.cell(row=start_idx + 50 +
                   len(offbal_value1_list[0]), column=j + 3, value=offbal_value2_list[j][-1])
        sheet.cell(row=start_idx + 50 +
                   len(offbal_value1_list[0]), column=j + 9, value=pre_offbal_value2_list[j][-1])

    return wb
