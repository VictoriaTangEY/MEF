from input_handler.load_parameters import load_configuration_file
from input_handler.env_setting import run_setting
from pathlib import Path
import pandas as pd
from datetime import datetime
from reporting import report_basic

import re


def calculate_EAD_ECL(context, param, report_temp, ecl_result, bins, labels, name, column_name):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv
    bins: range bins for grouping
    labels: range labels for grouping
    name: column name for grouping
    column_name:  IFRS_AMOUNT_LCY or ECL_ULTIMATE_LCY

    return: 
    - green_business_value_list
    - online_business_value_list
    - other_business_value_list
    - green_consumer_value_list
    - online_consumer_value_list
    - other_consumer_value_list
    - total_value_list
    """
    # load parameters
    map_loan = param['LoanGroupSecond']
    map_loan['CATEGORY'] = map_loan['CATEGORY'].str.lower()
    map_loan['SUB_CATEGORY'] = map_loan['SUB_CATEGORY'].str.lower()

    # load report template
    tab_name = 'ECL_stage_report'
    # set condition
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    ecl_result_filter = rp_f.overall_filter(df_conditions, ecl_result)

    # calculate amount
    ecl_filtered = ecl_result_filter.filter(items=[column_name, name, 'CATEGORY', 'SUB_CATEGORY'])
    ecl_filtered['CATEGORY'] = ecl_filtered['CATEGORY'].str.lower()
    ecl_filtered['SUB_CATEGORY'] = ecl_filtered['SUB_CATEGORY'].str.lower()

    combined_ecl_df = pd.merge(ecl_filtered, map_loan, on=['CATEGORY', 'SUB_CATEGORY'], how='left')
    combined_ecl_df.loc[combined_ecl_df['CATEGORY'] == 'online_consumer', 'LOAN_SECOND_LAYER'] = 'Online consumer'
    combined_ecl_df.loc[combined_ecl_df['CATEGORY'] == 'other_consumer', 'LOAN_SECOND_LAYER'] = 'Other consumer'
    combined_ecl_df['LOAN_SECOND_LAYER'].fillna('Green consumer', inplace=True)

    # group
    combined_ecl_df['RANGE_GROUP'] = pd.cut(
        combined_ecl_df[name], bins=bins, labels=labels, right=False)

    green_business_df = combined_ecl_df[combined_ecl_df['LOAN_SECOND_LAYER']
                                  == 'Green business']
    online_business_df = combined_ecl_df[combined_ecl_df['LOAN_SECOND_LAYER']
                                  == 'Online business']
    other_business_df = combined_ecl_df[combined_ecl_df['LOAN_SECOND_LAYER']
                                  == 'Other business']
    green_consumer_df = combined_ecl_df[combined_ecl_df['LOAN_SECOND_LAYER']
                                  == 'Green consumer']
    online_consumer_df = combined_ecl_df[combined_ecl_df['LOAN_SECOND_LAYER']
                                  == 'Online consumer']
    other_consumer_df = combined_ecl_df[combined_ecl_df['LOAN_SECOND_LAYER']
                                  == 'Other consumer']
    corporate_df = combined_ecl_df[combined_ecl_df['LOAN_SECOND_LAYER']
                                  == 'Corporate']

    # loop for all range
    green_business_value_list = []
    online_business_value_list = []
    other_business_value_list = []
    green_consumer_value_list = []
    online_consumer_value_list = []
    other_consumer_value_list = []
    corporate_value_list = []
    total_value_list = []
    
    green_business_total_value = 0
    online_business_total_value = 0
    other_business_total_value = 0
    green_consumer_total_value = 0
    online_consumer_total_value = 0
    other_consumer_total_value = 0
    corporate_total_value = 0
    total_total_value = 0

    for i in labels:
        green_business_value = green_business_df[green_business_df['RANGE_GROUP'] == i][column_name].sum()
        green_business_value = rp_f.scaling_num(green_business_value, df_conditions)
        green_business_value_list.append(green_business_value)
        green_business_total_value += green_business_value

        online_business_value = online_business_df[online_business_df['RANGE_GROUP'] == i][column_name].sum()
        online_business_value = rp_f.scaling_num(online_business_value, df_conditions)
        online_business_value_list.append(online_business_value)
        online_business_total_value += online_business_value

        other_business_value = other_business_df[other_business_df['RANGE_GROUP'] == i][column_name].sum()
        other_business_value = rp_f.scaling_num(other_business_value, df_conditions)
        other_business_value_list.append(other_business_value)
        other_business_total_value += other_business_value

        green_consumer_value = green_consumer_df[green_consumer_df['RANGE_GROUP'] == i][column_name].sum()
        green_consumer_value = rp_f.scaling_num(green_consumer_value, df_conditions)
        green_consumer_value_list.append(green_consumer_value)
        green_consumer_total_value += green_consumer_value

        online_consumer_value = online_consumer_df[online_consumer_df['RANGE_GROUP'] == i][column_name].sum()
        online_consumer_value = rp_f.scaling_num(online_consumer_value, df_conditions)
        online_consumer_value_list.append(online_consumer_value)
        online_consumer_total_value += online_consumer_value

        other_consumer_value = other_consumer_df[other_consumer_df['RANGE_GROUP'] == i][column_name].sum()
        other_consumer_value = rp_f.scaling_num(other_consumer_value, df_conditions)
        other_consumer_value_list.append(other_consumer_value)
        other_consumer_total_value += other_consumer_value

        corporate_value = corporate_df[corporate_df['RANGE_GROUP'] == i][column_name].sum()
        corporate_value = rp_f.scaling_num(corporate_value, df_conditions)
        corporate_value_list.append(corporate_value)
        corporate_total_value += corporate_value

        total_value = green_business_value + online_business_value +  other_business_value + green_consumer_value + online_consumer_value + other_consumer_value + corporate_value
        total_value_list.append(total_value)
        total_total_value += total_value

    return green_business_value_list, online_business_value_list, other_business_value_list, green_consumer_value_list, online_consumer_value_list, other_consumer_value_list, corporate_value_list, total_value_list


def calculate_mik(context, param, report_temp, ecl_result, column_name):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv
    column_name: 'DRAWN_BAL_LCY'

    return:
    - mik_value
    """
    # load report template
    tab_name = 'ECL_stage_report'
    # set condition
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    ecl_result_filter = rp_f.overall_filter(df_conditions, ecl_result)

    # calculate amount
    mik_filtered = ecl_result_filter[ecl_result_filter['SUB_CATEGORY'].str.lower() == 'mik']

    mik_value = mik_filtered[column_name].sum()
    mik_value = rp_f.scaling_num(mik_value, df_conditions)

    return mik_value


def run_ecl_stage_report(context, param, report_temp, ecl_result, wb):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return: wb: openpyxl.workbook
    """
    # load report template
    tab_name = 'ECL_stage_report'

    loan_df = ecl_result[0][ecl_result[0]['DATA_SOURCE_CD'] == 'LOAN']
    pre_loan_df = ecl_result[1][ecl_result[1]['DATA_SOURCE_CD'] == 'LOAN']
                            
    onbal_df = loan_df[loan_df['ON_OFF_BAL_IND'] == 'ON']
    pre_onbal_df = pre_loan_df[pre_loan_df['ON_OFF_BAL_IND'] == 'ON']

    onbal_Res_F_df = onbal_df[onbal_df['RESTRUCTURED_IND'] == False]
    pre_onbal_Res_F_df = pre_onbal_df[pre_onbal_df['RESTRUCTURED_IND'] == False]

    onbal_Res_T_df = onbal_df[onbal_df['RESTRUCTURED_IND'] == True]
    pre_onbal_Res_T_df = pre_onbal_df[pre_onbal_df['RESTRUCTURED_IND'] == True]

    onbal_stage1_df = onbal_Res_F_df[onbal_Res_F_df['STAGE_FINAL'] == 1]
    pre_onbal_stage1_df = pre_onbal_Res_F_df[pre_onbal_Res_F_df['STAGE_FINAL'] == 1]
    onbal_stage2_df = onbal_Res_F_df[onbal_Res_F_df['STAGE_FINAL'] == 2]
    pre_onbal_stage2_df = pre_onbal_Res_F_df[pre_onbal_Res_F_df['STAGE_FINAL'] == 2]

    rest_stage2_df = onbal_Res_T_df[onbal_Res_T_df['STAGE_FINAL'] == 2]
    pre_rest_stage2_df = pre_onbal_Res_T_df[pre_onbal_Res_T_df['STAGE_FINAL'] == 2]
 
    onbal_stage3_df = onbal_df[onbal_df['STAGE_FINAL'] == 3]
    pre_onbal_stage3_df = onbal_df[onbal_df['STAGE_FINAL'] == 3]
 
    onbal_specific_df = onbal_df[onbal_df['ECL_APPROACH'] == 'SPECIFIC_ASSESSMENT']
    pre_onbal_specific_df = pre_onbal_df[pre_onbal_df['ECL_APPROACH'] == 'SPECIFIC_ASSESSMENT']

    # EAD
    # onbal stage 1
    bins_1 = [0, 1, 31]  
    labels_1 = [1, 2]
    
    EAD_value1_list = calculate_EAD_ECL(
        context, param, report_temp, onbal_stage1_df, bins_1, labels_1, 'PAST_DUE_DAYS', 'IFRS_AMOUNT_LCY')
    EAD_pre_value1_list = calculate_EAD_ECL(
        context, param, report_temp, pre_onbal_stage1_df, bins_1, labels_1, 'PAST_DUE_DAYS', 'IFRS_AMOUNT_LCY')
    
    # onbal stage 2
    bins_2 = [0, 1, 31, 61, 91]
    labels_2 = [1, 2, 3, 4]
    
    EAD_value2_list = calculate_EAD_ECL(
        context, param, report_temp, onbal_stage2_df, bins_2, labels_2, 'PAST_DUE_DAYS', 'IFRS_AMOUNT_LCY')
    EAD_pre_value2_list = calculate_EAD_ECL(
        context, param, report_temp, pre_onbal_stage2_df, bins_2, labels_2, 'PAST_DUE_DAYS', 'IFRS_AMOUNT_LCY')
    
    # onbal stage 2(rest) & stage 3 & specific
    bins_3 = [-float('inf'), float('inf')]
    labels_3 = [1]
    
    EAD_value2_list_rest_list = calculate_EAD_ECL(
        context, param, report_temp, rest_stage2_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'IFRS_AMOUNT_LCY')
    EAD_pre_value2_list_rest_list = calculate_EAD_ECL(
        context, param, report_temp, pre_rest_stage2_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'IFRS_AMOUNT_LCY')

    EAD_value3_list = calculate_EAD_ECL(
        context, param, report_temp, onbal_stage3_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'IFRS_AMOUNT_LCY')
    EAD_pre_value3_list = calculate_EAD_ECL(
        context, param, report_temp, pre_onbal_stage3_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'IFRS_AMOUNT_LCY')
    
    EAD_specific_list = calculate_EAD_ECL(
        context, param, report_temp, onbal_specific_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'IFRS_AMOUNT_LCY')
    EAD_pre_specific_list = calculate_EAD_ECL(
        context, param, report_temp, pre_onbal_specific_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'IFRS_AMOUNT_LCY')


    # ECL
    # onbal stage 1
    bins_1 = [0, 1, 31]  
    labels_1 = [1, 2]
    
    ECL_value1_list = calculate_EAD_ECL(
        context, param, report_temp, onbal_stage1_df, bins_1, labels_1, 'PAST_DUE_DAYS', 'ECL_ULTIMATE_LCY')
    ECL_pre_value1_list = calculate_EAD_ECL(
        context, param, report_temp, pre_onbal_stage1_df, bins_1, labels_1, 'PAST_DUE_DAYS', 'ECL_ULTIMATE_LCY')
    
    # onbal stage 2
    bins_2 = [0, 1, 31, 61, 91]
    labels_2 = [1, 2, 3, 4]
    
    ECL_value2_list = calculate_EAD_ECL(
        context, param, report_temp, onbal_stage2_df, bins_2, labels_2, 'PAST_DUE_DAYS', 'ECL_ULTIMATE_LCY')
    ECL_pre_value2_list = calculate_EAD_ECL(
        context, param, report_temp, pre_onbal_stage2_df, bins_2, labels_2, 'PAST_DUE_DAYS', 'ECL_ULTIMATE_LCY')
    
    # onbal stage 2(rest) & stage 3 & specific
    bins_3 = [-float('inf'), float('inf')]
    labels_3 = [1]
    
    ECL_value2_list_rest_list = calculate_EAD_ECL(
        context, param, report_temp, rest_stage2_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'ECL_ULTIMATE_LCY')
    ECL_pre_value2_list_rest_list = calculate_EAD_ECL(
        context, param, report_temp, pre_rest_stage2_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'ECL_ULTIMATE_LCY')

    ECL_value3_list = calculate_EAD_ECL(
        context, param, report_temp, onbal_stage3_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'ECL_ULTIMATE_LCY')
    ECL_pre_value3_list = calculate_EAD_ECL(
        context, param, report_temp, pre_onbal_stage3_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'ECL_ULTIMATE_LCY')
    
    ECL_specific_list = calculate_EAD_ECL(
        context, param, report_temp, onbal_specific_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'ECL_ULTIMATE_LCY')
    ECL_pre_specific_list = calculate_EAD_ECL(
        context, param, report_temp, pre_onbal_specific_df, bins_3, labels_3, 'PAST_DUE_DAYS', 'ECL_ULTIMATE_LCY')
    

    # MIK
    mik_value = calculate_mik(
        context, param, report_temp, onbal_df,'DRAWN_BAL_LCY')
    mik_value_pre = calculate_mik(
        context, param, report_temp, pre_onbal_df,'DRAWN_BAL_LCY')


    # write Excel
    sheet = wb[tab_name]
    rp_f = report_basic.report_cond_basic(context=context)
    RUN_YYMM = rp_f.format_date_slash(context.run_yymm)
    PREV_YYMM = rp_f.format_date_slash(context.prev_yymm)
    sheet['B1'] = RUN_YYMM
    sheet['B17'] = PREV_YYMM
    
    # Define row positions
    current_start_row = 5
    previous_start_row = 21
    diff_start_row = 36

    # Define column positions
    column_mapping = {
        "EAD_value1_list": 3, "EAD_value2_list": 5, "EAD_value2_list_rest_list": 9,
        "EAD_value3_list": 10, "EAD_specific_list": 11,
        "ECL_value1_list": 13, "ECL_value2_list": 15, "ECL_value2_list_rest_list": 19,
        "ECL_value3_list": 20, "ECL_specific_list": 21
    }

    # Write current period values
    for i in range(len(EAD_value1_list)):
        
        #Column B: sum of C to K
        row_idx = current_start_row + i
        sheet.cell(row=row_idx, column=2).value = f"=SUM(C{row_idx}:K{row_idx})"

        for j in range(len(EAD_value1_list[i])):
            sheet.cell(row=row_idx, column=column_mapping["EAD_value1_list"] + j, value=EAD_value1_list[i][j])
        for j in range(len(EAD_value2_list[i])):
            sheet.cell(row=row_idx, column=column_mapping["EAD_value2_list"] + j, value=EAD_value2_list[i][j])
        sheet.cell(row=row_idx, column=column_mapping["EAD_value2_list_rest_list"], value=EAD_value2_list_rest_list[i][0])
        sheet.cell(row=row_idx, column=column_mapping["EAD_value3_list"], value=EAD_value3_list[i][0])
        sheet.cell(row=row_idx, column=column_mapping["EAD_specific_list"], value=EAD_specific_list[i][0])

        #Column L: sum of M to U
        sheet.cell(row=row_idx, column=12).value = f"=SUM(M{row_idx}:U{row_idx})"

        for j in range(len(ECL_value1_list[i])):
            sheet.cell(row=row_idx, column=column_mapping["ECL_value1_list"] + j, value=ECL_value1_list[i][j])
        for j in range(len(ECL_value2_list[i])):
            sheet.cell(row=row_idx, column=column_mapping["ECL_value2_list"] + j, value=ECL_value2_list[i][j])
        sheet.cell(row=row_idx, column=column_mapping["ECL_value2_list_rest_list"], value=ECL_value2_list_rest_list[i][0])
        sheet.cell(row=row_idx, column=column_mapping["ECL_value3_list"], value=ECL_value3_list[i][0])
        sheet.cell(row=row_idx, column=column_mapping["ECL_specific_list"], value=ECL_specific_list[i][0])


    # Write previous period values
    for i in range(len(EAD_pre_value1_list)):

        #Column B: sum of C to K
        row_idx_pre = previous_start_row + i
        sheet.cell(row=row_idx_pre, column=2).value = f"=SUM(C{row_idx_pre}:K{row_idx_pre})"

        for j in range(len(EAD_pre_value1_list[i])):
            sheet.cell(row=row_idx_pre, column=column_mapping["EAD_value1_list"] + j, value=EAD_pre_value1_list[i][j])
        for j in range(len(EAD_pre_value2_list[i])):
            sheet.cell(row=row_idx_pre, column=column_mapping["EAD_value2_list"] + j, value=EAD_pre_value2_list[i][j])
        sheet.cell(row=row_idx_pre, column=column_mapping["EAD_value2_list_rest_list"], value=EAD_pre_value2_list_rest_list[i][0])
        sheet.cell(row=row_idx_pre, column=column_mapping["EAD_value3_list"], value=EAD_pre_value3_list[i][0])
        sheet.cell(row=row_idx_pre, column=column_mapping["EAD_specific_list"], value=EAD_pre_specific_list[i][0])

        #Column L: sum of M to U
        sheet.cell(row=row_idx_pre, column=12).value = f"=SUM(M{row_idx_pre}:U{row_idx_pre})"

        for j in range(len(ECL_pre_value1_list[i])):
            sheet.cell(row=row_idx_pre, column=column_mapping["ECL_value1_list"] + j, value=ECL_pre_value1_list[i][j])
        for j in range(len(ECL_pre_value2_list[i])):
            sheet.cell(row=row_idx_pre, column=column_mapping["ECL_value2_list"] + j, value=ECL_pre_value2_list[i][j])
        sheet.cell(row=row_idx_pre, column=column_mapping["ECL_value2_list_rest_list"], value=ECL_pre_value2_list_rest_list[i][0])
        sheet.cell(row=row_idx_pre, column=column_mapping["ECL_value3_list"], value=ECL_pre_value3_list[i][0])
        sheet.cell(row=row_idx_pre, column=column_mapping["ECL_specific_list"], value=ECL_pre_specific_list[i][0])


    # Write difference table values
    for i in range(len(EAD_value1_list) + 3):
        row_idx_diff = diff_start_row + i
        row_idx_curr = current_start_row + i
        row_idx_prev = previous_start_row + i

        # Column B: sum of C to K
        sheet.cell(row=row_idx_diff, column=2).value = f"=SUM(C{row_idx_diff}:K{row_idx_diff})"

        for col in range(column_mapping["EAD_value1_list"], column_mapping["EAD_specific_list"]+1):
            curr_cell = sheet.cell(row=row_idx_curr, column=col).coordinate
            prev_cell = sheet.cell(row=row_idx_prev, column=col).coordinate
            sheet.cell(row=row_idx_diff, column=col).value = f"={curr_cell}-{prev_cell}"

        # Column L: sum of M to U
        sheet.cell(row=row_idx_diff, column=12).value = f"=SUM(M{row_idx_diff}:U{row_idx_diff})"

        for col in range(column_mapping["ECL_value1_list"], column_mapping["ECL_specific_list"]+1):
            curr_cell = sheet.cell(row=row_idx_curr, column=col).coordinate
            prev_cell = sheet.cell(row=row_idx_prev, column=col).coordinate
            sheet.cell(row=row_idx_diff, column=col).value = f"={curr_cell}-{prev_cell}"


    # Write MIK value
    sheet["C13"] = mik_value
    sheet.cell(row=13, column=2).value = f"=SUM(C{13}:K{13})"
    sheet.cell(row=13, column=12).value = f"=SUM(C{13}:K{13})"

    sheet["C29"] = mik_value_pre
    sheet.cell(row=29, column=2).value = f"=SUM(M{29}:U{29})"
    sheet.cell(row=29, column=12).value = f"=SUM(M{29}:U{29})"


    # Write Repo
    sheet.cell(row=14, column=2).value = f"=SUM(C{14}:K{14})"
    sheet.cell(row=14, column=12).value = f"=SUM(C{14}:K{14})"
    sheet.cell(row=30, column=2).value = f"=SUM(M{30}:U{30})"
    sheet.cell(row=30, column=12).value = f"=SUM(M{30}:U{30})"


    # Write Grand Total
    for col in range(3, 12):
        sheet.cell(row=15, column=col).value = f"=SUM({chr(64+col)}12:{chr(64+col)}13)"
    for col in range(13, 22):
        sheet.cell(row=15, column=col).value = f"=SUM({chr(64+col)}12:{chr(64+col)}13)"
    sheet.cell(row=15, column=2).value = f"=SUM(C{15}:K{15})"
    sheet.cell(row=15, column=12).value = f"=SUM(C{15}:K{15})"

    for col in range(3, 12):
        sheet.cell(row=31, column=col).value = f"=SUM({chr(64+col)}28:{chr(64+col)}29)"
    for col in range(13, 22):
        sheet.cell(row=31, column=col).value = f"=SUM({chr(64+col)}28:{chr(64+col)}29)"
    sheet.cell(row=31, column=2).value = f"=SUM(M{31}:U{31})"
    sheet.cell(row=31, column=12).value = f"=SUM(M{31}:U{31})"

  
    return wb
