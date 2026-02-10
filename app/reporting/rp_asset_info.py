from pathlib import Path
import pandas as pd
import numpy as np
from reporting import report_basic

import re


def sum_value(context, param, result_df):
    # set condition
    tab_name = 'Asset_info'
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    df_filter = rp_f.overall_filter(df_conditions, result_df)

    # split df_filter into two types: AC/FVOCI and FVTPL
    df_ac_fvoci = df_filter[df_filter['IFRS9_MEAS_TYPE'].isin(['AC', 'FVOCI'])]
    df_fvtpl = df_filter[df_filter['IFRS9_MEAS_TYPE'] == 'FVTPL']

    # select category - AC/FVOCI
    df_other = df_ac_fvoci[df_ac_fvoci['SUB_CATEGORY'].isin(['SF', 'SL', 'C'])]
    df = df_ac_fvoci[~df_ac_fvoci['SUB_CATEGORY'].isin(['SF', 'SL', 'C'])]
    df_onbal = df[df['ON_OFF_BAL_IND'] == 'ON']
    df_offbal = df[df['ON_OFF_BAL_IND'] == 'OFF']

    # Gross carrying amount
    GCA_onbal_sum = df_onbal['IFRS_AMOUNT_LCY'].sum()
    GCA_offbal_sum = df_offbal['IFRS_AMOUNT_LCY'].sum()
    GCA_other_sum = df_other['IFRS_AMOUNT_LCY'].sum()
    FVTPL_sum = df_fvtpl['IFRS_AMOUNT_LCY'].sum()

    # Credit loss  allowance
    CLA_onbal_sum = -1 * df_onbal['ECL_ULTIMATE_LCY'].sum()
    CLA_offbal_sum = -1 * df_offbal['ECL_ULTIMATE_LCY'].sum()
    CLA_other_sum = -1 * df_other['ECL_ULTIMATE_LCY'].sum()

    sum_dic = {'GCA_onbal_sum': GCA_onbal_sum, 'GCA_offbal_sum': GCA_offbal_sum,
               'CLA_onbal_sum': CLA_onbal_sum, 'CLA_offbal_sum': CLA_offbal_sum,
               'GCA_other_sum': GCA_other_sum, 'CLA_other_sum': CLA_other_sum,
               'FVTPL_sum': FVTPL_sum}
    for key, value in sum_dic.items():
        value = rp_f.scaling_num(value, df_conditions)
        sum_dic[key] = value
    return sum_dic


def run_asset_info(context, report_temp, sum_dic, wb):
    """
    context: environment setting
    report_temp: reporting empty template
    sum_dic: sum of ECL calculation result

    return: wb: openpyxl.workbook
    """
    tab_name = 'Asset_info'

    GCA_total = sum_dic['GCA_onbal_sum'] + sum_dic['GCA_offbal_sum'] + \
        sum_dic['GCA_other_sum'] + sum_dic['FVTPL_sum']
    CLA_total = sum_dic['CLA_onbal_sum'] + \
        sum_dic['CLA_offbal_sum'] + sum_dic['CLA_other_sum']
    CA_total = GCA_total + CLA_total

    # create value list
    GCA_list = [sum_dic['GCA_onbal_sum'], sum_dic['GCA_offbal_sum'],
                sum_dic['GCA_other_sum'], sum_dic['FVTPL_sum'], GCA_total]
    CLA_list = [sum_dic['CLA_onbal_sum'], sum_dic['CLA_offbal_sum'],
                sum_dic['CLA_other_sum'], None, CLA_total]
    CA_onbal = sum_dic['GCA_onbal_sum'] + sum_dic['CLA_onbal_sum']
    CA_offbal = sum_dic['GCA_offbal_sum'] + sum_dic['CLA_offbal_sum']
    CA_other = sum_dic['GCA_other_sum'] + sum_dic['CLA_other_sum']
    CA_list = [CA_onbal, CA_offbal, CA_other, sum_dic['FVTPL_sum'], CA_total]

    sheet = wb[tab_name]
    rp_f = report_basic.report_cond_basic(context=context)
    RUN_YYMM = context.run_yymm
    sheet['B1'] = rp_f.format_date_slash(RUN_YYMM)
    
    for i in range(len(GCA_list)):
        sheet[f'B{i + 3}'] = GCA_list[i]
        sheet[f'C{i + 3}'] = CLA_list[i]
        sheet[f'D{i + 3}'] = CA_list[i]

    return wb
