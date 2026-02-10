from pathlib import Path
import pandas as pd
import numpy as np
from reporting import report_basic

import re


def run_forward_looking(context, param, report_temp, assumption, wb):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    assumption: last row in scenariodata

    return: wb: openpyxl.workbook
    """
    # load parameters
    mef = param['MEFmodel']
    pwa = param['pwa']
    map = param['FLName_mapping']

    mef_df = pd.DataFrame(mef)
    pwa_df = pd.DataFrame(pwa)

    # delete duplicate MEFname and intercept in MEF col
    Variable = mef_df['MEF_NAME'].drop_duplicates()
    Variable = Variable[Variable != 'INTERCEPT']

    Scenario = pwa_df['scenario']
    Assigned_weight = pwa_df['pwa']

    # fill in Variable, Scenario and Assigned_weight col
    Variable_repeated = np.repeat(Variable, len(Scenario))
    Scenario_tiled = np.tile(Scenario, len(Variable))
    Assigned_weight_repeated = np.tile(Assigned_weight, len(Variable))

    df = pd.DataFrame({
        'Variable': Variable_repeated,
        'Scenario': Scenario_tiled,
        'Assigned weight': Assigned_weight_repeated
    })
    df = df.reset_index(drop=True)

    # fill in Assumptions col
    assumption = assumption.melt(
        id_vars='Code:', var_name='combined_key', value_name='Assumptions')
    df['combined_key'] = df['Variable'] + '_' + df['Scenario']
    df = df.merge(
        assumption[['combined_key', 'Assumptions']], on='combined_key', how='left')
    df = df.drop('combined_key', axis=1)

    # transfer Scenario name
    df['Scenario'] = df['Scenario'].replace({
        'BASE': 'Base',
        'SEVE': 'Severe',
        'GROW': 'Optimistic'
    })

    # transfer MEF name
    name_mapping = dict(zip(map['MEF_NAME'], map['FL_NAME']))
    df['Variable'] = df['Variable'].map(name_mapping)

    tab_name = 'Forward_looking'

    # set condition
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    df['Assigned weight'] = df['Assigned weight'].apply(
        lambda x: rp_f.unit_represent_num(x, df_conditions))
    df['Assumptions'] = df['Assumptions'].apply(
        lambda x: rp_f.unit_represent_num(x, df_conditions))
    df_filter = rp_f.overall_filter(df_conditions, df)

    # output
    sheet = wb[tab_name]
    row = 2
    for _, row_data in df_filter.iterrows():
        sheet[f'A{row}'] = row_data['Variable']
        sheet[f'B{row}'] = row_data['Scenario']
        sheet[f'C{row}'] = row_data['Assigned weight']
        sheet[f'D{row}'] = row_data['Assumptions']
        row += 1

    # merge the variable cell
    total_rows = len(df)
    for i in range(2, total_rows + 2, 3):
        if i + 2 <= total_rows + 1:
            sheet.merge_cells(f'A{i}:A{i+2}')

    return wb
