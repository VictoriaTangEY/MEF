from input_handler.load_parameters import load_configuration_file
from input_handler.env_setting import run_setting
from pathlib import Path
import pandas as pd
from datetime import datetime
from reporting import report_basic
import re


def merge_ECL_results(context, param, ecl_result, ecl_result_prev):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return: merged dataframe

    """
    # load report template
    tab_name = 'ECL_movement_loan_on'
    # set condition
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    
    ecl_result_filter_prev = rp_f.overall_filter(df_conditions, ecl_result_prev)
    ecl_result_filter = rp_f.overall_filter(df_conditions, ecl_result)

    # select required columns
    cols = ['CONTRACT_ID', 'STAGE_FINAL', 'ECL_ULTIMATE_LCY', 'EAD_LCY', 'ACRU_INT_LCY', 'PD_POOL_ID', 'SUB_CATEGORY', 'IFRS_AMOUNT_LCY']
    df_curr = ecl_result_filter[cols].copy()
    df_prev = ecl_result_filter_prev[cols].copy().rename(
        columns={
            'STAGE_FINAL': 'STAGE_FINAL_prev',
            'ECL_ULTIMATE_LCY': 'ECL_ULTIMATE_LCY_prev',
            'EAD_LCY': 'EAD_LCY_prev',
            'ACRU_INT_LCY': 'ACRU_INT_LCY_prev',
            'PD_POOL_ID': 'PD_POOL_ID_prev',
            'SUB_CATEGORY': 'SUB_CATEGORY_prev',
            'IFRS_AMOUNT_LCY': 'IFRS_AMOUNT_LCY_prev'
        }
    )
    df_merged = df_curr.merge(df_prev, on='CONTRACT_ID', how='outer') 
    df_merged['PRINCIPAL_CHANGE'] = df_merged['IFRS_AMOUNT_LCY'] - df_merged['IFRS_AMOUNT_LCY_prev']
    df_merged['ECL_CHANGE'] = df_merged['ECL_ULTIMATE_LCY'] - df_merged['ECL_ULTIMATE_LCY_prev']

    # Mapping using SUB_CATEGORY
    def determine_sub_category(row):
        # new：df_prev no，df_curr yes
        if pd.isna(row['SUB_CATEGORY_prev']) and not pd.isna(row['SUB_CATEGORY']):
            return row['SUB_CATEGORY']
        # derecognized：df_prev yes，df_curr no
        elif not pd.isna(row['SUB_CATEGORY_prev']) and pd.isna(row['SUB_CATEGORY']):
            return row['SUB_CATEGORY_prev']
        # existing：df_prev yes, df_curr yes
        elif not pd.isna(row['SUB_CATEGORY_prev']) and not pd.isna(row['SUB_CATEGORY']):
            return row['SUB_CATEGORY']
        else:
            return None
    df_merged['SUB_CATEGORY_FINAL'] = df_merged.apply(determine_sub_category, axis=1)    
    
    # load parameters
    map = param['LoanGroupFirst']

    # Make SUB_CATEGORY case-insensitive for mapping
    df_merged['SUB_CATEGORY_FINAL_upper'] = df_merged['SUB_CATEGORY_FINAL'].str.upper()
    map_upper = map.copy()
    map_upper['SUB_CATEGORY_FINAL_upper'] = map_upper['SUB_CATEGORY'].str.upper()
    
    df_merged_main = pd.merge(df_merged, map_upper, on='SUB_CATEGORY_FINAL_upper', how='left')
    df_merged_main = df_merged_main.drop(columns=['SUB_CATEGORY_FINAL_upper'])

    group_map = {
        'business': 'Business loans',
        'consumer': 'Consumer loans',
        'agricultural': 'Agricultural loans',
    }
    group_dfs = {k: df_merged_main[df_merged_main['LOAN_FIRST_LAYER'] == v] for k, v in group_map.items()}
    
    return group_dfs

def compute_ECL_PRIN_total_prev(df_merged, name, stage):
    """
    row 6: Compute ECL or Gross Carrying Amount
    NAME: ECL_ULTIMATE_LCY or IFRS_AMOUNT_LCY
    stage: 1, 2, 3
    """
    # Filter for the specified stage
    stage_filter = df_merged['STAGE_FINAL_prev'] == stage
    
    # Select the appropriate column based on name parameter
    if name == 'ECL_ULTIMATE_LCY':
        col_name = 'ECL_ULTIMATE_LCY_prev'
    elif name == 'IFRS_AMOUNT_LCY':
        col_name = 'IFRS_AMOUNT_LCY_prev'
    else:
        raise ValueError("name must be either 'ECL_ULTIMATE_LCY' or 'IFRS_AMOUNT_LCY'")
    
    # Calculate sum for the filtered data
    total = df_merged.loc[stage_filter, col_name].sum()
    return total

def compute_ECL_PRIN_new(df_merged, name, stage):
    """
    row 7: Compute ECL or Gross Carrying Amount
    NAME: ECL_ULTIMATE_LCY or IFRS_AMOUNT_LCY
    stage: 1, 2, 3
    """
    if stage == 2 or stage == 3:
        return 0
    
    if name not in ['ECL_ULTIMATE_LCY', 'IFRS_AMOUNT_LCY']:
        raise ValueError("name must be either 'ECL_ULTIMATE_LCY' or 'IFRS_AMOUNT_LCY'")
    
    # contract_id in curr but not in prev
    new_entries = df_merged[df_merged['STAGE_FINAL_prev'].isna()]
    new_total = new_entries[name].sum()

    # contract_id in curr and prev, stage_final is the same, principal_change > 0
    existing_entries = df_merged[
        (df_merged['STAGE_FINAL_prev'] == df_merged['STAGE_FINAL']) &
        (df_merged['PRINCIPAL_CHANGE'] > 0)
    ]

    if name == 'ECL_ULTIMATE_LCY':
        increase_total = existing_entries['ECL_CHANGE'].sum()
    else:
        increase_total = existing_entries['PRINCIPAL_CHANGE'].sum()

    total = new_total + increase_total
    return total

def compute_ECL_PRIN_exposures(df_merged, name, stage):
    """
    row 8: Compute ECL or Gross Carrying Amount
    NAME: ECL_ULTIMATE_LCY or IFRS_AMOUNT_LCY
    stage: 1, 2, 3
    """
    if name not in ['ECL_ULTIMATE_LCY', 'IFRS_AMOUNT_LCY']:
        raise ValueError("name must be either 'ECL_ULTIMATE_LCY' or 'IFRS_AMOUNT_LCY'")
    
    col_prev = name + '_prev'

    # contract_id in prev but not in curr -> closed
    closed_entries = df_merged[
        (df_merged['STAGE_FINAL_prev'] == stage) &
        (df_merged['STAGE_FINAL'].isna())
    ]

    closed_total = -closed_entries[col_prev].sum()

    # contract_id in prev and curr, stage_final is the same, principal_change < 0
    existing_entries = df_merged[
        (df_merged['STAGE_FINAL_prev'] == stage) &
        (df_merged['STAGE_FINAL'] == stage) &
        (df_merged['PRINCIPAL_CHANGE'] < 0)
    ]

    if name == 'ECL_ULTIMATE_LCY':
        decrease_total = existing_entries['ECL_CHANGE'].sum()
    else:
        decrease_total = existing_entries['PRINCIPAL_CHANGE'].sum()

    total = closed_total + decrease_total
    return total
    
def compute_ECL_PRIN_transfer1(df_merged, name, stage):
    if stage == 3:
        return 0
    
    if stage == 1:
        # stage_final is 1, stage_final_prev is not 1 and new
        transfered = df_merged[
            (df_merged['STAGE_FINAL'] == 1) &
            (df_merged['STAGE_FINAL_prev'].notna()) &
            (df_merged['STAGE_FINAL_prev'] != 1)
        ]
        return transfered[name].sum()
    
    if stage == 2:
        # stage_final is 1, stage_final_prev is 2 or 3
        transfered = df_merged[
            (df_merged['STAGE_FINAL'] == 1) &
            (df_merged['STAGE_FINAL_prev'].isin([2, 3]))
        ]
        col_prev = name + '_prev'
        return -transfered[col_prev].sum()
    
def compute_ECL_PRIN_transfer2(df_merged, name, stage):
    col_prev = name + '_prev'
    if stage == 1:
        # stage_prev is 1, stage_curr is 2 or 3
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'] == 1) &
            (df_merged['STAGE_FINAL'].isin([2, 3]))
        ]
        value1 = -transfered[col_prev].sum()

        # stage_prev is new, stage_curr is 2 or 3
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'].isna()) &
            (df_merged['STAGE_FINAL'].isin([2, 3]))
        ]
        value2 = -transfered[name].sum()
        
        # stage_prev==stage_curr==2 or 3, principal_change > 0
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'] == df_merged['STAGE_FINAL']) &
            (df_merged['STAGE_FINAL'].isin([2, 3])) &
            (df_merged['PRINCIPAL_CHANGE'] > 0)
        ]
        if name == 'ECL_ULTIMATE_LCY':
            value3 = -transfered['ECL_CHANGE'].sum()
        else:
            value3 = -transfered['PRINCIPAL_CHANGE'].sum()
        
        return value1 + value2 + value3
    
    if stage == 2:
        # stage_prev is 1 or 3, stage_curr is 2
        transfered = df_merged[
            (df_merged['STAGE_FINAL'] == 2) &
            (df_merged['STAGE_FINAL_prev'].isin([1, 3]))
        ]
        value1 = transfered[name].sum()

        # stage_prev is 3, stage_curr is 1
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'] == 3) &
            (df_merged['STAGE_FINAL'] == 1)
        ]
        value2 = -transfered[col_prev].sum()

        # stage_prev is 1, stage_curr is 3
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'] == 1) &
            (df_merged['STAGE_FINAL'] == 3)
        ]
        value3 = -transfered[col_prev].sum()

        # stage_prev is new, stage_curr is 2 or 3
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'].isna()) &
            (df_merged['STAGE_FINAL'].isin([2, 3]))
        ]
        value4 = transfered[name].sum()
        
        # stage_prev==stage_curr==2 or 3, principal_change > 0
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'] == df_merged['STAGE_FINAL']) &
            (df_merged['STAGE_FINAL'].isin([2, 3])) &
            (df_merged['PRINCIPAL_CHANGE'] > 0)
        ]
        if name == 'ECL_ULTIMATE_LCY':
            value5 = transfered['ECL_CHANGE'].sum()
        else:
            value5 = transfered['PRINCIPAL_CHANGE'].sum()

        return value1 - value2 -value3 + value4 + value5
        

    if stage == 3:
        # stage_prev is 3, stage_curr is 1 or 2
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'] == 3) &
            (df_merged['STAGE_FINAL'].isin([1, 2]))
        ]
        value1 = -transfered[col_prev].sum()
        return value1
    
def compute_ECL_PRIN_transfer3(df_merged, name, stage):
    col_prev = name + '_prev'
    if stage == 1:
        return 0
    
    if stage == 2:
        # stage_prev is 1 or 2, stage_curr is 3
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'].isin([1, 2])) &
            (df_merged['STAGE_FINAL'] == 3)
        ]
        value1 = -transfered[col_prev].sum()

        # stage_prev is new, stage_curr is 3
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'].isna()) &
            (df_merged['STAGE_FINAL'] == 3)
        ]
        value2 = -transfered[name].sum()

        # stage_prev==stage_curr==3, principal_change > 0
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'] == df_merged['STAGE_FINAL']) &
            (df_merged['STAGE_FINAL'] == 3) &
            (df_merged['PRINCIPAL_CHANGE'] > 0)
        ]
        if name == 'ECL_ULTIMATE_LCY':
            value3 = -transfered['ECL_CHANGE'].sum()
        else:
            value3 = -transfered['PRINCIPAL_CHANGE'].sum()

        return value1 + value2 + value3
        

    if stage == 3:
        # stage_prev is 1 or 2, stage_curr is 3
        transfered = df_merged[
            (df_merged['STAGE_FINAL'] == 3) &
            (df_merged['STAGE_FINAL_prev'].isin([1, 2]))
        ]
        value1 = transfered[name].sum()

        # stage_prev is new, stage_curr is 3
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'].isna()) &
            (df_merged['STAGE_FINAL'] == 3)
        ]
        value2 = transfered[name].sum()
        
        # stage_prev==stage_curr==3, principal_change > 0
        transfered = df_merged[
            (df_merged['STAGE_FINAL_prev'] == df_merged['STAGE_FINAL']) &
            (df_merged['STAGE_FINAL'] == 3) &
            (df_merged['PRINCIPAL_CHANGE'] > 0)
        ]
        if name == 'ECL_ULTIMATE_LCY':
            value3 = transfered['ECL_CHANGE'].sum()
        else:
            value3 = transfered['PRINCIPAL_CHANGE'].sum()

        return value1 + value2 + value3

def compute_Changes_AccruedInterest(df_merged, name, stage):
    """
    row 15: Compute ECL or Gross Carrying Amount impact from accrued interest.
    NAME: ECL_ULTIMATE_LCY or IFRS_AMOUNT_LCY
    stage: 1, 2, 3
    """
    # Filter for the specified stage and principal change
    # filter_cond = (
    #     (df_merged['STAGE_FINAL_prev'] == stage) &
    #     (df_merged['STAGE_FINAL'] == stage) &
    #     (df_merged['PRINCIPAL_CHANGE'] == 0)
    # )
    filter_cond = (
        (df_merged['STAGE_FINAL_prev'] == stage) &
        (df_merged['STAGE_FINAL'] == stage)
    )

    filtered = df_merged[filter_cond]

    name_prev = name + '_prev'
    valid_curr = (filtered['EAD_LCY'] != 0) & filtered['EAD_LCY'].notna()
    valid_prev = (filtered['EAD_LCY_prev'] != 0) & filtered['EAD_LCY_prev'].notna()

    changesAI_curr = ((filtered[name] / filtered['EAD_LCY']) * filtered['ACRU_INT_LCY']).where(valid_curr, 0)
    changesAI_prev = ((filtered[name_prev] / filtered['EAD_LCY_prev']) * filtered['ACRU_INT_LCY_prev']).where(valid_prev, 0)

    value = changesAI_curr.sum() - changesAI_prev.sum()
    return value

def compute_Changes_ModelAssumptions(df_merged, name, stage):
    """
    row 16: Compute ECL or Gross Carrying Amount impact from model assumptions.
    NAME: ECL_ULTIMATE_LCY or IFRS_AMOUNT_LCY
    stage: 1, 2, 3
    """
    filter_cond = (
        (df_merged['STAGE_FINAL_prev'] == stage) &
        (df_merged['STAGE_FINAL'] == stage) &
        (df_merged['PRINCIPAL_CHANGE'] == 0) &
        (df_merged['PD_POOL_ID_prev'] == df_merged['PD_POOL_ID'])
    )

    filtered = df_merged[filter_cond]

    name_prev = name + '_prev'
    value = filtered[name].sum() - filtered[name_prev].sum()
    return value

def run_ecl_movement_loan_on(context, param, report_temp, ecl_result, ecl_result_prev, wb):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    ecl_result: ECL_calculation_result_files_deal.csv

    return: wb: openpyxl.workbook
    """
     # load report template
    tab_name = 'ECL_movement_loan_on'
    sheet = wb[tab_name]
    group_dfs = merge_ECL_results(context, param, ecl_result, ecl_result_prev)
    start_rows = {
        'business': 5,
        'consumer': 32,
        'agricultural': 59,
    }
    ECL_columns = ['B', 'C', 'D']
    GCM_columns = ['F', 'G', 'H']
    columns_to_sum = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

    functions = [
        compute_ECL_PRIN_total_prev,
        compute_ECL_PRIN_new,
        compute_ECL_PRIN_exposures,
        compute_ECL_PRIN_transfer1,
        compute_ECL_PRIN_transfer2,
        compute_ECL_PRIN_transfer3,
        compute_Changes_AccruedInterest,
        compute_Changes_ModelAssumptions
    ]

    for group in group_dfs:
        df = group_dfs[group]
        start_row = start_rows[group]
        
        for row_idx, func in enumerate(functions, start=start_row):
            for stage, col in enumerate(ECL_columns, start=1):
                sheet[f'{col}{row_idx}'] = func(df, 'ECL_ULTIMATE_LCY', stage)
        for row_idx, func in enumerate(functions, start=start_row):
            for stage, col in enumerate(GCM_columns, start=1):
                sheet[f'{col}{row_idx}'] = func(df, 'IFRS_AMOUNT_LCY', stage)
        
        # write zeros
        for row in [start_row+8, start_row+15, start_row+16]:
            for col in ECL_columns + GCM_columns:
                sheet[f'{col}{row}'] = 0
            sheet[f'E{row}'] = f'=SUM(B{row}:D{row})'
            sheet[f'I{row}'] = f'=SUM(F{row}:H{row})'

        # write sums
        for row in range(start_row, start_row+9):
            sheet[f'E{row}'] = f'=SUM(B{row}:D{row})'
            sheet[f'I{row}'] = f'=SUM(F{row}:H{row})'    

        for col in columns_to_sum:
            sheet[f'{col}{start_row+10}'] = f'=SUM({col}{start_row}:{col}{start_row+8})'
            sheet[f'{col}{start_row+18}'] = f'=SUM({col}{start_row},{col}{start_row+10},{col}{start_row+15},{col}{start_row+16})'

    return wb

def run_ecl_movement_loan_on_summed(context, param, report_temp, all_movement_sheets, wb):
    """
    context: enviorment setting
    param: all parameter list
    report_temp: reporting empty template
    all_movement_sheets: list of movement sheets from adjacent months
    wb: workbook to write to

    return: wb: openpyxl.workbook
    """
    # load report template
    tab_name = 'ECL_movement_loan_on'
    sheet = wb[tab_name]
    rp_f = report_basic.report_cond_basic(context=context)
    RUN_YYMM = rp_f.format_date_slash(context.run_yymm)
    PREV_YYMM = rp_f.format_date_slash(context.prev_yymm)
    sheet['A5'] = PREV_YYMM
    sheet['A23'] = RUN_YYMM
    sheet['A32'] = PREV_YYMM
    sheet['A50'] = RUN_YYMM
    sheet['A59'] = PREV_YYMM
    sheet['A77'] = RUN_YYMM
    
    def write_values_summed(all_movement_sheets, start_row):
        ECL_columns = ['B', 'C', 'D']
        GCM_columns = ['F', 'G', 'H']
        columns_to_sum = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        
        # Get the first sheet for total_prev row (start_row)
        first_sheet = all_movement_sheets[0]
        
        # For total_prev row (start_row), use the first sheet's values
        for col in ECL_columns + GCM_columns:
            cell_value = first_sheet[f'{col}{start_row}'].value
            sheet[f'{col}{start_row}'] = cell_value
        
        # For all other rows, sum up values from all movement sheets
        # Define the row ranges for this section (excluding the total_prev row)
        section_rows = list(range(start_row+1, start_row+9)) + list(range(start_row+15, start_row+17))
        
        # Sum up values for all rows except the total_prev row
        for row in section_rows:
            for col in ECL_columns + GCM_columns:
                total_value = 0
                for i, movement_sheet in enumerate(all_movement_sheets):
                    cell_value = movement_sheet[f'{col}{row}'].value
                    if cell_value is not None and cell_value != '':
                        # Handle formula cells by extracting numeric value
                        if isinstance(cell_value, str) and cell_value.startswith('='):
                            # For formula cells, try to get the calculated value
                            try:
                                total_value += float(cell_value)
                            except:
                                # If we can't parse the formula, skip this sheet
                                continue
                        else:
                            try:
                                total_value += float(cell_value)
                            except:
                                # If we can't convert to float, skip this sheet
                                continue
                sheet[f'{col}{row}'] = total_value
        
        # Write sum formulas for the total_prev row
        sheet[f'E{start_row}'] = f'=SUM(B{start_row}:D{start_row})'
        sheet[f'I{start_row}'] = f'=SUM(F{start_row}:H{start_row})'
        
        # Write sum formulas for all other rows
        for row in section_rows:
            sheet[f'E{row}'] = f'=SUM(B{row}:D{row})'
            sheet[f'I{row}'] = f'=SUM(F{row}:H{row})'
        
        # Write zeros
        zero_rows = [start_row+8, start_row+15, start_row+16]  
        for row in zero_rows:
            for col in ECL_columns + GCM_columns:
                sheet[f'{col}{row}'] = 0
            sheet[f'E{row}'] = f'=SUM(B{row}:D{row})'
            sheet[f'I{row}'] = f'=SUM(F{row}:H{row})'
        
        # Write sums
        for col in columns_to_sum:
            sheet[f'{col}{start_row+10}'] = f'=SUM({col}{start_row+1}:{col}{start_row+9})'
            sheet[f'{col}{start_row+18}'] = f'=SUM({col}{start_row},{col}{start_row+10},{col}{start_row+15},{col}{start_row+16})'
        
        return

    # Write values for each section using start_row
    write_values_summed(all_movement_sheets, 5)   # Business section
    write_values_summed(all_movement_sheets, 32)  # Consumer section  
    write_values_summed(all_movement_sheets, 59)  # Agricultural section

    return wb
