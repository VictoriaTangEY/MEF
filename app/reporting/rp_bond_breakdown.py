from input_handler.load_parameters import load_configuration_file
from input_handler.env_setting import run_setting
from pathlib import Path
import pandas as pd
from datetime import datetime
from reporting import report_basic
import numpy as np
import os
import re
from input_handler import env_setting

# Data loading ================================================================
interim_path = None
def set_interim_path(context):
    global interim_path
    production_path = context.resultPath.parents[2]
    interim_dir = Path(production_path) / str(context.run_yymm) / 'data' / '03_report' / 'interim' / 'bondbreakdown'
    interim_dir.mkdir(parents=True, exist_ok=True)
    interim_path = interim_dir
    return interim_path

def load_bond_data(context, param):
    # Load bond breakdown data
    bond_file_path = context.inDataPath / 'bondbreakdown.xlsx'
    bond_df = pd.read_excel(bond_file_path, sheet_name='bond')
    daily_coupon_df = pd.read_excel(bond_file_path, sheet_name='daily_coupon')
    
    # Load exchange rate data
    exchange_rate_file_path = context.inDataPath / 'exchange_rate_table_daily.csv'
    exchange_rate_df = pd.read_csv(exchange_rate_file_path)
    
    # Process date format conversion
    # Convert Date column from M/D/YYYY format to datetime
    exchange_rate_df['DATE'] = pd.to_datetime(exchange_rate_df['Date'], format='%d/%m/%Y')
    
    # Ensure Exchange Rate column is numeric type
    exchange_rate_df['Exchange Rate'] = pd.to_numeric(exchange_rate_df['Exchange Rate'], errors='coerce')
 
    return bond_df, daily_coupon_df, exchange_rate_df

def load_prev_bond_data(context, param, prev_yymm):
    # Load prev bond breakdown data
    previnDataPath = context.previnDataPath
    prev_bond_file_path = previnDataPath / 'bondbreakdown.xlsx'
    
    try:
        bond_df_prev = pd.read_excel(prev_bond_file_path, sheet_name='bond')
        daily_coupon_df_prev = pd.read_excel(prev_bond_file_path, sheet_name='daily_coupon')
    except Exception as e:
        print(f"Warning: Failed to load prev_yymm {prev_yymm} bondbreakdown data: {e}")
        print(f"File path attempted: {prev_bond_file_path}")
        print(f"Using empty DataFrames as fallback")
        bond_df_prev = pd.DataFrame()
        daily_coupon_df_prev = pd.DataFrame()
    
    return bond_df_prev, daily_coupon_df_prev

def get_exchange_rate(exchange_rate_df, input_date):
    # input_date must be of format YYYY-MM-DD datetime
    match = exchange_rate_df.loc[exchange_rate_df['DATE'] == input_date, 'Exchange Rate']
    if not match.empty:
        return match.iloc[0]
    else:
        return None

def merge_ecl_results_with_bond(context, param, ecl_result, ecl_result_prev, bond_df):
    """
    Merge ECL results and bond data
    
    Strategy: Use bond tab as the main table, merge ECL results into it
    This ensures all required bond columns are preserved

    Args:
        context: environment setting
        param: all parameter list
        ecl_result: current ECL results
        ecl_result_prev: previous ECL results
        bond_df: bond tab data

    Returns:
        merged_df: merged data with bond tab as base
    """
    # First add _prev suffix to all columns in ecl_result_prev (except CONTRACT_ID and CUST_ID)
    ecl_result_prev_renamed = ecl_result_prev.copy()
    rename_cols = {}
    for col in ecl_result_prev_renamed.columns:
        if col not in ['CONTRACT_ID', 'CUST_ID']:
            rename_cols[col] = f"{col}_prev"
    
    ecl_result_prev_renamed = ecl_result_prev_renamed.rename(columns=rename_cols)
    
    # Select required columns and merge ECL results
    cols_curr = ['CONTRACT_ID', 'CUST_ID', 'IFRS9_MEAS_TYPE', 'INIT_DATE', 'MAT_DATE', 'DRAWN_BAL_OCY', 'IFRS_AMOUNT_LCY', 'EFF_INT_RT', 'AMORTIZATION_OCY', 'IFRS9_PD_12M_MADJ', 'IFRS9_LGD_12M', 'ECL_ULTIMATE_LCY']
    cols_prev = ['CONTRACT_ID', 'CUST_ID', 'IFRS9_MEAS_TYPE_prev', 'INIT_DATE_prev', 'MAT_DATE_prev', 'DRAWN_BAL_OCY_prev', 'IFRS_AMOUNT_LCY_prev', 'EFF_INT_RT_prev', 'AMORTIZATION_OCY_prev', 'IFRS9_PD_12M_MADJ_prev', 'IFRS9_LGD_12M_prev', 'ECL_ULTIMATE_LCY_prev']
    
    # Ensure columns exist, skip if not
    available_cols_curr = [col for col in cols_curr if col in ecl_result.columns]
    available_cols_prev = [col for col in cols_prev if col in ecl_result_prev_renamed.columns]
    
    df_curr = ecl_result[available_cols_curr].copy()
    df_prev = ecl_result_prev_renamed[available_cols_prev].copy()
    
    # Merge ECL results
    merged_ecl = df_curr.merge(df_prev, on=['CONTRACT_ID', 'CUST_ID'], how='outer')
    
    # Start with bond_df as the base table (preserves all bond columns)
    merged_df = bond_df.copy()
    
    # Merge current ECL results into bond_df
    merged_df = merged_df.merge(df_curr, on=['CONTRACT_ID', 'CUST_ID'], how='left')
    
    # Merge previous ECL results into the result
    merged_df = merged_df.merge(df_prev, on=['CONTRACT_ID', 'CUST_ID'], how='left')
    
    # Ensure numeric columns have correct types
    numeric_columns = [
        'DRAWN_BAL_OCY', 'DRAWN_BAL_OCY_prev', 'IFRS_AMOUNT_LCY', 'IFRS_AMOUNT_LCY_prev',
        'EFF_INT_RT', 'EFF_INT_RT_prev', 'AMORTIZATION_OCY', 'AMORTIZATION_OCY_prev', 
        'interest', 'ECL_ULTIMATE_LCY', 'ECL_ULTIMATE_LCY_prev'
    ]
    
    for col in numeric_columns:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
    
    # # Filter out invalid date data
    # # 1899-12-30 usually indicates invalid dates in Excel
    # if 'INIT_DATE' in merged_df.columns:
    #     merged_df = merged_df[merged_df['INIT_DATE'] != '1899-12-30']
    #     print(f"Filtered out bonds with invalid INIT_DATE, remaining: {len(merged_df)}")

    # if 'MAT_DATE' in merged_df.columns:
    #     merged_df = merged_df[merged_df['MAT_DATE'] != '1899-12-30']
    #     print(f"Filtered out bonds with invalid MAT_DATE, remaining: {len(merged_df)}")

    try:
        output_path = interim_path / "merged_ecl_bond.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"Warning: Failed to save to interim folder: {e}")

    return merged_df

def calculate_daily_discount_premium(merged_df, exchange_rate_df, run_yymm, prev_yymm):
    prev_date = datetime.strptime(str(prev_yymm), "%Y%m%d")
    run_date = datetime.strptime(str(run_yymm), "%Y%m%d")
    total_days = (run_date - prev_date).days + 1

    all_dates = [(prev_date + pd.Timedelta(days=day)).strftime('%Y-%m-%d') for day in range(total_days)]
    
    for date_str in all_dates:
        merged_df[date_str] = 0.0
    
    for idx, row in merged_df.iterrows():
        contract_id = row['CONTRACT_ID']
        cust_id = row['CUST_ID']
        
        drawn_balance = row['DRAWN_BAL_OCY']
        eff_int_rate = row['EFF_INT_RT']
        amortization = row['AMORTIZATION_OCY']
        interest_rate = row['interest']
        
        if pd.isna(drawn_balance) or pd.isna(interest_rate) or pd.isna(eff_int_rate) or pd.isna(amortization):
            continue
        
        interest_receivable = drawn_balance * interest_rate / 360
        npv_current = amortization
        
        try:
            payment_date = pd.to_datetime(row['payment_date']) if pd.notna(row['payment_date']) else prev_date
            row_start_date = max(prev_date, payment_date)
            
            maturity_date = pd.to_datetime(row['MAT_DATE']) if pd.notna(row['MAT_DATE']) else run_date
            row_end_date = min(run_date, maturity_date)
            
            row_num_days = (row_end_date - row_start_date).days + 1
            if row_num_days <= 0:
                row_num_days = 1
                
        except Exception as e:
            print(f"Warning: Error calculating date range for contract {contract_id}: {e}")
            row_num_days = 1
            row_start_date = prev_date
            row_end_date = prev_date
        
        for day in range(row_num_days):
            interest_income = npv_current * eff_int_rate / 360
            daily_discount_premium = interest_receivable - interest_income
            
            npv_next = npv_current + daily_discount_premium
            
            day_date = (row_start_date + pd.Timedelta(days=day)).strftime('%Y-%m-%d')
            
            if day_date in merged_df.columns:
                merged_df.at[idx, day_date] = daily_discount_premium
            
            npv_current = npv_next
    
    try:
        output_path = interim_path / "merged_daily_bond.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"Warning: Failed to save to interim folder: {e}")
    
    return merged_df

def merge_ecl_and_bond_with_prev(context, param, ecl_result, ecl_result_prev, bond_df, bond_df_prev):
    # -------------------------------
    # Step 0: Define ECL columns to keep
    # -------------------------------
    ecl_cols = ['CONTRACT_ID', 'CUST_ID', 'IFRS9_MEAS_TYPE', 'INIT_DATE', 'MAT_DATE', 
                'DRAWN_BAL_OCY', 'IFRS_AMOUNT_LCY', 'EFF_INT_RT', 'AMORTIZATION_OCY', 
                'IFRS9_PD_12M_MADJ', 'IFRS9_LGD_12M', 'ECL_ULTIMATE_LCY']

    # -------------------------------
    # Step 1: Prepare previous ECL with _prev suffix
    # -------------------------------
    ecl_prev_selected = ecl_result_prev[ecl_cols].copy()
    rename_prev = {col: f"{col}_prev" for col in ecl_prev_selected.columns if col not in ['CONTRACT_ID', 'CUST_ID']}
    ecl_prev_selected = ecl_prev_selected.rename(columns=rename_prev)

    # -------------------------------
    # Step 2: Prepare previous bond with _prev suffix
    # -------------------------------
    bond_prev_renamed = bond_df_prev.copy()
    rename_prev_bond = {col: f"{col}_prev" for col in bond_prev_renamed.columns if col not in ['CONTRACT_ID', 'CUST_ID']}
    bond_prev_renamed = bond_prev_renamed.rename(columns=rename_prev_bond)

    # -------------------------------
    # Step 3: Merge current and previous ECL
    # -------------------------------
    df_curr = ecl_result[ecl_cols].copy()
    merged_ecl = df_curr.merge(ecl_prev_selected, on=['CONTRACT_ID', 'CUST_ID'], how='outer')

    # -------------------------------
    # Step 4: Start with bond_df as base
    # -------------------------------
    merged_df = bond_df.copy()
    merged_df = merged_df.merge(df_curr, on=['CONTRACT_ID', 'CUST_ID'], how='left')
    merged_df = merged_df.merge(ecl_prev_selected, on=['CONTRACT_ID', 'CUST_ID'], how='left')
    merged_df = merged_df.merge(bond_prev_renamed, on=['CONTRACT_ID', 'CUST_ID'], how='left')

    # -------------------------------
    # Step 5: Ensure numeric columns
    # -------------------------------
    numeric_columns = ['DRAWN_BAL_OCY', 'DRAWN_BAL_OCY_prev', 'IFRS_AMOUNT_LCY', 'IFRS_AMOUNT_LCY_prev',
                       'EFF_INT_RT', 'EFF_INT_RT_prev', 'AMORTIZATION_OCY', 'AMORTIZATION_OCY_prev',
                       'interest', 'ECL_ULTIMATE_LCY', 'ECL_ULTIMATE_LCY_prev']
    for col in numeric_columns:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    return merged_df

def add_ead_columns(merged_df, exchange_rate_df, run_yymm, prev_yymm):
    # -------------------------------
    # Step 0: Prepare exchange rate mapping
    # -------------------------------
    exchange_rate_df['DATE_DT'] = pd.to_datetime(exchange_rate_df['DATE'], dayfirst=True, errors='coerce')
    exrate_map = dict(zip(exchange_rate_df['DATE_DT'], exchange_rate_df['Exchange Rate']))

    run_date = pd.to_datetime(str(run_yymm), format="%Y%m%d")
    prev_date = pd.to_datetime(str(prev_yymm), format="%Y%m%d")

    current_exrate = exrate_map.get(run_date, 1.0)
    prev_exrate = exrate_map.get(prev_date, 1.0)

    # -------------------------------
    # Step 1: Calculate EAD_curr
    # -------------------------------
    for col in ['current_balance', 'accrued_interest_movement', 'unamortized_discount_movement']:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)

    merged_df['EAD_curr'] = (
        merged_df['current_balance'] + 
        merged_df['accrued_interest_movement'] + 
        merged_df['unamortized_discount_movement']
    ) * current_exrate

    # -------------------------------
    # Step 2: Calculate EAD_prev
    # -------------------------------
    for col in ['current_balance_prev', 'accrued_interest_movement_prev', 'unamortized_discount_movement_prev']:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
        else:
            merged_df[col] = 0

    merged_df['EAD_prev'] = (
        merged_df['current_balance_prev'] +
        merged_df['accrued_interest_movement_prev'] +
        merged_df['unamortized_discount_movement_prev']
    ) * prev_exrate

    # -------------------------------
    # Step 3: Save CSV
    # -------------------------------
    try:
        output_path = interim_path / "merged_ecl_bond_full.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"Warning: Failed to save to interim folder: {e}")

    return merged_df

# Gross carrying amount ==============================================================
def calculate_gross_carrying_amount_purchase_addition(merged_df, exchange_rate_df, run_yymm, prev_yymm, meas_type):
    # Filter condition: IFRS9_MEAS_TYPE = meas_type and payment_date between RUN_YYMM and PREV_YYMM
    filtered_df = merged_df[
        (merged_df['IFRS9_MEAS_TYPE'] == meas_type) &
        (merged_df['payment_date'] >= str(prev_yymm)) &
        (merged_df['payment_date'] <= str(run_yymm))
    ]
    
    if filtered_df.empty:
        return 0.0
    
    result = 0.0
    
    for _, row in filtered_df.iterrows():
        contract_id = row['CONTRACT_ID']
        cust_id = row['CUST_ID']
        drawn_bal_prev = row['DRAWN_BAL_OCY_prev']
        
        # Get exchange rate for purchased date
        payment_date = row['payment_date']
        payment_exchange_rate = get_exchange_rate(exchange_rate_df, payment_date)
        
        # Calculate: DRAWN_BAL_OCY_prev * EXRATE_purchased / 1000
        amount = drawn_bal_prev * payment_exchange_rate / 1000
        result += amount
    
    return result

def calculate_gross_carrying_amount_changes_accrued_interest(merged_df, daily_coupon_df, exchange_rate_df, run_yymm, prev_yymm, meas_type):
    # -------------------------------
    # Step 0: Filter by meas_type
    # -------------------------------
    filtered_df = merged_df[merged_df['IFRS9_MEAS_TYPE'] == meas_type].copy()
    if filtered_df.empty:
        return 0.0

    # -------------------------------
    # Step 1: Prepare exchange_rate_map
    # -------------------------------
    exchange_rate_df['DATE_DT'] = pd.to_datetime(exchange_rate_df['DATE'], dayfirst=True)
    exrate_map = dict(zip(exchange_rate_df['DATE_DT'], exchange_rate_df['Exchange Rate']))

    # -------------------------------
    # Step 2: Convert date columns to datetime
    # -------------------------------
    def convert_cols_to_datetime(df):
        new_cols = {}
        for col in df.columns:
            if isinstance(col, str) and (col.count('-') == 2 or '/' in col):
                try:
                    new_cols[col] = pd.to_datetime(col, dayfirst=True)
                except:
                    new_cols[col] = col
        df.rename(columns=new_cols, inplace=True)
        return df

    filtered_df = convert_cols_to_datetime(filtered_df)
    daily_coupon_df = convert_cols_to_datetime(daily_coupon_df)

    # -------------------------------
    # Step 3: SUM_DAILY_EXCHANGE
    # -------------------------------
    daily_cols = [col for col in filtered_df.columns if isinstance(col, pd.Timestamp)]
    sum_daily_exchange = []
    for idx, row in filtered_df.iterrows():
        total = 0
        for date_col in daily_cols:
            premium = row[date_col]
            exrate = exrate_map.get(date_col)
            if exrate is None or pd.isna(premium):
                continue
            total += premium * exrate
        sum_daily_exchange.append(total)
    filtered_df['SUM_DAILY_EXCHANGE'] = sum_daily_exchange

    # -------------------------------
    # Step 4: SUM_ADJ_EXCHANGE
    # -------------------------------
    sum_adj_exchange = []
    for idx, row in filtered_df.iterrows():
        payment_date = row.get('payment_date')
        if pd.notna(payment_date):
            if isinstance(payment_date, str):
                try:
                    payment_date_dt = pd.to_datetime(payment_date, dayfirst=True)
                except:
                    payment_date_dt = pd.to_datetime(payment_date)
            else:
                payment_date_dt = pd.to_datetime(payment_date)
            exrate = exrate_map.get(payment_date_dt, 1.0)
        else:
            exrate = 1.0

        adj_val = (-row.get('DRAWN_BAL_OCY_prev', 0)
                   + row.get('coupon_received', 0)
                   - row.get('accrued_interest', 0)) * exrate
        sum_adj_exchange.append(adj_val)
    filtered_df['SUM_ADJ_EXCHANGE'] = sum_adj_exchange

    # -------------------------------
    # Step 5: SUM_DAILY_COUPON
    # -------------------------------
    coupon_cols = [col for col in filtered_df.columns if isinstance(col, pd.Timestamp) and col in daily_coupon_df.columns]
    sum_daily_coupon = []

    for idx, row in filtered_df.iterrows():
        cust_id = row['CUST_ID']
        contract_id = row['CONTRACT_ID']
        coupon_row = daily_coupon_df[(daily_coupon_df['CUST_ID'] == cust_id) &
                                     (daily_coupon_df['CONTRACT_ID'] == contract_id)]
        total = 0
        if not coupon_row.empty:
            coupon_row = coupon_row.iloc[0]
            for date_col in coupon_cols:
                coupon_val = coupon_row[date_col]
                exrate = exrate_map.get(date_col)
                if exrate is None or pd.isna(coupon_val):
                    continue
                total += coupon_val * exrate
        sum_daily_coupon.append(total)
    filtered_df['SUM_DAILY_COUPON'] = sum_daily_coupon

    # -------------------------------
    # Step 6: CHANGES_ACCRUED_INTEREST
    # -------------------------------
    filtered_df['CHANGES_ACCRUED_INTEREST'] = (
        filtered_df['SUM_DAILY_EXCHANGE'] +
        filtered_df['SUM_ADJ_EXCHANGE'] +
        filtered_df['SUM_DAILY_COUPON']
    )

    # -------------------------------
    # Step 7: Save filtered_df
    # -------------------------------
    try:
        output_path = interim_path / f"merged_accrued_interest_{meas_type}.csv"
        filtered_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"Warning: Failed to save to interim folder: {e}")

    # -------------------------------
    # Step 8: Return total sum
    # -------------------------------
    result = filtered_df['CHANGES_ACCRUED_INTEREST'].sum()
    return result

def calculate_gross_carrying_amount_matured_repaid(merged_df, exchange_rate_df, run_yymm, prev_yymm, meas_type):
    # Filter condition: IFRS9_MEAS_TYPE = meas_type and MAT_DATE between RUN_YYMM and PREV_YYMM
    # Ensure type consistency for date comparison
    filtered_df = merged_df[
        (merged_df['IFRS9_MEAS_TYPE'] == meas_type) &
        (merged_df['MAT_DATE'] >= str(prev_yymm)) &
        (merged_df['MAT_DATE'] <= str(run_yymm))
    ]
    
    if filtered_df.empty:
        return 0.0
    
    result = 0.0
    
    for _, row in filtered_df.iterrows():
        drawn_bal = row['DRAWN_BAL_OCY']
        mat_date = row['MAT_DATE']
        
        # Get exchange rate for maturity date
        mat_exchange_rate = get_exchange_rate(exchange_rate_df, mat_date)
        
        # Calculate: -sum(DRAWN_BAL_OCY * EXRATE_MAT / 1000)
        amount = -(drawn_bal * mat_exchange_rate / 1000)
        result += amount
    
    return result

def calculate_gross_carrying_amount_decrease_fair_value(merged_df, exchange_rate_df, run_yymm, prev_yymm, meas_type, net_reversal_charge_period):
    # Filter condition: IFRS9_MEAS_TYPE = meas_type
    filtered_df = merged_df[merged_df['IFRS9_MEAS_TYPE'] == meas_type]
    
    if filtered_df.empty:
        return 0.0
    
    result = 0.0
    
    for _, row in filtered_df.iterrows():
        drawn_bal = row['DRAWN_BAL_OCY']
        drawn_bal_prev = row['DRAWN_BAL_OCY_prev']
        
        # Get exchange rates
        run_date = pd.to_datetime(str(run_yymm), format="%Y%m%d")
        prev_date = pd.to_datetime(str(prev_yymm), format="%Y%m%d")

        run_exchange_rate = get_exchange_rate(exchange_rate_df, run_date)
        prev_exchange_rate = get_exchange_rate(exchange_rate_df, prev_date)
        
        # Calculate: (DRAWN_BAL_OCY * RUN_YYMM - DRAWN_BAL_OCY_prev * PREV_YYMM) / 1000
        amount = (drawn_bal * run_exchange_rate - drawn_bal_prev * prev_exchange_rate) / 1000
        result += amount
    
    # Add Net (reversal)/charge for the period (Note 16)
    result += net_reversal_charge_period
    
    return result

# ECL ==============================================================
def calculate_ecl_purchase_new_investments(merged_df, meas_type):
    # -------------------------------
    # Step 0: Filter by IFRS9_MEAS_TYPE
    # -------------------------------
    filtered_df = merged_df[merged_df['IFRS9_MEAS_TYPE'] == meas_type].copy()
    
    if filtered_df.empty:
        return 0.0

    # -------------------------------
    # Step 1: Ensure numeric columns
    # -------------------------------
    for col in ['EAD_curr', 'EAD_prev', 'IFRS9_PD_12M_MADJ', 'IFRS9_LGD_12M']:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)
        else:
            filtered_df[col] = 0

    # -------------------------------
    # Step 2: Only rows where EAD_curr > EAD_prev
    # -------------------------------
    filtered_df = filtered_df[filtered_df['EAD_curr'] > filtered_df['EAD_prev']]

    if filtered_df.empty:
        return 0.0

    # -------------------------------
    # Step 3: Calculate ECL contribution
    # -------------------------------
    ecl_values = (filtered_df['EAD_curr'] - filtered_df['EAD_prev']) * \
                 filtered_df['IFRS9_PD_12M_MADJ'] * \
                 filtered_df['IFRS9_LGD_12M'] / 1000

    total_ecl = ecl_values.sum()

    return total_ecl

def calculate_ecl_changes_input_ecl_calculation(merged_df, meas_type='FVOCI'):
    # -------------------------------
    # Step 0: Filter by IFRS9_MEAS_TYPE
    # -------------------------------
    filtered_df = merged_df[merged_df['IFRS9_MEAS_TYPE'] == meas_type].copy()
    
    if filtered_df.empty:
        return 0.0

    # -------------------------------
    # Step 1: Ensure numeric columns
    # -------------------------------
    for col in ['EAD_prev', 'IFRS9_PD_12M_MADJ', 'IFRS9_LGD_12M', 'IFRS9_PD_12M_MADJ_prev', 'IFRS9_LGD_12M_prev']:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)
        else:
            filtered_df[col] = 0

    # -------------------------------
    # Step 2: Calculate ECL contribution
    # -------------------------------
    ecl_values = filtered_df['EAD_prev'] * (
        filtered_df['IFRS9_PD_12M_MADJ'] * filtered_df['IFRS9_LGD_12M'] -
        filtered_df['IFRS9_PD_12M_MADJ_prev'] * filtered_df['IFRS9_LGD_12M_prev']
    )

    total_ecl = ecl_values.sum()

    return total_ecl

def calculate_ecl_recoveries_matured_repaid(merged_df, meas_type):
    # -------------------------------
    # Step 0: Filter by IFRS9_MEAS_TYPE
    # -------------------------------
    filtered_df = merged_df[merged_df['IFRS9_MEAS_TYPE'] == meas_type].copy()
    
    if filtered_df.empty:
        return 0.0

    # -------------------------------
    # Step 1: Ensure numeric columns
    # -------------------------------
    for col in ['EAD_curr', 'EAD_prev', 'IFRS9_PD_12M_MADJ', 'IFRS9_LGD_12M']:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)
        else:
            filtered_df[col] = 0

    # -------------------------------
    # Step 2: Only rows where EAD_curr > EAD_prev
    # -------------------------------
    filtered_df = filtered_df[filtered_df['EAD_curr'] < filtered_df['EAD_prev']]

    if filtered_df.empty:
        return 0.0

    # -------------------------------
    # Step 3: Calculate ECL contribution
    # -------------------------------
    ecl_values = (filtered_df['EAD_curr'] - filtered_df['EAD_prev']) * \
                 filtered_df['IFRS9_PD_12M_MADJ'] * \
                 filtered_df['IFRS9_LGD_12M'] / 1000

    total_ecl = ecl_values.sum()

    return total_ecl

def calculate_net_reversal_charge_period(ecl_purchase_new, ecl_changes_input, ecl_recoveries_matured):
    # Net (reversal)/charge for the period = ECL Purchase of new investments + ECL Changes to input used for ECL calculation + ECL Recoveries/matured/repaid
    result = ecl_purchase_new + ecl_changes_input + ecl_recoveries_matured
    return result

# Run ==============================================================
def run_bond_breakdown(context, param, report_temp, ecl_result, ecl_result_prev, wb):
    """
    Run the bond breakdown report: merges ECL with bond data, calculates daily discount/premium,
    EAD, FVOCI/AC components, and writes results to Excel sheet.
    """

    tab_name = 'Bond_breakdown'
    rp_f = report_basic.report_cond_basic(context=context)
    df_conditions = param['Conditions'].query("REPORT_NAME == @tab_name")
    
    interim_path = set_interim_path(context)
    
    # Filter ECL results
    ecl_result_filtered = rp_f.overall_filter(df_conditions, ecl_result)
    ecl_result_prev_filtered = rp_f.overall_filter(df_conditions, ecl_result_prev)

    # Load bond data
    bond_df, daily_coupon_df, exchange_rate_df = load_bond_data(context, param)
    bond_df_prev, daily_coupon_df_prev = load_prev_bond_data(context, param, context.prev_yymm)

    # Merge current and previous ECL results with bond data
    merged_df = merge_ecl_results_with_bond(context, param, ecl_result_filtered, ecl_result_prev_filtered, bond_df)
    merged_df_prev = merge_ecl_and_bond_with_prev(context, param, ecl_result_filtered, ecl_result_prev_filtered, bond_df, bond_df_prev)

    # Calculate daily discount/premium and EAD columns
    merged_df = calculate_daily_discount_premium(merged_df, exchange_rate_df, context.run_yymm, context.prev_yymm)
    merged_df_prev = add_ead_columns(merged_df_prev, exchange_rate_df, context.run_yymm, context.prev_yymm)

    # Helper to calculate carrying amounts and ECL for a measurement type
    def calc_meas_type_components(meas_type):
        curr_mask = merged_df['IFRS9_MEAS_TYPE'] == meas_type
        prev_mask = merged_df['IFRS9_MEAS_TYPE'] == meas_type

        gross_curr = merged_df.loc[curr_mask, 'IFRS_AMOUNT_LCY'].sum()
        gross_prev = merged_df.loc[curr_mask, 'IFRS_AMOUNT_LCY_prev'].sum()
        ecl_final_prev = merged_df.loc[curr_mask, 'ECL_ULTIMATE_LCY_prev'].sum()

        purchase_addition = calculate_gross_carrying_amount_purchase_addition(merged_df, exchange_rate_df, context.run_yymm, context.prev_yymm, meas_type)
        changes_accrued_interest = calculate_gross_carrying_amount_changes_accrued_interest(merged_df, daily_coupon_df, exchange_rate_df, context.run_yymm, context.prev_yymm, meas_type)
        matured_repaid = calculate_gross_carrying_amount_matured_repaid(merged_df, exchange_rate_df, context.run_yymm, context.prev_yymm, meas_type)
        decrease_fair_value = calculate_gross_carrying_amount_decrease_fair_value(merged_df, exchange_rate_df, context.run_yymm, context.prev_yymm, meas_type, 0)

        ecl_purchase_new = calculate_ecl_purchase_new_investments(merged_df_prev, meas_type)
        ecl_changes_input = calculate_ecl_changes_input_ecl_calculation(merged_df_prev, meas_type)
        ecl_recoveries_matured = calculate_ecl_recoveries_matured_repaid(merged_df_prev, meas_type)
        net_reversal_charge = calculate_net_reversal_charge_period(ecl_purchase_new, ecl_changes_input, ecl_recoveries_matured)

        decrease_fair_value = calculate_gross_carrying_amount_decrease_fair_value(merged_df, exchange_rate_df, context.run_yymm, context.prev_yymm, meas_type, net_reversal_charge)

        gross_changes = gross_curr - gross_prev - purchase_addition - changes_accrued_interest - matured_repaid - decrease_fair_value
        ecl_total = ecl_final_prev + ecl_purchase_new + ecl_changes_input + ecl_recoveries_matured + net_reversal_charge
        net_carrying_amount = gross_curr - ecl_total

        return {
            'gross_prev': gross_prev,
            'gross_curr': gross_curr,
            'ecl_final_prev': ecl_final_prev,
            'purchase_addition': purchase_addition,
            'changes_accrued_interest': changes_accrued_interest,
            'matured_repaid': matured_repaid,
            'decrease_fair_value': decrease_fair_value,
            'gross_changes': gross_changes,
            'ecl_purchase_new': ecl_purchase_new,
            'ecl_changes_input': ecl_changes_input,
            'ecl_recoveries_matured': ecl_recoveries_matured,
            'net_reversal_charge': net_reversal_charge,
            'ecl_total': ecl_total,
            'net_carrying_amount': net_carrying_amount
        }

    # Calculate components for FVOCI and AC
    fvoci = calc_meas_type_components('FVOCI')
    ac = calc_meas_type_components('AC')

    # Write results to Excel sheet
    sheet = wb[tab_name]
    rp_f = report_basic.report_cond_basic(context=context)
    RUN_YYMM = rp_f.format_date_slash(context.run_yymm)
    PREV_YYMM = rp_f.format_date_slash(context.prev_yymm)
    sheet['A4'] = f'Gross carrying amount at {PREV_YYMM}'
    sheet['A10'] = f'At {RUN_YYMM}'
    sheet['A12'] = f'ECL allowance at {PREV_YYMM}'
    sheet['A17'] = f'At {RUN_YYMM}'
    sheet['A19'] = f'At {RUN_YYMM}'
    
    # Gross carrying amounts
    sheet['B4'], sheet['C4'] = fvoci['gross_prev'], ac['gross_prev']
    sheet['B10'], sheet['C10'] = fvoci['gross_curr'], ac['gross_curr']

    # ECL FINAL prev
    sheet['B12'], sheet['C12'] = fvoci['ecl_final_prev'], ac['ecl_final_prev']

    # FVOCI components
    sheet['B5'] = fvoci['purchase_addition']
    sheet['B6'] = fvoci['changes_accrued_interest']
    sheet['B7'] = fvoci['matured_repaid']
    sheet['B8'] = fvoci['gross_changes']
    sheet['B9'] = fvoci['decrease_fair_value']
    sheet['B13'] = fvoci['ecl_purchase_new']
    sheet['B14'] = fvoci['ecl_changes_input']
    sheet['B15'] = fvoci['ecl_recoveries_matured']
    sheet['B16'] = fvoci['net_reversal_charge']
    sheet['B17'] = fvoci['ecl_total']
    sheet['B19'] = fvoci['net_carrying_amount']

    # AC components
    sheet['C5'] = ac['purchase_addition']
    sheet['C6'] = ac['changes_accrued_interest']
    sheet['C7'] = ac['matured_repaid']
    sheet['C8'] = ac['gross_changes']
    sheet['C9'] = ac['decrease_fair_value']
    sheet['C13'] = ac['ecl_purchase_new']
    sheet['C14'] = ac['ecl_changes_input']
    sheet['C15'] = ac['ecl_recoveries_matured']
    sheet['C16'] = ac['net_reversal_charge']
    sheet['C17'] = ac['ecl_total']
    sheet['C19'] = ac['net_carrying_amount']

    return wb
