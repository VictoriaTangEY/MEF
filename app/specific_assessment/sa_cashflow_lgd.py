# Load packages
# Load packages
from util.loggers import createLogHandler
from input_handler.data_preprocessor import data_preprocessor
from input_handler.load_parameters import load_configuration_file, load_parameters
from input_handler.env_setting import run_setting
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
from scipy.stats import norm
from typing import Tuple, Dict, Optional
from datetime import date
from pandas.tseries.offsets import MonthEnd


import warnings
warnings.filterwarnings("ignore")


# show all columns and rows of a dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class SACashflowLGDParam():
    def __init__(self, context):
        self.run_yymm = context.run_yymm
        self.prev_yymm = context.prev_yymm
        self.SCENARIO_VERSION = context.SCENARIO_VERSION
        self.prtflo_scope = context.prtflo_scope
        self.scenario_set = context.scenario_set
        self.days_in_year = context.days_in_year
        self.days_in_month = context.days_in_month
        self.dtype_tbl = context.dtype_tbl
        self.total_yr = context.total_yr

        self.masterPath = context.masterPath
        self.dataPathScen = context.dataPathScen
        self.parmPath = context.parmPath
        self.inDataPath = context.inDataPath
        self.resultPath = context.resultPath

        self.inputDataExtECL = context.inputDataExtECL

        self.instrument_table_name = context.instrument_table_name
        self.exchange_rate_table_name = context.exchange_rate_table_name
        self.repayment_table_name = context.repayment_table_name
        self.sa_fs_table_name = context.sa_fs_table_name
        self.sa_other_debt_table_name = context.sa_other_debt_table_name

    def get_adjustment_factor(self, sa_df_cust, other_df, cust):
        # Get adjustment factor = bal in sa_table / (bal in other_debt + bal in sa_table)
        other_df_bal = other_df[other_df['CUST_ID']
                                == cust].TOTAL_DRAWN_BAL_LCY.sum()
        sa_df_bal = sa_df_cust[sa_df_cust['CUST_ID']
                               == cust].DRAWN_BAL_LCY.sum()
        # print('\nother_df_bal')
        # print(other_df_bal)
        # print('\nsa_df_bal')
        # print(sa_df_bal)
        adj_factor = sa_df_bal / (sa_df_bal + other_df_bal)
        return adj_factor

    def generate_fy_keys(self, base_year: int, lookback: int = 2, lookforward: int = 3) -> dict:
        """Generate financial year keys with clear temporal relationships."""
        fy_dict = {}

        # Previous years
        for i in range(1, lookback + 1):
            fy_dict[f'prev_{i}'] = date(base_year - i, 12, 31)

        # Current year
        fy_dict['curr'] = date(base_year, 12, 31)

        # Future years
        for i in range(1, lookforward + 1):
            fy_dict[f'next_{i}'] = date(base_year + i, 12, 31)

        return fy_dict

    def prepare_financial_data(self, fs_df, cust):
        """
        Prepare financial data by cleaning ITEM_NAME, ensuring STATEMENT_DATE is in date format,
        and separating the DataFrame into working capital items (wc_df) and cash-related items (cash_df).

        Parameters:
            fs_df (pd.DataFrame): The financial statement DataFrame containing columns 'ITEM_NAME', 'STATEMENT_TYPE', and 'STATEMENT_DATE'.

        Returns:
            wc_df (pd.DataFrame): DataFrame containing working capital items from the balance sheet.
            cash_df (pd.DataFrame): DataFrame containing cash-related items from the income statement.
        """
        # TODO: incorporate the ITEM_NAME check in pre_run_validation
        fs_df_ = fs_df[fs_df['CUST_ID'] == cust].copy()

        # Clean up ITEM_NAME by stripping spaces and replacing them with underscores
        fs_df_['ITEM_NAME'] = fs_df_[
            'ITEM_NAME'].str.strip().str.replace(' ', '_')

        # Ensure STATEMENT_DATE is in date format (not datetime)
        fs_df_['STATEMENT_DATE'] = pd.to_datetime(
            fs_df_['STATEMENT_DATE']).dt.date

        # Get unique ITEM_NAME lists for BS (Balance Sheet) and IS (Income Statement)
        wc_list = fs_df_[fs_df_['STATEMENT_TYPE']
                         == 'BS']['ITEM_NAME'].unique().tolist()
        cash_list = fs_df_[fs_df_['STATEMENT_TYPE']
                           == 'IS']['ITEM_NAME'].unique().tolist()

        # Pivot the table to reshape it
        fs_df_1 = fs_df_.pivot_table(
            index=['ITEM_NAME'],
            columns='STATEMENT_DATE',
            values='ITEM_AMOUNT',
            aggfunc='sum'
        ).reset_index()

        # Separate into working capital DataFrame (wc_df)
        wc_df = fs_df_1[fs_df_1['ITEM_NAME'].isin(wc_list)]

        # Separate into cash-related DataFrame (cash_df)
        cash_df = fs_df_1[fs_df_1['ITEM_NAME'].isin(cash_list)]

        return wc_df, cash_df

    def get_cash_profit(self, cash_df, fy_dict):
        """Calculate cash-based net profit using rolling window averages and clear temporal references."""
        # Calculate rolling averages using explicit temporal labels
        cash_df[fy_dict['next_1']] = cash_df[[fy_dict['prev_2'],
                                              fy_dict['prev_1'], fy_dict['curr']]].mean(axis=1)
        # print(cash_df[fy_dict['next_1']])
        cash_df[fy_dict['next_2']] = cash_df[[fy_dict['prev_1'],
                                              fy_dict['curr'], fy_dict['next_1']]].mean(axis=1)
        # print(cash_df[fy_dict['next_2']])
        cash_df[fy_dict['next_3']] = cash_df[[fy_dict['curr'],
                                              fy_dict['next_1'], fy_dict['next_2']]].mean(axis=1)
        # print(cash_df[fy_dict['next_3']])

        # Initialize results dataframe with clear structure
        cash_profit_df = pd.DataFrame({
            'ITEM_NAME': ['Net_profit', 'Depreciation']
        })

        # Unified calculation for each forecast year
        for year_offset in (1, 2, 3):
            year_key = f'next_{year_offset}'

            # Explicit column reference for readability
            data = {
                'Sales': cash_df.loc[cash_df['ITEM_NAME'] == 'Sales', fy_dict[year_key]].values[0],
                'COGS': cash_df.loc[cash_df['ITEM_NAME'] == 'COGS', fy_dict[year_key]].values[0],
                'Operating_expense': cash_df.loc[cash_df['ITEM_NAME'] == 'Operating_expense', fy_dict[year_key]].values[0],
                'Depreciation_expense': cash_df.loc[cash_df['ITEM_NAME'] == 'Depreciation_expense', fy_dict[year_key]].values[0],
                'Other_income': cash_df.loc[cash_df['ITEM_NAME'] == 'Other_income', fy_dict[year_key]].values[0],
                'Other_expense': cash_df.loc[cash_df['ITEM_NAME'] == 'Other_expense', fy_dict[year_key]].values[0],
                'Realized_FX_gain_(loss)': cash_df.loc[cash_df['ITEM_NAME'] == 'Realized_FX_gain_(loss)', fy_dict[year_key]].values[0],
                'Interest_expense': cash_df.loc[cash_df['ITEM_NAME'] == 'Interest_expense', fy_dict[year_key]].values[0],
                'Taxes': cash_df.loc[cash_df['ITEM_NAME'] == 'Taxes', fy_dict[year_key]].values[0]
            }

            # Clear financial calculations
            ebit = (
                data['Sales']
                - data['COGS']
                - data['Operating_expense']
                - data['Depreciation_expense']
                + data['Other_income']
                - data['Other_expense']
                + data['Realized_FX_gain_(loss)']
            )
            net_profit = ebit - data['Interest_expense'] - data['Taxes']

            # Structured assignment
            cash_profit_df.loc[
                cash_profit_df['ITEM_NAME'] == 'Net_profit',
                fy_dict[year_key]
            ] = net_profit

            cash_profit_df.loc[
                cash_profit_df['ITEM_NAME'] == 'Depreciation',
                fy_dict[year_key]
            ] = data['Depreciation_expense']

        return cash_profit_df

    def get_growth_rate(self, df, numerator_col, denominator_col):
        """Calculate growth rate with robust error handling"""
        result = df[numerator_col] / df[denominator_col] - 1
        return result.replace([np.inf, -np.inf, np.nan], 0)

    def get_wc_growth_rates(self, wc_df: pd.DataFrame, fy_dict: dict) -> pd.DataFrame:
        """
        Step 1: Calculate historical and projected growth rates
        Returns DataFrame with growth rates for future periods using fiscal year labels
        """
        df = wc_df.copy()

        # Calculate historical growth rates
        df['gr_prev_1'] = self.get_growth_rate(
            df, fy_dict['prev_1'], fy_dict['prev_2'])
        df['gr_curr'] = self.get_growth_rate(
            df, fy_dict['curr'], fy_dict['prev_1'])

        # Project future growth rates using moving average
        df['gr_next_1'] = df[['gr_prev_1', 'gr_curr']].mean(axis=1)
        df['gr_next_2'] = df[['gr_curr', 'gr_next_1']].mean(axis=1)
        df['gr_next_3'] = df[['gr_next_1', 'gr_next_2']].mean(axis=1)

        # Create column name mapping
        column_mapping = {
            'gr_next_1': fy_dict['next_1'],
            'gr_next_2': fy_dict['next_2'],
            'gr_next_3': fy_dict['next_3']
        }

        return df[['ITEM_NAME', 'gr_next_1', 'gr_next_2', 'gr_next_3']].rename(columns=column_mapping)

    def get_changes_in_working_capital(self, wc_df: pd.DataFrame,
                                       growth_rates_df: pd.DataFrame,
                                       fy_dict: dict) -> pd.DataFrame:
        """
        Step 2: Calculate changes in working capital using growth rates DF
        Returns DataFrame with projected changes in working capital
        """
        # Merge with growth rates using actual fiscal year column names
        df = wc_df.merge(
            growth_rates_df,
            on='ITEM_NAME',
            how='left',
            suffixes=('', '_gr')
        )

        # Asset items requiring sign inversion
        ASSET_ITEMS = [
            'Accounts/trade_receivables',
            'Other_receivables',
            'Intercompany_receivables',
            'Inventory',
            'Advance_to_suppliers',
            'Other_short_term_assets'
        ]

        # Get growth rate columns from fy_dict mapping
        gr_cols = [
            fy_dict['next_1'],
            fy_dict['next_2'],
            fy_dict['next_3']
        ]

        # Calculate projected balances
        df['tmp_1'] = df[fy_dict['curr']] * (1 + df[gr_cols[0]])
        df['tmp_2'] = df['tmp_1'] * (1 + df[gr_cols[1]])
        df['tmp_3'] = df['tmp_2'] * (1 + df[gr_cols[2]])

        # Compute period-to-period changes
        df[fy_dict['next_1']] = df['tmp_1'] - df[fy_dict['curr']]
        df[fy_dict['next_2']] = df['tmp_2'] - df['tmp_1']
        df[fy_dict['next_3']] = df['tmp_3'] - df['tmp_2']

        # Apply negative sign convention for asset accounts
        asset_mask = df['ITEM_NAME'].isin(ASSET_ITEMS)
        for year_col in [fy_dict[k] for k in ['next_1', 'next_2', 'next_3']]:
            df.loc[asset_mask, year_col] *= -1

        return df[['ITEM_NAME', fy_dict['next_1'], fy_dict['next_2'], fy_dict['next_3']]]

    def get_aggregate_fs(self, cash_profit_df: pd.DataFrame, ciwc_df: pd.DataFrame, fy_dict: dict) -> pd.DataFrame:
        """
        Calculate strategic analysis parameters for cash flow and working capital changes.

        Parameters:
            cash_profit_df (pd.DataFrame): DataFrame containing Net_profit and Depreciation
            ciwc_df (pd.DataFrame): DataFrame with working capital changes
            fy_dict (dict): Fiscal year mapping with 'next_1', 'next_2', 'next_3' keys

        Returns:
            pd.DataFrame: Parameters DataFrame with cash and ciwc values for each forecast period
        """

        # Validate fiscal year keys
        required_keys = ['next_1', 'next_2', 'next_3']
        if not all(k in fy_dict for k in required_keys):
            missing = [k for k in required_keys if k not in fy_dict]
            raise ValueError(f"Missing fiscal year keys: {missing}")

        # Component definitions
        WC_COMPONENTS = {
            'negative': [
                'Accounts/trade_receivables',
                'Intercompany_receivables',
                'Other_receivables',
                'Inventory',
                'Advance_to_suppliers',
                'Other_short_term_assets'
            ],
            'positive': [
                'Accounts_payables',
                'Intercompany_payables',
                'Tax_payable',
                'Advances_from_customer',
                'Other_current_liabilities'
            ]
        }

        # 1. Calculate cash parameters (Net Profit + Depreciation)
        cash_values = []
        for year_key in required_keys:
            year_col = fy_dict[year_key]
            try:
                np = cash_profit_df.loc[cash_profit_df['ITEM_NAME']
                                        == 'Net_profit', year_col].values[0]
                dp = cash_profit_df.loc[cash_profit_df['ITEM_NAME']
                                        == 'Depreciation', year_col].values[0]
                cash_values.append(np + dp)
            except IndexError:
                raise ValueError(
                    f"Missing Net_profit or Depreciation in {year_key}")

        # 2. Calculate working capital changes
        ciwc_values = []
        for year_key in required_keys:
            year_col = fy_dict[year_key]
            try:
                negative = ciwc_df.loc[ciwc_df['ITEM_NAME'].isin(
                    WC_COMPONENTS['negative']), year_col].sum()
                positive = ciwc_df.loc[ciwc_df['ITEM_NAME'].isin(
                    WC_COMPONENTS['positive']), year_col].sum()
                ciwc_values.append(positive + negative)
            except KeyError:
                raise ValueError(
                    f"Missing working capital components in {year_col}")

        # 3. Construct final DataFrame
        return pd.DataFrame({
            'Parameter': ['cash', 'ciwc'],
            fy_dict['next_1']: [cash_values[0], ciwc_values[0]],
            fy_dict['next_2']: [cash_values[1], ciwc_values[1]],
            fy_dict['next_3']: [cash_values[2], ciwc_values[2]]
        })

    def calculate_noc(self, fs_agg_df: pd.DataFrame,
                      fy_dict: Dict) -> pd.DataFrame:
        """
        Aggregates all parameter rows into a 'noc' (Net Operating Cash) row

        Parameters:
            fs_agg_df: DataFrame with columns ['Parameter', 'next_1', 'next_2', 'next_3']

        Returns:
            DataFrame with original rows plus new 'noc' row containing column sums

        Example Input:
            | Parameter | next_1 | next_2 | next_3 |
            |-----------|--------|--------|--------|
            | cash      | 150    | 165    | 182    |
            | ciwc      | -50    | -55    | -60    |

        Example Output:
            | Parameter | next_1 | next_2 | next_3 |
            |-----------|--------|--------|--------|

            | noc       | 100    | 110    | 122    |
        """
        # Create a copy to avoid modifying original data
        df = fs_agg_df.copy()

        # Calculate sums for each forecast period
        noc_values = {
            'Parameter': 'noc',
            fy_dict['next_1']: df[fy_dict['next_1']].sum(),
            fy_dict['next_2']: df[fy_dict['next_2']].sum(),
            fy_dict['next_3']: df[fy_dict['next_3']].sum()
        }

        # Append new row using concat for pandas>=1.4.0 compatibility
        noc_row = pd.DataFrame([noc_values])
        return noc_row

    # get sa params table
    def create_sa_adj_params(self, cash_profit_df: pd.DataFrame,
                             wc_gr_df: pd.DataFrame,
                             cust_id: str) -> pd.DataFrame:
        """
        Consolidate financial parameters with all projection values set to 1.0

        Args:
            cash_profit_df: Income statement parameters (structure only)
            wc_gr_df: Balance sheet parameters (structure only)
            cust_id: Customer identifier

        Returns:
            Consolidated DataFrame with:
            - Original metadata columns preserved
            - All projection columns set to 1.0 (float)
        """
        # Add identifiers to each dataset
        cash_profit = cash_profit_df.assign(
            CUST_ID=cust_id,
            STATEMENT_TYPE="IS"  # Income Statement
        )

        wc_growth = wc_gr_df.assign(
            CUST_ID=cust_id,
            STATEMENT_TYPE="BS"  # Balance Sheet
        )

        # Validate column compatibility
        required_columns = {'ITEM_NAME', 'CUST_ID', 'STATEMENT_TYPE'}
        if not required_columns.issubset(cash_profit.columns):
            raise ValueError("Cash profit DF missing required columns")
        if not required_columns.issubset(wc_growth.columns):
            raise ValueError("WC growth DF missing required columns")

        # Concatenate vertically
        consolidated = pd.concat(
            [cash_profit, wc_growth],
            axis=0,
            ignore_index=True
        )

        # Identify column types
        base_columns = ['CUST_ID', 'STATEMENT_TYPE', 'ITEM_NAME']
        projection_columns = [col for col in consolidated.columns
                              if col not in base_columns]

        # Set all projection values to 1.0
        consolidated[projection_columns] = 1.0

        # Ensure numeric type for projections
        for col in projection_columns:
            consolidated[col] = pd.to_numeric(
                consolidated[col], errors='ignore')

        return consolidated[base_columns + projection_columns]

    def run(self,
            fs_df,
            sa_df,
            other_df,
            custs: str) -> pd.DataFrame:

        #fy_dict = self.generate_fy_keys(int(str(self.prev_yymm)[:4]))
        fy_dict = self.generate_fy_keys(int(str(self.run_yymm)[:4]) - 1)

        # initialize
        cash_profit_list = []
        ciwc_list = []
        noc_list = []
        sa_params_list = []

        for cust in custs:
            sa_df_cust = sa_df[sa_df['CUST_ID'] == cust].copy()

            wc_df, cash_df = self.prepare_financial_data(fs_df, cust)

            cash_profit_df = self.get_cash_profit(cash_df, fy_dict)
            wc_gr_df = self.get_wc_growth_rates(wc_df, fy_dict)
            ciwc_df = self.get_changes_in_working_capital(
                wc_df, wc_gr_df, fy_dict)

            # sum cash and ciwc separately
            fs_agg_df = self.get_aggregate_fs(cash_profit_df, ciwc_df, fy_dict)

            # calc net operating cash flow
            adj_factor = self.get_adjustment_factor(sa_df_cust, other_df, cust)
            noc_df = self.calculate_noc(fs_agg_df, fy_dict)
            noc_df = noc_df.applymap(
                lambda x: x * adj_factor if isinstance(x, (int, float)) else x)

            cust_params_df = self.create_sa_adj_params(
                cash_profit_df, wc_gr_df, cust)

            cash_profit_list.append(
                cash_profit_df.assign(CUST_ID=cust)[
                    ['CUST_ID'] + [col for col in cash_profit_df.columns]]
            )
            ciwc_list.append(
                ciwc_df.assign(CUST_ID=cust)[
                    ['CUST_ID'] + [col for col in ciwc_df.columns]]
            )
            noc_list.append(
                noc_df.assign(CUST_ID=cust)[
                    ['CUST_ID'] + [col for col in noc_df.columns]]
            )
            sa_params_list.append(cust_params_df)

        sa_params_df = pd.concat(
            sa_params_list, ignore_index=True).reset_index(drop=True)
        cash_profit_all = pd.concat(
            cash_profit_list, ignore_index=True).reset_index(drop=True)
        ciwc_all = pd.concat(
            ciwc_list, ignore_index=True).reset_index(drop=True)
        noc_all = pd.concat(noc_list, ignore_index=True).reset_index(drop=True)

        # Formatting SA fs adjustment param
        header = {
            "INDEX": ["Parameter name", "Parameter description", "Mandatory input", "Parameter data type"],
            "CUST_ID": ["CUST_ID", "The customer ID", "Y", "str"],
            "ITEM_NAME": ["ITEM_NAME", "The financial statement item name", "Y", "str"],
            "STATEMENT_TYPE": ["STATEMENT_TYPE", "The fianancial statement type", "Y", "str"],
            "next_1_year": ["next_1_year", "For statement type'BS', this parameter is multiplier applied to the projected growth rate of FS ITEM in Year 1. \n For statement type 'IS', this parameter is the multiplier applied to the projected amount of FS ITEM in Year 1", "Y", "float"],
            "next_2_year": ["next_2_year", "For statement type'BS', this parameter is multiplier applied to the projected growth rate of FS ITEM in Year 2. \n For statement type 'IS', this parameter is the multiplier applied to the projected amount of FS ITEM in Year 2", "Y", "float"],
            "next_3_year": ["next_3_year", "For statement type'BS', this parameter is multiplier applied to the projected growth rate of FS ITEM in Year 3. \n For statement type 'IS', this parameter is the multiplier applied to the projected amount of FS ITEM in Year 3", "Y", "float"]
        }
        table_header = pd.DataFrame(header)

        table = sa_params_df.copy()
        table['INDEX'] = range(1, len(table) + 1)
        # Rename columns before concatenation - for param col name standardization
        table = table.rename(columns={
            fy_dict['next_1']: 'next_1_year',
            fy_dict['next_2']: 'next_2_year',
            fy_dict['next_3']: 'next_3_year'
        })

        table = table.reindex(columns=table_header.columns)
        sa_fs_param = pd.concat([table_header, table], ignore_index=True)
        return sa_fs_param, cash_profit_all, ciwc_all, noc_all


class SACashflowLGD(SACashflowLGDParam):
    def __init__(self, context):
        super().__init__(context)

    # recalc the cash_profit_df by apply the cash_mult to the cash_profit_df
    def refresh_fs_df(
        self,
        fs_df: pd.DataFrame,
        mult_df: pd.DataFrame,
        fy_dict: dict,
    ) -> pd.DataFrame:

        merged_df = pd.merge(
            fs_df,
            mult_df,
            on="ITEM_NAME",
            suffixes=("_df1", "_df2")
        )

        result_df = merged_df[["ITEM_NAME"]].copy()
        for col in [fy_dict['next_1'], fy_dict['next_2'], fy_dict['next_3']]:
            result_df[col] = merged_df[f"{col}_df1"] * merged_df[f"{col}_df2"]

        return result_df

    # calc NPL statistics

    def _calculate_npl_stats(self, data: pd.Series, alpha: float) -> Tuple[float, float, float]:
        """Helper function to calculate NPL statistics"""
        n = len(data)
        mean = data.mean()
        std = data.std()

        z_score = norm.ppf(1 - alpha/2)
        std_err = std / (n ** 0.5)

        return (
            mean,
            mean - z_score * std_err,
            mean + z_score * std_err
        )

    def calculate_sector_npl_stats(self,
                                   es_df: pd.DataFrame,
                                   npl_data: pd.DataFrame,
                                   alpha: float = 0.1
                                   ) -> pd.DataFrame:
        """
        Calculate NPL statistics with confidence intervals for economic sectors.

        Args:
            es_df: DataFrame containing economic sector information with 'ECON_SECTOR_TYPE' column
            npl_data: DataFrame containing NPL data with columns named 'SECTOR_NPL_{sector}'
            alpha: Significance level for confidence intervals (default: 0.1)

        Returns:
            DataFrame with columns ['sector', 'mean', 'ci_lower', 'ci_upper']
        """
        # Validate inputs
        if 'ECON_SECTOR_TYPE' not in es_df:
            raise ValueError("es_df must contain 'ECON_SECTOR_TYPE' column")

        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")

        # Generate sector names
        sectors = es_df['ECON_SECTOR_TYPE'].unique().tolist()
        npl_columns = [f'SECTOR_NPL_{sec}' for sec in sectors]

        # Verify all required columns exist in npl_data
        missing_cols = [col for col in npl_columns if col not in npl_data]
        if missing_cols:
            raise ValueError(
                f"Missing NPL columns in npl_data: {missing_cols}")

        # Calculate statistics for each sector
        stats = []
        for sector, col in zip(sectors, npl_columns):
            data = npl_data[col].dropna()
            if len(data) < 2:
                raise ValueError(f"Insufficient data for sector {sector}")

            mean, ci_lower, ci_upper = self._calculate_npl_stats(data, alpha)
            stats.append({
                'npl_sector': sector,
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

        return pd.DataFrame(stats)

    # calc LGD
    def get_total_repay_df(self, df, repay_df):
        contract_ids = df.CONTRACT_ID.unique().tolist()
        cust_repay = repay_df[repay_df['CONTRACT_ID'].isin(contract_ids)]

        ttl_repay = cust_repay.pivot_table(
            index='CF_DATE', columns='CONTRACT_ID', values='PRIN_PMT_OCY')
        ttl_repay['ttl_repay'] = ttl_repay.sum(axis=1)

        return ttl_repay[['ttl_repay']]

    def get_cash_flow_dates(self, scenario_date: str, period_years: int) -> pd.DatetimeIndex:
        """
        Generate monthly cash flow dates starting from the quarter following the scenario date.

        Args:
            scenario_date: Format 'YYYYQX' where X is quarter (1-4)
            period_years: Number of years to project forward

        Returns:
            Monthly DatetimeIndex through end of projection period

        Example:
            >>> get_cash_flow_dates('2023Q4', 5)
            DatetimeIndex(['2024-01-31', '2024-02-29', ..., '2027-12-31'], dtype='datetime64[ns]', freq='M')
        """
        # Validate input format
        if len(scenario_date) != 6 or not scenario_date[4] == 'Q':
            raise ValueError(
                "Scenario date must be in format 'YYYYQX' (e.g. '2023Q4')")

        year = int(scenario_date[:4])
        quarter = int(scenario_date[-1])

        if not 1 <= quarter <= 4:
            raise ValueError("Quarter must be between 1-4")

        # Calculate start date
        if quarter == 4:
            start_year = year + 1
            start_month = 1
        else:
            start_year = year
            start_month = quarter * 3 + 1  # Next quarter start month

        start_date = pd.Timestamp(
            year=start_year, month=start_month, day=1) + MonthEnd(1)

        # Calculate end date
        end_date = pd.Timestamp(year + period_years - 1, 12, 31)

        return pd.date_range(start=start_date, end=end_date, freq='M')

    def get_pwa_noc(
        self,
        fy_dict: Dict,
        sa_df_cust: pd.DataFrame,
        noc_df: pd.DataFrame,
        npl_stats: pd.DataFrame,
        pwa_df: pd.DataFrame,
        es_df: pd.DataFrame,
        cust: str
    ) -> Optional[pd.DataFrame]:
        """
        Calculate Probability-Weighted Average Net Operating Cash (NOC) for a customer.

        Args:
            fy_dict: Fiscal year mapping {'next_1': 2024, ...}
            sa_df_cust: Strategic analysis dataframe with customer-sector mapping
            noc_df: NOC projections dataframe
            npl_stats: Sector NPL statistics with confidence intervals
            pwa_df: Scenario probability weights
            cust_id: Target customer ID

        Returns:
            DataFrame with PWA NOC projections and scenario-specific NOC values
        """
        def get_cust_sector_type(sa_df_cust, es_df, drawn_balance_col='DRAWN_BAL_LCY'):
            # chk empty data
            if sa_df_cust.empty:
                return None

            # total withdrawn balance by economic sector grouping
            sector_balances = sa_df_cust.groupby(
                'ECON_SECTOR')[drawn_balance_col].sum()

            # access to the most heavily weighted sectors
            if not sector_balances.empty:
                # returns the ECON_SECTOR with the largest balance
                dominant_sector = sector_balances.idxmax()
                sector_type = es_df.loc[
                    es_df['ECON_SECTOR'] == dominant_sector,
                    'ECON_SECTOR_TYPE'
                ].iloc[0]
                return sector_type

        sector_type = get_cust_sector_type(sa_df_cust, es_df)

        # Get NPL risk factor
        try:
            sector_stats = npl_stats.query(
                f"npl_sector == '{sector_type}'").iloc[0]
            risk_factor = (
                sector_stats['mean'] - sector_stats['ci_lower']) / sector_stats['mean']

        except IndexError:
            print(f"No NPL stats found for sector {sector_type}")
            return None

        # Calculate scenario-adjusted NOC
        results = []
        for year_key in ['next_1', 'next_2', 'next_3']:
            if year_key not in fy_dict:
                print(f"Missing fiscal year key {year_key}")
                return None

            fiscal_year = fy_dict[year_key]

            # divided by 12: monthly net operating cashflow
            base_noc = noc_df.loc[noc_df['Parameter']
                                  == 'noc', fiscal_year].values[0] / 12

            # Scenario adjustments
            scenarios = {
                'SEVE': base_noc * (1 - risk_factor),
                'BASE': base_noc,
                'GROW': base_noc * (1 + risk_factor)
            }

            # Apply probability weights
            pwa_noc = sum(
                scenarios[scenario] * pwa_df.loc[pwa_df['scenario']
                                                 == scenario, 'pwa'].values[0]
                for scenario in ['SEVE', 'BASE', 'GROW']
            )

            results.append({
                'prediction_year': fiscal_year,
                'SEVE_NOC': scenarios['SEVE'],
                'BASE_NOC': scenarios['BASE'],
                'GROW_NOC': scenarios['GROW'],
                'pwa_noc': pwa_noc
            })

        return pd.DataFrame(results)

    def get_total_exposure(self, df):
        return df['PRIN_BAL_LCY'].sum() + df['ACRU_INT_LCY'].sum()

    def get_cust_EIR(self, df):
        eir = df['EFF_INT_RT']
        exposure = df['PRIN_BAL_LCY'] + df['ACRU_INT_LCY']
        return (eir * exposure).sum() / exposure.sum()

    def get_lgd_df(self, cash_flow_dates: pd.DatetimeIndex, pwa_noc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Loss Given Date (LGD) DataFrame with datetime index preservation.

        Args:
            cash_flow_dates: Monthly cash flow dates (DatetimeIndex)
            pwa_noc_df: Annual NOC projections with prediction years

        Returns:
            DataFrame with cash flow dates as index and total NOC values
        """
        # Convert prediction years to datetime and extract year
        annual_noc = pwa_noc_df.copy()
        annual_noc['year'] = pd.to_datetime(
            annual_noc['prediction_year']).dt.year

        # Create base dataframe with cash flow dates as index
        lgd_df = pd.DataFrame(index=cash_flow_dates)
        lgd_df.index.name = 'cash_flow_date'

        # Extract year from index and prepare for merge
        lgd_df = lgd_df.assign(year=lgd_df.index.year).reset_index()

        # Merge with annual NOC values
        lgd_df = lgd_df.merge(
            annual_noc[['year', 'pwa_noc']],
            on='year',
            how='left'
        )

        # Restore datetime index and clean up
        lgd_df = lgd_df.set_index('cash_flow_date').sort_index()
        lgd_df = lgd_df[['pwa_noc']].rename(columns={'pwa_noc': 'ttl_noc'})

        # Forward fill within each year group and fill remaining NAs
        lgd_df['ttl_noc'] = lgd_df['ttl_noc'].ffill().fillna(0)

        return lgd_df

    def add_discounted_col(
        self,
        df: pd.DataFrame,
        lgd_df: pd.DataFrame,
        col: str,
        eir: float
    ) -> pd.DataFrame:
        """
        Add discounted NOC column using time-aware discounting.

        Args:
            lgd_df: DataFrame with cash_flow_date index and ttl_noc column
            reporting_date: Base date for discounting (e.g. '2024-06-30')
            eir: Effective interest rate (e.g. 0.05 for 5%)

        Returns:
            Modified DataFrame with discounted_noc column
        """
        # Convert reporting date to pandas timestamp if not already
        # TODO: check if the report_date of one cust is unique and = the scenario date
        reporting_date = df['REPORT_DATE'].unique()[0]

        # Calculate time delta in years
        delta_days = (lgd_df.index - reporting_date).days
        delta_years = delta_days / 365

        # Calculate discount factor
        discount_factor = (1 + eir) ** delta_years

        # Calculate discounted NOC
        lgd_df[f'discounted_{col}'] = lgd_df[col] / discount_factor

        return lgd_df

    def get_total_recovery(self, lgd_df_1, reporting_date, eir):
        """
        Calculate the total recoverable amount (delayed repayment plan version)
        New logic: If the cash flow is insufficient in a given month, the entire repayment plan is shifted back by one month until the cumulative cash flow is sufficient to cover the repayment.
        Discounting is now based on actual payment date, not theoretical repayment date.
        """
        total_recovery = 0.0
        carry_over = 0.0      # Cumulative available cash flow
        pending_repayments = []  # Queue for pending repayment plans with their original dates

        # Ensure reporting_date is a pandas timestamp
        if not isinstance(reporting_date, pd.Timestamp):
            reporting_date = pd.Timestamp(reporting_date)

        # Process data in chronological order
        df = lgd_df_1.sort_index()

        for date, row in df.iterrows():
            discounted_noc = row["discounted_ttl_noc"]
            ttl_repay = row["ttl_repay"]  # Use original repayment amount, not discounted

            # Add the current month's repayment plan to the queue with its original date
            pending_repayments.append((ttl_repay, date))

            # Handle negative cash flow
            current_noc = max(discounted_noc, 0.0)
            carry_over += current_noc  # Accumulate cash flow

            # Attempt to process the first repayment plan in the queue
            while len(pending_repayments) > 0 and carry_over >= pending_repayments[0][0]:
                # Repay the earliest outstanding debt
                repaid_amount, original_date = pending_repayments.pop(0)
                
                # Calculate discount based on actual payment date (current date)
                delta_days = (date - reporting_date).days
                delta_years = delta_days / 365
                discount_factor = (1 + eir) ** delta_years
                
                # Discount the repaid amount based on actual payment date
                discounted_repaid = repaid_amount / discount_factor
                
                total_recovery += discounted_repaid
                carry_over -= repaid_amount

        return total_recovery

    def run(
            self,
            sa_df,
            fs_df,
            other_df,
            repay_df,
            param,
            npl_data,
            custs,
            sa_fs_param_1,):

        # initialize the LGD result
        lgd_results = []
        sa_intrim_output = []
        noc_intrim_output = []

        # get param dfs
        es_df = param['Economic_sector']
        pwa_df = param['pwa']
        period = param['CFL_LGD_period'].Period.values[0]

        #fy_dict = self.generate_fy_keys(int(str(self.prev_yymm)[:4]))
        fy_dict = self.generate_fy_keys(int(str(self.run_yymm)[:4]) - 1)

        # change the column name in sa_fs_param_1 after reloading the sa_fs_params
        sa_fs_param_1 = sa_fs_param_1.rename(columns={
            'next_1_year': fy_dict['next_1'],
            'next_2_year': fy_dict['next_2'],
            'next_3_year': fy_dict['next_3']
        })

        for cust in custs:
            # print('\ncust')
            # print(cust)
            sa_df_cust = sa_df[sa_df['CUST_ID'] == cust].copy()
            sa_fs_param_cust = sa_fs_param_1[sa_fs_param_1['CUST_ID'] == cust].copy(
            )

            # get the updated multipliers for cash and growth rate of working capital
            cash_mult = sa_fs_param_cust[sa_fs_param_cust['STATEMENT_TYPE'] == 'IS'][[
                'ITEM_NAME', fy_dict['next_1'], fy_dict['next_2'], fy_dict['next_3']]]
            wc_gr_mult = sa_fs_param_cust[sa_fs_param_cust['STATEMENT_TYPE'] == 'BS'][[
                'ITEM_NAME', fy_dict['next_1'], fy_dict['next_2'], fy_dict['next_3']]]
            # print('\ncash_mult')
            # print(cash_mult)
            # print('\nwc_gr_mult')
            # print(wc_gr_mult)
            # recalculate the cash and changes in working capital
            wc_df, cash_df = self.prepare_financial_data(fs_df, cust)
            # print('\cash_df')
            # print(cash_df)
            # print('\wc_df')
            # print(wc_df)

            cash_profit_df = self.get_cash_profit(cash_df, fy_dict)
            cash_profit_df = self.refresh_fs_df(
                cash_profit_df, cash_mult, fy_dict)

            # print('\wc_df')
            # print(wc_df)

            wc_gr_df = self.get_wc_growth_rates(wc_df, fy_dict)
            wc_gr_df = self.refresh_fs_df(wc_gr_df, wc_gr_mult, fy_dict)

            ciwc_df = self.get_changes_in_working_capital(
                wc_df, wc_gr_df, fy_dict)
            # print('\ciwc_df')
            # print(ciwc_df)

            # sum cash and ciwc separately
            fs_agg_df = self.get_aggregate_fs(cash_profit_df, ciwc_df, fy_dict)
            # print('\fs_agg_df')
            # print(fs_agg_df)
            # calc net operating cash flow
            adj_factor = self.get_adjustment_factor(sa_df_cust, other_df, cust)
            # print('\fadj_factor')
            # print(adj_factor)
            noc_df = self.calculate_noc(fs_agg_df, fy_dict)
            noc_df = noc_df.applymap(
                lambda x: x * adj_factor if isinstance(x, (int, float)) else x)
            # print('\fnoc_df')
            # print(noc_df)
            # LGD calculation
            # NPL statistics
            npl_stats = self.calculate_sector_npl_stats(
                es_df, npl_data, alpha=0.1)
            # print('\fnpl_stats')
            # print(npl_stats)
            # LGD
            ttl_repay_df = self.get_total_repay_df(sa_df_cust, repay_df)
            # print(ttl_repay_df)
            cash_flow_dates = self.get_cash_flow_dates(
                self.SCENARIO_VERSION, period)

            pwa_noc_df = self.get_pwa_noc(
                fy_dict, sa_df_cust, noc_df, npl_stats, pwa_df, es_df, cust)

            ttl_exposure = self.get_total_exposure(sa_df_cust)
            eir = self.get_cust_EIR(sa_df_cust)

            lgd_df = self.get_lgd_df(cash_flow_dates, pwa_noc_df)

            lgd_df_1 = lgd_df.merge(
                ttl_repay_df, how='left', left_index=True, right_index=True).copy()
            lgd_df_1 = lgd_df_1.fillna(0)  # Fill NaN values with 0

            lgd_df_2 = self.add_discounted_col(
                sa_df_cust,
                lgd_df_1,
                'ttl_noc',
                eir=eir
            )

            lgd_df_2 = self.add_discounted_col(
                sa_df_cust,
                lgd_df_1,
                'ttl_repay',
                eir=eir
            )

            ttl_recover = self.get_total_recovery(lgd_df_2, sa_df_cust['REPORT_DATE'].unique()[0], eir)
            lgd = 1 - ttl_recover/ttl_exposure

            # Create lgd_results
            lgd_result = {
                'CUST_ID': cust,
                'LGD': lgd,
            }
            lgd_results.append(lgd_result)

            # Create a dictionary for the customer's interim output
            cust_intrim = {
                'CUST_ID': cust,
                'LGD': lgd,
                'Total_Exposure': ttl_exposure,
                'Total_Recovery': ttl_recover,
                'EIR': eir,
                'ECL': lgd * ttl_exposure
            }

            sa_intrim_output.append(cust_intrim)

            # Create NOC interim output for each year
            if pwa_noc_df is not None:
                for _, row in pwa_noc_df.iterrows():
                    noc_intrim = {
                        'CUST_ID': cust,
                        'prediction_year': row['prediction_year'],
                        'noc': noc_df.loc[noc_df['Parameter'] == 'noc', row['prediction_year']].values[0],
                        'SEVE_NOC': row['SEVE_NOC'],
                        'BASE_NOC': row['BASE_NOC'],
                        'GROW_NOC': row['GROW_NOC'],
                        'pwa_noc': row['pwa_noc']
                    }
                    noc_intrim_output.append(noc_intrim)

        final_lgd_df = pd.DataFrame(lgd_results)
        sa_intrim_output_df = pd.DataFrame(sa_intrim_output)
        noc_intrim_output_df = pd.DataFrame(noc_intrim_output)

        return final_lgd_df, sa_intrim_output_df, noc_intrim_output_df


class SACashflowLGDPrecomputed(SACashflowLGD):
    def __init__(self, context):
        super().__init__(context)
        self.noc_file_path = Path(context.inDataPath) / 'sa_manual_noc.csv'

    def load_precomputed_noc(self) -> pd.DataFrame:
        """
        Load pre-computed NOC values from CSV file.

        Expected CSV structure:
        CUST_ID,prediction_year,noc
        CUST001,2024-12-31,1000000
        CUST001,2025-12-31,1100000
        CUST001,2026-12-31,1210000
        CUST002,2024-12-31,2000000
        ...

        Returns:
            DataFrame with columns:
            - CUST_ID: Customer identifier (string)
            - prediction_year: Year-end date for the prediction (datetime.date)
            - noc: Base NOC value
        """
        if not self.noc_file_path.exists():
            raise FileNotFoundError(
                f"NOC file not found at {self.noc_file_path}")

        # Read CSV with CUST_ID as string
        noc_df = pd.read_csv(self.noc_file_path, dtype={'CUST_ID': str})
        required_cols = ['CUST_ID', 'prediction_year', 'noc']

        # Check for required columns
        missing_cols = [
            col for col in required_cols if col not in noc_df.columns]
        if missing_cols:
            raise ValueError(f"NOC file must contain columns: {missing_cols}")

        # Convert prediction_year to datetime.date to match fy_dict format
        noc_df['prediction_year'] = pd.to_datetime(
            noc_df['prediction_year']).dt.date

        return noc_df

    def run(
            self,
            sa_df,
            repay_df,
            param,
            npl_data,
            custs):
        """
        Run LGD calculation using pre-computed NOC values.

        Args:
            sa_df: Strategic analysis dataframe
            repay_df: Repayment dataframe
            param: Parameter dictionary
            npl_data: NPL data dataframe
            custs: List of customer IDs

        Returns:
            Tuple of (final_lgd_df, sa_intrim_output_df, noc_intrim_output_df)
        """
        # Load pre-computed NOC values
        precomputed_noc = self.load_precomputed_noc()

        # Initialize results
        lgd_results = []
        sa_intrim_output = []
        noc_intrim_output = []

        # Get parameter dataframes
        es_df = param['Economic_sector']
        pwa_df = param['pwa']
        period = param['CFL_LGD_period'].Period.values[0]

        #fy_dict = self.generate_fy_keys(int(str(self.prev_yymm)[:4]))
        fy_dict = self.generate_fy_keys(int(str(self.run_yymm)[:4]) - 1)

        for cust in custs:
            sa_df_cust = sa_df[sa_df['CUST_ID'] == cust].copy()

            # Get pre-computed NOC for this customer
            cust_noc = precomputed_noc[precomputed_noc['CUST_ID'] == cust]
            if cust_noc.empty:
                print(
                    f"Warning: No pre-computed NOC found for customer {cust}")
                continue

            # Create noc_df in the format expected by parent class methods
            noc_df = pd.DataFrame({
                'Parameter': ['noc'],
                fy_dict['next_1']: [cust_noc[cust_noc['prediction_year'] == fy_dict['next_1']]['noc'].iloc[0]],
                fy_dict['next_2']: [cust_noc[cust_noc['prediction_year'] == fy_dict['next_2']]['noc'].iloc[0]],
                fy_dict['next_3']: [
                    cust_noc[cust_noc['prediction_year'] == fy_dict['next_3']]['noc'].iloc[0]]
            })

            # LGD calculation using parent class methods
            npl_stats = self.calculate_sector_npl_stats(
                es_df, npl_data, alpha=0.1)
            
            ttl_repay_df = self.get_total_repay_df(sa_df_cust, repay_df)

            cash_flow_dates = self.get_cash_flow_dates(
                self.SCENARIO_VERSION, period)

            pwa_noc_df = self.get_pwa_noc(
                fy_dict, sa_df_cust, noc_df, npl_stats, pwa_df, es_df, cust)

            ttl_exposure = self.get_total_exposure(sa_df_cust)
            eir = self.get_cust_EIR(sa_df_cust)

            lgd_df = self.get_lgd_df(cash_flow_dates, pwa_noc_df)
            lgd_df_1 = lgd_df.merge(
                ttl_repay_df, how='left', left_index=True, right_index=True).copy()
            lgd_df_1 = lgd_df_1.fillna(0)

            lgd_df_2 = self.add_discounted_col(
                sa_df_cust, lgd_df_1, 'ttl_noc', eir=eir)
            lgd_df_2 = self.add_discounted_col(
                sa_df_cust, lgd_df_1, 'ttl_repay', eir=eir)

            ttl_recover = self.get_total_recovery(lgd_df_2, sa_df_cust['REPORT_DATE'].unique()[0], eir)
            lgd = 1 - ttl_recover/ttl_exposure

            # Create lgd_results
            lgd_result = {
                'CUST_ID': cust,
                'LGD': lgd,
            }
            lgd_results.append(lgd_result)

            # Create a dictionary for the customer's interim output
            cust_intrim = {
                'CUST_ID': cust,
                'LGD': lgd,
                'Total_Exposure': ttl_exposure,
                'Total_Recovery': ttl_recover,
                'EIR': eir,
                'ECL': lgd * ttl_exposure
            }

            sa_intrim_output.append(cust_intrim)

            # Create NOC interim output for each year
            if pwa_noc_df is not None:
                for _, row in pwa_noc_df.iterrows():
                    noc_intrim = {
                        'CUST_ID': cust,
                        'prediction_year': row['prediction_year'],
                        'noc': noc_df.loc[noc_df['Parameter'] == 'noc', row['prediction_year']].values[0],
                        'SEVE_NOC': row['SEVE_NOC'],
                        'BASE_NOC': row['BASE_NOC'],
                        'GROW_NOC': row['GROW_NOC'],
                        'pwa_noc': row['pwa_noc']
                    }
                    noc_intrim_output.append(noc_intrim)

        final_lgd_df = pd.DataFrame(lgd_results)
        sa_intrim_output_df = pd.DataFrame(sa_intrim_output)
        noc_intrim_output_df = pd.DataFrame(noc_intrim_output)

        return final_lgd_df, sa_intrim_output_df, noc_intrim_output_df
