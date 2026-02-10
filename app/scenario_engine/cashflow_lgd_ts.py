# Load packages
from pathlib import Path
import calendar
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
from input_handler.data_preprocessor import data_preprocessor
from typing import Dict, Tuple


import warnings
warnings.filterwarnings("ignore")


# show all columns and rows of a dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# setting up environment
class env_setting():
    def __init__(self, context):
        self.run_yymm = context.run_yymm
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

        self.mute_eir = context.mute_eir
        self.mute_stage_consistency = context.mute_stage_consistency


class data_preparation(env_setting):
    def __init__(self, context):
        super().__init__(context)

    def load_data(self):
        dp = data_preprocessor(context=self)
        raw_data = dp.load_scenario_data(
            data_path=self.dataPathScen, file_pattern='LGD')
        return raw_data

    def prepare_data(self, raw_data):
        data = raw_data.copy()
        data.columns = data.columns.str.lower()
        data.columns = data.columns.str.replace(' ', '_')

        # only keep the columns that are needed
        data = data[['acct_no', 'sub_category', 'pdd_bal', 'value',
                     'dte_pdd', 'dte', 'loan_bal', 'eir', 'type']]

        # delete rows where sub_category is null
        data = data[~data['sub_category'].isnull()]

        # convert the date columns to datetime
        data['dte'] = pd.to_datetime(data['dte'], format='%m/%d/%Y')
        data['dte_pdd'] = pd.to_datetime(data['dte_pdd'], format='%m/%d/%Y')

        # sort df by sub_category, acct_no, and dte
        data.sort_values(by=['sub_category', 'acct_no', 'dte'], inplace=True)
        return data

    def data_preparation_run(self):
        raw_data = self.load_data()
        data = self.prepare_data(raw_data)
        return raw_data, data


# term structure projection
class cashflow_lgd(data_preparation):
    def __init__(self, context):
        super().__init__(context)
        data_prepared = self.data_preparation_run()
        self.data = data_prepared[1].copy()

    def get_cashflow_lgd(self, param: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        data = self.data.copy()
        lgd_approach = param['LGD_approach']

        cats_approach = lgd_approach[lgd_approach['LGD_APPROACH']
                                     == 'CFL']['LGD_CATEGORY'].to_list()
        cats_default = data.sub_category.unique().tolist()
        # If the cat has no default cases, use the assigned LGD directly.
        cats_need_assign = [
            cat for cat in cats_approach if cat not in cats_default]
        cats = [cat for cat in cats_approach if cat not in cats_need_assign]

        lgd_output = pd.DataFrame()  # create an empty dataframe to store the output

        # TODO: change back to the full list after testing
        for cat in tqdm(cats, desc='calculating LGD of each sub_category: ', position=0, leave=True):
            ################## filter the data by sub_category ##################
            df = data[data['sub_category'] == cat].copy()

            ################## calculate repayment ##################
            # calculate the month between the dte_pdd and dte
            def months_between(date2, date1):
                year_diff = date2.year - date1.year
                month_diff = date2.month - date1.month

                total_month_diff = year_diff * 12 + month_diff
                return total_month_diff

            df['md'] = df.apply(lambda x: months_between(
                x['dte'], x['dte_pdd']), axis=1)

            # calculate the new pdd_bal = pdd_bal*value
            df['pdd_bal'] = df['pdd_bal'] * df['value']
            pdd_bal = df[df['type'] == 'PD']['pdd_bal'].sum()

            # calculate the payment = case when type='Pm' then loan_bal else 0 end / power(1+eir/12,months_between(dte, dte_pdd))) payment
            df['payment'] = np.where(
                df['type'] == 'Pm', df['loan_bal'], 0) / np.power(1 + df['eir'] / 12, df['md'])

            # aggregate payment by all other columns
            df['payment'] = df.groupby(['acct_no', 'sub_category', 'pdd_bal', 'value', 'dte_pdd', 'dte', 'md'])[
                'payment'].transform('sum')

            # drop duplicated dte
            df.drop_duplicates(subset=['acct_no', 'sub_category', 'pdd_bal',
                               'value', 'dte_pdd', 'dte', 'md'], keep='first', inplace=True)

            # calculate cumulative payment
            df['cum_payment'] = df.groupby(['acct_no'])['payment'].cumsum()

            # adjust the cum_payment: if -cum_payment > pdd_bal then -pdd_bal else cum_payment
            df['cum_payment'] = np.where(-df['cum_payment'] >
                                         df['pdd_bal'], -df['pdd_bal'], df['cum_payment'])

            # keep only the columns that are needed
            df = df[['acct_no', 'sub_category', 'md',
                     'pdd_bal', 'payment', 'cum_payment']]

            ###### extend the LGD liftime to month between 20150601 to reporting date ######

            # month between 2015-6-1 and latest payment DTE
            max_pay_dte = pd.to_datetime(data['dte'].max())
            mth_add = months_between(max_pay_dte, datetime(2015, 6, 1))

            # Get unique accounts
            unique_accts = df['acct_no'].unique()

            # Create a DataFrame with all possible md values for each account
            complete_md = pd.DataFrame({
                'acct_no': [acct_no for acct_no in unique_accts for _ in range(mth_add+1)],
                'md': list(range(mth_add+1)) * len(unique_accts)
            })

            # Merge with the original DataFrame
            df_lgd = pd.merge(complete_md, df, on=[
                              'acct_no', 'md'], how='left')

            # Fill missing cum_payment values with the last known pdd_bal, cum_payment and sub_category for each acct
            df_lgd['pdd_bal'] = df_lgd.groupby('acct_no')['pdd_bal'].transform(
                lambda group: group.ffill().bfill())
            df_lgd['cum_payment'] = df_lgd.groupby('acct_no')['cum_payment'].transform(
                lambda group: group.ffill().bfill())
            df_lgd['sub_category'] = df_lgd.groupby('acct_no')['sub_category'].transform(
                lambda group: group.ffill().bfill())

            ################## aggregate the cum_payment by sub_category, md ##################
            # df_lgd = df_lgd.groupby(['sub_category', 'md'])['cum_payment'].sum().reset_index()
            df_lgd = df_lgd.groupby(['sub_category', 'md']).agg({
                'pdd_bal': 'first',
                'cum_payment': 'sum',
                'payment': 'sum'
            }).reset_index()

            # calculate the total pdd_bal by sub_category
            # df_lgd['pdd_bal'] = df.pdd_bal.unique().sum()
            df_lgd['pdd_bal'] = pdd_bal

            # calculate cumulative recovery rate = -cum_payment/pdd_bal
            df_lgd['cum_rr'] = -df_lgd['cum_payment'] / df_lgd['pdd_bal']

            # calculate marginal recovery rate
            df_lgd['marg_rr'] = df_lgd['cum_rr'] - df_lgd['cum_rr'].shift(1)
            df_lgd.fillna(method='ffill', axis=1)

            # expected recovery rate
            df_lgd['exp_rr'] = df_lgd.groupby(
                ['sub_category'])['cum_rr'].transform('max')
            df_lgd['shifted_cum_rr'] = df_lgd.groupby(['sub_category'])[
                'cum_rr'].shift(1)
            df_lgd['exp_rr'] = np.where(df_lgd['shifted_cum_rr'].isnull(), df_lgd['exp_rr'], (
                df_lgd['exp_rr'] - df_lgd['shifted_cum_rr']) / (1 - df_lgd['shifted_cum_rr']))

            # lgd = 1 - exp_rr
            df_lgd['lgd'] = 1 - df_lgd['exp_rr']

            # keep only the columns that are needed
            df_lgd = df_lgd[['sub_category', 'md', 'pdd_bal', 'payment',
                             'cum_payment', 'cum_rr', 'marg_rr', 'exp_rr', 'lgd']]

            lgd_output = pd.concat([lgd_output, df_lgd])

        # Assign the cashflow LGD for categories that with no defualt cases but are in the LGD CFL approach list
        if cats_need_assign != []:
            for cat in cats_need_assign:
                cat_assign = lgd_approach[lgd_approach['LGD_CATEGORY']
                                          == cat]['CFL_LGD_ASSIGNMENT'].values[0]
                print(
                    (f"Assigned the LGD of \"{cat_assign}\" to \"{cat}\"."))
                df_cat = lgd_output[lgd_output['sub_category']
                                    == cat_assign]
                df_cat['sub_category'] = cat
                lgd_output = pd.concat(
                    [df_cat, lgd_output], ignore_index=True)

        ################## format AutoLGD ##################
        header = {
            "index": ["Parameter name", "Parameter description", "Mandatory input", "Parameter data type"],
            "LGD_pool_id": ["LGD_POOL_ID", "LGD pool ID", "Y", "category"],
            # "pool_id_description": ["POOL_ID_DESCRIPTION", "LGD pool ID description", "N", "str"],
            "md": ["MONTH_IN_DEFAULT", "Month in default", "Y", "integer"],
            "sub_category": ["SUB_CATEGORY", "sub_category", "Y", "str"],
            "lgd_0": ["LGD_0", "Year 0 Cumulative LGD", "Y", "float"]
        }

        # create header for LGD_1 to LGD_30, the same as LGD_0
        for i in range(1, 31):
            header[f'lgd_{i}'] = [f'LGD_{i}',
                                  f"Year {i} Cumulative LGD", "Y", "float"]

        table_header = pd.DataFrame(header)

        if not lgd_output.empty:
            lgd_output['md'] = lgd_output['md'].astype(int)

            lgd_product_digit = lgd_approach[[
                'LGD_CATEGORY', 'LGD_PRODUCT_DIGIT']]
            lgd_product_digit.rename(
                columns={'LGD_CATEGORY': 'sub_category'}, inplace=True)

            lgd_table = pd.merge(lgd_output, lgd_product_digit,
                                 on='sub_category', how='left')
            # lgd_table['pool_id_description'] = lgd_table['sub_category']

            lgd_table.reset_index(drop=True, inplace=True)
            lgd_table = lgd_table[['LGD_PRODUCT_DIGIT',
                                   'sub_category', 'md', 'lgd']].copy()
            lgd_table.rename(columns={'lgd': 'lgd_0'}, inplace=True)

            # create LGD_1 to LGD_30, the same as LGD_0
            for i in range(1, 31):
                lgd_table[f'lgd_{i}'] = lgd_table['lgd_0']

            lgd_table['index'] = range(1, len(lgd_output) + 1)
            lgd_table['LGD_pool_id'] = lgd_table.apply(
                lambda row: f"CFL{int(row['LGD_PRODUCT_DIGIT']):02d}{int(row['md']):03d}XXX", axis=1
            )
            lgd_table = lgd_table.reindex(columns=table_header.columns)

            auto_lgd = pd.concat([table_header, lgd_table], ignore_index=True)
        else:
            auto_lgd = table_header

        return auto_lgd
