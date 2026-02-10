# Load packages
from util.loggers import createLogHandler
from input_handler.data_preprocessor import data_preprocessor
from input_handler.load_parameters import load_configuration_file, load_parameters
from input_handler.env_setting import run_setting
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict

import warnings
warnings.filterwarnings("ignore")


# show all columns and rows of a dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class SACollateralLGD():
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

    def get_collateral_lgd(self,
                           sa_df: pd.DataFrame,
                           param: Dict[str, pd.DataFrame],
                           custs: list) -> pd.DataFrame:

        coll_param = param['SA_COL_param']
        coll_list = coll_param.collateral_type.to_list()
        pwa_df = param['pwa']

        def get_scenario_wt(pwa_df):
            wt_best = pwa_df[pwa_df['scenario'] == 'GROW']['pwa'].values[0]
            wt_base = pwa_df[pwa_df['scenario'] == 'BASE']['pwa'].values[0]
            wt_worst = pwa_df[pwa_df['scenario'] == 'SEVE']['pwa'].values[0]
            return wt_best, wt_base, wt_worst

        def calc_wt_eir(cust_df):
            # Calculate weighted sum and total balance
            weighted_sum = (cust_df['DRAWN_BAL_LCY'] *
                            cust_df['EFF_INT_RT']).sum()
            total_balance = cust_df['DRAWN_BAL_LCY'].sum()

            # Handle case where total balance is zero to avoid division by zero
            if total_balance == 0:
                return 0.0

            # Calculate weighted average
            wt_eir = weighted_sum / total_balance

            return float(wt_eir)

        def apply_adjustment(colls, coll_param, wt_eir, scenario):
            # validate scenario
            scenario = scenario.lower()
            if scenario not in ["best", "base", "worst"]:
                raise ValueError(
                    "Invalid scenario parameter, must be 'best', 'base' or 'worst'")

            # preparation
            haircut_col = f"haircut_{scenario}"
            time_col = f"time_to_sell_{scenario}"
            cost_col = f"cost_to_sell_{scenario}"

            # initialize result series
            adjusted_colls = pd.Series(index=colls.index, dtype=float)

            # loop each coll
            for coll_type in colls.index:
                # get the param row for the coll_type
                param_row = coll_param[coll_param["collateral_type"]
                                       == coll_type]

                if param_row.empty:
                    adjusted_colls[coll_type] = 0.0
                    continue

                # get params
                haircut = param_row[haircut_col].values[0]
                time = param_row[time_col].values[0]
                cost = param_row[cost_col].values[0]

                # calculate adjusted value
                original_value = colls[coll_type]
                adjusted_value = original_value * \
                    (1 - haircut) * ((1 - cost) / (1 + wt_eir) ** time)

                adjusted_colls[coll_type] = adjusted_value

            return adjusted_colls

        def calc_loss_amt(colls_snr, drawn_bal, snr_wt):
            exposure = drawn_bal
            colls_npv = colls_snr.sum()
            # print('colls_npv:', colls_npv)
            if exposure <= colls_npv:
                loss_amt = 0.0
            else:
                loss_amt = snr_wt * (exposure - colls_npv)
            # print('loss_amt:', loss_amt)
            return loss_amt

        # loop each customer
        lgd_dict = dict()

        for cust in custs:

            cust_df = sa_df[sa_df['CUST_ID'] == cust].copy()

            # get the sum of drawn balance of the cust
            drawn_bal = cust_df['DRAWN_BAL_LCY'].sum()
            # print(drawn_bal)

            # get drawn balance weighted eir
            wt_eir = calc_wt_eir(cust_df)
            # print(wt_eir)

            # get collaterals groupby cust
            colls = cust_df[coll_list].sum(axis=0)
            # print(colls)

            # get scenario weight
            wt_best, wt_base, wt_worst = get_scenario_wt(pwa_df)
            # print(wt_best, wt_base, wt_worst)

            # apply haircut, time_to_sell, cost_to_sell to colllaterals by scenarios
            colls_best = apply_adjustment(colls, coll_param, wt_eir, 'best')
            colls_base = apply_adjustment(colls, coll_param, wt_eir, 'base')
            colls_worst = apply_adjustment(colls, coll_param, wt_eir, 'worst')
            # print(colls_best, colls_base, colls_worst)

            # calc loss amt for scenarios
            loss_best = calc_loss_amt(colls_best, drawn_bal, wt_best)
            loss_base = calc_loss_amt(colls_base, drawn_bal, wt_base)
            loss_worst = calc_loss_amt(colls_worst, drawn_bal, wt_worst)

            ttl_loss = loss_best + loss_base + loss_worst
            # print('ttl_loss: ', ttl_loss)

            # calc lgd
            if drawn_bal > 0:
                lgd = ttl_loss / drawn_bal
            else:
                lgd = 0.0

            # store result
            lgd_dict[cust] = lgd

        lgd_df = pd.DataFrame({
            'CUST_ID': lgd_dict.keys(),
            'LGD': lgd_dict.values()
        }).set_index('CUST_ID').reset_index()

        return lgd_df
