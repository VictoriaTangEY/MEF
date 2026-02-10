# Load packages
from typing import Dict
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings("ignore")


# show all columns and rows of a dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class collateral_lgd():
    def __init__(self, context):
        self.run_yymm = context.run_yymm
        self.prtflo_scope = context.prtflo_scope
        self.scenario_set = context.scenario_set
        self.days_in_year = context.days_in_year
        self.days_in_month = context.days_in_month
        self.dtype_tbl = context.dtype_tbl
        self.total_yr = context.total_yr
        self.t_zero = context.T_ZERO

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

    def _vectorize_decorator(self, func):
        return np.vectorize(func)

    def get_LGD_model_ID(self, df: pd.DataFrame, param: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_ = df.assign(
            SUB_CATEGORY=df.SUB_CATEGORY.str.upper(),
            MONTH_IN_DEFT=0,
        )

        param_ = (param['LGD_approach'].assign(
            LGD_CATEGORY=param['LGD_approach'].LGD_CATEGORY.str.upper(),
        ).drop(labels=['Parameter name'], axis=1))

        df_1 = (df_.merge(
            param_, how='left',
            left_on=['SUB_CATEGORY'],
            right_on=['LGD_CATEGORY']
        ))

        @self._vectorize_decorator
        def _assign_mid_digit() -> str:
            return '000'

        @self._vectorize_decorator
        def _assign_cq_digit() -> str:
            return 'XXX'

        df_2 = df_1.assign(
            LGD_MID_DIGIT=_assign_mid_digit(),

            LGD_CQ_DIGIT=_assign_cq_digit(),
        )

        df_3 = (df_2.assign(
            LGD_POOL_ID=(df_2.LGD_APPROACH +
                         df_2.LGD_PRODUCT_DIGIT +
                         df_2.LGD_MID_DIGIT +
                         df_2.LGD_CQ_DIGIT)
        ).drop(labels=['LGD_PRODUCT_DIGIT',
                       'LGD_MID_DIGIT',
                       'LGD_CQ_DIGIT',
                       'LGD_CATEGORY'], axis=1))

        return df_3

    def get_mef_multiplier(self, df: pd.DataFrame, t_zero: str, mef_list: list, pwa_df: pd.DataFrame):

        # get the observed Quarter (=T_ZERO) and the predict Quarter
        obs_Q = datetime.strptime(str(t_zero), "%Y%m%d")
        pred_Q = obs_Q + relativedelta(years=1)
        obs_Q = obs_Q.strftime("%Y/%m/%d")
        pred_Q = pred_Q.strftime("%Y/%m/%d")

        result_dict = {}

        for mef in mef_list:
            pwa_multiplier = 0
            for scenario in ('GROW', 'BASE', 'SEVE'):
                scenario_rate = (df[df['Code:'] == pred_Q][f'{mef}_{scenario}'].values[0]) / (
                    df[df['Code:'] == obs_Q][f'{mef}_{scenario}'].values[0])
                pwa_scenario_rate = scenario_rate * \
                    pwa_df[pwa_df['scenario'] == scenario]['pwa'].values[0]
                pwa_multiplier += pwa_scenario_rate

            result_dict[mef] = pwa_multiplier

        return result_dict

    def get_collateral_lgd(self, instr_df: pd.DataFrame,
                           snr_df: pd.DataFrame,
                           ca_df: pd.DataFrame,
                           param: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        # read in params
        lgd_approach = param['LGD_approach']
        coll_param = param['Collateral_parameter']
        cure_param = param['Cure_rate']
        pwa_df = param['pwa']

        # can add more mefs in the future enhancement
        mef_list = ['MNG_PROP_PPI']

        cats = lgd_approach[lgd_approach['LGD_APPROACH']
                            == 'COL'].LGD_CATEGORY.to_list()
        colls = coll_param.collateral_type.to_list()

        # get collateral table from instrument table
        coll_df = ca_df.copy()
        coll_df['SUB_CATEGORY'] = coll_df['SUB_CATEGORY'].str.lower()
        coll_df = coll_df[coll_df['SUB_CATEGORY'].isin(
            cats)][['SUB_CATEGORY'] + colls]
        coll_df = coll_df.groupby('SUB_CATEGORY')[colls].sum().reset_index()

        # get LGD_POOL_ID
        coll_df_1 = coll_df.copy()
        coll_df_1 = self.get_LGD_model_ID(coll_df_1, param)

        ########## calculation process ################
        # in the collective assessment table, sum DRAWN_BAL_LCY group by the sub_segment and become a new dataframe drawn_bal_df
        # Confirmed using LCY
        drawn_bal_df = ca_df[ca_df['SA_IND'] == False].copy()
        drawn_bal_df['SUB_CATEGORY'] = drawn_bal_df['SUB_CATEGORY'].str.lower()

        drawn_bal_df = drawn_bal_df[drawn_bal_df['SUB_CATEGORY'].isin(
            cats)].groupby('SUB_CATEGORY')['DRAWN_BAL_LCY'].sum()
        drawn_bal_df = drawn_bal_df.reset_index()
        drawn_bal_df.columns = ['SUB_CATEGORY', 'DRAWN_BAL_LCY']
        drawn_bal_df = drawn_bal_df.set_index('SUB_CATEGORY').T

        # left join the DRAWN_BAL_LCY in the drawn_bal to the collateral table by the GROUP_ID
        coll_value = coll_df[coll_df['SUB_CATEGORY'].isin(cats)].copy()

        coll_value = coll_value.set_index('SUB_CATEGORY').T
        coll_value.reset_index(inplace=True)
        coll_value.rename(columns={'index': 'collateral_type'}, inplace=True)

        # left join the coll_portion to the map_df by the collateral type
        coll_value = coll_param.merge(
            coll_value, on='collateral_type', how='left')

        # set collaterl_id as index
        coll_value = coll_value.set_index('collateral_id')

        # calculate adjusted collateral values
        coll_val_adj = coll_value.copy()

        multipliers = self.get_mef_multiplier(
            df=snr_df, t_zero=self.t_zero, mef_list=mef_list, pwa_df=pwa_df)

        # using iterrows to loop every row of coll_val_adj to adjust the value of column name in groups
        for i, row in coll_val_adj.iterrows():
            for cat in cats:
                if cat in row:
                    # adjust the collateral value by haircut
                    coll_val_adj.at[i, cat] = row[cat] * (1 - row['haircut'])
                    # adjust the collateral value by mev and FL
                    if row['forward_looking_adj'] == "Y":
                        coll_val_adj.at[i, cat] *= multipliers['MNG_PROP_PPI']

        # cap the collateral value by the drawn balance
        coll_prt_cap = coll_val_adj.copy()
        # used for capping the secured portion by priority
        coll_prt_cap.sort_values(by='priority', ascending=True, inplace=True)

        for cat in cats:
            # calculate secured portion
            coll_prt_cap[cat] /= drawn_bal_df[cat].values[0]
            # cap the secured portion by priority
            remaining = 1  # fully secured

            for index, row in coll_prt_cap.iterrows():
                coll_prt = row[cat]
                if coll_prt <= remaining:
                    remaining -= coll_prt
                else:
                    coll_prt_cap.at[index, cat] = remaining
                    remaining = 0

        # calculate the final LGD
        sec_prt = coll_prt_cap[cats].copy()
        sec_prt = sec_prt.T
        sec_prt['unsec'] = 1 - sec_prt.sum(axis=1)

        lgd = coll_prt_cap[['LGD']].copy()
        lgd = lgd.T
        lgd['unsec'] = 1

        # repossession LGD
        repo = sec_prt.copy()

        for col in repo.columns:
            repo[col] = sec_prt[col] * lgd.loc['LGD', col]

        repo_lgd = pd.DataFrame({
            'lgd_category': repo.index,
            'repo_lgd': repo.sum(axis=1)
        })

        # cure rate
        cr = cure_param[cure_param['lgd_category'].isin(
            cats)][['lgd_category', 'cure_rate']]

        # final LGD
        final_lgd = repo_lgd.copy()
        final_lgd = final_lgd.merge(cr, on='lgd_category', how='left')
        final_lgd['lgd_0'] = (1 - final_lgd['cure_rate']) * \
            final_lgd['repo_lgd'] + final_lgd['cure_rate'] * 0

        ################## format AutoLGD ##################
        lgd_output = final_lgd.copy()
        lgd_output.columns = lgd_output.columns.str.upper()
        lgd_output = lgd_output.applymap(
            lambda x: x.upper() if isinstance(x, str) else x)

        # left join the LGD_POOL_ID to the
        coll_df_1.rename(
            columns={'SUB_CATEGORY': 'LGD_CATEGORY'}, inplace=True)
        lgd_output = lgd_output.merge(
            coll_df_1[['LGD_CATEGORY', 'LGD_POOL_ID']], on='LGD_CATEGORY', how='left')
        lgd_output.insert(0, 'LGD_POOL_ID', lgd_output.pop('LGD_POOL_ID'))

        # add MD: month_in_default = 0
        # month in default = 0
        lgd_output['MD'] = 0
        lgd_output.insert(1, 'MD', lgd_output.pop('MD'))

        # create LGD_1 to LGD_30, the same as LGD_0
        for i in range(1, 31):
            lgd_output[f'LGD_{i}'] = lgd_output['LGD_0']

        header = {
            "INDEX": ["Parameter name", "Parameter description", "Mandatory input", "Parameter data type"],
            "LGD_POOL_ID": ["LGD_POOL_ID", "lgd_pool_id", "Y", "category"],
            "MD": ["MONTH_IN_DEFAULT", "Month in default", "Y", "integer"],
            "LGD_CATEGORY": ["LGD_CATEGORY", "lgd category", "Y", "str"],
            "LGD_0": ["LGD_0", "Year 0 Cumulative LGD", "Y", "float"]
        }

        # create header for LGD_1 to LGD_30, the same as LGD_0
        for i in range(1, 31):
            header[f'LGD_{i}'] = [f'LGD_{i}',
                                  f"Year {i} Cumulative LGD", "Y", "float"]

        table_header = pd.DataFrame(header)
        table_header.columns = table_header.columns.str.upper()

        # concat table header and lgd output
        lgd_table = lgd_output[[
            'LGD_POOL_ID', 'MD', 'LGD_CATEGORY'] + [f'LGD_{i}' for i in range(31)]].copy()
        lgd_table['INDEX'] = range(1, len(lgd_output) + 1)
        lgd_table = lgd_table.reindex(columns=table_header.columns)

        coll_lgd = pd.concat([table_header, lgd_table], ignore_index=True)

        return coll_lgd
