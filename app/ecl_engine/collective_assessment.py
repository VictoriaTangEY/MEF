# Load packages
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from input_handler.data_preprocessor import data_preprocessor
from input_handler.load_parameters import (load_configuration_file,
                                           load_parameters)
from input_handler.env_setting import run_setting
from ecl_engine import output_file_handler as ofh

# TBD: Just for analysis purpose
from memory_profiler import profile


class ecl_engine():
    def __init__(self, context):
        self.run_yymm = context.run_yymm
        self.prtflo_scope = context.prtflo_scope
        self.scenario_set = context.scenario_set
        self.days_in_year = context.days_in_year
        self.days_in_month = context.days_in_month
        self.dtype_tbl = context.dtype_tbl
        self.total_yr = context.total_yr

        self.masterPath = context.masterPath
        self.parmPath = context.parmPath
        self.inDataPath = context.inDataPath
        self.resultPath = context.resultPath

    def _vectorize_decorator(self, func):
        return np.vectorize(func)

    def _get_valid_cashflow_scope(self,
                                  instr_df: pd.DataFrame,
                                repay_df: pd.DataFrame)-> pd.DataFrame:
        
        # to filter out those zero repayment contract
        # as these contract may still have interest need to pay (ead_ocy is not zero)
        repay_df_grp = repay_df.groupby('CONTRACT_ID').agg(
                        SUM_REPAY_BAL_OCY=('UNPAID_PRIN_BAL_BEG_OCY', 'sum')
                        ).reset_index()
        repay_df_grp['KEY']=repay_df_grp['CONTRACT_ID'].astype(str)+str('ON') # repayment schedule is only on-bal
        repay_df_grp = repay_df_grp[['KEY', 'SUM_REPAY_BAL_OCY']]

        instr_df['KEY'] = instr_df['CONTRACT_ID'].astype(str) + instr_df['ON_OFF_BAL_IND'].astype(str).str.upper()

        df_ = pd.merge(instr_df,repay_df_grp, how='left',left_on = 'KEY', right_on = 'KEY')

        cond = ((df_.ON_OFF_BAL_IND.str.upper() == 'ON')
                & (~df_.PRTFLO_ID.str.startswith(('6000', '6010', '3000', '3010'))) #sure there is no nan in prtflio_id
                & (df_.DATA_SOURCE_CD.str.upper() == 'LOAN')
                & (df_.STAGE_FINAL < 3)
                & (df_.SUM_REPAY_BAL_OCY > 0))
        
        df_1 = (df_.assign(
            is_valid_cashflow=np.where(cond, True, False)
        ))

        return df_1

    def _check_key_consistency(self, instr_df: pd.DataFrame,
                               repay_df: pd.DataFrame) -> pd.DataFrame:
        """
        Check if all valid contract ID in instrument table
        are fully captured in repayment table.
        """
        repay_df_keys = (repay_df
                         .filter(items=['CONTRACT_ID'])
                         .drop_duplicates(subset=['CONTRACT_ID'])
                         .assign(
                             is_in_repay_df=True,
                         ))

        df_ = (instr_df.merge(repay_df_keys,
                              how='left',
                              left_on=['CONTRACT_ID'],
                              right_on=['CONTRACT_ID']
                              ))
        df_1 = (df_.assign(
            is_in_repay_df=df_.is_in_repay_df.fillna(False)
        ))

        return df_1

    def _check_maturity_sufficiency(self,
                                    df: pd.DataFrame,
                                    repay_df: pd.DataFrame) -> pd.DataFrame:
        """
        To check if cash flow data is sufficient to support
        the adjusted maturity date
        """
        repay_df_grp = (repay_df.groupby(by=['CONTRACT_ID'])
                        .agg(
                            CF_DATE_MAX=pd.NamedAgg('CF_DATE', 'max'),
        )
            .reset_index())

        df_ = (df.merge(
            repay_df_grp, how='left',
            left_on=['CONTRACT_ID'],
            right_on=['CONTRACT_ID']
        ))

        cond = (df.MAT_DATE_ADJ <= df_.CF_DATE_MAX)

        df_1 = (df_.assign(
            is_suff_cf=np.where(cond, True, False),
        ))

        return df_1

    def filter_cashflow_valid_scope(self,
                                    df: pd.DataFrame,
                                    repay_df: pd.DataFrame) -> pd.DataFrame:

        df_ = (df.pipe(self._get_valid_cashflow_scope,repay_df=repay_df)
               .pipe(self._check_key_consistency, repay_df=repay_df)
               .pipe(self._check_maturity_sufficiency, repay_df=repay_df)
               )

        cond = ((df_.is_valid_cashflow)
                & (df_.is_in_repay_df)
                & (df_.is_suff_cf))

        df_valid = df_[cond]
        df_invalid = df_[~cond]

        return df_valid, df_invalid

    def _merge_repayment_schedule(self,
                                  instr_df: pd.DataFrame,
                                  repay_df: pd.DataFrame) -> pd.DataFrame:

        keep_cols = [
            'REPORT_DATE',
            'CONTRACT_ID',
            'ON_OFF_BAL_IND',
            'STAGE_FINAL',
            'EFF_INT_RT',
            'MAT_DATE',
            'MAT_DATE_ADJ',
            'PD_POOL_ID',
            'LGD_POOL_ID',
            'PREPAYMENT_POOL_ID',
            'PENALTY_OCY',
            'OTHER_FEE_AND_CHARGES_OCY',
            'ACRU_INT_OCY'
        ]
        #250422:update unpaid to include interest
        repay_df['EAD_UNADJ'] = repay_df['UNPAID_PRIN_BAL_BEG_OCY'] + repay_df['INT_ACCR']

        keep_cols_repay = [
            'CONTRACT_ID',
            'CF_DATE',
            'EAD_UNADJ'
        ]

        ead_ = (instr_df[keep_cols].merge(
            repay_df[keep_cols_repay], how='left',
            left_on=['CONTRACT_ID'],
            right_on=['CONTRACT_ID'])
            .sort_values(by=['CONTRACT_ID', 'CF_DATE'])
            #.rename(columns={'UNPAID_PRIN_BAL_BEG_OCY': 'EAD_UNADJ'})
        )

        return ead_

    def _calculate_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function to generate intra period days for 
        - Discounting purpose
        - Pro-rata on PD and prepayment rate
        """

        ead_df_ = (df.assign(
            cum_days=(df.CF_DATE - df.REPORT_DATE).dt.days,
        ))

        ead_df_1 = (ead_df_.assign(
            cum_days_in_year=(ead_df_.cum_days) / self.days_in_year,
            cum_days_cap_12m=np.minimum(ead_df_.cum_days, self.days_in_year),
            cum_days_cap_12m_in_year=np.minimum(ead_df_.cum_days
                                                / self.days_in_year, 1),
        ))

        ead_df_2 = (ead_df_1.assign(
            years=(np.ceil(ead_df_1.cum_days_in_year)
                     .astype(int)
                     .astype(str)),

            cum_days_lag=(ead_df_1
                          .groupby(by=['CONTRACT_ID'])['cum_days']
                          .shift()
                          .fillna(0)),

            cum_days_cap_12m_lag=(ead_df_1
                                  .groupby(by=['CONTRACT_ID'])['cum_days_cap_12m']
                                  .shift()
                                  .fillna(0)),
        ))

        ead_df_3 = (ead_df_2.assign(
            mar_days=(ead_df_2.cum_days-ead_df_2.cum_days_lag),
            mar_days_cap_12m=(ead_df_2.cum_days_cap_12m
                              - ead_df_2.cum_days_cap_12m_lag),
        ))

        filter_12m_cond = (ead_df_3.cum_days_lag <= self.days_in_year)

        # 241126: cast date
        # ead_df_3['CF_DATE'] = pd.to_datetime(ead_df_3['CF_DATE'], errors='coerce')
        # ead_df_3['MAT_DATE_ADJ'] = pd.to_datetime(ead_df_3['MAT_DATE_ADJ'], errors='coerce')
        filter_mature_cond = (ead_df_3.CF_DATE <= ead_df_3.MAT_DATE_ADJ)

        ead_df_4 = (ead_df_3.assign(
            mar_days_in_year=(ead_df_3.mar_days) / self.days_in_year,
            mar_days_cap_12m_in_year=(
                ead_df_3.mar_days_cap_12m) / self.days_in_year,

            filter_12m=np.where(filter_12m_cond, 1, 0),
            filter_mature=np.where(filter_mature_cond, 1, 0),
        ))

        return ead_df_4
    
    def _calculate_days_months(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function to generate intra period days and months for 
        - Discounting purpose
        - Pro-rata on PD and prepayment rate
                """
        ead_df_ = (df.assign(
                        #cum_days=(df.CF_DATE - df.REPORT_DATE).dt.days,
                    cum_months=(df.CF_DATE.dt.to_period('M') - df.REPORT_DATE.dt.to_period('M')).apply(lambda x: x.n),
                ))

        ead_df_1 = (ead_df_.assign(
            #cum_days_in_year=(ead_df_.cum_days) / self.days_in_year,
            # cum_days_cap_12m=np.minimum(ead_df_.cum_days, self.days_in_year),
            # cum_days_cap_12m_in_year=np.minimum(ead_df_.cum_days
            #                                     / self.days_in_year, 1),
            cum_months_in_year=(ead_df_.cum_months) / 12,
            cum_months_cap_12m=np.minimum(ead_df_.cum_months, 12),
            cum_months_cap_12m_in_year=np.minimum(ead_df_.cum_months
                                                / 12, 1),
            cum_days_in_month=ead_df_['CF_DATE'].dt.day/ead_df_.CF_DATE.dt.days_in_month.astype(int) 
            #250307: calculate daily cum PD for bond
        ))

        ead_df_2 = (ead_df_1.assign(
            years=(np.ceil(ead_df_1.cum_months_in_year)
                        .astype(int)
                        .astype(str)),
            # cum_days_lag=(ead_df_1
            #                 .groupby(by=['CONTRACT_ID'])['cum_days']
            #                 .shift()
            #                 .fillna(0)),
            # cum_days_cap_12m_lag=(ead_df_1
            #                         .groupby(by=['CONTRACT_ID'])['cum_days_cap_12m']
            #                         .shift()
            #                         .fillna(0)),
            months=(ead_df_1.cum_months.astype(int).astype(str)),
            cum_months_lag=(ead_df_1
                    .groupby(by=['CONTRACT_ID'])['cum_months']
                    .shift()
                    .fillna(0)),
            cum_months_cap_12m_lag=(ead_df_1
                        .groupby(by=['CONTRACT_ID'])['cum_months_cap_12m']
                        .shift()
                        .fillna(0)),
            
        ))

        ead_df_3 = (ead_df_2.assign(
            # mar_days=(ead_df_2.cum_days-ead_df_2.cum_days_lag),
            # mar_days_cap_12m=(ead_df_2.cum_days_cap_12m
            #                     - ead_df_2.cum_days_cap_12m_lag),
            mar_months=(ead_df_2.cum_months-ead_df_2.cum_months_lag),
            mar_months_cap_12m=(ead_df_2.cum_months_cap_12m
                    - ead_df_2.cum_months_cap_12m_lag),
        ))

        #filter_12m_cond = (ead_df_3.cum_days_lag <= self.days_in_year)
        filter_12m_cond = (ead_df_3.cum_months_lag < 12)

        # 241126: cast date
        # ead_df_3['CF_DATE'] = pd.to_datetime(ead_df_3['CF_DATE'], errors='coerce')
        # ead_df_3['MAT_DATE_ADJ'] = pd.to_datetime(ead_df_3['MAT_DATE_ADJ'], errors='coerce')
        filter_mature_cond = (ead_df_3.CF_DATE <= ead_df_3.MAT_DATE_ADJ)

        ead_df_4 = (ead_df_3.assign(
            # mar_days_in_year=(ead_df_3.mar_days) / self.days_in_year,
            mar_months_in_year=(ead_df_3.mar_months) /12,
            # mar_days_cap_12m_in_year=(
            #     ead_df_3.mar_days_cap_12m) / self.days_in_year,
            mar_months_cap_12m_in_year=(
                ead_df_3.mar_months_cap_12m) / 12,
            filter_12m=np.where(filter_12m_cond, 1, 0),
            filter_mature=np.where(filter_mature_cond, 1, 0),
        ))

        return ead_df_4

    def _adjust_prepayment(self,
                           df: pd.DataFrame,
                           param=Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_ = (df.merge(
            param['Prepayment_parameter'], how='left',  # type: ignore
            left_on=['PREPAYMENT_POOL_ID'],
            right_on=['PREPAYMENT_ID']
        )
            .drop(labels=['PREPAYMENT_ID',
                          'Parameter name',
                          'DESCRIPTION'], axis=1)
        )

        df_1 = (df_.assign(
            SMM=1-(1-df_.PREPAYMENT_RATE_VAL)**(df_.mar_months_in_year),
            SMM_cap_12m=1 -
                (1-df_.PREPAYMENT_RATE_VAL)**(df_.mar_months_cap_12m_in_year),
                ))

        #250113: avoid double count on CRU_INT_OCY, 
        # as EAD_OCY for ON already includes it based on nature of drawn_bal
        df_2 = (df_1.assign(
            EAD_PPADJ_12M=df_1.EAD_UNADJ *
                (1-df_1.SMM_cap_12m), # + df_1.ACRU_INT_OCY,
                EAD_PPADJ_LT=df_1.EAD_UNADJ*(1-df_1.SMM), # + df_1.ACRU_INT_OCY, 
                ))

        return df_2
    
    def  _adjust_prepayment_old(self,
                           df: pd.DataFrame,
                           param=Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_ = (df.merge(
            param['Prepayment_parameter'], how='left',  # type: ignore
            left_on=['PREPAYMENT_POOL_ID'],
            right_on=['PREPAYMENT_ID']
        )
            .drop(labels=['PREPAYMENT_ID',
                          'Parameter name',
                          'DESCRIPTION'], axis=1)
        )

        df_1 = (df_.assign(
            SMM=1-(1-df_.PREPAYMENT_RATE_VAL)**(df_.mar_days_in_year),
            SMM_cap_12m=1 -
                (1-df_.PREPAYMENT_RATE_VAL)**(df_.mar_days_cap_12m_in_year),
                ))

        df_2 = (df_1.assign(
            EAD_PPADJ_12M=df_1.EAD_UNADJ *
                (1-df_1.SMM_cap_12m) + df_1.ACRU_INT_OCY,
                EAD_PPADJ_LT=df_1.EAD_UNADJ*(1-df_1.SMM) + df_1.ACRU_INT_OCY, 
                ))

        return df_2
    
    def _adjust_ead_valid(self,
                          df_2:pd.DataFrame):
        #250113: as the UNPAID_PRIN_BAL_BEG_OCY does not contain other 3 unlike EAD_OCY, add 3 items back for valid cases
        df_3 = (df_2.assign(
                EAD_PPADJ_12M=df_2.EAD_PPADJ_12M + 
                df_2.ACRU_INT_OCY + 
                df_2.PENALTY_OCY +
                df_2.OTHER_FEE_AND_CHARGES_OCY,
                EAD_PPADJ_LT=df_2.EAD_PPADJ_LT + 
                df_2.ACRU_INT_OCY + 
                df_2.PENALTY_OCY +
                df_2.OTHER_FEE_AND_CHARGES_OCY,
                ))
        return df_3

    def _consol_pd_param(self,
                         param_bond: pd.DataFrame,
                         param: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        keep_cols = ['PD_POOL_ID'] + [f'PD_{i}' for i in range(361)]
        full_pd_param_ = pd.DataFrame()

        # 241118 yj: to use bond pd applied mef, we dun merge new param into param large list here
        # consol_param_list = ['AutoPDParam',
        #                      'PD_term_structure',]
        # for pd_param in consol_param_list:
        #     _param = (param[pd_param][keep_cols])
        #     full_pd_param_ = pd.concat([full_pd_param_, _param], axis=0)

        pd_bond_param = (param_bond[keep_cols])
        pd_loan_param = (param['AutoPDParam'][keep_cols])
        full_pd_param_ = pd.concat([pd_bond_param, pd_loan_param], axis=0)

        full_pd_param_['PD_POOL_ID'] = (full_pd_param_['PD_POOL_ID']
                                        .astype('category'))

        return full_pd_param_

    def _consol_lgd_param(self,
                          param: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        keep_cols = ['LGD_POOL_ID'] + [f'LGD_{i}' for i in range(31)]

        consol_param_list = ['AutoLGDParam',
                             'LGD_term_structure',]

        full_lgd_param_ = pd.DataFrame()

        for lgd_param in consol_param_list:
            _param = (param[lgd_param][keep_cols])
            full_lgd_param_ = pd.concat([full_lgd_param_, _param], axis=0)

        full_lgd_param_['LGD_POOL_ID'] = (full_lgd_param_['LGD_POOL_ID']
                                          .astype('category'))

        return full_lgd_param_

    # 241118 yj: for bond pd, the mef multiplier is applied seperately from autopd, which is loan pd already considered mef
    def _apply_mef(self,
                   param_mef: pd.DataFrame,
                   pd_param_bond: pd.DataFrame) -> pd.DataFrame:

        pd_param_ = pd_param_bond.copy()

        columns_to_multiply = [f'PD_{i}' for i in range(
            31) if f'PD_{i}' in pd_param_.columns]

        mef_bond = param_mef.query("segment == 'NNCRD'").multiplier.values

        pd_param_[columns_to_multiply] = pd_param_[columns_to_multiply].apply(
            lambda x: (x * mef_bond).apply(
                lambda y: min(y, 1.0)))  # cap 100%
        return pd_param_

    def _transpose_pd(self,
                      pd_param: pd.DataFrame) -> pd.DataFrame:

        pd_t_ = (pd_param.melt(id_vars=['PD_POOL_ID'],
                                var_name='term',
                                value_name='cum_PD'))

        pd_t_ = (pd_t_.assign(
            months=pd_t_.term.str.extract(r'(\d+)').astype(int),)
            .sort_values(by=['PD_POOL_ID', 'months'])
            .drop(labels=['term'], axis=1)
        )

        pd_t_ = pd_t_.assign(
            cum_PD_lag=(pd_t_
                        .groupby(by=['PD_POOL_ID'])['cum_PD']
                        .shift()
                        .fillna(0)),)

        pd_t_ = (pd_t_.assign(
            mar_PD=pd_t_.cum_PD - pd_t_.cum_PD_lag,
            months=pd_t_.months.astype(str),
        )
            .drop(labels=['cum_PD_lag'], axis=1))

        return pd_t_

    def _transpose_lgd(self,
                       lgd_param: pd.DataFrame) -> pd.DataFrame:

        lgd_t_ = (lgd_param.melt(id_vars=['LGD_POOL_ID'],
                                 var_name='term',
                                 value_name='LGD'))

        lgd_t_ = (lgd_t_.assign(
            years=lgd_t_.term.str.extract(r'(\d+)').astype(int),)
            .sort_values(by=['LGD_POOL_ID', 'years'])
            .drop(labels=['term'], axis=1))

        lgd_t_ = (lgd_t_.assign(
            years=lgd_t_.years.astype(str),))

        return lgd_t_
    
    def _convert_bond_pd_param_monthly(self,
                                       pd_param_bond_mef: pd.DataFrame,
                                        ) -> pd.DataFrame:
        # Create a new DataFrame to store the monthly data
        monthly_PD_bond = pd.DataFrame()
        monthly_PD_bond[['Parameter name', 'PD_POOL_ID', 'POOL_DESCRIPTION', 'PD_0']] = pd_param_bond_mef[['Parameter name', 'PD_POOL_ID', 'POOL_DESCRIPTION', 'PD_0']]
        #monthly_PD_bond = monthly_PD_bond.assign(**pd_param_bond_mef[['Parameter name', 'PD_POOL_ID', 'POOL_DESCRIPTION', 'PD_0']])
        # Loop through each pair of columns and calculate monthly values
        for i in range(0, 30):#year
            pd_prev = pd_param_bond_mef[f'PD_{i}']
            pd_current = pd_param_bond_mef[f'PD_{i+1}']
            for j in range(1, 13):#month
                new_col_name = f'PD_{i*12 + j}'
                monthly_PD_bond[new_col_name] = pd_prev + (pd_current - pd_prev) * j / 12 #250307
        mask = monthly_PD_bond['PD_POOL_ID'].str.contains('018', na=False)
        monthly_PD_bond.loc[mask, 'PD_1':'PD_360'] = 0

        return(monthly_PD_bond)

    def calculate_PD(self,
                     ead_df: pd.DataFrame,
                     param: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        pd_param_bond_mef = self._apply_mef(
            param_mef=param['MEFmultipliers'], pd_param_bond=param['PD_term_structure'])
        pd_param_consol = self._consol_pd_param(
            param_bond=pd_param_bond_mef, param=param)
        pd_t = self._transpose_pd(pd_param=pd_param_consol)

        ead_df_ = (ead_df.merge(
            pd_t, how='left',
            left_on=['PD_POOL_ID', 'years'],
            right_on=['PD_POOL_ID', 'years']))

        ead_df_ = (ead_df_.assign(
            mPDLn_12m=np.log((1 - ead_df_.mar_PD) **
                             (ead_df_.mar_days_cap_12m_in_year)),
            mPDLn_lt=np.log((1 - ead_df_.mar_PD)**(ead_df_.mar_days_in_year)),
        ))

        ead_df_ = (ead_df_.assign(
            mPDLnCum_12m=(ead_df_
                          .groupby(by=['CONTRACT_ID'])
                          .agg(
                              mPDLnCum_12m=pd.NamedAgg('mPDLn_12m', 'cumsum'))),

            mPDLnCum_lt=(ead_df_
                         .groupby(by=['CONTRACT_ID'])
                         .agg(
                             mPDLnCum_lt=pd.NamedAgg('mPDLn_lt', 'cumsum')))))

        ead_df_ = ead_df_.assign(
            cumPD_12m=(1-np.exp(ead_df_.mPDLnCum_12m)),
            cumPD_lt=(1-np.exp(ead_df_.mPDLnCum_lt)),
        )

        ead_df_ = (ead_df_.assign(
            cumPD_12m_prev=(ead_df_
                            .groupby(by=['CONTRACT_ID'])['cumPD_12m']
                            .shift()
                            .fillna(0)),
            cumPD_lt_prev=(ead_df_
                           .groupby(by=['CONTRACT_ID'])['cumPD_lt']
                           .shift()
                           .fillna(0))))

        ead_df_ = (ead_df_.assign(
            marPD_12m_freq=ead_df_['cumPD_12m']-ead_df_['cumPD_12m_prev'],
            marPD_lt_freq=ead_df_['cumPD_lt']-ead_df_['cumPD_lt_prev']))

        return ead_df_

    def calculate_PD_new(self,
                        ead_df: pd.DataFrame,
                        param: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        pd_param_bond_mef = self._apply_mef(
            param_mef=param['MEFmultipliers'], pd_param_bond=param['PD_term_structure'])
        pd_param_bond_mef_monthly = self._convert_bond_pd_param_monthly(pd_param_bond_mef=pd_param_bond_mef)
        pd_param_consol = self._consol_pd_param(
            param_bond=pd_param_bond_mef_monthly, param=param)
        pd_t = self._transpose_pd(pd_param=pd_param_consol)


        ead_df_ = (ead_df.merge(
            pd_t, how='left',
            left_on=['PD_POOL_ID', 'months'],
            right_on=['PD_POOL_ID', 'months']))

        # 20250213 yj: as the mar_PD is monthly PD already, depreciated the calculation of monthly PD 
        # ead_df_ = (ead_df_.assign(
        #     mPDLn_12m=np.log((1 - ead_df_.mar_PD) **
        #                         (ead_df_.mar_months_cap_12m_in_year)),
        #     mPDLn_lt=np.log((1 - ead_df_.mar_PD)**(ead_df_.mar_months_in_year)),
        # ))

        # ead_df_ = (ead_df_.assign(
        #     mPDLnCum_12m=(ead_df_
        #                     .groupby(by=['CONTRACT_ID'])
        #                     .agg(
        #                         mPDLnCum_12m=pd.NamedAgg('mPDLn_12m', 'cumsum'))),

        #     mPDLnCum_lt=(ead_df_
        #                     .groupby(by=['CONTRACT_ID'])
        #                     .agg(
        #                         mPDLnCum_lt=pd.NamedAgg('mPDLn_lt', 'cumsum')))))

        # ead_df_ = ead_df_.assign(
        #     cumPD_12m=(1-np.exp(ead_df_.mPDLnCum_12m)),
        #     cumPD_lt=(1-np.exp(ead_df_.mPDLnCum_lt)),
        # )

        # ead_df_ = (ead_df_.assign(
        #     cumPD_12m_prev=(ead_df_
        #                     .groupby(by=['CONTRACT_ID'])['cumPD_12m']
        #                     .shift()
        #                     .fillna(0)),
        #     cumPD_lt_prev=(ead_df_
        #                     .groupby(by=['CONTRACT_ID'])['cumPD_lt']
        #                     .shift()
        #                     .fillna(0))))

        # ead_df_ = (ead_df_.assign(
        #     marPD_12m_freq=ead_df_['cumPD_12m']-ead_df_['cumPD_12m_prev'],
        #     marPD_lt_freq=ead_df_['cumPD_lt']-ead_df_['cumPD_lt_prev']))
        
        ead_df_ = (ead_df_.assign(
            marPD_12m_freq=np.where(
                ead_df_.PD_POOL_ID.str[:2].isin(['GC', 'SL', 'SF']),
                ead_df_['mar_PD'] * ead_df_['filter_12m'] * ead_df_.cum_days_in_month,
                ead_df_['mar_PD'] * ead_df_['filter_12m']
            )
        ,
                marPD_lt_freq=np.where(
            ead_df_.PD_POOL_ID.str[:2].isin(['GC', 'SL', 'SF']), 
            ead_df_['mar_PD'] * ead_df_.cum_days_in_month,       
            ead_df_['mar_PD']                                     
            )
        ))
        return ead_df_
    
    def calculate_LGD(self,
                      ead_df: pd.DataFrame,
                      param: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        lgd_param_consol = self._consol_lgd_param(param=param)
        lgd_t = self._transpose_lgd(lgd_param=lgd_param_consol)

        ead_df_ = (ead_df.merge(
            lgd_t, how='left',
            left_on=['LGD_POOL_ID', 'years'],
            right_on=['LGD_POOL_ID', 'years']))

        return ead_df_

    def calculate_discount_factor(self,
                                  df: pd.DataFrame,
                                  is_mute: bool = False) -> pd.DataFrame:

        if is_mute:
            df_ = df.assign(
                disc_factor=1,
            )
        else:
            df_ = df.assign(
                disc_factor=(1+df.EFF_INT_RT)**(-df.cum_months_in_year), # 250704 250613 update to cumulative month
            )

        return df_

    def calculate_engine_ecl(self,
                             ecl_df: pd.DataFrame,
                             instr_df: pd.DataFrame
                             ) -> pd.DataFrame:
        """
        To calculate the engine ECL amount
        """
        ecl_df_ = (ecl_df.assign(

            # For Model validation purpose
            IFRS9_PD_12M=(ecl_df.mar_PD *
                          ecl_df.filter_12m
                          ),

            IFRS9_PD_12M_MADJ=(ecl_df.marPD_12m_freq *
                               ecl_df.filter_12m
                               ),

            IFRS9_PD_LT=(ecl_df.marPD_lt_freq),

            IFRS9_LGD_12M=(ecl_df.LGD *
                           ecl_df.filter_12m),

            IFRS9_LGD_LT=(ecl_df.LGD),

            ECL_ENGINE_12M_OCY=(ecl_df.EAD_PPADJ_12M
                                * ecl_df.marPD_12m_freq
                                * ecl_df.LGD
                                * ecl_df.filter_12m
                                * ecl_df.filter_mature
                                * ecl_df.disc_factor),

            ECL_ENGINE_LT_OCY=(ecl_df.EAD_PPADJ_LT
                               * ecl_df.marPD_lt_freq
                               * ecl_df.LGD
                               * ecl_df.filter_mature
                               * ecl_df.disc_factor),
        ))

        ecl_df_agg = (ecl_df_.groupby(by=['CONTRACT_ID'])
                      .agg(
            IFRS9_PD_12M=pd.NamedAgg('IFRS9_PD_12M', 'max'),
            IFRS9_PD_12M_MADJ=pd.NamedAgg(
                'IFRS9_PD_12M_MADJ', 'sum'),
            
            IFRS9_PD_LT=pd.NamedAgg('IFRS9_PD_LT', 'sum'),

            IFRS9_LGD_12M=pd.NamedAgg('IFRS9_LGD_12M', 'max'),
            IFRS9_LGD_LT=pd.NamedAgg('IFRS9_LGD_LT', 'mean'),

            ECL_ENGINE_12M_OCY=pd.NamedAgg(
                'ECL_ENGINE_12M_OCY', 'sum'),
            ECL_ENGINE_LT_OCY=pd.NamedAgg('ECL_ENGINE_LT_OCY', 'sum'),)
            .reset_index())

        ecl_final = (instr_df.merge(
            ecl_df_agg, how='left',
            left_on=['CONTRACT_ID'],
            right_on=['CONTRACT_ID']
        ))

        # Choose final engine ECL based on stage assessment results
        cond = [
            ecl_final.STAGE_FINAL == 1,
            ecl_final.STAGE_FINAL >= 2,
        ]

        choices = [
            ecl_final.ECL_ENGINE_12M_OCY,
            ecl_final.ECL_ENGINE_LT_OCY,
        ]

        ecl_final['ECL_ENGINE_OCY'] = (np.select(cond,
                                                 choices,
                                                 default=-999))

        return ecl_final, ecl_df_

    # create dummy repayment list for invalid records
    def _create_repay_df_dummy(self, 
                               repay_df_invalid_: pd.DataFrame, 
                               lastrowZERO=True):
        df_ = pd.DataFrame(
            columns=['MAT_DATE_ADJ', 'CONTRACT_ID', 'UNPAID_PRIN_BAL_BEG_OCY', 'CF_DATE'])
        for index, row in repay_df_invalid_.iterrows():
            date_range = pd.date_range(
                start=row['REPORT_DATE'], end=row['MAT_DATE_ADJ']+pd.offsets.MonthEnd(0), freq='M')
            new_rows = []
            for date_del in date_range[1:]:  # exclude the report date row
                # assign ead_ocy to each month
                new_rows.append({'MAT_DATE_ADJ': row['MAT_DATE_ADJ'], 'CONTRACT_ID': row['CONTRACT_ID'],
                                'CF_DATE': date_del, 'UNPAID_PRIN_BAL_BEG_OCY': row['EAD_OCY']})

            df_ = pd.concat([df_, pd.DataFrame(new_rows)], ignore_index=True)

        # 20241224: Cap the CF date with Adjusted maturity date
        df_1 = (df_.assign(
            # 241126: since EAD_OCY may equal to zero, makes ead NAN
            UNPAID_PRIN_BAL_BEG_OCY = df_.UNPAID_PRIN_BAL_BEG_OCY.fillna(0),
            CF_DATE = np.minimum(df_.MAT_DATE_ADJ, df_.CF_DATE),
            is_last_row = (df_.groupby('CONTRACT_ID').cumcount(ascending=False) == 0),
        ))

        # 20241224: Optimized a bit on this parts
        # For expired records, set UNPAID_PRIN_BAL_BEG_OCY at last month is zero
        df_2 = (df_1.assign(
            UNPAID_PRIN_BAL_BEG_OCY = np.where(
                (df_1.is_last_row) & lastrowZERO, 0, df_1.UNPAID_PRIN_BAL_BEG_OCY
            ),
        ))
        
        # 20250422: create interest column to align with repayment table
        df_2['INT_ACCR'] = 0
        return df_2
    
    def repay_df_creation_bycondition(self, df_bal_invalid: pd.DataFrame):
        ca_df_invalid_1_ex = df_bal_invalid.query(
            "MAT_DATE_ADJ <= REPORT_DATE")
        ca_df_invalid_1_nex = df_bal_invalid.query(
            "MAT_DATE_ADJ > REPORT_DATE")
        keep_cols_construct = [
            'CONTRACT_ID',
            'REPORT_DATE',
            'MAT_DATE_ADJ',
            'EAD_OCY',]
        # expired stage 1, MAT date <- report date + 1 month
        ca_df_invalid_1_ex_1 = ca_df_invalid_1_ex.query("STAGE_FINAL == 1")
        repay_df_invalid_1_ex_1 = ca_df_invalid_1_ex_1[keep_cols_construct]
        # repay_df_invalid_1_ex_1.MAT_DATE_ADJ = ca_df_invalid_1_ex.REPORT_DATE + pd.offsets.MonthEnd(1)
        repay_df_invalid_1_ex_1 = repay_df_invalid_1_ex_1.reset_index(
            drop=True)
        # expired stage 2, MAT date <- report date + 12 month
        ca_df_invalid_1_ex_2 = ca_df_invalid_1_ex.query("STAGE_FINAL == 2")
        repay_df_invalid_1_ex_2 = ca_df_invalid_1_ex_2[keep_cols_construct]
        # repay_df_invalid_1_ex_2.MAT_DATE_ADJ = ca_df_invalid_1_ex.REPORT_DATE + pd.DateOffset(months=12)
        repay_df_invalid_1_ex_2 = repay_df_invalid_1_ex_2.reset_index(
            drop=True)
        # not expired stage 1, MAT date > report date, use MAT_date but cap 12 month
        ca_df_invalid_1_nex_1 = ca_df_invalid_1_nex.query("STAGE_FINAL == 1")
        repay_df_invalid_1_nex_1 = ca_df_invalid_1_nex_1[keep_cols_construct]
        # repay_df_invalid_1_nex_1.MAT_DATE_ADJ = ca_df_invalid_1_nex.REPORT_DATE + pd.offsets.MonthEnd(12)
        repay_df_invalid_1_nex_1 = repay_df_invalid_1_nex_1.reset_index(
            drop=True)
        # not expired stage 2, MAT date > report date, do not adjust
        ca_df_invalid_1_nex_2 = ca_df_invalid_1_nex.query("STAGE_FINAL == 2")
        repay_df_invalid_1_nex_2 = ca_df_invalid_1_nex_2[keep_cols_construct]
        repay_df_invalid_1_nex_2 = repay_df_invalid_1_nex_2.reset_index(
            drop=True)

        repay_df_invalid_ex_1_final = self._create_repay_df_dummy(
            repay_df_invalid_1_ex_1, lastrowZERO=True)
        repay_df_invalid_ex_2_final = self._create_repay_df_dummy(
            repay_df_invalid_1_ex_2, lastrowZERO=True)
        # TODO: Why only this not last row Zero? YJ: for stage 1 the repayment continue at last month
        repay_df_invalid_nex_1_final = self._create_repay_df_dummy(
            repay_df_invalid_1_nex_1, lastrowZERO=False)
        repay_df_invalid_nex_2_final = self._create_repay_df_dummy(
            repay_df_invalid_1_nex_2, lastrowZERO=True)

        return (pd.concat([repay_df_invalid_ex_1_final, repay_df_invalid_ex_2_final,
                          repay_df_invalid_nex_1_final, repay_df_invalid_nex_2_final])
                .reset_index(drop=True))
    
    #250102: stage 3 calculation
    def _calculate_engine_stage3_ecl(self,
                                     ecl_df: pd.DataFrame,
                                    instr_df: pd.DataFrame
                                        ) -> pd.DataFrame:
        """
        To calculate the engine ECL amount for stage 3
        """
        ecl_df_1_ = (ecl_df.assign(
            
            # For Model validation purpose
            IFRS9_PD_12M=1,

            IFRS9_PD_12M_MADJ=1, #stage 3 IFRS9_PD_12M_MADJ shall be 1 at first record

            IFRS9_PD_LT=1,

            IFRS9_LGD_12M=(ecl_df.LGD),

            IFRS9_LGD_LT=(ecl_df.LGD),


            EAD_PPADJ_12M=(ecl_df.EAD_OCY),
            EAD_PPADJ_LT=(ecl_df.EAD_OCY),
        ))

        ecl_df_ = (ecl_df_1_.assign(
            ECL_ENGINE_12M_OCY=(ecl_df_1_.EAD_PPADJ_12M
                                * ecl_df_1_.LGD),

            ECL_ENGINE_LT_OCY=(ecl_df_1_.EAD_PPADJ_LT
                                * ecl_df_1_.LGD),
        ))

        ecl_df_agg = (ecl_df_.groupby(by=['CONTRACT_ID', 'ON_OFF_BAL_IND'])
                    .agg(
            IFRS9_PD_12M=pd.NamedAgg('IFRS9_PD_12M', 'max'),
            IFRS9_PD_12M_MADJ=pd.NamedAgg(
                'IFRS9_PD_12M_MADJ', 'sum'),
            IFRS9_PD_LT=pd.NamedAgg('IFRS9_PD_LT', 'sum'),

            IFRS9_LGD_12M=pd.NamedAgg('IFRS9_LGD_12M', 'max'),
            IFRS9_LGD_LT=pd.NamedAgg('IFRS9_LGD_LT', 'mean'),

            ECL_ENGINE_12M_OCY=pd.NamedAgg(
                'ECL_ENGINE_12M_OCY', 'sum'),
            ECL_ENGINE_LT_OCY=pd.NamedAgg('ECL_ENGINE_LT_OCY', 'sum'),)
            .reset_index())

        ecl_final = (instr_df.merge(
            ecl_df_agg, how='left',
            left_on=['CONTRACT_ID', 'ON_OFF_BAL_IND'],
            right_on=['CONTRACT_ID', 'ON_OFF_BAL_IND']
        ))

        # To align with stage 1&2 data storage way
        cond = [
            ecl_final.STAGE_FINAL < 3,
            ecl_final.STAGE_FINAL == 3,
        ]

        choices = [
            ecl_final.ECL_ENGINE_12M_OCY,
            ecl_final.ECL_ENGINE_LT_OCY,
        ]

        ecl_final['ECL_ENGINE_OCY'] = (np.select(cond,
                                                    choices,
                                                    default=-999))

        return ecl_final, ecl_df_
    
    # ECL calculation result files handler
    # @profile

    def _run(self,
            instr_df: pd.DataFrame,
            repay_df: pd.DataFrame,
            param: Dict[str, pd.DataFrame]
            ) -> pd.DataFrame:

        # Split the collective assessment run into 2 routes
        # One with complete cash flow data
        # The other without valid cash flow data
        ca_df_valid, ca_df_invalid = self.filter_cashflow_valid_scope(df=instr_df,
                                                                      repay_df=repay_df)

        # 20241227 - Added the count condition to avoid crashing 
        # if not records extracted
        # Collective assessment (Valid cash flow route)
        # EAD model
        if ca_df_valid.shape[0] > 0:
            ecl_df_valid, ecl_interim_df_valid = (ca_df_valid
                                                .pipe(self._merge_repayment_schedule,
                                                        repay_df=repay_df)
                                                .pipe(self._calculate_days)
                                                .pipe(self._adjust_prepayment, param=param)
                                                .pipe(self.calculate_PD, param=param)
                                                .pipe(self.calculate_LGD, param=param)
                                                .pipe(self.calculate_discount_factor)
                                                .pipe(self.calculate_engine_ecl,
                                                        instr_df=ca_df_valid)
                                                )
        else:
            ecl_df_valid = pd.DataFrame()
            ecl_interim_df_valid = pd.DataFrame()

        # Invalid cash flow route
        # 241112: we seperate on-bal and off-bal calculation due to duplicate keys in credit cards and credit lines
        # 1. ON
        
        ca_df_invalid_on = ca_df_invalid.query(
            "ON_OFF_BAL_IND =='ON' and DRAWN_BAL_OCY > 0 and STAGE_FINAL < 3")
        
        if ca_df_invalid_on.shape[0] > 0:
            repay_df_on_invalid = self.repay_df_creation_bycondition(
                ca_df_invalid_on)

            ecl_df_invalid_on, ecl_df_interim_invalid_on = (ca_df_invalid_on
                                                            .pipe(self._merge_repayment_schedule, repay_df=repay_df_on_invalid)
                                                            .pipe(self._calculate_days)
                                                            .pipe(self._adjust_prepayment, param=param)
                                                            .pipe(self.calculate_PD, param=param)
                                                            .pipe(self.calculate_LGD, param=param)
                                                            .pipe(self.calculate_discount_factor)
                                                            .pipe(self.calculate_engine_ecl, instr_df=ca_df_invalid_on))
        else:
            ecl_df_invalid_on = pd.DataFrame()
            ecl_df_interim_invalid_on = pd.DataFrame()
        
        # 2. OFF
        ca_df_invalid_off = ca_df_invalid.query(
                "ON_OFF_BAL_IND == 'OFF' and UNDRAWN_BAL_OCY > 0 and STAGE_FINAL < 3")
        
        if ca_df_invalid_off.shape[0] > 0:
            repay_df_off_invalid = self.repay_df_creation_bycondition(
                ca_df_invalid_off)

            ecl_df_invalid_off, ecl_df_interim_invalid_off = (ca_df_invalid_off
                                                            .pipe(self._merge_repayment_schedule, repay_df=repay_df_off_invalid)
                                                            .pipe(self._calculate_days)
                                                            .pipe(self._adjust_prepayment, param=param)
                                                            .pipe(self.calculate_PD, param=param)
                                                            .pipe(self.calculate_LGD, param=param)
                                                            .pipe(self.calculate_discount_factor)
                                                            .pipe(self.calculate_engine_ecl, instr_df=ca_df_invalid_off))
        else:
            ecl_df_invalid_off = pd.DataFrame()
            ecl_df_interim_invalid_off = pd.DataFrame()
                  
        ecl_df = []
        ecl_interim_df = []
        ecl_df = pd.concat([ecl_df_valid, ecl_df_invalid_on,
                           ecl_df_invalid_off]).reset_index(drop=True)
        ecl_interim_df = pd.concat(
            [ecl_interim_df_valid, ecl_df_interim_invalid_on, ecl_df_interim_invalid_off]).reset_index(drop=True)

        return ecl_df, ecl_interim_df

    def run(self,
            instr_df: pd.DataFrame,
            repay_df: pd.DataFrame,
            param: Dict[str, pd.DataFrame]
            ) -> pd.DataFrame:


        #20241227 - Carve out scope does not require cash flow
        
        # # 3. STAGE 3 ECL = EAD*LGD, no interim
        need_cf_cond = (instr_df.STAGE_FINAL < 3)
        
        ca_cf_input = instr_df[need_cf_cond]
        ca_nocf_input = instr_df[~need_cf_cond]

        # Below are stage 1&2:
        # Split the collective assessment run into 2 routes
        # One with complete cash flow data
        # The other without valid cash flow data     
        ca_df_valid, ca_df_invalid = self.filter_cashflow_valid_scope(df=ca_cf_input,
                                                                      repay_df=repay_df)
        
        # Collective assessment (Valid cash flow route)
        # EAD model
        if ca_df_valid.shape[0] > 0:
            ecl_df_valid, ecl_interim_df_valid = (ca_df_valid
                                                .pipe(self._merge_repayment_schedule,
                                                        repay_df=repay_df)
                                                .pipe(self._calculate_days_months)
                                                .pipe(self._adjust_prepayment, param=param)
                                                .pipe(self._adjust_ead_valid)
                                                .pipe(self.calculate_PD_new, param=param)
                                                .pipe(self.calculate_LGD, param=param)
                                                .pipe(self.calculate_discount_factor)
                                                .pipe(self.calculate_engine_ecl,
                                                        instr_df=ca_df_valid)
                                                )
        else:
            ecl_df_valid = pd.DataFrame()
            ecl_interim_df_valid = pd.DataFrame()

        # # Invalid cash flow route
        # # 241112: we seperate on-bal and off-bal calculation due to duplicate keys in credit cards and credit lines
        # # 1. ON
        ca_df_invalid_on = ca_df_invalid.query(
            "ON_OFF_BAL_IND =='ON' and DRAWN_BAL_OCY > 0 and STAGE_FINAL < 3")
        
        if ca_df_invalid_on.shape[0] > 0:
            repay_df_on_invalid = self.repay_df_creation_bycondition(
                ca_df_invalid_on)

            ecl_df_invalid_on, ecl_df_interim_invalid_on = (ca_df_invalid_on
                                                            .pipe(self._merge_repayment_schedule, repay_df=repay_df_on_invalid)
                                                            .pipe(self._calculate_days_months)
                                                            .pipe(self._adjust_prepayment, param=param)
                                                            .pipe(self.calculate_PD_new, param=param)
                                                            .pipe(self.calculate_LGD, param=param)
                                                            .pipe(self.calculate_discount_factor)
                                                            .pipe(self.calculate_engine_ecl, instr_df=ca_df_invalid_on))
        else:
            ecl_df_invalid_on = pd.DataFrame()
            ecl_df_interim_invalid_on = pd.DataFrame()
        
        # # 2. OFF
        ca_df_invalid_off = ca_df_invalid.query(
            "ON_OFF_BAL_IND == 'OFF' and UNDRAWN_BAL_OCY > 0 and STAGE_FINAL < 3")
        if ca_df_invalid_off.shape[0] > 0:
            repay_df_off_invalid = self.repay_df_creation_bycondition(
                ca_df_invalid_off)

            ecl_df_invalid_off, ecl_df_interim_invalid_off = (ca_df_invalid_off
                                                            .pipe(self._merge_repayment_schedule, repay_df=repay_df_off_invalid)
                                                            .pipe(self._calculate_days_months)
                                                            .pipe(self._adjust_prepayment, param=param)
                                                            .pipe(self.calculate_PD_new, param=param)
                                                            .pipe(self.calculate_LGD, param=param)
                                                            .pipe(self.calculate_discount_factor)
                                                            .pipe(self.calculate_engine_ecl, instr_df=ca_df_invalid_off))
        else:
            ecl_df_invalid_off = pd.DataFrame()
            ecl_df_interim_invalid_off = pd.DataFrame()
        
        # Below are stage 3:
        if ca_nocf_input.shape[0] > 0:
            ca_nocf_df_1 = (ca_nocf_input.assign(years = str(1))) # as LGD does not hv term structure, set 1 to all
            ca_nocf_df_2 = self.calculate_LGD(ca_nocf_df_1,param=param)
            ca_nocf_df_2.LGD = ca_nocf_df_2.LGD.fillna(1) # as there are few records out of the Month in default's boundary, their LGD are all set to 1
            ecl_nocf_df, ecl_interim_nocf_df = self._calculate_engine_stage3_ecl(ecl_df=ca_nocf_df_2,instr_df=ca_nocf_input)
        else:
            ecl_nocf_df = pd.DataFrame()
            ecl_interim_nocf_df = pd.DataFrame()

        # Aggregate as a whole ecl result table
        ecl_df = []
        ecl_interim_df = []
        ecl_df = pd.concat([ecl_df_valid, ecl_df_invalid_on,
                           ecl_df_invalid_off,
                           ecl_nocf_df]).reset_index(drop=True)

        
        ecl_interim_df = pd.concat(
            [ecl_interim_df_valid, 
             ecl_df_interim_invalid_on, 
             ecl_df_interim_invalid_off,
             ecl_interim_nocf_df])#.reset_index(drop=True) TODO 250205 to save memory
        
        return ecl_df, ecl_interim_df
        

if __name__ == '__main__':
    configPath = Path(
        r'C:\Users\WH947CH\Engagement\Khan Bank\03_ECL_engine\02_Development\khb_engine\run_config_file.json')
    c = load_configuration_file(configPath=configPath)
    rc = run_setting(run_config=c)

    ca_engine = ecl_engine(context=rc)

    result = ca_engine.run()

    print(result.keys())
