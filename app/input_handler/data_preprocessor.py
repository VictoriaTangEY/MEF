import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, List
from datetime import datetime

from pathlib import Path
import logging

from input_handler import model_segmentation

# TBD: Just for analysis purpose
from memory_profiler import profile


class data_preprocessor():
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

    def _vectorize_decorator(self, func):
        return np.vectorize(func)

    def standardize_data(self,
                         param_dict: Dict[str, pd.DataFrame],
                         inDataPath: Path,
                         rawDataName: str,
                         inputDataExt: str,
                         dtype_tbl: Dict[str, type]) -> pd.DataFrame:
        param = param_dict.copy()
        # Data Type casting and renaming
        # The tables start from 4th row
        rawKeepList = param[rawDataName].query("keep_ind == 'Y'")[
            'colname'].to_list()
        renameList = param[rawDataName].query("keep_ind == 'Y'")[
            'colname_std'].to_list()
        dtypeList = param[rawDataName].query("keep_ind == 'Y'")[
            'data_type'].to_list()
        str_list = param[rawDataName].query("data_type == 'str' and keep_ind == 'Y'")[
            'colname'].to_list()

        dtype_mapping = {}
        init_dtype_mapping = {}
        rename_mapping = {}

        for var, dtype in zip(rawKeepList, dtypeList):
            dtype_mapping[var] = dtype_tbl[dtype]

        for var in str_list:
            init_dtype_mapping[var] = str

        for rawVarName, stdVarName in zip(rawKeepList, renameList):
            rename_mapping[rawVarName] = stdVarName

        try:
            # Special treatment for no sa_fs_table
            if rawDataName == self.sa_fs_table_name:
                data_path = f'{inDataPath}/{rawDataName}.{inputDataExt}'
                if not os.path.exists(data_path):
                    return pd.DataFrame(), False
                else:
                    df = pd.read_csv(data_path,
                        usecols=rawKeepList,
                        dtype=init_dtype_mapping,
                        encoding='utf-8-sig'
                    )
                    return df, True
            else:
                df = (pd.read_csv(f'{inDataPath}/{rawDataName}.{inputDataExt}',  # file name contains 'instrument'
                                usecols=rawKeepList,
                                dtype=init_dtype_mapping,
                              encoding='utf-8-sig')
                    )

        except FileNotFoundError:
            print(
                f"File not found. Please check if the input file name, location and file extension is correct.")
            return (pd.DataFrame(), True)

        except PermissionError:
            print("Please check if the file is being opened by another users.")
            return (pd.DataFrame(), True)

        except Exception as e:
            print(e)
            return (pd.DataFrame(), True)

        # Type Casting
        for key, val in dtype_mapping.items():
            try:
                if val == 'datetime64[ns]':
                    df[key] = pd.to_datetime(df[key], errors='coerce')
                elif val == 'float':
                    df[key] = pd.to_numeric(df[key], errors='coerce')
                else:
                    df[key] = df[key].astype(val)
            except Exception as e:
                print(f'Report error on data casting for {key} to type {val}.')
                print(e)
                return (pd.DataFrame(), True)

        return (df.rename(columns=rename_mapping), False)

    def load_input_data(self,
                        param: Dict[str, pd.DataFrame]
                        ) -> pd.DataFrame:
        """
        ### load_input_data

        A centralized funciton to load all necessary input data
        for ECL calculation.

        ### Parameters
        - param (Dict): A dictionary of parameters load from load_parameter function

        ### Return
        A tuple of raw input data \n
        0: instrument table \n
        1: exchange rate table \n
        2: repayment table \n
        """
        instr_df, is_error = self.standardize_data(param_dict=param,
                                                   inDataPath=self.inDataPath,
                                                   rawDataName=self.instrument_table_name,
                                                   inputDataExt=self.inputDataExtECL,
                                                   dtype_tbl=self.dtype_tbl,
                                                   )

        fx_df, is_error = self.standardize_data(param_dict=param,
                                                inDataPath=self.inDataPath,
                                                rawDataName=self.exchange_rate_table_name,
                                                inputDataExt=self.inputDataExtECL,
                                                dtype_tbl=self.dtype_tbl)

        repay_df, is_error = self.standardize_data(param_dict=param,
                                                   inDataPath=self.inDataPath,
                                                   rawDataName=self.repayment_table_name,
                                                   inputDataExt=self.inputDataExtECL,
                                                   dtype_tbl=self.dtype_tbl)
        return (instr_df, fx_df, repay_df)

    def _get_facility_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        To perform facility level stage allocation based on the logic
        specified in the model documentaiton

        The input data must have the following columns
        - LOAN_CLASS_GRD
        - PAST_DUE_DAYS
        - CREDIT_RATING_CURRENT
        - DEFT_IND
        - INSTR_TYPE
        - WATCHLIST_IND
        - RESTRUCTURED_IND
        - CUST_TYPE

        '''
        df_ = df.copy()

        loan_class_default_grade_list = [
            'SUBSTANDART',
            'DOUBTFUL',
            'LOSS'
        ]

        cond = [
            ((df.LOAN_CLASS_GRD.str.upper().str.strip().isin(loan_class_default_grade_list)) |
             (df.PAST_DUE_DAYS > 90) |  # 250224: update to >90 DPD
             (df.CREDIT_RATING_CURRENT.str.upper().str.strip().isin(['D'])) |
             (df.DEFT_IND)
             ),

            ((df.LOAN_CLASS_GRD.str.upper().str.strip().isin(['SPECIAL MENTION'])) |
             # 250224: update to >90 DPD
             # 250324 YJ: fix rules
             # 20250704: convert type
             (df.CUST_TYPE.astype(str).str.upper().str.strip().isin(['1'])) & (df.PAST_DUE_DAYS > 15) |
             (df.CUST_TYPE.astype(str).str.upper().str.strip().isin(['2'])) & (df.PAST_DUE_DAYS > 30) |
             #  20241229: Updated notch difference allocation
             # - Current in speculative grade
             # - Drop > 3 notches compare with initial rating
             ((df.DATA_SOURCE_CD.str.upper().str.strip().isin(['NON_LOAN'])) &
              (df.CREDIT_RATING_NOTCH >= 11) &
              (df.NOTCH_DIFF_ACT > df.NOTCH_DIFF_THRESHOLD)) |
             (df.WATCHLIST_IND) |
             (df.RESTRUCTURED_IND)
             )
        ]

        choices = [3, 2]

        df_1 = (df_.assign(
            STAGE_FAC=np.select(cond, choices, default=1)
        ))

        return df_1

    def _get_customer_stage_old(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        To perform stage consistency on the instrument table

        The input data must have the following columns
        - CUST_ID: Customer ID
        - STAGE_FAC: Facility level stage

        '''

        df_ = df.copy()

        # Stage consistency aggregation process
        # 20241023: All customers are subject to state consistency.
        # 20250313 YJ: The highest stage of ON-bal records are taken for stage consistency.Ignore OFF-bal.
        agg_stage = (df_[['CUST_ID', 'STAGE_FAC']]
                     .groupby(by=['CUST_ID'])
                     .agg(
                         STAGE_CUST=pd.NamedAgg('STAGE_FAC', 'max'),
        )
            .reset_index()
            .query("CUST_ID != '' and CUST_ID.notnull() and CUST_ID != 'nan'", engine='python')
        )

        df_1 = df_.merge(right=agg_stage,
                         how='left',
                         left_on=['CUST_ID'],
                         right_on=['CUST_ID'])

        return df_1

    # TODO: remove later. as bank didnt add restructured date into instr data, interimly load and merge in engine
    def _load_interim_restru_date(self,
                                  inDataPath: Path,
                                  rawDataName: str,
                                  inputDataExt: str):
        try:
            df = (pd.read_csv(f'{inDataPath}/{rawDataName}.{inputDataExt}',
                              dtype={'CONTRACT_ID': str},
                              encoding='utf-8-sig'))
        except FileNotFoundError:
            print(
                f"File not found. Please check if the input file name, location and file extension is correct.")
            return (pd.DataFrame(), True)
        except PermissionError:
            print("Please check if the file is being opened by another users.")
            return (pd.DataFrame(), True)

        except Exception as e:
            print(e)
            return (pd.DataFrame(), True)

        # Type casting
        try:
            # df['CONTRACT_ID'] = df['CONTRACT_ID'].astype(str)
            df['CUST_ID'] = df['CUST_ID'].astype(str)
            df['RESTRUCTURED_IND'] = df['RESTRUCTURED_IND'].astype(bool)
            # df['RESTRUCT_DATE'] = pd.to_datetime(df['RESTRUCT_DATE'], errors='coerce')
            df['RESTRUCT_DATE'] = pd.to_datetime(
                df['RESTRUCT_DATE'], format='%d/%m/%Y', errors='coerce')
        except Exception as e:
            print(f'Report error on data casting .')
            print(e)
            return (pd.DataFrame(), True)
        return df
    # 250331: update the stage consistency logic

    def _get_customer_stage_wrong(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Exclude off-bal, zero ead, sec by deposit, and fvtpl
        df_ = df
        ~((df.ON_OFF_BAL_IND.str.upper() == 'OFF') |
            (df.DRAWN_BAL_OCY <= 0) |
            (df.SUB_CATEGORY.str.upper() == 'SECURED_BY_DEPOSIT') |
            (df.IFRS9_MEAS_TYPE.str.upper() == 'FVTPL'))

        # 2. Identify and exclude duplicates
        duplicates = df_.duplicated(subset='CONTRACT_ID', keep=False)
        df_ = df_[~duplicates]  # Exclude duplicates from df_

        # 3. Apply KB designed rules
        agg_stage_0 = df_[['CONTRACT_ID', 'ON_OFF_BAL_IND','CUST_ID', 'INIT_DATE', 'STAGE_FAC',
                            'DEFT_IND', 'RESTRUCTURED_IND', 'RESTRUCTURED_DATE']].copy()

        for cust_id, group in agg_stage_0.groupby(by=['CUST_ID']):
            mask_1 = (group.DEFT_IND.astype(str).str.upper() == 'TRUE')
            mask_2 = (group.RESTRUCTURED_IND.astype(str).str.upper() == 'TRUE')

            if mask_2.any():
                restr_row = group[mask_2]
                other_rows_2 = group[~mask_2]
                
                other_rows_2['STAGE_CUST'] = np.where(
                    (other_rows_2['STAGE_FAC'] == 1) &
                    (other_rows_2['INIT_DATE'] < max(restr_row['RESTRUCTURED_DATE'].values)),
                    2,
                    other_rows_2['STAGE_FAC']
                )
                agg_stage_0.loc[other_rows_2.index, 'STAGE_CUST'] = other_rows_2['STAGE_CUST']

            elif mask_1.any():
                deft_row = group[mask_1]
                other_rows = group[~mask_1]

                other_rows['STAGE_CUST'] = other_rows['STAGE_FAC'].apply(lambda x: min(x + 1, 3))
                agg_stage_0.loc[other_rows.index, 'STAGE_CUST'] = other_rows['STAGE_CUST']

        agg_stage_0['STAGE_CUST'] = agg_stage_0['STAGE_CUST'].fillna(agg_stage_0['STAGE_FAC'])
        agg_stage_0 = agg_stage_0.reset_index().query(
            "CUST_ID != '' and CUST_ID.notnull() and CUST_ID != 'nan'", engine='python'
        )
        # Merge with agg_stage_0 first
        merged_df = pd.merge(
            df_,
            agg_stage_0[['CONTRACT_ID', 'STAGE_CUST']],
            on='CONTRACT_ID',
            how='left'
        )

        excluded_rows = df[~df.index.isin(df_.index)]

        #Assign STAGE_CUST of excluded rows as STAGE_FAC
        excluded_rows['STAGE_CUST'] = excluded_rows['STAGE_FAC']

        #Combine both DataFrames
        df_1 = pd.concat([merged_df, excluded_rows], ignore_index=True)

        # Ensure the output DataFrame has same rows as the input
        if len(df_1) != len(df):
            raise ValueError("Output DataFrame does not have the same length as the input DataFrame.")

        return df_1

    def _get_customer_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Exclude off-bal, zero ead, sec by deposit, and fvtpl
        df_ = df[
        ~((df.ON_OFF_BAL_IND.str.upper() == 'OFF') |
            (df.DRAWN_BAL_OCY <= 0) |
            (df.SUB_CATEGORY.str.upper() == 'SECURED_BY_DEPOSIT') |
            (df.IFRS9_MEAS_TYPE.str.upper() == 'FVTPL'))]

        # 2. Identify and exclude duplicates
        duplicates = df_.duplicated(subset='CONTRACT_ID', keep=False)
        df_ = df_[~duplicates]  # Exclude duplicates from df_

        # 3. Extract useful columns
        agg_stage_0 = df_[['CONTRACT_ID', 'ON_OFF_BAL_IND','CUST_ID','LOAN_CLASS_GRD', 
                   'INIT_DATE','RESTRUCTURED_IND', 'RESTRUCTURED_DATE', 'STAGE_FAC']].copy()
        # 4. assign grd number based on loan_class grd
        # Mapping of loan class grades to numbers
        loan_class_map = {
            'Performing': 1,
            'Special mention': 2,
            'Doubtful': 3,
            'Substandart': 4,
            'Loss': 5
        }
        # Adding a new column for numerical values
        agg_stage_0['LOAN_CLASS_NUM'] = agg_stage_0['LOAN_CLASS_GRD'].replace(loan_class_map).astype(int)

        # 5. Perform default rule first     
        # Calculate WORST_GRD for each customer
        agg_stage_0['worst_grd'] = agg_stage_0.groupby('CUST_ID')['LOAN_CLASS_NUM'].transform('max')

        # Determine CROSS_STAGE
        agg_stage_0['CROSS_STAGE'] = np.where(
            agg_stage_0['worst_grd'] == 1,
            agg_stage_0['STAGE_FAC'],
            np.where(
                agg_stage_0['worst_grd'] == 2,
                2,
                np.minimum(3, agg_stage_0['STAGE_FAC'] + 1)
            )
        )
        # 6. Perform restrucutre rule
        # Determine REST_STAGE # updated on 250612
        # Create REST_IND_CROSS column that inherits the customer-level restructuring status
        agg_stage_0['REST_IND_CROSS'] = agg_stage_0.groupby('CUST_ID')['RESTRUCTURED_IND'].transform(
            lambda x: x.astype(str).str.upper().eq('TRUE').any()
        )
        # Get customer-level latest restructure date (without modifying original column)
        agg_stage_0['CUST_LATEST_REST_DATE'] = agg_stage_0.groupby('CUST_ID')['RESTRUCTURED_DATE'].transform('max')
        
        # Calculate REST_STAGE using customer-level rules
        agg_stage_0['REST_STAGE'] = np.where(
            (agg_stage_0['REST_IND_CROSS'].astype(str).str.upper() == 'TRUE') &
            (agg_stage_0['CUST_LATEST_REST_DATE'] >= agg_stage_0['INIT_DATE']),
            2,
            0
        )
        # agg_stage_0['REST_STAGE'] = 0
        # mask_2 = (
        #     (agg_stage_0['RESTRUCTURED_IND'].astype(str).str.upper() == 'TRUE') &
        #     (agg_stage_0['RESTRUCTURED_DATE'] >= agg_stage_0['INIT_DATE'])
        # )

        # agg_stage_0.loc[mask_2, 'REST_STAGE'] = 2

        # 7. Max stage as STAGE_CUST
        agg_stage_0['STAGE_CUST'] = agg_stage_0[['STAGE_FAC', 'CROSS_STAGE', 'REST_STAGE']].max(axis=1)
        # Merge with agg_stage_0 first
        merged_df = pd.merge(
            df_,
            agg_stage_0[['CONTRACT_ID', 'STAGE_CUST']],
            on='CONTRACT_ID',
            how='left'
        )

        #8. Assign STAGE_CUST of excluded rows as STAGE_FAC
        excluded_rows = df[~df.index.isin(df_.index)]
        excluded_rows['STAGE_CUST'] = excluded_rows['STAGE_FAC']

        #9. Combine both DataFrames
        df_1 = pd.concat([merged_df, excluded_rows], ignore_index=True)

        # Ensure the output DataFrame has same rows as the input
        if len(df_1) != len(df):
            raise ValueError("Output DataFrame does not have the same length as the input DataFrame.")

        return df_1
    
    def _get_overlaid_stage(self, df: pd.DataFrame,
                            param: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        '''
        To perform stage overlaid on the instrument table

        '''
        keep_cols = ['CONTRACT_ID', 'STAGE_OVR']

        mo_param = (param['Overlay'][keep_cols])

        df_1 = df.merge(mo_param, how='left',
                        left_on=['CONTRACT_ID'],
                        right_on=['CONTRACT_ID'],
                        )

        df_2 = df_1.assign(
            STAGE_OVR=df_1.STAGE_OVR.fillna(0).astype(int)
        )

        return df_2

    def _get_final_stage(self, df: pd.DataFrame) -> pd.DataFrame:

        cond = [
            df.STAGE_OVR != 0,
        ]

        choices = [
            df.STAGE_OVR,
        ]

        if self.mute_stage_consistency == "Y":
            df_ = df.assign(
                STAGE_FINAL=np.select(cond, choices, default=df.STAGE_FAC)
            )
        else:
            df_ = df.assign(
                STAGE_FINAL=np.select(cond, choices, default=df.STAGE_CUST)
            )

        return df_

    def allocate_stage(self,
                       df: pd.DataFrame,
                       param: Dict[str, pd.DataFrame],
                       main_logger=None) -> pd.DataFrame:

        if main_logger == None:
            main_logger = logging.getLogger(__name__)

        try:
            main_logger.info("Performing stage allocation...")
            df_ = (df.pipe(self._get_facility_stage)
                   .pipe(self._get_customer_stage)
                   .pipe(self._get_overlaid_stage, param=param)
                   .pipe(self._get_final_stage)
                   )
        except Exception as e:
            main_logger.exception(f"Stage allocation error. Error message {e}")

        else:
            main_logger.info("Stage allocation successful.")
            return df_

    def allocate_ecl_approach(self,
                              df: pd.DataFrame,
                              param: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # ECL approach assignment
        # TODO: To automate in the parameter files - updated 20241223
        proxy_portfolio_list = param['Proxy_mapping']['PRTFLO_ID'].to_list()

        # 20241203: Calculate cust level columns
        cust_os = (df[['CUST_ID', 'PRIN_BAL_LCY']]
                   .groupby(by=['CUST_ID'])
                   .agg(
                       CUST_PRIN_BAL_LCY=pd.NamedAgg('PRIN_BAL_LCY', 'sum'),
        )
            .reset_index()
            .query("CUST_ID != '' and CUST_ID.notnull() and CUST_ID != 'nan'", engine='python')
        )

        df = df.merge(right=cust_os,
                      how='left',
                      left_on=['CUST_ID'],
                      right_on=['CUST_ID'])

        # cond = [
        #     # ECL Proxy approach logic
        #     df['PRTFLO_ID'].isin(proxy_portfolio_list),

        #     # Specific assessment approach: On-balalance quantitative logic
        #     (df['ON_OFF_BAL_IND'].isin(['ON']) &
        #      df['STAGE_FINAL'].isin([2, 3]) &
        #      (df['CUST_PRIN_BAL_LCY'] >= 2_000_000_000) &
        #      (df['PAST_DUE_DAYS'] >= 30)),

        #     # Specific assessment approach: On-balane qualitative logic
        #     (df['ON_OFF_BAL_IND'].isin(['ON']) &
        #      (df['WATCHLIST_SA'])) | (df['SA_IND']),        
        # ]

        # choices = [
        #     'PROXY',
        #     'SPECIFIC_ASSESSMENT',
        #     'SPECIFIC_ASSESSMENT',
        # ]

        # df_1 = df.assign(
        #     ECL_APPROACH=np.select(
        #         cond, choices, default='COLLECTIVE_ASSESSMENT'),
        # )
        
        # 1. Identify SA customers (customer level logic - unchanged)
        # Specific assessment approach: On-balalance quantitative logic
        quant_sa_customers = df[
            (df['ON_OFF_BAL_IND'].isin(['ON'])) & 
            (df['CUST_PRIN_BAL_LCY'] >= 2_000_000_000) & 
            (df['PAST_DUE_DAYS'] >= 30)
        ]['CUST_ID'].unique()
        
        # Specific assessment approach: On-balane qualitative logic
        qual_sa_customers = df[
            ((df['ON_OFF_BAL_IND'].isin(['ON']) & 
              df['WATCHLIST_SA']) | df['SA_IND'])
        ]['CUST_ID'].unique()

        # 2. Create customer level mapping for SA
        sa_customers = np.union1d(quant_sa_customers, qual_sa_customers)
        customer_sa_map = {cust: 'SPECIFIC_ASSESSMENT' for cust in sa_customers}

        # 3. Apply SA mapping to all contracts of SA customers
        df_1 = df.assign(
            ECL_APPROACH=df['CUST_ID'].map(customer_sa_map).fillna('')
        )

        # 4. Apply proxy logic at contract level (only for non-SA contracts)
        proxy_mask = (
            (df_1['ECL_APPROACH'] == '') &  # Only for contracts not already assigned SA
            (df_1['PRTFLO_ID'].isin(proxy_portfolio_list))
        )
        
        # 5. Create final ECL_APPROACH assignment
        conditions = [
            df_1['ECL_APPROACH'] == 'SPECIFIC_ASSESSMENT',
            proxy_mask
        ]
        choices = [
            'SPECIFIC_ASSESSMENT',
            'PROXY'
        ]
        
        df_1 = df_1.assign(
            ECL_APPROACH=np.select(conditions, choices, default='COLLECTIVE_ASSESSMENT')
        )
        
        return df_1

    def split_on_off_data(self,
                          df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        df_on_loan = (df.query('''
                            ON_OFF_BAL_IND.str.upper() == 'ON' \
                            and DATA_SOURCE_CD.str.upper() == 'LOAN'
                            '''))

        df_on_nloan = (df.query('''
                            ON_OFF_BAL_IND.str.upper() == 'ON' \
                            and DATA_SOURCE_CD.str.upper() == 'NON_LOAN'
                            '''))

        df_off = df.query("ON_OFF_BAL_IND.str.upper() == 'OFF'")

        return (df_on_loan, df_on_nloan, df_off)

    # def _map_credit_notch(self,
    #                       df: pd.DataFrame,
    #                       param: Dict[str, pd.DataFrame]) -> pd.DataFrame:

    #     df_1 = df.assign(
    #         CREDIT_RATING_CURRENT=df.CREDIT_RATING_CURRENT.str.upper(),
    #         CREDIT_RATING_SOURCE=df.CREDIT_RATING_SOURCE.str.upper(),
    #     )

    #     df_2 = (df_1.merge(
    #         param['Notch_mapping'].drop(['Parameter name'], axis=1), how='left',
    #         left_on=['CREDIT_RATING_CURRENT'],
    #         right_on=['CREDIT_RATING']
    #     )
    #     )

    #     df_3 = (df_2.assign(
    #         CREDIT_RATING_NOTCH=df_2.CREDIT_RATING_NOTCH.fillna(0)
    #     ))

    #     return df_3

    # 20241229: Expand the scope to initial rating too for notch difference
    def _map_credit_notch(self,
                          df: pd.DataFrame,
                          param: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        df_1 = df.assign(
            CREDIT_RATING_CURRENT=df.CREDIT_RATING_CURRENT.str.upper().str.strip(),
            CREDIT_RATING_SOURCE=df.CREDIT_RATING_SOURCE.str.upper().str.strip(),
        )

        df_2 = (df_1.merge(
            param['Notch_mapping'].drop(['Parameter name',
                                         'NOTCH_DIFF_THRESHOLD'], axis=1), how='left',
            left_on=['CREDIT_RATING_CURRENT'],
            right_on=['CREDIT_RATING']
        )
            .merge(
            param['Notch_mapping'].drop(['Parameter name'], axis=1), how='left',
            left_on=['CREDIT_RATING_INITIAL'],
            right_on=['CREDIT_RATING'],
            suffixes=('', '_INIT')
        )
            .drop(columns=[
                'CREDIT_RATING',
                'CREDIT_RATING_INIT',
            ])
        )

        df_3 = (df_2.assign(
            CREDIT_RATING_NOTCH=df_2.CREDIT_RATING_NOTCH.fillna(0),
            CREDIT_RATING_NOTCH_INIT=df_2.CREDIT_RATING_NOTCH_INIT.fillna(0),
        ))

        df_4 = (df_3.assign(
            NOTCH_DIFF_ACT=(df_3.CREDIT_RATING_NOTCH
                            - df_3.CREDIT_RATING_NOTCH_INIT).fillna(0),
            NOTCH_DIFF_THRESHOLD=df_3.NOTCH_DIFF_THRESHOLD.fillna(0),
        ))

        return df_4

    def _adjust_lifetime(self, df: pd.DataFrame,
                         param: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        # TODO: Need to add some exceptional handling functions
        """
        To generate the adjusted maturity date to consider the following
        - Revolving lifetime
        - Missing maturity date
        - Expired accounts
        """

        df_ = (df.merge(
            param['Lifetime_parameter'], how='left',
            left_on=['LIFETIME_POOL_ID'],
            right_on=['LIFETIME_ID']
        )
            .drop(labels=['Parameter name',
                          'LIFETIME_ID',
                          'DESCRIPTION',], axis=1)
        )

        @self._vectorize_decorator
        def _adjust_mat_date(original_mat_date: np.datetime64,
                             report_date: np.datetime64,
                             lifetime_val: float,
                             stage: float) -> np.datetime64:

            report_date_plus_365d = report_date + pd.DateOffset(months=12)
            # report_date_plus_365d = np.datetime64(report_date
            #                                       + np.timedelta64(365, 'D'))
            original_mat_date_cast = np.datetime64(original_mat_date)

            # Exception handling: Possible missing maturity date
            if np.isnan(original_mat_date_cast):
                return report_date_plus_365d

            elif not np.isnan(lifetime_val):
                # return np.datetime64(report_date
                #                      + np.timedelta64(int(lifetime_val), 'D'))
                # 241126 update YJ: to keep mature date the end day of a month
                return (report_date + pd.DateOffset(days=int(lifetime_val)) + pd.offsets.MonthEnd(0))

            # Exception handling: Expired accounts
            elif original_mat_date_cast <= report_date:
                if stage == 1:
                    return report_date + pd.offsets.MonthEnd(1)

                    # return np.datetime64(report_date
                    #                      + np.timedelta64(30, 'D'))
                else:
                    return report_date_plus_365d
            # Handle account with valid maturity date
            # Stage 2 flooring
            else:
                if stage >= 2:
                    return max(original_mat_date_cast, report_date_plus_365d)
                else:
                    return original_mat_date_cast

        df_1 = (df_.assign(
            MAT_DATE_ADJ=_adjust_mat_date(original_mat_date=df_.MAT_DATE,
                                          report_date=df_.REPORT_DATE,
                                          lifetime_val=df_.LIFETIME_VAL,
                                          stage=df_.STAGE_FINAL)
        ))

        return df_1

    def _calculate_ead(self,
                       instr_df: pd.DataFrame,
                       param: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        @self._vectorize_decorator
        def define_ead(data_source_cd: str,
                       on_off_bal_ind: str,
                       ead_pool_cd: str,
                       drawn_bal: float,
                       undrawn_bal: float,
                       credit_lmt: float,
                       amort: float,
                       ccf: float) -> float:
            if data_source_cd.upper() == 'NON_LOAN':
                return (drawn_bal) 
            #250626 remove amort; confirmed with Undrakh, the drawn_bal for non-loan already included amort
            else:
                if on_off_bal_ind.upper() == 'OFF':
                    if ead_pool_cd.startswith('ULF'):
                        return (undrawn_bal*ccf)
                    elif ead_pool_cd.startswith('LMF'):
                        return max(credit_lmt*ccf
                                   - (credit_lmt-undrawn_bal), 0)
                else:
                    return (drawn_bal)

        ccf_param = (param['EAD_parameter']
                     .filter(items=['EAD_POOL_ID',
                                    'CCF_VAL']))
        df_ = (instr_df.merge(
            ccf_param, how='left',
            left_on=['EAD_POOL_ID'],
            right_on=['EAD_POOL_ID']
        ))

        df_ = (df_.assign(
            EAD_OCY=define_ead(data_source_cd=df_.DATA_SOURCE_CD,
                               on_off_bal_ind=df_.ON_OFF_BAL_IND,
                               ead_pool_cd=df_.EAD_POOL_ID,
                               drawn_bal=df_.DRAWN_BAL_OCY,
                               undrawn_bal=df_.UNDRAWN_BAL_OCY,
                               credit_lmt=df_.CREDIT_LIMIT_OCY,
                               amort=df_.AMORTIZATION_OCY,
                               ccf=df_.CCF_VAL,),
        ))

        return df_

    def _mute_eir(self, instr_df: pd.DataFrame) -> pd.DataFrame:
        if self.mute_eir == "Y":
            instr_df['EFF_INT_RT'] = 0

        return instr_df

    # @profile
    def run(self,
            instr_df_raw: pd.DataFrame,
            run_scope: List[str],
            param: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        The master process to run the data preprocessing functions
        The output would be a tuple with index representing the following
        files:

        0: Full instrument table \n
        1: Table for collective assessment \n
        2: Table for specific assessment \n
        3: Table for proxy assessment \n
        4: Out of scope table \n
        """
        # #TODO: remove later. as bank didnt add restructured date into instr data, interimly load and merge in engine
        # rest_date_df = self._load_interim_restru_date(inDataPath=self.inDataPath,
        #                             rawDataName='Interim_instrument_table_Rest date_20240630',
        #                             inputDataExt=self.inputDataExtECL
        #                                     )
        # # merge with instr table
        # instr_df_merged = pd.merge(instr_df_raw, rest_date_df[['CONTRACT_ID', 'RESTRUCT_DATE']], on='CONTRACT_ID', how='left')

        # #250127 YJ: as the drawn bal is not exact in 240630 instr table, override Drawn bal by its components
        # instr_df = (instr_df_merged.assign(
        #     DRAWN_BAL_OCY = instr_df_merged.PRIN_BAL_OCY
        #                     +instr_df_merged.ACRU_INT_OCY
        #                     +instr_df_merged.PENALTY_OCY
        #                     +instr_df_merged.OTHER_FEE_AND_CHARGES_OCY))
        
        #20250704: to update for Loan only
        instr_df = instr_df_raw.copy()
        loan_mask = instr_df["DATA_SOURCE_CD"] == "LOAN"

        instr_df.loc[loan_mask, "DRAWN_BAL_OCY"] = (
            instr_df.loc[loan_mask, "PRIN_BAL_OCY"]
            + instr_df.loc[loan_mask, "ACRU_INT_OCY"]
            + instr_df.loc[loan_mask, "PENALTY_OCY"]
            + instr_df.loc[loan_mask, "OTHER_FEE_AND_CHARGES_OCY"]
                                                                )
        # 20241227: Update the excluded rules
        # Exclude MIK but as fallback, also exclude those
        # On balance with 0 drawn balance oct or off balance with 0 undrawn ocy
        out_scope_logic = ("SUB_CATEGORY.str.upper()=='MIK'"
                           + "or (ON_OFF_BAL_IND == 'ON' and DRAWN_BAL_OCY == 0)"
                           + "or (ON_OFF_BAL_IND == 'OFF' and UNDRAWN_BAL_OCY == 0)"
                           )

        out_scope_df_ = instr_df.query(out_scope_logic)

        # Filter out required run scope by users
        instr_df_filter = (instr_df
                           .query("DATA_SOURCE_CD.str.upper().isin(@run_scope)"))

        # 20241227: Fix exclusion bugs owing to duplicated contract ID by on / off bal ind
        instr_df_filter_1 = (instr_df_filter.merge(
            out_scope_df_[['CONTRACT_ID', 'ON_OFF_BAL_IND']],
            how='left',
            left_on=['CONTRACT_ID', 'ON_OFF_BAL_IND'],
            right_on=['CONTRACT_ID', 'ON_OFF_BAL_IND'],
            indicator=True,
        ))

        instr_df_in_scope = (instr_df_filter_1.query(
            "_merge == 'left_only'"
        )
            .drop(columns=['_merge'])
        )

        # Map credit notch number
        instr_df_1 = self._map_credit_notch(df=instr_df_in_scope, param=param)

        # Perform Stage allocation
        instr_df_2 = self.allocate_stage(df=instr_df_1,
                                         param=param)

        # Splitting instrument table by various ECL calculation approach
        instr_df_3 = self.allocate_ecl_approach(df=instr_df_2, param=param)

        # Remove interim tables to free up memory
        del instr_df_filter, instr_df_filter_1, instr_df_in_scope,
        instr_df_1, instr_df_2

        # Assign model IDs
        ms = model_segmentation.model_segmentation()
        instr_df_4 = ms.get_all_model_ID(df=instr_df_3, param=param)
        instr_df_5 = self._adjust_lifetime(df=instr_df_4, param=param)
        instr_df_6 = self._calculate_ead(instr_df=instr_df_5, param=param)

        # For testing purpose
        instr_df_7 = self._mute_eir(instr_df=instr_df_6)

        # TODO: Need to merge back to final ECL data at the end
        ca_df_ = instr_df_7.query("ECL_APPROACH == 'COLLECTIVE_ASSESSMENT'")
        sa_df_ = instr_df_7.query("ECL_APPROACH == 'SPECIFIC_ASSESSMENT'")
        px_df_ = instr_df_7.query("ECL_APPROACH == 'PROXY'")

        return instr_df_7, ca_df_, sa_df_, px_df_, out_scope_df_

    # 20241227 - For checking purpose (no use for ECL calculation)
    def _check_in_out_count(self,
                            input_df: pd.DataFrame,
                            output_df_set: Dict) -> int:
        rec_cnt = 0
        for idx in range(1, len(output_df_set)):
            rec_cnt = rec_cnt + output_df_set[idx].shape[0]

        error_msg = '''
        Input record count is not match with output record
        '''
        assert input_df.shape[0] == rec_cnt, error_msg

        return rec_cnt

    def convert_to_last_day_of_quarter(self, quarter_str):
        QUARTER_TO_MONTHS = {
            "Q1": (3, 31),  # 3/31
            "Q2": (6, 30),  # 6/30
            "Q3": (9, 30),  # 9/30
            "Q4": (12, 31)  # 12/31
        }

        year = int(quarter_str[:4])
        quarter = quarter_str[4:]
        month, day = QUARTER_TO_MONTHS[quarter]
        return datetime(year, month, day).strftime('%Y/%m/%d')

    def convert_data(self, item):
        return 0 if item == 'ND' else float(item)

    def deal_sheet(self, df):
        for i in range(len(df)):
            if df.iloc[i]['Code:'] == 'Scenario Version:':
                fixed_end = i + 1
                break

        try:
            d = df.iloc[fixed_end:, :]
        except Exception as e:
            print("Issue observed: no variable count number is Scenario Version:")
            print(f"Error as {e}")

        # change date type
        try:
            d['Code:'] = d['Code:'].apply(
                self.convert_to_last_day_of_quarter)
        except Exception as e:
            print("Issue observed in change date type")
            print(f"Error as {e}")

        # change data type
        try:
            d.loc[:, d.columns[1:]] = d.loc[:,
                                            d.columns[1:]].applymap(self.convert_data)
        except Exception as e:
            print("Issue observed in change data type")
            print(f"Error as {e}")

        return d

    def load_scenario_data(self, data_path, file_pattern):
        if file_pattern == 'MEF':
            for _wbFile in data_path.glob('[!~]*RawData.xlsx'):
                df = pd.read_excel(_wbFile, sheet_name='Data')
                try:
                    data = self.deal_sheet(df)
                except Exception as e:
                    print(f"Error as {e}")

        if file_pattern == 'LGD':
            for _wbFile in data_path.glob('*lgd*.csv'):
                try:
                    data = pd.read_csv(_wbFile)
                except Exception as e:
                    print(f"Error as {e}")

        if file_pattern == 'PD':
            for _wbFile in data_path.glob('*pdd_gg_zad*.csv'):
                try:
                    # temp adj, delete the "skiprows=3" after 202412 testing
                    # data = pd.read_csv(_wbFile, skiprows=3)
                    data = pd.read_csv(_wbFile)
                except Exception as e:
                    print(f"Error as {e}")

        if file_pattern == 'NPL':
            for _wbFile in data_path.glob('NPL*.xlsx'):
                df = pd.read_excel(_wbFile, sheet_name='NPL_data')
                try:
                    data = self.deal_sheet(df)
                except Exception as e:
                    # print(f"Error as {e}")
                    print(f"Error in deal_sheet for file {_wbFile}: {e}")
        return data
