import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import csv

from util.loggers import createLogHandler
from input_handler.load_parameters import load_configuration_file, load_parameters
from input_handler.env_setting import run_setting
from input_handler.data_preprocessor import data_preprocessor
from ecl_engine import output_file_handler as ofh


class ProxyModel():
    def __init__(self, context):
        self.parmPath = context.parmPath
        self.resultPath = context.resultPath
        self.parameters = load_parameters(self.parmPath)
        self.ECLparam = self.parameters['ProxyECL']
        self.Portfparam = self.parameters['Proxy_mapping']

    def _vectorize_decorator(self, func):
        return np.vectorize(func)

    def get_proxy_ecl_pct_old(self, ecl_result_df, param):

        @self._vectorize_decorator
        def get_proxy_key(LGD_POOL_ID):
            if str(LGD_POOL_ID).startswith('COL12'):
                return 'KB92'  # CORP - KB92
            elif str(LGD_POOL_ID).startswith('COL22'):
                return 'KB91'  # SME - KB91
            else:
                return 'Out of scope'

        ecl_result_df['PROXY_PRTFLO_ID'] = get_proxy_key(
            ecl_result_df['LGD_POOL_ID'])

        # get PROXY_PRTFLO_ID (e.g., KB91) and PROXY_ECL_KEY (e.g., KB91-1)
        ecl_result_df = ecl_result_df[ecl_result_df['PROXY_PRTFLO_ID']
                                      != 'Out of scope']
        ecl_result_df['PROXY_ECL_KEY'] = ecl_result_df['PROXY_PRTFLO_ID'].astype(
            str) + "-" + ecl_result_df['STAGE_FINAL'].astype(int).astype(str)

        ################### original ##################################
        # # calculate the avg ECL percentage for KB91 and KB92
        # ecl_result_df['PROXY_ECL_PCT_COL'] = ecl_result_df['ECL_FINAL_LCY'] / \
        #     ecl_result_df['EAD_LCY']

        # # get the calculated proxy ecl percentge from the ECL result
        # px_ecl_pct_col = ecl_result_df.groupby(
        #     'PROXY_ECL_KEY')['PROXY_ECL_PCT_COL'].mean().reset_index()
        # px_ecl_pct_col
        ##################  changed  ##################################
        px_ecl_pct_col = ecl_result_df.groupby('PROXY_ECL_KEY').apply(
            lambda x: x['ECL_FINAL_LCY'].sum() / x['EAD_LCY'].sum()).reset_index(name='PROXY_ECL_PCT_COL')
        ###############################################################

        # get the assigned proxy ecl percentatge from the param table "ProxyECL"
        px_ecl_pct_assign = param['ProxyECL']
        px_ecl_pct_assign['PROXY_ECL_KEY'] = px_ecl_pct_assign['PORTFOLIO_CODE'] + \
            "-" + px_ecl_pct_assign['STAGE_FINAL'].astype(int).astype(str)


        px_ecl_pct = pd.merge(
            px_ecl_pct_col, px_ecl_pct_assign, on='PROXY_ECL_KEY', how='right')
        px_ecl_pct['PROXY_ECL_PCT'] = np.where(
            px_ecl_pct['PROXY_ECL_PCT_COL'] <= 0, px_ecl_pct['PROXY_ECL_PCT_COL'], px_ecl_pct['PROXY_ECL_PCT'])
        px_ecl_pct = px_ecl_pct[['PROXY_ECL_KEY', 'PROXY_ECL_PCT']]
        # px_ecl_pct.to_csv(self.resultPath/'proxy_output.csv', index=False, encoding='utf-8')

        return px_ecl_pct
    
    def get_proxy_ecl_pct(self, param):
        px_ecl_pct_assign = param['ProxyECL']
        px_ecl_pct_assign['PROXY_ECL_KEY'] = px_ecl_pct_assign['PORTFOLIO_CODE'] + \
            "-" + px_ecl_pct_assign['STAGE_FINAL'].astype(int).astype(str)
        px_ecl_pct = px_ecl_pct_assign[['PROXY_ECL_KEY', 'PROXY_ECL_PCT']]
        
        return px_ecl_pct        

    def get_proxy_ecl_key(self, px_df_, param):

        # ensue the same data types to merge
        px_df_['PRTFLO_ID'] = px_df_['PRTFLO_ID'].astype(str)
        px_mapping_param = param['Proxy_mapping']
        px_mapping_param['PRTFLO_ID'] = px_mapping_param['PRTFLO_ID'].astype(
            str)
        px_df_1 = pd.merge(px_df_, px_mapping_param,
                           how='left', on='PRTFLO_ID')

        @self._vectorize_decorator
        def get_portfolio_code(PROXY_PORTFOLIO_ID, SUB_CATEGORY):
            if str(PROXY_PORTFOLIO_ID).upper() == 'IMPORT LETTERS OF CREDIT':
                return 'KB91'
            elif str(PROXY_PORTFOLIO_ID).upper() == 'PERFORMANCE GUARANTEE ISSUED':
                return 'KB91'
            elif str(PROXY_PORTFOLIO_ID).upper() == 'FINANCIAL GUARANTEE ISSUED':
                return 'KB91'
            elif str(PROXY_PORTFOLIO_ID).upper() == 'FACTORING' and str(SUB_CATEGORY).upper() == 'SME':
                return 'KB91'
            elif str(PROXY_PORTFOLIO_ID).upper() == 'FACTORING' and str(SUB_CATEGORY).upper() == 'CORP':
                return 'KB92'
            else:
                return 'stat'  # Default value

        # TODO to update the vectorize in class
        px_df_1['PROXY_PORTFOLIO_CODE'] = get_portfolio_code(
            px_df_1['PROXY_PORTFOLIO_ID'], px_df_1['SUB_CATEGORY'])

        # get PROXY_ECL_KEY (e.g., KB91-1)
        px_df_1['PROXY_ECL_KEY'] = px_df_1['PROXY_PORTFOLIO_CODE'] + \
            "-" + px_df_1['STAGE_FINAL'].astype(int).astype(str)

        return px_df_1

    def _adjust_ead(self,df):
        df_ = df.copy()
        mask_guarantee = (
            (df_.PROXY_PORTFOLIO_ID.astype(str).str.upper() == 'PERFORMANCE GUARANTEE ISSUED') |
            (df_.PROXY_PORTFOLIO_ID.astype(str).str.upper() == 'FINANCIAL GUARANTEE ISSUED')
        )

        df_.loc[mask_guarantee, 'EAD_OCY'] = df_.loc[mask_guarantee, 'EAD_OCY'] - df_.loc[mask_guarantee, 'CASH']
        return df_


    def get_proxy_result(self, px_df_1, fx_df, param):
        
        px_ecl_pct = self.get_proxy_ecl_pct(param=param)

        proxy_result = pd.merge(px_df_1, px_ecl_pct,
                                how='left', on='PROXY_ECL_KEY')
        proxy_result_adj = self._adjust_ead(proxy_result)

        proxy_result_adj_2 = ofh.convert_to_LCY(df=proxy_result_adj, fx_tbl=fx_df, param=param)
        proxy_result_adj_2['PROXY_ECL_LCY'] = proxy_result_adj_2['EAD_LCY'] * \
            proxy_result['PROXY_ECL_PCT']
        proxy_result_adj_2['PROXY_ECL_OCY'] = proxy_result_adj_2['EAD_OCY'] * \
            proxy_result['PROXY_ECL_PCT']

        return proxy_result_adj_2

    def run(self, px_df_, fx_df, param):
        px_df_1 = self.get_proxy_ecl_key(px_df_, param)
        proxy_result = self.get_proxy_result(
           px_df_1=px_df_1, fx_df=fx_df, param=param)

        # export to result folder
        proxy_result.to_csv(
            self.resultPath/'ProxyECL_calculation_result_files_deal.csv', index=False, encoding='utf-8-sig',quoting=csv.QUOTE_NONNUMERIC)

        return proxy_result
