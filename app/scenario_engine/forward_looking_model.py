from input_handler.data_preprocessor import data_preprocessor
from input_handler.load_parameters import load_parameters
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
import statsmodels.api as sm
import re
import sys
import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(project_root)


class ForwardLookingModel():
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

        # TODO update the load_data to the run_scenario_engine
        # FIXME: the sheet name align a bit on pwa (upper case in data checking parameter)
        self.parameters = load_parameters(self.parmPath)
        self.mef = self.parameters['MEFmodel']
        self.pwa = self.parameters['pwa']
        self.moving_avg = self.parameters['FLMoving_avg']

    def calculate_output(self, multipliers, weights):
        output = {}
        output['CNSMR'] = (
            multipliers['CNSMR_Growth'] * weights['Growth'] +
            multipliers['CNSMR_Base'] * weights['Base'] +
            multipliers['CNSMR_Severe'] * weights['Severe']
        )
        output['BUSNS'] = (
            multipliers['BUSNS_Growth'] * weights['Growth'] +
            multipliers['BUSNS_Base'] * weights['Base'] +
            multipliers['BUSNS_Severe'] * weights['Severe']
        )
        output['NNCRD'] = (
            multipliers['NNCRD_Growth'] * weights['Growth'] +
            multipliers['NNCRD_Base'] * weights['Base'] +
            multipliers['NNCRD_Severe'] * weights['Severe']
        )
        return output

    def calculate_pd(self, dfs, z_dic, z_dic2):
        pd_dict = {}
        for key, value in dfs.items():
            z = z_dic[key]
            latest_observed_PD = z_dic2[key]
            predicted_PD = norm.cdf(-z)
            if latest_observed_PD == 0:
                latest_observed_PD += 10e-12
            multipliers = predicted_PD / latest_observed_PD
            pd_dict[key] = multipliers
        return pd_dict

    def calculate_weights(self):
        weights = {
            'Base': self.pwa.loc[self.pwa['scenario'] == 'BASE', 'pwa'].values[0],
            'Severe': self.pwa.loc[self.pwa['scenario'] == 'SEVE', 'pwa'].values[0],
            'Growth': self.pwa.loc[self.pwa['scenario'] == 'GROW', 'pwa'].values[0]
        }
        return weights

    def calculate_y(self, dfs):
        for key, value in dfs.items():
            model = key.split('_')[0]
            value['y'] = self.mef.loc[
                (self.mef['MEF_MODEL_ID'] == model) &
                (self.mef['MEF_NAME'] == 'INTERCEPT'), 'MODEL_COEF'
            ].values[0]

            for col in value.columns:
                if col in ['Code:', 'y']:
                    continue
                try:
                    name_col = '_'.join(col.split('_')[:3])
                    trans_col = '_'.join(col.split('_')[4:])
                    value['y'] += (
                        value[col] * self.mef.loc[
                            (self.mef['MEF_MODEL_ID'] == model) &
                            (self.mef['TRANSFORMATION'] == trans_col) &
                            (self.mef['MEF_NAME'] == name_col), 'MODEL_COEF'
                        ].values[0]
                    )
                except Exception as e:
                    print(f"Issue in calculation y: {col}")
                    print(f"Error: {e}")

    def calculate_z(self, dfs):
        z_dic = {}
        z_dic2 = {}
        for key, value in dfs.items():
            y = value['y']
            y = np.asarray(y, dtype=float)
            y_real = y[:-1]
            y_pre = y[-1]
            # moving average PD base
            moving_avg = self.moving_avg['MOVING_AVG']
            last_four_quarters = y_real[-int(moving_avg):]
            last_four_quarters_norm = norm.cdf(-last_four_quarters)
            z2 = np.mean(last_four_quarters_norm)
            z_dic[key] = y_pre
            z_dic2[key] = z2
        return z_dic, z_dic2

    def transformation_act(self, mef, model_id, ids, data):
        df = pd.DataFrame()
        df_pre = pd.DataFrame()
        df['Code:'] = data['Code:']
        df_pre['Code:'] = data['Code:']
        for i in range(mef.loc[ids == model_id, 'MEF_NAME'].size):
            try:
                name = mef.loc[ids == model_id, 'MEF_NAME'].values[i]
                methods = mef.loc[ids == model_id, 'TRANSFORMATION'].values[i]
                method = methods.split("_")
            except Exception as e:
                print(f"Issue in MEF: {model_id}")
                print(f"Error: {e}")

            for col in data.columns:
                if name in col:
                    df[col + '_' + methods] = None
                    df_pre[col + '_' + methods] = data[col]
                    for j in method:
                        act = self.choose_calculation_methods(j)
                        if j == 'SQ':
                            df_pre[col + '_' + methods] = act(data[col])

                        if df[col + '_' + methods].isnull().all():
                            try:
                                df[col + '_' + methods] = act(data[col])
                            except Exception as e:
                                print(f"Issue in calculation: {j}")
                                print(f"Error: {e}")
                        else:
                            try:
                                df[col + '_' +
                                    methods] = act(df[col + '_' + methods])
                            except Exception as e:
                                print(f"Issue in calculation: {j}")
                                print(f"Error: {e}")
        return df, df_pre

    def choose_calculation_methods(self, method):
        if method == 'YOY':
            def yoy(col):
                return col.pct_change(periods=4)
            return yoy
        elif method == 'ID':
            def identity(col):
                return col
            return identity
        elif method == 'SQ':
            def square(col):
                return col ** 2
            return square
        else:
            match = re.search(r"(\w+)\((\d+)\)", method)
            func = match.group(1)
            n = int(match.group(2))
            if func == 'LAG':
                def lag(col):
                    return col.shift(n)
                return lag
            elif func == 'DIV':
                def divide(col):
                    return col / n
                return divide
            elif func == 'DIFF':
                def difference(col):
                    return col.diff(n)
                return difference
            else:
                print(f"Unknown method: {func}")

    def choose_time_period(self, df, df_pre, end):
        try:
            end_date = datetime.strptime(end, '%Y/%m/%d')
            end_plus_one_year = end_date.replace(
                year=end_date.year + 1).strftime('%Y/%m/%d')
            df = df[(df['Code:'] <= end) |
                    (df['Code:'] == end_plus_one_year)]
            df[df['Code:'] == end_plus_one_year] = df_pre[df_pre['Code:']
                                                          == end_plus_one_year]
            # df['Code:'] = pd.to_datetime(df['Code:']).apply(lambda x: f"{x.year}/{x.month}/{x.day}")
        except Exception as e:
            print("Issue in choosing date period")
            print(f"Error: {e}")
        return df

    def transformation(self, end):
        processor = data_preprocessor(context=self)
        data = processor.load_scenario_data(
            data_path=self.dataPathScen, file_pattern='MEF')

        ids = self.mef['MEF_MODEL_ID']
        cnsmr_df = None
        busns_df = None
        nncrd_df = None

        for model_id in ids:
            if model_id == 'CNSMR' and cnsmr_df is None:
                cnsmr_df, cnsmr_df_pre = self.transformation_act(
                    self.mef, model_id, ids, data)
                cnsmr_df = self.choose_time_period(cnsmr_df, cnsmr_df_pre, end)
            elif model_id == 'BUSNS' and busns_df is None:
                busns_df, busns_df_pre = self.transformation_act(
                    self.mef, model_id, ids, data)
                busns_df = self.choose_time_period(busns_df, busns_df_pre, end)
            elif model_id == 'NNCRD' and nncrd_df is None:
                nncrd_df, nncrd_df_pre = self.transformation_act(
                    self.mef, model_id, ids, data)
                nncrd_df = self.choose_time_period(nncrd_df, nncrd_df_pre, end)
            elif model_id not in ['CNSMR', 'BUSNS', 'NNCRD']:
                print(f"Unknown segment: {model_id}")
        return cnsmr_df, busns_df, nncrd_df

    def devision(self, df):
        df_grow = pd.DataFrame()
        df_base = pd.DataFrame()
        df_seve = pd.DataFrame()
        df_grow['Code:'] = df['Code:']
        df_base['Code:'] = df['Code:']
        df_seve['Code:'] = df['Code:']
        for col in df.columns:
            if 'GROW' in col:
                df_grow[col] = df[col]
            elif 'BASE' in col:
                df_base[col] = df[col]
            elif 'SEVE' in col:
                df_seve[col] = df[col]
            elif col != 'Code:':
                print(f"Unknown scenario: {col}")
        return df_grow, df_base, df_seve

    def multiplier(self, end):
        cnsmr_df, busns_df, nncrd_df = self.transformation(end)
        cnsmr_growth, cnsmr_base, cnsmr_severe = self.devision(cnsmr_df)
        busns_growth, busns_base, busns_severe = self.devision(busns_df)
        nncrd_growth, nncrd_base, nncrd_severe = self.devision(nncrd_df)

        dfs = {
            'CNSMR_Growth': cnsmr_growth, 'CNSMR_Base': cnsmr_base, 'CNSMR_Severe': cnsmr_severe,
            'BUSNS_Growth': busns_growth, 'BUSNS_Base': busns_base, 'BUSNS_Severe': busns_severe,
            'NNCRD_Growth': nncrd_growth, 'NNCRD_Base': nncrd_base, 'NNCRD_Severe': nncrd_severe
        }

        self.calculate_y(dfs)
        z_dic, z_dic2 = self.calculate_z(dfs)
        pd_dict = self.calculate_pd(dfs, z_dic, z_dic2)
        weights = self.calculate_weights()
        output = self.calculate_output(pd_dict, weights)
        df_output = pd.DataFrame(list(output.items()), columns=[
                                 'segment', 'multiplier'])
        df_output['index'] = range(1, len(df_output) + 1)

        ################## format MEF ##################
        header = {
            "index": ["Parameter name", "Parameter description", "Mandatory input", "Parameter data type"],
            "segment": ["segment", "segment", "Y", "str"],
            "multiplier": ["multiplier", "MEF multiplier", "Y", "float"]
        }

        header = pd.DataFrame(header)

        mef_output = pd.concat([header, df_output], ignore_index=True)

        return mef_output


# if __name__ == "__main__":
#     model = ForwardLookingModel(
#         r'C:\Users\SV783QD\Downloads\20240920 - To Stella FL multiplier\99_Data_Server')
#     end = '2024/03/31'
#     model.multiplier(start, end)
