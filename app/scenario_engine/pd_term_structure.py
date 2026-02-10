# Load packages
import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from sklearn.linear_model import LinearRegression
from input_handler.data_preprocessor import data_preprocessor
from datetime import datetime
from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings("ignore")


# show all columns and rows of a dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# setting up environment


class env_setting():
    def __init__(self, context):
        self.run_yymm = context.run_yymm
        self.scenario_version = context.SCENARIO_VERSION
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


# prepare data
class data_preparation(env_setting):
    def __init__(self, context):
        super().__init__(context)

    def load_data(self):
        dp = data_preprocessor(context=self)
        raw_data = dp.load_scenario_data(
            data_path=self.dataPathScen, file_pattern='PD')
        return raw_data

    def calculate_start_date(self, sd2_dates):
        """
        Calculate the start date based on the latest SD2 date minus 72 months.
        Args:
            sd2_dates: Series or list of SD2 dates
        Returns:
            str: Start date in YYYY-MM-DD format
        """
        # Find the latest SD2 date
        latest_sd2 = pd.to_datetime(sd2_dates).max()

        # Calculate start date by subtracting 72 months
        start_date = latest_sd2 - pd.DateOffset(months=72)

        # Format the date to YYYY-MM-DD
        return start_date.strftime("%Y-%m-%d")

    def prepare_data(self, raw_data):
        data = raw_data.copy()
        # drop data where pdd_cat == P90+
        data = data[data['PDD_CAT'] != 'P90+']

        # convert date columns to date
        data['DTE_THEN'] = pd.to_datetime(
            data['DTE_THEN'], format='%m/%d/%Y', errors='coerce')
        data['SD2'] = pd.to_datetime(
            data['SD2'], format='%m/%d/%Y', errors='coerce')

        # only keep the latest 72 months SD2
        start_date = self.calculate_start_date(
            sd2_dates=data['SD2'])
        # chk
        # print('start_date: ', start_date)
        data = data[(data['SD2'] > pd.Timestamp(start_date))]

        # keep the cols needed
        data = data[['CATEGORY', 'PDD_CAT', 'M', 'PDD_BAL', 'BAL']]

        # sort data and reset index
        # reorder
        data.rename(columns={'CATEGORY': 'Segment',
                    'PDD_CAT': 'Category'}, inplace=True)

        segment_order = ['micro', 'other_consumer',
                         'sme', 'corp', 'online_consumer']
        category_order = ['P0', 'P1-30', 'P31-60', 'P61-90', 'Rest']
        data['Segment'] = pd.Categorical(
            data['Segment'], categories=segment_order, ordered=True)
        data['Category'] = pd.Categorical(
            data['Category'], categories=category_order, ordered=True)

        data.sort_values(by=['Segment', 'Category', 'M'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    def data_preparation_run(self):
        raw_data = self.load_data()
        data = self.prepare_data(raw_data)
        return raw_data, data


# term structure projection
class scenario_engine(data_preparation):
    def __init__(self, context):
        super().__init__(context)
        data_prepared = self.data_preparation_run()

        self.data = data_prepared[1].copy()
        self.segs = self.data.Segment.unique()
        self.cats = self.data.Category.unique()

        self.category_to_index = {
            'P0': 3,
            'P1-30': 2,
            'P31-60': 1,
            'P61-90': 0,
            'Rest': 0
        }

        self.mef_seg_mapping = {
            'micro': 'BUSNS',
            'sme': 'BUSNS',
            'corp': 'BUSNS',
            'online_consumer': 'CNSMR',
            'other_consumer': 'CNSMR'
        }

    # 20250116: Added for code optimization
    def _vectorize_decorator(self, func):
        return np.vectorize(func)

    def adjust_and_check_monotonic(self, pd):
        # monotonic increasing (including equality) adjust: horizontally
        for index, row in pd.iterrows():
            for i in range(1+2, len(row)):
                # if the current value is less than the previous value
                if row.iloc[i] < row.iloc[i - 1]:
                    # adjust: set the current value = the previous value
                    row.iloc[i] = row.iloc[i - 1]

        # monotinic increasing (including equality) adjust: vertically
        for seg in self.segs:
            # fot category = P0 to _90: vertical monotonic adjust
            mask = (pd['Segment'] == seg) & (
                pd['Category'].isin(['P0', 'P1-30', 'P31-60', 'P61-90']))
            for col in pd.columns[2:]:
                indices = pd[mask].index
                for i in range(1, len(indices)):
                    # if the current value is less than the previous
                    if pd.loc[indices[i], col] < pd.loc[indices[i - 1], col]:
                        # adjust: set the current value = previous value
                        pd.loc[indices[i], col] = pd.loc[indices[i - 1], col]

            # for category = Rest: vertical monotonic adjust
            mask = (pd['Segment'] == seg) & (
                pd['Category'].isin(['P0', 'Rest']))
            for col in pd.columns[2:]:
                indices = pd[mask].index
                for i in range(1, len(indices)):
                    # if the current value is less than the previous
                    if pd.loc[indices[i], col] < pd.loc[indices[i - 1], col]:
                        # adjust: set the current value = previous value
                        pd.loc[indices[i], col] = pd.loc[indices[i - 1], col]

        # monotonic increasing (including equality) chk: horizontally
        horizontally_monotonic = all(all(pd.iloc[i][col] <= pd.iloc[i][col + 1]
                                     for col in range(2, len(pd.columns) - 2)) for i in range(len(pd)))
        if not horizontally_monotonic:
            print("The DataFrame is not horizontally monotonic.")

        # monotonic increasing (including equality) chk: vertically
        mono_list = []
        for seg in self.segs:
            # for category = P0 to P90: chk if the pd is monotonically increasing vertically (including equality)
            mask = (pd['Segment'] == seg) & (
                pd['Category'].isin(['P0', 'P1-30', 'P31-60', 'P61-90']))
            vertically_monotonic = all(
                all(pd.loc[mask, col].iloc[i] <= pd.loc[mask, col].iloc[i + 1]
                    for i in range(len(pd.loc[mask, col]) - 1))
                for col in pd.columns[2:]
            )
            # for interim test: print(f"\nCheck if PD of Segment '{seg}' and Category 'P0' to 'P90' is monotonically increasing vertically (including equality): {vertically_monotonic}")
            # add to mono_list for final chk
            mono_list.append(vertically_monotonic)

            # for category = P0 and Rest: chk if the pd is monotonically increasing vertically (including equality) -> Rest >= P0
            mask = (pd['Segment'] == seg) & (
                pd['Category'].isin(['P0', 'Rest']))
            vertically_monotonic = all(
                all(pd.loc[mask, col].iloc[i] <= pd.loc[mask, col].iloc[i + 1]
                    for i in range(len(pd.loc[mask, col]) - 1))
                for col in pd.columns[2:]
            )
            # for interim test: print(f"Check if PD of Segment '{seg}' and Category 'P0' and 'Rest' is monotonically increasing vertically (including equality): {vertically_monotonic}")
            mono_list.append(vertically_monotonic)
            if not {all(mono_list)}:
                print("The DataFrame is not horizontally monotonic.")

        return pd

    def process_weibull_unconditional_PD(self, df1, select_multiplier):
        # Create a copy of df1 to df2
        df2 = df1.copy()

        # Multiply the first column of df2 by the multiplier and cap at 1
        df2.iloc[:, 0] = (df2.iloc[:, 0] * select_multiplier).clip(upper=1)

        # Iterate over columns 2 to 12 (indexes 1 to 11)
        for col in range(1, 12):  # Column indexes 1 to 11 correspond to columns 2 to 12
            # Calculate the sum of all previous columns for each row
            previous_sum = df2.iloc[:, :col].sum(axis=1)

            # Iterate through each row in the DataFrame
            for i in range(len(df2)):
                # If the sum of previous columns is greater than 1
                if previous_sum[i] > 1:
                    df2.iloc[i, col] = 0  # Set current column to 0
                else:
                    # If the remaining value (1 - previous_sum) is less than the corresponding value in df1
                    if (1 - previous_sum[i]) < df1.iloc[i, col]:
                        # Set current column to the remaining value
                        df2.iloc[i, col] = 1 - previous_sum[i]
                    else:
                        # If the total with the multiplier exceeds 1
                        if (previous_sum[i] + select_multiplier * df1.iloc[i, col]) > 1:
                            # Set to remaining value
                            df2.iloc[i, col] = 1 - previous_sum[i]
                        else:
                            # Otherwise, multiply the value
                            df2.iloc[i, col] = select_multiplier * \
                                df1.iloc[i, col]

        # Iterate over columns 13 and beyond (starting from index 12 to 359)
        for col in range(12, 360):  # Column indexes 12 to 359 correspond to columns 13 to 360
            # Calculate the sum of all previous columns for each row
            previous_sum = df2.iloc[:, :col].sum(axis=1)

            # Iterate through each row in the DataFrame
            for i in range(len(df2)):
                # If the sum of previous columns is greater than or equal to 1
                if previous_sum[i] >= 1:
                    df2.iloc[i, col] = 0  # Set current column to 0
                else:
                    # If the remaining value (1 - previous_sum) is less than the corresponding value in df1
                    if (1 - previous_sum[i]) < df1.iloc[i, col]:
                        # Set current column to the remaining value
                        df2.iloc[i, col] = 1 - previous_sum[i]
                    else:
                        # Otherwise, set to the corresponding value from df1
                        df2.iloc[i, col] = df1.iloc[i, col]
        return df2

    def _ts_calculation(self, param):
        data = self.data.copy()

        #################################### 1. Hist_pd ####################################
        pdd_bal_df = data.pivot_table(
            index=['Segment', 'Category'],
            columns='M',
            values='PDD_BAL',
            aggfunc='sum',
            fill_value=0
        ).reset_index()

        bal_df = data.pivot_table(
            index=['Segment', 'Category'],
            columns='M',
            values='BAL',
            aggfunc='sum',
            fill_value=0
        ).reset_index()

        hist_pd = pdd_bal_df.iloc[:, 2:] / bal_df.iloc[:, 2:]
        hist_pd.fillna(0, inplace=True)
        hist_pd = pd.concat([pdd_bal_df.iloc[:, 0:2], hist_pd], axis=1)

        # hist_pd adjust
        for seg in self.segs:
            hist_pd.loc[(hist_pd['Segment'] == seg) & (
                hist_pd['Category'] == 'P0'), hist_pd.columns[2:5]] = 0
            hist_pd.loc[(hist_pd['Segment'] == seg) & (
                hist_pd['Category'] == 'P1-30'), hist_pd.columns[2:4]] = 0
            hist_pd.loc[(hist_pd['Segment'] == seg) & (
                hist_pd['Category'] == 'P31-60'), hist_pd.columns[2:3]] = 0

        #################################### 2. Hist PD adjust ####################################
        # unconditional PD
        uncon_pd = hist_pd.iloc[:, 2:2+72].copy()
        uncon_pd_cum = (1 - uncon_pd).shift(axis=1).fillna(1).cumprod(axis=1)
        uncon_pd = uncon_pd_cum * uncon_pd
        uncon_pd.columns = range(1, len(uncon_pd.columns) + 1)
        uncon_pd = pd.concat([hist_pd.iloc[:, :2], uncon_pd], axis=1)

        # cumulative PD
        cum_pd = uncon_pd.iloc[:, 2:].copy()
        cum_pd = cum_pd.cumsum(axis=1)
        cum_pd = pd.concat([uncon_pd.iloc[:, :2], cum_pd], axis=1)
        self.adjust_and_check_monotonic(cum_pd)

        #################################### 3. Weibull dist ####################################
        # Table 1
        # cum_pd adjustment: if PD > 1, set PD = 0.99999999
        for index, row in cum_pd.iterrows():
            for i in range(2, len(row)):
                if row.iloc[i] > 1:
                    cum_pd.iloc[index, i] = 0.99999999

        # Table 2
        # Weibull distribution
        num_columns = 72

        weibull_list = []
        for i in range(4):
            num_zeros = 3 - i
            row = [0] * num_zeros + list(range(1, num_columns - num_zeros + 1))
            row += [None] * (num_columns - len(row))
            weibull_list.append(row)

        weibull_dist = pd.DataFrame(
            weibull_list, columns=range(1, num_columns + 1))
        weibull_dist = np.log(weibull_dist)

        # copy the last row (for fitting Category 'Rest')
        last_row = weibull_dist.iloc[-1].to_frame().T
        weibull_dist = pd.concat([weibull_dist, last_row], ignore_index=True)

        # weibull pd
        # base on cum_pd, if PD > 0, set PD = LN(LN(1/(1-PD))), else set PD = 0
        weibull_param1 = cum_pd.copy()

        for index, row in cum_pd.iterrows():
            for i in range(2, len(row)):
                if row.iloc[i] > 0:
                    weibull_param1.iloc[index, i] = np.log(
                        np.log(1/(1-weibull_param1.iloc[index, i])))
                else:
                    weibull_param1.iloc[index, i] = 0

        # Table 3
        # generate weibull parameters
        weibull_param2 = pd.DataFrame(columns=['intercept', 'slope'])

        # linear regression
        for seg in self.segs:
            weibull_pd_seg = weibull_param1.loc[weibull_param1['Segment'] == seg].copy(
            )

            # calculate weibull param for Category: P0 to P90
            for i in range(3, -1, -1):
                # extract y and x and convert to np arrays
                y = pd.to_numeric(
                    weibull_dist.iloc[3-i, i:], errors='coerce').to_numpy()
                x = pd.to_numeric(
                    weibull_pd_seg.iloc[3-i, 2+i:], errors='coerce').to_numpy().reshape(-1, 1)
                model = LinearRegression().fit(x, y)
                weibull_param2 = pd.concat([weibull_param2, pd.DataFrame(
                    {'intercept': [model.intercept_], 'slope': [model.coef_[0]]})], ignore_index=True)

            # calculate weibull param for Category: Rest
            y = pd.to_numeric(
                weibull_dist.iloc[4, :], errors='coerce').to_numpy()
            x = pd.to_numeric(
                weibull_pd_seg.iloc[4, 2:], errors='coerce').to_numpy().reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            weibull_param2 = pd.concat([weibull_param2, pd.DataFrame(
                {'intercept': [model.intercept_], 'slope': [model.coef_[0]]})], ignore_index=True)

        weibull_param2 = pd.concat(
            [weibull_param1.iloc[:, :2], weibull_param2], axis=1)
        weibull_param2['beta'] = np.exp(weibull_param2['intercept'])
        weibull_param2['alpha'] = 1/weibull_param2['slope']

        # Table 4
        # weibull pd - PD fitted by weibull distribution
        weibull_pd = cum_pd[['Segment', 'Category']]
        for i in range(1, 361):
            weibull_pd[i] = 0

        mob = np.array(range(1, 361))

        for seg in self.segs:
            for cat in self.cats:

                n = 0  # starting point of mob

                # starting point of columns of weibull_pd, if cat is not in the dict, return None
                i = self.category_to_index.get(cat, None)

                beta = weibull_param2.loc[
                    (weibull_param2['Segment'] == seg) & (
                        weibull_param2['Category'] == cat),
                    'beta'
                ].values[0]

                alpha = weibull_param2.loc[
                    (weibull_param2['Segment'] == seg) & (
                        weibull_param2['Category'] == cat),
                    'alpha'
                ].values[0]

                mask = (weibull_pd['Segment'] == seg) & (
                    weibull_pd['Category'] == cat)

                for j in range(2 + i, weibull_pd.shape[1]):

                    if j <= cum_pd.shape[1]-1:
                        cum_value = cum_pd.loc[(cum_pd['Segment'] == seg) & (
                            cum_pd['Category'] == cat), cum_pd.columns[j]].values[0]
                    else:
                        cum_value = ""

                    if (alpha < 0 or np.isinf(alpha)):
                        # print(cum_value)
                        if cum_value == "":
                            weibull_pd.loc[mask, weibull_pd.columns[j]
                                           ] = weibull_pd.loc[mask, weibull_pd.columns[j - 1]]
                        else:
                            weibull_pd.loc[mask,
                                           weibull_pd.columns[j]] = cum_value
                    else:
                        weibull_pd.loc[mask, weibull_pd.columns[j]] = weibull_min.cdf(
                            mob[n], alpha, scale=beta)
                        # print(weibull_min.cdf(mob[0], alpha, scale=beta))
                    n += 1

        # Table 5
        # weibull annual PD
        weibull_pd_left = weibull_pd.iloc[:, :2].copy()

        weibull_pd_right = weibull_pd.iloc[:, 2:].copy()
        weibull_pd_right = weibull_pd_right.loc[:,
                                                weibull_pd_right.columns % 12 == 0]
        weibull_pd_right.columns = range(1, len(weibull_pd_right.columns) + 1)

        annual_pd = pd.concat([weibull_pd_left, weibull_pd_right], axis=1)

        #################################### 3. Life-time PD ####################################
        weibull_pd = self.adjust_and_check_monotonic(weibull_pd)

        # weibull unconditional PD
        wb_uncon_pd = weibull_pd.copy()
        wb_uncon_pd_shif = wb_uncon_pd.iloc[:, 2:].shift(1, axis=1)
        wb_uncon_pd_shif.fillna(0, inplace=True)
        wb_uncon_pd.iloc[:, 2:] = wb_uncon_pd.iloc[:, 2:] - wb_uncon_pd_shif

        # Table 9
        # weibull life PD
        wb_life_pd = pd.DataFrame()

        # load MEF params
        mef_df = param['MEFmultipliers']

        multiplier = pd.DataFrame({
            'Segment': self.segs,
            'Multiplier': [mef_df.loc[mef_df['segment'] == self.mef_seg_mapping[seg], 'multiplier'].values[0] for seg in self.segs]
        })

        for seg in self.segs:
            df1 = wb_uncon_pd[wb_uncon_pd['Segment'] == seg].copy()
            df1 = df1.iloc[:, 2:].reset_index(drop=True)

            select_multiplier = multiplier[multiplier['Segment']
                                           == seg]['Multiplier'].values[0]

            # print(select_multiplier)

            df2 = self.process_weibull_unconditional_PD(df1, select_multiplier)
            df2 = pd.concat([wb_uncon_pd[wb_uncon_pd['Segment'] ==
                            seg].iloc[:, :2].reset_index(drop=True), df2], axis=1)

            wb_life_pd = pd.concat([wb_life_pd, df2], ignore_index=True)

        wb_life_pd.reset_index(drop=True)

        # forward-looking cumulative PD - monthly
        fl_cum_pd_mth = wb_life_pd.iloc[:, 2:].copy()
        fl_cum_pd_mth = fl_cum_pd_mth.cumsum(axis=1)
        fl_cum_pd_mth = pd.concat(
            [wb_life_pd.iloc[:, :2], fl_cum_pd_mth], axis=1)

        # forward-looking cumulative PD - annually
        fl_cum_pd_anl_left = fl_cum_pd_mth.iloc[:, :2].copy()

        fl_cum_pd_anl_right = fl_cum_pd_mth.iloc[:, 2:].copy()
        fl_cum_pd_anl_right = fl_cum_pd_anl_right.loc[:,
                                                      fl_cum_pd_anl_right.columns % 12 == 0]
        fl_cum_pd_anl_right.columns = range(
            1, len(fl_cum_pd_anl_right.columns) + 1)

        fl_cum_pd_anl = pd.concat(
            [fl_cum_pd_anl_left, fl_cum_pd_anl_right], axis=1)

        #################################### Format AutoPD ####################################
        def get_seg_id(seg):
            if seg == 'micro':
                return 'MS'
            elif seg == 'other_consumer':
                return 'CS'
            elif seg == 'sme':
                return 'SM'
            elif seg == 'corp':
                return 'CL'
            elif seg == 'online_consumer':
                return 'OC'

        def get_cat_id(cat):
            if cat == 'P0':
                return '001'
            elif cat == 'P1-30':
                return '002'
            elif cat == 'P31-60':
                return '003'
            elif cat == 'P61-90':
                return '004'
            elif cat == 'Rest':
                return '00R'

        # Copy the DataFrame and add an index column
        pd_table = fl_cum_pd_anl.copy()

        # change the "Other" in column Segment to "Corporate business"
        pd_table['Segment'] = pd_table['Segment'].replace(
            'Other', 'Corporate business')

        # Initialize lists for new data
        pd_pool_id_list = []
        pd_pool_desc_list = []

        # Iterate over rows to generate PD_pool_id and pool_description
        for index, row in pd_table.iterrows():
            seg = row['Segment']
            cat = row['Category']

            pd_pool_id = get_seg_id(seg) + get_cat_id(cat)
            pd_pool_desc = seg + " " + cat

            pd_pool_id_list.append(pd_pool_id)
            pd_pool_desc_list.append(pd_pool_desc)

        # Add the new columns to the DataFrame
        pd_table['PD_pool_id'] = pd_pool_id_list
        pd_table['pool_description'] = pd_pool_desc_list

        # PD_0 is column with hard code as 0% - for some stage 3 facilities / fallback segments, assign 100% for PD_0
        pd_table['PD_0'] = 0

        # Select only the relevant columns
        pd_table = pd_table[['PD_pool_id', 'pool_description', 'PD_0']]

        pd_table = pd.concat([pd_table, fl_cum_pd_anl.iloc[:, 2:]], axis=1)
        pd_table.insert(0, 'index', range(1, len(pd_table) + 1))

        # Rename columns from the second onward
        pd_table.columns = ['index', 'PD_pool_id', 'pool_description',
                            'PD_0'] + [f'PD_{i}' for i in range(1, 31)]

        # create a table header
        header = {
            'index': [
                'Parameter name',
                'Parameter description',
                'Mandatory input',
                'Parameter data type'
            ],
            'PD_pool_id': [
                'PD_POOL_ID',
                'PD segment pool ID for PD parameter mapping purpose',
                'Y',
                'category'
            ],
            'pool_description': [
                'POOL_DESCRIPTION',
                'Description of the PD segment',
                'N',
                'str'
            ]
        }

        # add col PD_0 to PD_30 (1 + 30 months)
        for i in range(31):
            col_name = f'PD_{i}'
            if i == 0:
                header[col_name] = [
                    col_name,
                    'Year 0 Cumulative PD',
                    'Y',
                    'float'
                ]
            else:
                header[col_name] = [
                    col_name,
                    f'Year {i} Pre-forward looking cumulative PD',
                    'Y',
                    'float'
                ]

        table_header = pd.DataFrame(header)

        auto_pd = pd.concat([table_header, pd_table], ignore_index=True)
        auto_pd.columns = auto_pd.columns.str.upper()
        return auto_pd

    # 20250128: Enhanced to monthly output
    def ts_calculation(self, param):
        data = self.data.copy()

        #################################### 1. Hist_pd ####################################
        pdd_bal_df = data.pivot_table(
            index=['Segment', 'Category'],
            columns='M',
            values='PDD_BAL',
            aggfunc='sum',
            fill_value=0
        ).reset_index()

        bal_df = data.pivot_table(
            index=['Segment', 'Category'],
            columns='M',
            values='BAL',
            aggfunc='sum',
            fill_value=0
        ).reset_index()

        hist_pd = pdd_bal_df.iloc[:, 2:] / bal_df.iloc[:, 2:]
        hist_pd.fillna(0, inplace=True)
        hist_pd = pd.concat([pdd_bal_df.iloc[:, 0:2], hist_pd], axis=1)

        # hist_pd adjust
        for seg in self.segs:
            hist_pd.loc[(hist_pd['Segment'] == seg) & (
                hist_pd['Category'] == 'P0'), hist_pd.columns[2:5]] = 0
            hist_pd.loc[(hist_pd['Segment'] == seg) & (
                hist_pd['Category'] == 'P1-30'), hist_pd.columns[2:4]] = 0
            hist_pd.loc[(hist_pd['Segment'] == seg) & (
                hist_pd['Category'] == 'P31-60'), hist_pd.columns[2:3]] = 0

        #################################### 2. Hist PD adjust ####################################
        # unconditional PD
        # TODO update the hard-code 72: change the starting date or the mth len
        uncon_pd = hist_pd.iloc[:, 2:2+72].copy()
        uncon_pd_cum = (1 - uncon_pd).shift(axis=1).fillna(1).cumprod(axis=1)
        uncon_pd = uncon_pd_cum * uncon_pd
        uncon_pd.columns = range(1, len(uncon_pd.columns) + 1)
        uncon_pd = pd.concat([hist_pd.iloc[:, :2], uncon_pd], axis=1)

        # cumulative PD
        cum_pd = uncon_pd.iloc[:, 2:].copy()
        cum_pd = cum_pd.cumsum(axis=1)
        cum_pd = pd.concat([uncon_pd.iloc[:, :2], cum_pd], axis=1)
        self.adjust_and_check_monotonic(cum_pd)

        #################################### 3. Weibull dist ####################################
        # Table 1
        # cum_pd adjustment: if PD > 1, set PD = 0.99999999
        for index, row in cum_pd.iterrows():
            for i in range(2, len(row)):
                if row.iloc[i] > 1:
                    cum_pd.iloc[index, i] = 0.99999999

        # Table 2
        # Weibull distribution
        num_columns = 72

        weibull_list = []
        for i in range(4):
            num_zeros = 3 - i
            row = [0] * num_zeros + list(range(1, num_columns - num_zeros + 1))
            row += [None] * (num_columns - len(row))
            weibull_list.append(row)

        weibull_dist = pd.DataFrame(
            weibull_list, columns=range(1, num_columns + 1))
        weibull_dist = np.log(weibull_dist)

        # copy the last row (for fitting Category 'Rest')
        last_row = weibull_dist.iloc[-1].to_frame().T
        weibull_dist = pd.concat([weibull_dist, last_row], ignore_index=True)

        # weibull pd
        # base on cum_pd, if PD > 0, set PD = LN(LN(1/(1-PD))), else set PD = 0
        weibull_param1 = cum_pd.copy()

        for index, row in cum_pd.iterrows():
            for i in range(2, len(row)):
                if row.iloc[i] > 0:
                    weibull_param1.iloc[index, i] = np.log(
                        np.log(1/(1-weibull_param1.iloc[index, i])))
                else:
                    weibull_param1.iloc[index, i] = 0

        # Table 3
        # generate weibull parameters
        weibull_param2 = pd.DataFrame(columns=['intercept', 'slope'])

        # linear regression
        for seg in self.segs:
            weibull_pd_seg = weibull_param1.loc[weibull_param1['Segment'] == seg].copy(
            )

            # calculate weibull param for Category: P0 to P90
            for i in range(3, -1, -1):
                # extract y and x and convert to np arrays
                y = pd.to_numeric(
                    weibull_dist.iloc[3-i, i:], errors='coerce').to_numpy()
                x = pd.to_numeric(
                    weibull_pd_seg.iloc[3-i, 2+i:], errors='coerce').to_numpy().reshape(-1, 1)
                model = LinearRegression().fit(x, y)
                weibull_param2 = pd.concat([weibull_param2, pd.DataFrame(
                    {'intercept': [model.intercept_], 'slope': [model.coef_[0]]})], ignore_index=True)

            # calculate weibull param for Category: Rest
            y = pd.to_numeric(
                weibull_dist.iloc[4, :], errors='coerce').to_numpy()
            x = pd.to_numeric(
                weibull_pd_seg.iloc[4, 2:], errors='coerce').to_numpy().reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            weibull_param2 = pd.concat([weibull_param2, pd.DataFrame(
                {'intercept': [model.intercept_], 'slope': [model.coef_[0]]})], ignore_index=True)

        weibull_param2 = pd.concat(
            [weibull_param1.iloc[:, :2], weibull_param2], axis=1)
        weibull_param2['beta'] = np.exp(weibull_param2['intercept'])
        weibull_param2['alpha'] = 1/weibull_param2['slope']

        # Table 4
        # weibull pd - PD fitted by weibull distribution
        weibull_pd = cum_pd[['Segment', 'Category']]
        for i in range(1, 361):
            weibull_pd[i] = 0

        mob = np.array(range(1, 361))

        for seg in self.segs:
            for cat in self.cats:

                n = 0  # starting point of mob

                # starting point of columns of weibull_pd, if cat is not in the dict, return None
                i = self.category_to_index.get(cat, None)

                beta = weibull_param2.loc[
                    (weibull_param2['Segment'] == seg) & (
                        weibull_param2['Category'] == cat),
                    'beta'
                ].values[0]

                alpha = weibull_param2.loc[
                    (weibull_param2['Segment'] == seg) & (
                        weibull_param2['Category'] == cat),
                    'alpha'
                ].values[0]

                mask = (weibull_pd['Segment'] == seg) & (
                    weibull_pd['Category'] == cat)

                for j in range(2 + i, weibull_pd.shape[1]):

                    if j <= cum_pd.shape[1]-1:
                        cum_value = cum_pd.loc[(cum_pd['Segment'] == seg) & (
                            cum_pd['Category'] == cat), cum_pd.columns[j]].values[0]
                    else:
                        cum_value = ""

                    if (alpha < 0 or np.isinf(alpha)):
                        # print(cum_value)
                        if cum_value == "":
                            weibull_pd.loc[mask, weibull_pd.columns[j]
                                           ] = weibull_pd.loc[mask, weibull_pd.columns[j - 1]]
                        else:
                            weibull_pd.loc[mask,
                                           weibull_pd.columns[j]] = cum_value
                    else:
                        weibull_pd.loc[mask, weibull_pd.columns[j]] = weibull_min.cdf(
                            mob[n], alpha, scale=beta)
                        # print(weibull_min.cdf(mob[0], alpha, scale=beta))
                    n += 1

        # Table 5
        # weibull annual PD
        weibull_pd_left = weibull_pd.iloc[:, :2].copy()

        weibull_pd_right = weibull_pd.iloc[:, 2:].copy()
        weibull_pd_right = weibull_pd_right.loc[:,
                                                weibull_pd_right.columns % 12 == 0]
        weibull_pd_right.columns = range(1, len(weibull_pd_right.columns) + 1)

        annual_pd = pd.concat([weibull_pd_left, weibull_pd_right], axis=1)

        #################################### 3. Life-time PD ####################################
        weibull_pd = self.adjust_and_check_monotonic(weibull_pd)

        # weibull unconditional PD
        wb_uncon_pd = weibull_pd.copy()
        wb_uncon_pd_shif = wb_uncon_pd.iloc[:, 2:].shift(1, axis=1)
        wb_uncon_pd_shif.fillna(0, inplace=True)
        wb_uncon_pd.iloc[:, 2:] = wb_uncon_pd.iloc[:, 2:] - wb_uncon_pd_shif

        # Table 9
        # weibull life PD
        wb_life_pd = pd.DataFrame()

        # load MEF params
        mef_df = param['MEFmultipliers']

        multiplier = pd.DataFrame({
            'Segment': self.segs,
            'Multiplier': [mef_df.loc[mef_df['segment'] == self.mef_seg_mapping[seg], 'multiplier'].values[0] for seg in self.segs]
        })

        for seg in self.segs:
            df1 = wb_uncon_pd[wb_uncon_pd['Segment'] == seg].copy()
            df1 = df1.iloc[:, 2:].reset_index(drop=True)

            select_multiplier = multiplier[multiplier['Segment']
                                           == seg]['Multiplier'].values[0]

            # print(select_multiplier)

            df2 = self.process_weibull_unconditional_PD(df1, select_multiplier)
            df2 = pd.concat([wb_uncon_pd[wb_uncon_pd['Segment'] ==
                            seg].iloc[:, :2].reset_index(drop=True), df2], axis=1)

            wb_life_pd = pd.concat([wb_life_pd, df2], ignore_index=True)

        wb_life_pd.reset_index(drop=True)

        # 20250116: Keep monthly PD as per user request
        # Quick fix approach
        fl_cum_pd_anl = wb_life_pd.copy()

        # 20250123: Optimize code for PD POOL ID mapping
        def _get_PD_model_ID(df: pd.DataFrame) -> pd.DataFrame:

            @self._vectorize_decorator
            def _assign_prod_digit(category: str) -> str:
                if category.upper() == 'CORP':
                    return 'CL'
                elif category.upper() == 'SME':
                    return 'SM'
                elif category.upper() == 'MICRO':
                    return 'MS'
                elif category.upper() == 'ONLINE_CONSUMER':
                    return 'OC'
                elif category.upper() == 'OTHER_CONSUMER':
                    return 'CS'

                return 'XX'

            @self._vectorize_decorator
            def _assign_cq_digit(category: str) -> str:
                if category.upper() == 'REST':
                    return '00R'
                elif category.upper() == 'P0':
                    return '001'
                elif category.upper() == 'P1-30':
                    return '002'
                elif category.upper() == 'P31-60':
                    return '003'
                elif category.upper() == 'P61-90':
                    return '004'

                return 'XXX'

            df_ = (df.assign(
                _product_digit=_assign_prod_digit(category=df.Segment),

                _cq_digit=_assign_cq_digit(category=df.Category),
            ))

            df_1 = (df_.assign(
                PD_POOL_ID=(df_['_product_digit'] +
                            df_['_cq_digit']).astype('category'),
            )
                .drop(labels=['_product_digit', '_cq_digit'], axis=1)
            )

            return df_1

        # Copy the DataFrame and add an index column
        pd_table = fl_cum_pd_anl.copy()

        # Rename column
        rename_dict = {}
        for i in range(1, 361):
            rename_dict[i] = f'PD_{i}'

        pd_table = (pd_table.rename(columns=rename_dict)
                    .pipe(_get_PD_model_ID))

        pd_table = (pd_table.assign(
            POOL_DESCRIPTION=(pd_table['Segment'].astype(str) +
                              ' ' +
                              pd_table['Category'].astype(str)),
            PD_0=0,
        ))

        # Re-order columns
        col_order = (['PD_POOL_ID', 'POOL_DESCRIPTION', 'PD_0'] +
                     [f'PD_{i}' for i in range(1, 361)])

        pd_table = pd_table[col_order].reset_index()

        # To cumulative
        for i in range(1, 361):
            pd_table[f'PD_{i}'] = pd_table[f'PD_{i}'] + pd_table[f'PD_{i-1}']

        # create a table header
        header = {
            'index': [
                'Parameter name',
                'Parameter description',
                'Mandatory input',
                'Parameter data type'
            ],
            'PD_POOL_ID': [
                'PD_POOL_ID',
                'PD segment pool ID for PD parameter mapping purpose',
                'Y',
                'category'
            ],
            'POOL_DESCRIPTION': [
                'POOL_DESCRIPTION',
                'Description of the PD segment',
                'N',
                'str'
            ]
        }
        pd_table['index'] = range(1, len(pd_table) + 1)
        # add col PD_0 to PD_360 (1 + 360 months)
        for i in range(361):
            col_name = f'PD_{i}'
            if i == 0:
                header[col_name] = [
                    col_name,
                    'Monthly 0 Cumulative PD',
                    'Y',
                    'float'
                ]
            else:
                header[col_name] = [
                    col_name,
                    f'Monthly {i} Pre-forward looking cumulative PD',
                    'Y',
                    'float'
                ]

        table_header = pd.DataFrame(header)

        auto_pd = pd.concat([table_header, pd_table], ignore_index=True)
        auto_pd.columns = auto_pd.columns.str.upper()

        return auto_pd
