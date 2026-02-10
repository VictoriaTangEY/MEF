# Load packages
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# setting up environment

class run_setting():
    def __init__(self, run_config):

        c = run_config.copy()

        # General run purpose
        self.run_yymm = c['RUN_SETTING']['RUN_YYMM']
        self.prev_yymm = c['RUN_SETTING']['PREV_YYMM']

        self.masterPath = Path(c['RUN_SETTING']['MASTERPATH'])
        self.outputPath = Path(c['RUN_SETTING']['OUTPUTPATH'])

        self.PARAM_VERSION = c['RUN_SETTING']['PARAM_VERSION']
        self.SCENARIO_VERSION = c['RUN_SETTING']['SCENARIO_VERSION']

        self.RUN_MODE = c['RUN_SETTING']['RUN_MODE']
        self.RUN_MODE_SA = c['RUN_SETTING']['RUN_MODE_SA']

        # Path for input: data and parameter
        self.dataPathScen = self.masterPath / 'scenario' / \
            str(self.SCENARIO_VERSION) / 'data'
        self.inDataPath = self.masterPath / 'production' / \
            str(self.run_yymm) / 'data' / '01_input'
        self.previnDataPath = self.masterPath / 'production' / \
            str(self.prev_yymm) / 'data' / '01_input'
        self.parmPath = self.masterPath / 'parameter' / str(self.PARAM_VERSION)

        # Path for output: log, result and report
        self.resultPath = self.outputPath / 'production' / \
            str(self.run_yymm) / 'data' / '02_result'
        self.prevResultPath = self.outputPath / 'production' / \
            str(self.prev_yymm) / 'data' / '02_result'
        self.reportPath = self.outputPath / 'production' / \
            str(self.run_yymm) / 'data' / '03_report'
        self.logPath = self.outputPath / \
            'production' / str(self.run_yymm) / 'log'

        # For scenario generation
        self.T_ZERO = c['SCENARIO_SETTING']['T_ZERO']
        self.extend_yr = c['SCENARIO_SETTING']['EXTEND_YR']

        self.total_yr = c['SCENARIO_SETTING']['TOTAL_YR']

        self.inputDataExtScen = c['SCENARIO_SETTING']['INPUT_DATA_EXT']
        self.dataNameScen = f'scenarioRawData.{self.inputDataExtScen}'

        # For ECL Calculation
        self.run_option = ['Data check only']

        self.run_scope_input = c['RUN_SETTING']['RUN_SCOPE']
        self.run_scope = [x.upper() for x in self.run_scope_input]

        self.valid_run_scope = ['NON_LOAN', 'LOAN']

        self.prtflo_scope = [
            m for m in self.run_scope if m in self.valid_run_scope]

        self.scenario_set = ['BASE', 'GOOD', 'BAD']

        # For Data Preprocessing
        self.dtype_tbl = {
            # DEBUG Comment-out, to check
            # 'date': np.datetime64,
            'date': 'datetime64[ns]',
            'str': str,
            'int': int,
            'float': np.float64,
            'bool': bool,
            'category': 'category',
        }
        self.days_in_year = c['CONSTANT']['days_in_year']
        self.days_in_month = c['CONSTANT']['days_in_month']
        self.inputDataExtECL = c['DATA_IO_SETTING']['INPUT_DATA_EXT']
        self.instrument_table_name = c['DATA_IO_SETTING']['INSTRUMENT_TABLE_NAME']
        self.exchange_rate_table_name = c['DATA_IO_SETTING']['EXCHANGE_RATE_TABLE_NAME']
        self.repayment_table_name = c['DATA_IO_SETTING']['REPAYMENT_TABLE_NAME']
        self.sa_fs_table_name = c['DATA_IO_SETTING']['SA_FS_TABLE_NAME']
        self.sa_other_debt_table_name = c['DATA_IO_SETTING']['SA_OTHER_DEBT_TABLE_NAME']

        # For Testing
        self.mute_eir = c['TEST_SETTING']['MUTE_EIR']
        self.mute_stage_consistency = c['TEST_SETTING']['MUTE_STAGE_CONSISTENCY']

        # print(self.dataPath)
        # self.logPath = masterPath / 'log'
