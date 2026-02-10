# Load packages
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from input_handler.data_preprocessor import data_preprocessor
from input_handler.load_parameters import (load_configuration_file,
                                           load_parameters)
from input_handler.env_setting import run_setting

# TBD: Just for analysis purpose
from memory_profiler import profile

class overlay_engine():
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

    def extract_overlay_info(self,
                             param: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        To extract and format the overlay information
        under the parameter dictionary
        """
        keep_cols = [
            'CONTRACT_ID',
            'ECL_RATE_OVR',
            'ECL_OVR',
            'ADJ_RSN',
            'PREPARED_BY'
        ]
        # TODO: check if need keep all cols
        overlay_param = (param['Overlay']
                         .filter(items=keep_cols,
                                 axis=1))

        return overlay_param

    def calculate_overlay_ecl(self,
                              df: pd.DataFrame,
                              overlay_param: pd.DataFrame) -> pd.DataFrame:
        """
        ### calculate_overlay_ecl

        Perform override on engine calculated ECL 
        based on management decision

        ### Parameters
        - param (Dict): A dictionary of parameters load from load_parameter function

        ### Return
        A tuple of raw input data \n
        0: instrument table \n
        1: exchange rate table \n
        2: repayment table \n
        """
        df_ = (df.merge(
            overlay_param, how='left',
            left_on=['CONTRACT_ID'],
            right_on=['CONTRACT_ID']
        ))

        # Overlay hierachy
        # Priority: Engine ECL < Individual assessment < Management Overlay - ECL rate < Management Overlay - ECL absolute
        cond = [
            df_.ECL_OVR.notna(),
            df_.ECL_RATE_OVR.notna(),
        ]

        choice = [
            df_.ECL_OVR,
            df_.ECL_RATE_OVR * df_.EAD_OCY,
        ]
        
        choices_flag = [
            'ECL OVERRIDE', 
            'ECL RATE OVERRIDE',
        ]

        df_ = (df_.assign(
            ECL_FINAL_OCY=np.select(cond,
                                    choice,
                                    default=df_.ECL_ENGINE_OCY),
            MO_TYPE=np.select(cond,
                              choices_flag,
                              default='NONE')
        ))

        return df_

    @profile
    def run(self,
            ecl_df: pd.DataFrame,
            param: Dict[str, pd.DataFrame]
            ) -> pd.DataFrame:

        overlay_param = self.extract_overlay_info(param=param)
        ecl_df_final = self.calculate_overlay_ecl(ecl_df,
                                                  overlay_param=overlay_param)

        return ecl_df_final


if __name__ == '__main__':
    configPath = Path(
        r'C:\Users\WH947CH\Engagement\Khan Bank\03_ECL_engine\02_Development\khb_engine\run_config_file.json')
    c = load_configuration_file(configPath=configPath)
    rc = run_setting(run_config=c)

    ca_engine = ecl_engine(context=rc)

    result = ca_engine.run()

    print(result.keys())
