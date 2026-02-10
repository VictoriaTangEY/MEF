import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from util.loggers import createLogHandler
from input_handler.data_preprocessor import data_preprocessor
from input_handler.load_parameters import load_parameters
from ecl_engine.output_file_handler import convert_to_LCY
from specific_assessment import sa_collateral_lgd as col_lgd_ts
from specific_assessment import sa_cashflow_lgd as cfl_lgd_ts
import warnings

warnings.filterwarnings("ignore")

class SAProcessor:
    def __init__(self, rc, parent_logger=None):
        self.rc = rc
        self.logger = parent_logger or createLogHandler(
            'SAProcessor', rc.logPath/'Log_file_specific_assessment.log')
        self._data_loaded = False
        self.param = None
        self.instr_df = None
        self.fx_df = None
        self.repay_df = None
        self.sa_df = None
        self.fs_df = None
        self.other_df = None
        self.npl_data = None

    def _load_common_data(self):
        """Load data once and reuse across all parts"""
        if not self._data_loaded:
            self.logger.info('Loading parameters')
            self.param = load_parameters(parmPath=self.rc.parmPath)
            
            dp = data_preprocessor(context=self.rc)
            
            self.logger.info('Loading scenario data')
            self.npl_data = dp.load_scenario_data(
                data_path=self.rc.dataPathScen, file_pattern='NPL')
            
            self.logger.info('Reading ECL input files')
            self.instr_df, self.fx_df, self.repay_df = dp.load_input_data(param=self.param)
            
            _, _, self.sa_df, _, _ = dp.run(
                instr_df_raw=self.instr_df,
                run_scope=self.rc.run_scope,
                param=self.param)
            
            self.fs_df, _ = dp.standardize_data(
                param_dict=self.param,
                inDataPath=self.rc.inDataPath,
                rawDataName=self.rc.sa_fs_table_name,
                inputDataExt=self.rc.inputDataExtECL,
                dtype_tbl=self.rc.dtype_tbl)
            
            self.other_df, _ = dp.standardize_data(
                param_dict=self.param,
                inDataPath=self.rc.inDataPath,
                rawDataName=self.rc.sa_other_debt_table_name,
                inputDataExt=self.rc.inputDataExtECL,
                dtype_tbl=self.rc.dtype_tbl)
            
            self._data_loaded = True

    def _run_part1(self):
        """Part 1 logic with loaded data"""
        self.logger.info('-' * 50)
        self.logger.info('Starting Part 1: SA Parameter Generation')

        try:
            # Special treatment for no sa_fs_table
            if not self.fs_df.empty:
                cfl_custs = self.fs_df['CUST_ID'].unique()
                lgd_param_p = cfl_lgd_ts.SACashflowLGDParam(context=self.rc)
                lgd_param, _, _, _ = lgd_param_p.run(
                    fs_df=self.fs_df,
                    sa_df=self.sa_df,
                    other_df=self.other_df,
                    custs=cfl_custs)

                lgd_param.to_excel(
                    self.rc.parmPath/'SAFSParam.xlsx',
                    header=False, index=False, sheet_name='sa_fs_param')
                self.param['sa_fs_param'] = lgd_param
                self.logger.info('SA FS parameters saved successfully')
                

                # Save interim outputs
                self._save_interim_outputs()
            
            else:
                self.logger.warning('No sa_fs_table available')
                self.sa_df.to_csv(self.rc.resultPath/'interim/sa_df.csv', index=False)

            return True

        except Exception as e:
            self.logger.exception("Error in Part 1")
            return False

    def _save_interim_outputs(self):
        """Save common interim files"""
        self.fx_df.to_csv(self.rc.resultPath/'interim/fx_df.csv', index=False)
        self.sa_df.to_csv(self.rc.resultPath/'interim/sa_df.csv', index=False)
        self.fs_df.to_csv(self.rc.resultPath/'interim/fs_df.csv', index=False)
        self.other_df.to_csv(self.rc.resultPath/'interim/other_df.csv', index=False)
        self.npl_data.to_excel(self.rc.resultPath/'interim/npl_data.xlsx', index=False)
        
        repay_df_sa = self.repay_df[
            self.repay_df['CONTRACT_ID'].isin(self.sa_df['CONTRACT_ID'].unique())]
        repay_df_sa.to_csv(self.rc.resultPath/'interim/repay_df.csv', index=False)

    def _run_part2(self):
        """Part 2 logic with loaded data"""
        self.logger.info('Starting Part 2: SA LGD Calculation')

        try:
            # Special treatment for no sa_fs_table
            if not self.fs_df.empty:
                cfl_custs = self.fs_df['CUST_ID'].astype(str).unique()
                _, sa_df_on, _, _ = self._off_bal(self.sa_df)

                manual_noc_path = Path(self.rc.inDataPath) / 'sa_manual_noc.csv'
                if manual_noc_path.exists():
                    self.logger.info('Using precomputed NOC values')
                    cfl_lgd_p = cfl_lgd_ts.SACashflowLGDPrecomputed(context=self.rc)
                    cfl_lgd, sa_interim_output_df, noc_interim_output_df = cfl_lgd_p.run(
                        sa_df=sa_df_on,
                        repay_df=self.repay_df,
                        param=self.param,
                        npl_data=self.npl_data,
                        custs=cfl_custs)
                else:
                    self.logger.info('Calculating NOC from financial statements')
                    cfl_lgd_p = cfl_lgd_ts.SACashflowLGD(context=self.rc)
                    cfl_lgd, sa_interim_output_df, noc_interim_output_df = cfl_lgd_p.run(
                        sa_df=sa_df_on,
                        fs_df=self.fs_df,
                        other_df=self.other_df,
                        repay_df=self.repay_df,
                        param=self.param,
                        npl_data=self.npl_data,
                        custs=cfl_custs,
                        sa_fs_param_1=self.param['sa_fs_param'])

                sa_interim_output_df.to_csv(
                    self.rc.resultPath/'interim/sa_cfl_lgd_interim_output.csv', 
                    index=False)
                noc_interim_output_df.to_csv(
                    self.rc.resultPath/'interim/noc_interim_output.csv', 
                    index=False)
                cfl_lgd.to_csv(
                    self.rc.resultPath/'interim/cfl_lgd.csv', 
                    index=False)
                self.logger.info('CFL LGD calculation completed and saved')
            
            else:
                pass

            return True

        except Exception as e:
            self.logger.exception("Error in Part 2")
            return False

    def _off_bal(self, sa_df):
        """Shared off-balance logic"""
        excluded_instr_types = [
            'credit card', 'credit line', 'Partially allotted term loan',
            'Trade credit facility', 'Domestic guarantee facility',
            'Corporate/business card facility', 'Multi purpose credit facility committed',
            'Overdraft credit facility', 'Multi purpose credit facility non committed',
            'factoring'
        ]
        
        sa_df_adj = sa_df.copy()
        sa_df_on = sa_df_adj[sa_df_adj['ON_OFF_BAL_IND'] == 'ON'].copy()
        sa_df_off = sa_df_adj[sa_df_adj['ON_OFF_BAL_IND'] == 'OFF'].copy()
        
        sa_df_off_no = sa_df_off[
            (sa_df_off['STAGE_FINAL'] == 3) & 
            (sa_df_off['INSTR_TYPE'].isin(excluded_instr_types))
        ].copy()
        sa_df_off_no['ECL_APPROACH'] = 'SA_OUTSCOPE'
        
        sa_df_adj.loc[sa_df_off_no.index, 'ECL_APPROACH'] = 'SA_OUTSCOPE'
        return sa_df_adj, sa_df_on, sa_df_off[~sa_df_off.index.isin(sa_df_off_no.index)].copy(), sa_df_off_no

    def _run_part3(self):
        """Part 3 logic with loaded data"""
        self.logger.info('Starting Part 3: Final SA ECL Calculation')
        self.logger.info('-' * 50)

        try:
            # Special treatment if no sa_fs_table
            if not self.fs_df.empty:
                cfl_custs = self.fs_df['CUST_ID'].astype(str).unique()
                cfl_lgd = pd.read_csv(
                    self.rc.resultPath/'interim/cfl_lgd.csv', 
                    dtype={'CUST_ID': str})
            else:
                cfl_custs = []
                cfl_lgd = pd.DataFrame()
            
            # Calculate collateral LGD
            coll_list = self.param['SA_COL_param'].collateral_type.to_list()
            cust_coll_sum = self.sa_df.groupby('CUST_ID')[coll_list].sum().sum(axis=1)
            col_custs = [c for c in cust_coll_sum[cust_coll_sum > 0].index 
                        if c not in cfl_custs]
            
            col_lgd = col_lgd_ts.SACollateralLGD(context=self.rc).get_collateral_lgd(
                sa_df=self.sa_df, 
                param=self.param, 
                custs=col_custs)

            # Process LGD approaches
            sa_ecl_df = self._add_lgd_approach(self.sa_df, cfl_lgd, col_lgd)
            
            # Handle off-balance loans
            sa_ecl_df_adj, sa_ecl_df_on, sa_ecl_df_off_yes, sa_ecl_df_off_no = self._off_bal(sa_ecl_df)
            sa_ecl_df_adj.loc[sa_ecl_df_off_no.index, 'LGD'] = None
            
            for cust_id in sa_ecl_df_adj['CUST_ID'].unique():
                on_rows = sa_ecl_df_on[sa_ecl_df_on['CUST_ID'] == cust_id]
                off_yes_rows = sa_ecl_df_off_yes[sa_ecl_df_off_yes['CUST_ID'] == cust_id]
                if not on_rows.empty and not off_yes_rows.empty:
                    sa_ecl_df_adj.loc[off_yes_rows.index, 'LGD'] = on_rows['LGD'].iloc[0]

            # Final ECL calculation
            sa_ecl_df_adj['ECL_FINAL_OCY'] = sa_ecl_df_adj['EAD_OCY'] * sa_ecl_df_adj['LGD']
            sa_ecl_df_adj = convert_to_LCY(
                df=sa_ecl_df_adj, 
                fx_tbl=self.fx_df, 
                param=self.param)
            
            sa_ecl_df_adj.to_csv(
                self.rc.resultPath/"SAECL_calculation_result_files_deal.csv", 
                index=False, 
                encoding='utf-8-sig')
            return True

        except Exception as e:
            self.logger.exception("Error in Part 3")
            return False

    def _add_lgd_approach(self, sa_df, cfl_lgd, col_lgd):
        """Priority: CFL > COL > ASSIGNED (100% LGD)"""
        sa_ecl_df = sa_df.copy()

        # Special treatment for no sa_fs_table
        if not cfl_lgd.empty:
            cust_lgd = pd.concat([
                cfl_lgd[['CUST_ID', 'LGD']].assign(LGD_SOURCE='CFL'),
                col_lgd[['CUST_ID', 'LGD']].assign(LGD_SOURCE='COL')
            ]).drop_duplicates('CUST_ID', keep='first')
        else:
            cust_lgd = col_lgd[['CUST_ID', 'LGD']].assign(LGD_SOURCE='COL')

        sa_ecl_df = pd.merge(sa_ecl_df, cust_lgd, on='CUST_ID', how='left')
        
        conditions = [
            sa_ecl_df['LGD_SOURCE'] == 'CFL',
            sa_ecl_df['LGD_SOURCE'] == 'COL',
            True
        ]
        choices = [sa_ecl_df['LGD'], sa_ecl_df['LGD'], 1.0]
        sa_ecl_df['LGD'] = np.select(conditions, choices, default=np.nan)
        
        approach_choices = ['CFL', 'COL', 'ASSIGNED']
        sa_ecl_df['LGD_APPROACH'] = np.select(conditions, approach_choices, default='UNDEFINED')
        
        return sa_ecl_df.drop(columns=['LGD_SOURCE'])

    def run(self):
        """Single entry point that loads data once and runs all parts"""
        try:
            self._load_common_data()
            if not self._run_part1():
                return False
            if not self._run_part2():
                return False
            if not self._run_part3():
                return False
            return True
        except Exception as e:
            self.logger.exception("Fatal error in SAProcessor")
            return False

def run(rc, parent_logger=None):
    """Module-level function for running 604"""
    if parent_logger:
        logger = parent_logger
    else:
        logger = createLogHandler(
            'sa_part4', rc.logPath/'Log_file_specific_assessment.log')
    logger.info('-' * 50)
    logger.info('Starting Part 4: Overall SA run')
    logger.info('-' * 50)
    return SAProcessor(rc, parent_logger).run()