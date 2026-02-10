import pandas as pd
import numpy as np
from typing import Dict

from pathlib import Path


class model_segmentation():
    def __init__(self):
        pass

    def _vectorize_decorator(self, func):
        return np.vectorize(func)

    # 20241023: Updated based on new Group ID logic
    def _get_PD_model_ID(self, df: pd.DataFrame) -> pd.DataFrame:

        @self._vectorize_decorator
        def _assign_prod_digit(data_source_cd: str,
                               category: str) -> str:

            if data_source_cd.upper() == 'LOAN':
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
            # TODO: To update after the new instrument tables
            # (Splitting off balance and approved but yet disbursed)
                elif category.upper() == 'OTHER':
                    return 'SM'

            elif data_source_cd.upper() == 'NON_LOAN':
                if category.upper() == 'C':
                    return 'GC'
                elif category.upper() == 'SL':
                    return 'SL'
                elif category.upper() == 'SF':
                    return 'SF'
            return 'XX'

        @self._vectorize_decorator
        def _assign_cq_digit(data_source_cd: str,
                             stage: float,
                             rest_ind: float,
                             past_due_days: float,
                             rating_notch: float) -> str:

            if stage == 3:
                return '00D'
            else:
                if data_source_cd.upper() == 'NON_LOAN':
                    # 250414: update 099 for NR
                    if pd.isnull(rating_notch) or rating_notch == 0:
                        return '099'
                    else:
                        return str(int(rating_notch)).zfill(3)
                elif data_source_cd.upper() == 'LOAN':
                    #250414 update the logic to only rest_ind true and 0 dpd as 00R
                    if rest_ind and past_due_days == 0: 
                        return '00R'
                    elif not rest_ind and past_due_days == 0:
                        return '001'
                    elif past_due_days <= 30:
                        return '002'
                    elif past_due_days <= 60:
                        return '003'
                    elif past_due_days <= 90:
                        return '004'
            return 'XXX'

        df_ = (df.assign(
            _product_digit=_assign_prod_digit(data_source_cd=df.DATA_SOURCE_CD,
                                              category=df.CATEGORY),

            _cq_digit=_assign_cq_digit(data_source_cd=df.DATA_SOURCE_CD,
                                       stage=df.STAGE_FINAL,
                                       rest_ind=df.RESTRUCTURED_IND,
                                       past_due_days=df.PAST_DUE_DAYS,
                                       rating_notch=df.CREDIT_RATING_NOTCH),
        ))

        df_1 = (df_.assign(
            PD_POOL_ID=(df_['_product_digit'] +
                        df_['_cq_digit']).astype('category'),
        )
            .drop(labels=['_product_digit', '_cq_digit'], axis=1)
        )

        return df_1
    def _get_LGD_model_ID(self, df: pd.DataFrame,
                          param: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        df_ = df.assign(
            SUB_CATEGORY=(df.SUB_CATEGORY.str.upper()).astype('category'),
            MONTH_IN_DEFT=df.MONTH_IN_DEFT.fillna(0),
        )

        param_ = (param['LGD_approach'].assign(
            LGD_CATEGORY=param['LGD_approach'].LGD_CATEGORY.str.upper(),
        )
            .drop(labels=['Parameter name'], axis=1)
        )

        df_1 = (df_.merge(
            param_, how='left',
            left_on=['SUB_CATEGORY'],
            right_on=['LGD_CATEGORY']
        ))

        # TODO: Might need to discuss any cap is needed on month in default
        @self._vectorize_decorator
        def _assign_mid_digit(lgd_approach: str,
                              month_in_default: int) -> str:
            if lgd_approach.upper() == 'CFL':
                return str(int(month_in_default)).zfill(3)
            else:
                return '000'

        @self._vectorize_decorator
        def _assign_cq_digit(lgd_approach: str,
                             rating_notch: int) -> str:
            if lgd_approach.upper() == 'NNL':
                if pd.isnull(rating_notch) or rating_notch == 0:
                    return '099'
                else:
                    return str(int(rating_notch)).zfill(3)
                    # TODO: update NR for 099
            else:
                return 'XXX'

        df_2 = df_1.assign(
            LGD_MID_DIGIT=_assign_mid_digit(lgd_approach=df_1.LGD_APPROACH,
                                            month_in_default=df_1.MONTH_IN_DEFT),

            LGD_CQ_DIGIT=_assign_cq_digit(lgd_approach=df_1.LGD_APPROACH,
                                          rating_notch=df_1.CREDIT_RATING_NOTCH),
        )

        df_3 = (df_2.assign(
            LGD_POOL_ID=(df_2.LGD_APPROACH
                         + df_2.LGD_PRODUCT_DIGIT
                         + df_2.LGD_MID_DIGIT
                         + df_2.LGD_CQ_DIGIT).astype('category')
        )
            .drop(labels=[  # 'LGD_PRODUCT_DIGIT', 241127 YJ: to use it on prepayment id
                'LGD_MID_DIGIT',
                'LGD_CQ_DIGIT',
                'LGD_CATEGORY'], axis=1)
        )

        return df_3

    def _get_EAD_model_ID(self, df: pd.DataFrame,
                          param: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        param_ead = param['EAD_approach']
        ead_dict = param_ead.groupby('EAD_POOL_ID')['INSTR_TYPE'].apply(list).to_dict()

        @self._vectorize_decorator
        def _assign_ead_id(ead_dict: Dict[str, list[str]],
                           on_off_ind: str,
                           product_cd: str,
                           instr_type: str,
                           undrawn_lmt: float,
                           credit_lmt: float) -> str:
            if on_off_ind.upper() == 'ON':
                return 'ULF099'
            elif on_off_ind.upper() == 'OFF':
                if instr_type in ead_dict['ULF001']:
                    return 'ULF001'
                elif instr_type in ead_dict['ULF002']:
                    return 'ULF002'
                elif instr_type in ead_dict['ULF003']:
                    return 'ULF003'
                elif instr_type in ead_dict['ULF004']:
                    return 'ULF004'
                elif instr_type in ead_dict['ULF005']:
                    return 'ULF005'
                else:
                    return 'ULF005'

        df_ = df.assign(
            EAD_POOL_ID=(_assign_ead_id(ead_dict,
                                        on_off_ind=df.ON_OFF_BAL_IND,
                                        product_cd=df.PRTFLO_ID,
                                        instr_type=df.INSTR_TYPE,
                                        undrawn_lmt=df.UNDRAWN_BAL_LCY,
                                        credit_lmt=df.CREDIT_LIMIT_LCY))
        )

        df_['EAD_POOL_ID'] = df_['EAD_POOL_ID'].astype('category')

        return df_

    def _get_lifetime_model_ID(self, df: pd.DataFrame) -> pd.DataFrame:

        @self._vectorize_decorator
        def _assign_lifetime_id(stage: float,
                                product_cd: str) -> str:

            if (product_cd.startswith(('6000', '6010'))):
                return 'CC' + str(int(stage)).zfill(3)
            elif (product_cd.startswith(('3000', '3010'))):
                return 'CL' + str(int(stage)).zfill(3)
            else:
                return 'XX099'

        df_ = (df.assign(
            LIFETIME_POOL_ID=(_assign_lifetime_id(stage=df.STAGE_FINAL,
                                                  product_cd=df.PRTFLO_ID)),
        ))

        df_['LIFETIME_POOL_ID'] = df_['LIFETIME_POOL_ID'].astype('category')

        return df_

    def _get_prepayment_model_ID(self, df: pd.DataFrame) -> pd.DataFrame:

        @self._vectorize_decorator
        def _assign_prepayment_id(stage: float,
                                  data_source_cd: str,
                                  on_off_ind: str,
                                  lgd_product_digit: str) -> str:
            if ((data_source_cd.upper() == 'LOAN')
                and (stage == 1)
                    and (on_off_ind.upper() == 'ON')):

                return 'LL' + lgd_product_digit.zfill(3)

            else:
                return 'XX099'

        df_ = (df.assign(
            PREPAYMENT_POOL_ID=(_assign_prepayment_id(stage=df.STAGE_FINAL,
                                                      data_source_cd=df.DATA_SOURCE_CD,
                                                      on_off_ind=df.ON_OFF_BAL_IND,
                                                      lgd_product_digit=df.LGD_PRODUCT_DIGIT)),
        ))

        df_['PREPAYMENT_POOL_ID'] = df_[
            'PREPAYMENT_POOL_ID'].astype('category')

        return df_

    def get_all_model_ID(self, df: pd.DataFrame,
                         param: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        df_ = (df.pipe(self._get_PD_model_ID)
               .pipe(self._get_LGD_model_ID, param=param)
               .pipe(self._get_EAD_model_ID, param=param)
               .pipe(self._get_lifetime_model_ID)
               .pipe(self._get_prepayment_model_ID)
               )

        return df_
