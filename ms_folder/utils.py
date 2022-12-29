"""
title     = 'SKT'
synopsis  = 'utils for explanatory data analysis and day-ahead forecasts of EV'
version   = '3.7.13'
date      = '2022–12–27'
author    = '장문석'
"""


import numpy as np
import pandas as pd
from scipy import stats



def corr_model(df, target, method='pearson'):
    """Method of correlation.

    Pearson, Kendall, Spearman correlation analysis.

    Parameters
    ----------
    df : dataframe
        Composed of explanatory variable(s) and a response variable.

    target : str
        Column name of a response variable.

    method : {'pearson', 'kendall', 'spearman'}, default='pearson'
        Method of correlation. See more in the ref [1].

    Returns
    -------
    corr_mat : dataframe
        Correlation matrix for all variables.

    corr_target : dataframe
        Correlation results for the response variable in descending order.

    References
    ----------
    .. [1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
    """
    corr_mat = df.corr(method=method)
    corr_target = corr_mat[target].sort_values(ascending=False)

    return corr_mat, pd.DataFrame(corr_target)
  
def remove_df_col(df, coln):
    """Dataframe의 컬럼(들) 제거.

    Parameters
    ----------
    coln : list
        제거할 컬럼명, ex) ["컬럼명1", "컬럼명2", ...].
    """
    dfc = df.copy()
    dfc.drop(coln, axis=1, inplace=True)
    return dfc

def null_add_general_info(df):
    """Null 값이 있어도 동작하는
    Get data information for Exploratory Data Analysis.

    함수 eda_add_general_info()와 동일함. 다만, Null 값이 있어도 동작.
    """
    dfc = df.copy()

    dfc['year'] = dfc.index.year
    dfc['month'] = dfc.index.month
    dfc['day'] = dfc.index.day
    dfc['week'] = dfc.index.dayofweek
    dfc['hour'] = dfc.index.hour
    dfc['minute'] = dfc.index.minute

    return dfc

def replace_null_ymhavg(df, coln):
    """Null 값 대체.

    누락된 시간과 동일한 연, 월에서 누락된 시간과 동일한 시간 값들의 평균으로 대체.

    Parameters
    ----------
    df : dataframe with datetime index
        datetime 형식의 날짜정보를 index로 가진 dataframe.

    coln : str
        null 값을 대체하고자 하는 컬럼명.

    Returns
    -------
    dfc : dataframe
        Null 값이 대체된 데이터프레임.

    References
    ----------
    .. [1] Ceci, Michelangelo, et al. "Predictive modeling of PV energy production:
        How to set up the learning task for a better prediction?." IEEE Transactions
        on Industrial Informatics 13.3 (2016): 956-966.
    """
    dfc = df.copy()

    target = dfc[coln]
    idx_null = np.where(target.isnull().values == 1)[0]   # null 값 인덱스
    date_null = dfc.index[idx_null]                       # null 값 datetime 인덱스

    for dn in date_null:
        y, m, h = dfc.loc[dn, 'year'], dfc.loc[dn, 'month'], dfc.loc[dn, 'hour']

        v = dfc.loc[(dfc['year'] == y) & (dfc['month'] == m) & (dfc['hour'] == h), coln]
        v = v.dropna()
        avg = np.mean(v)

        dfc.loc[dn, coln] = avg

    print(f"[*] Null 값 대체")
    print(f"{coln}의 Null 값 개수: {len(date_null)}")
    print(f"{coln}의 Null 값 날짜 : {date_null}")

    return dfc
  
