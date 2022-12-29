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
import json
import folium
from sklearn.preprocessing import OneHotEncoder


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

def df_time_diff_seconds(df_indices):
    """Calculate the time difference in seconds between
    consecutive dataframe indices(datetime).

    Only available to 1 day, 1hour, 30min, 15min, 1min interval.
    - 1 day interval : 86400 s
    - 1 hour interval : 3600 s
    - 30 min interval : 1800 s
    - 15 min interval : 900 s
    - 1 min interval : 60 s

    Parameters
    ----------
    df_indices : datetime
        Dataframe indices(datetime format).

    Returns
    -------
    tdelta_sec[0] : int
        Time difference in seconds.
    """
    # 시간 간격 계산 (초 차이)
    tdelta_sec = (df_indices[1:] - df_indices[:-1]).seconds
    tdelta_sec = list(tdelta_sec.values)

    # 예외 처리 : 시간 간격이 동일하지 않는 경우
    is_tdelta_same = all(element == tdelta_sec[0] for element in tdelta_sec)
    if is_tdelta_same is False:
        raise ValueError("datetime difference is not equal.")

    # 출력
    if tdelta_sec[0] == 86400:
        print("Time difference : 1 day")
    elif tdelta_sec[0] == 3600:
        print("Time difference : 1 hour")
    elif tdelta_sec[0] == 1800:
        print("Time difference : 30 min")
    elif tdelta_sec[0] == 900:
        print("Time difference : 15 min")
    elif tdelta_sec[0] == 60:
        print("Time difference : 1 min")
    else:
        raise ValueError("Time difference of dataframe should be checked.")

    return tdelta_sec[0]

def get_ymd(df, ymd_coln='ymd'):
    """Get the year-month-day information from dataframe index.

    Parameters
    ----------
    df : dataFrame
        dataFrame with datetime index.

    ymd_coln : str, default='ymd'
        데이터프레임 날짜 인덱스를 "년-월-일"로 추출한 new 컬럼명.

    Returns
    -------
    dfc : dataframe
        original dataframe with year-month-day information.
    """
    dfc = df.copy()

    ymd_set = []
    for i in range(len(dfc)):
        ymd = dfc.index[i].date()
        ymd_set.append(ymd)

    dfc[ymd_coln] = ymd_set
    assert dfc.isnull().sum().sum() == 0, "NaN values exist"

    return dfc

def get_week(df, week_coln='week'):
    """Get the week information from dataframe index.

    Mon: 0, Tue: 1, Wed : 2, ..., Sun : 6.

    Parameters
    ----------
    df : dataFrame
        dataFrame with datetime index.

    week_coln : str, default='week'
        데이터프레임 날짜 인덱스를 "요일"로 추출한 new 컬럼명.

    Returns
    -------
    dfc : dataframe
        original dataframe with week information.
    """
    dfc = df.copy()

    dfc[week_coln] = dfc.index.dayofweek
    assert dfc.isnull().sum().sum() == 0, "NaN values exist"

    return dfc

def holiday_feature(df, holidays, is_onehot=True):
    """Additional input features corresponding holiday.

    Binary holiday marks, each of which can either be 0 or 1.
    1 for holiday, 0 for not.

    Parameters
    ----------
    df : dataFrame
        Dataframe with datetime index. 1-D scalar time series.

    holidays : list or datetime
        Holidays information.

    is_onehot : bool, default=True
        Whether to use 1-of-K coding scheme.
    """
    dfc = df.copy()

    dfc = get_ymd(dfc)  # get the year-month-day information

    dfc['holiday'] = 0
    for h in holidays:
        dfc.loc[dfc['ymd'] == h, ['holiday']] = 1

    assert dfc.isnull().sum().sum() == 0, "NaN values exist"

    holi_categ = dfc['holiday'].values.reshape(-1, 1)

    if not is_onehot:
        return holi_categ
    else:
        return OneHotEncoder().fit_transform(holi_categ).toarray()

def redday_feature(df, holidays, is_onehot=True):
    """Additional input features corresponding saturday, sunday, holiday.

    Red days represent the saturday, sunday, holiday.
    Binary red day marks, each of which can either be 0 or 1.
    1 for red day, 0 for not.

    Parameters
    ----------
    df : dataFrame
        Dataframe with datetime index. 1-D scalar time series.

    holidays : list or datetime
        Holidays information.

    is_onehot : bool, default=True
        Whether to use 1-of-K coding scheme.
    """
    dfc = df.copy()

    dfc = get_ymd(dfc)  # get the year-month-day information
    dfc = get_week(dfc)  # get the week information

    dfc['redday'] = 0
    dfc.loc[dfc['week'] >= 5, ['redday']] = 1

    for h in holidays:
        dfc.loc[dfc['ymd'] == h, ['redday']] = 1

    assert dfc.isnull().sum().sum() == 0, "NaN values exist"

    red_categ = dfc['redday'].values.reshape(-1, 1)

    if not is_onehot:
        return red_categ
    else:
        return OneHotEncoder().fit_transform(red_categ).toarray()
