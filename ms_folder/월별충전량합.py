def get_monthly_ca(
        df,
        start_date: str = '2020-01-01',
        end_date: str = '2021-12-31',
        tsym_coln: str = 'tsym',
        ca_coln: str = 'chgAm',
):
    """입력한 기간 동안의 월별 충전량의 합.

    충전시작날짜를 기준.

    Parameters
    ----------
    start_date : str, default='2020-01-01'
        시작일자.

    end_date : str, default='2021-12-31'
        종료일자.

    tsym_coln : str, default='tsym'
        인수 "df"의 충전시작 "년-월" 정보가 담긴 컬럼명.

    ca_coln : str, default='chgAm'
        인수 "df"의 충전량 정보가 담긴 컬럼명.
        계산된 충전량 합에 대한 new 컬럼명.

    Returns
    -------
    res_df : dataframe
        입력한 기간 동안의 월별 충전량 합이 계산된 dataframe.
        인덱스 : 날짜(년-월-일), 컬럼정보 : 월별 충전량의 합.
    """
    month_set = pd.date_range(start=start_date, end=end_date, freq='M').to_period("M")
    res_set = []
    for m in month_set:
        res = df.loc[df[tsym_coln] == m, ca_coln].sum()
        res_set.append(res)
    res_df = pd.DataFrame(res_set, columns=[ca_coln], index=month_set)

    return res_df
    
    
    def get_ym(
        df,
        tsdt_coln: str = 'tsdt',
        tedt_coln: str = 'tedt',
        tsym_coln: str = 'tsym',
        teym_coln: str = 'teym',
):
    """충전시작시각 및 충전종료시각에서 "년-월" 추출.

    Parameters
    ----------
    tsdt_coln : str, default='tsdt'
        인수 "df"의 충전시작시각 정보가 담긴 컬럼명.

    tedt_coln : str, default='tedt'
        인수 "df"의 충전종료시각 정보가 담긴 컬럼명.

    tsym_coln : str, default='tsym'
        충전시작시각 정보를 "년-월"로 추출한 new 컬럼명.

    teym_coln : str, default='teym'
        충전종료시각 정보를 "년-월"로 추출한 new 컬럼명.

    Examples
    --------
    2021-01-31 12:08:00 -> 2020-01.
    """
    dfc = df.copy()

    start_ym = dfc[tsdt_coln].dt.to_period("M")
    end_ym = dfc[tedt_coln].dt.to_period("M")

    dfc[tsym_coln] = start_ym
    dfc[teym_coln] = end_ym

    assert dfc.isnull().sum().sum() == 0, "NaN 값 존재"

    return dfc
    
