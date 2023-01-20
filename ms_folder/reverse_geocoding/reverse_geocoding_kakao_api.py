"""
__Reverse Geocoding using Kakao API__

author: 장문석
date : 2023.01.11 (last updated)
PESL 연구원만 사용, 외부에 절대 배포 금지
출처 : https://developers.kakao.com/docs/latest/ko/local/dev-guide#coord-to-district
출처 : https://developers.kakao.com/docs/latest/ko/local/dev-guide#coord-to-address
"""


import numpy as np
import pandas as pd

import json
import requests


def reverse_geocoding_region(lng, lat):
    """역지오코딩(Reverse Geocoding).

    카카오 api를 이용하여 좌표(위도, 경도)로 행정구역정보 받기.
    Convert coordinates to administrative information using the kakao api.

    Parameters
    ----------
    lng : float
        Longitude.

    lat : float
        Latitude.

    Notes
    -----
    Converts the coordinates in the selected coordinate system into the
    administrative and legal-status area information. See more details
    in the ref[1], [2].

    References
    ----------
    .. [1] https://developers.kakao.com/docs/latest/ko/local/dev-guide#coord-to-district
    .. [2] https://developers.kakao.com/
    .. [3] https://cruddbdbdeep.github.io/python/2018/11/02/reverse-geocoding.html
    .. [4] https://mentha2.tistory.com/176
    """

    # base url(or address) without query parameters
    base_url = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"

    # headers with REST API key(personal) to request the query.
    rest_api_key =  ########################################## personal  카카오api에 들어가서 개인별로 rest_api_key 받아야함
    headers = {'Authorization': 'KakaoAK {}'.format(rest_api_key)}

    url = '%s?x=%s&y=%s' % (base_url, lng, lat)

    response = requests.get(url, headers=headers)
    json_dict = json.loads(response.text)

    return json_dict

def reverse_geocoding_address(lng, lat):
    """역지오코딩(Reverse Geocoding).

    카카오 api를 이용하여 좌표(위도, 경도)로 주소(2종류) 변환하기.
    Convert coordinates to the land-lot number address and road name address
    using the kakao api.

    Parameters
    ----------
    lng : float
        Longitude.

    lat : float
        Latitude.

    Notes
    -----
    There are two types of address. See more details in the ref[1], [2].

        Address(지번 주소) : Full land-lot number address.
        Road address(도로명 주소) : Full road name address. Some coordinates may not be
        converted to a road name address.

    References
    ----------
    .. [1] https://developers.kakao.com/docs/latest/ko/local/dev-guide#coord-to-address
    .. [2] https://developers.kakao.com/
    """

    # base url(or address) without query parameters
    base_url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"

    # headers with REST API key(personal) to request the query.
    rest_api_key =  ########################################## personal  카카오api에 들어가서 개인별로 rest_api_key 받아야함
    headers = {'Authorization': 'KakaoAK {}'.format(rest_api_key)}

    url = '%s?x=%s&y=%s' % (base_url, lng, lat)

    response = requests.get(url, headers=headers)
    json_dict = json.loads(response.text)

    return json_dict

def get_region_reverse_geocoding(
        json_dict,
        region_type='H',
        target='region_3depth_name',
):
    """카카오 api를 이용하여 좌표(위도, 경도)로 행정구역정보 받기.

    Parameters
    ----------
    json_dict : dict
        Response result of the auxiliary function 'reverse_geocoding_region()'.

    region_type : {'H', 'B'}, default='H'
        H (administrative) or B (legal-status).

    target : str, default='region_3depth_name'
        Target information such as address_name, region_1depth_name, etc.

    Notes
    -----
    Refer to ref [1] below for detailed information of the parameters.

    References
    ----------
    .. [1] https://developers.kakao.com/docs/latest/ko/local/dev-guide#coord-to-district
    """
    body_name = 'documents'
    body = json_dict[body_name]
    body = pd.DataFrame(body)

    region = body.loc[body['region_type'] == region_type, target].values

    return region

def get_address_reverse_geocoding(
        json_dict,
        address_type='address',
        target='address_name',
):
    """카카오 api를 이용하여 좌표(위도, 경도)로 주소(2종류) 변환하기.

    Parameters
    ----------
    json_dict : dict
        Response result of the auxiliary function 'reverse_geocoding_address()'.

    address_type : {'address', 'road_address'}, default='address'
        address (지번 주소) or road_address (도로명 주소).

    target : str, default='address_name'
        Target information such as address_name, region_1depth_name, etc.

    Notes
    -----
    Refer to ref [1] below for detailed information of the parameters.

    References
    ----------
    .. [1] https://developers.kakao.com/docs/latest/ko/local/dev-guide#coord-to-address
    """
    body_name = 'documents'
    body = json_dict[body_name][0]  # list -> dict
    body = [body[address_type]]  # address_type 선택 후, 데이터프레임 변환을 위하여 dict -> list
    body = pd.DataFrame(body)

    address = body[target].values

    return address