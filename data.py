# 의료 기초 데이터 분석 미니 프로젝트
# Numpy + pandas + Matplotlib

# [0]라이브러리 import, 출력 옵션(선택)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 줄  옵션
pd.set_option('display.unicode.east_asian_width', True) 
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 200)

# 1) 데이터 준비
data = {
    '측정일': [
        '2026-02-01','2026-02-01','2026-02-02',
        '2026-02-02','2026-02-03','2026-02-03',
        '2026-02-04','2026-02-04','2026-02-05',
        '2026-02-05','2026-02-06','2026-02-06',
        '2026-02-07','2026-02-07','2026-02-08'
    ],
    '환자ID': [
        'P001','P002','P003','P004','P005',
        'P006','P007','P008','P009','P010',
        'P011','P012','P013','P014','P015'
    ],
    '성별': [
        'F','M','M','F','F','M','M','F',
        'F','M','F','M','M','F','F'
    ],
    '나이': [
        25,31,44,np.nan,52,39,28,33,
        47,61,29,42,np.nan,36,55
    ],
    '키': [
        '162cm','175cm','168cm','160cm','158cm',
        '180cm','172cm','165cm','170cm','177cm',
        '159cm','174cm','169cm','163cm','157cm'
    ],
    '몸무게': [
        '54kg','78kg','70kg','52kg','60kg',
        '90kg','68kg','58kg','73kg','82kg',
        '50kg','76kg','71kg','57kg','66kg'
    ],
    'SBP(수축기)': [
        108,135,142,118,150,130,125,112,
        140,160,105,138,145,120,155
    ],
    'DBP(이완기)': [
        70,88,92,75,95,85,80,72,
        90,100,68,89,96,78,97
    ],
    '심박수': [
        72,84,90,68,95,78,70,75,
        88,92,66,80,86,73,89
    ],
    '공복혈당': [
        92,110,125,98,130,np.nan,105,90,
        118,140,88,112,128,100,np.nan
    ],
    '흡연': [
        'N','Y','Y','N','N','Y','N','N',
        'Y','Y','N','Y','Y','N','N'
    ],
    '운동빈도(주)': [
        3,1,0,2,np.nan,1,4,2,
        1,0,5,1,0,3,np.nan
    ],
}

df = pd.DataFrame(data)


print("==================================")
print("2) 데이터 탐색")
print("==================================")

print(df.head())

print(df.info())
num_cols = ['나이', 'SBP(수축기)', 'DBP(이완기)', '심박수', '공복혈당', '운동빈도(주)']
print(df[num_cols].describe())


print("==================================")
print("3) 데이터 정제 및 변환")
print("==================================")

# 3-1) 타입 변환 : 날짜 문자열 -> datetime
df['측정일'] = pd.to_datetime(df['측정일']) # pd.to_datetime(): 문자열을 날짜/시간 타입으로 변환 
print(df['측정일'].dtype) # 자료형 타입볼때 dtype씀

# 3-2) 단위 제거 : 170cm -> 170
df['키_cm'] = df['키'].str.replace("cm",'',regex = False).astype(int) #키의 cm를 없애고  regex는 정규표현식을 사용할거냐 말거냐 int형으로 바꾸겠다
df['몸무게_kg'] = df['몸무게'].str.replace("kg",'',regex = False).astype(int)
# 원본(키, 몸무게) 컬럼은 '문자열'이라 분석에 불리하므로 숫자형 컬럼으로 바꿔준다잉

# 3-3) 결측치 확인
print("3-3) 결측치 확인")
print(df.isnull().sum()) # 결측치의 합계를 알려줌

# 3-4) 결측치 처리(예시) --> 상황에 따라 0으로 하던가 평균으로 하던가 위에꺼하던가 머 그렇다(데이터 특성 마다 다름)

df['나이'] = df['나이'].fillna(df['나이'].mean())
df['공복혈당'] = df['공복혈당'].fillna(df['공복혈당'].median())
df['운동빈도(주)'] = df['운동빈도(주)'].fillna(0)

print(df.isnull().sum())

# 3-5) 파생 컬럼 생성(변환)
# BMI = 몸무게(kg) / 키(m)^2
df['키_m'] = df['키_cm'] / 100
df['BMI'] = df['몸무게_kg'] / (df['키_m'] ** 2)

# 데이터 확인
print(df[['키_cm', '키_m']].head())
print(df['BMI'].head())

# 맥박압(pp) = SBP - DBP(혈관 탄성/압력 차이를 보는 지표)
df['맥박압_pp'] = df['SBP(수축기)'] - df['DBP(이완기)']
print(df['맥박압_pp'].head())

# 평균동맥압(MAP) = (SBP + 2*DBP) / 3 (기초적인 참고 지표)
df['평균동맥압_MAP'] = (df['SBP(수축기)'] + 2 * df['DBP(이완기)']) / 3
print(df['평균동맥압_MAP'].head())

# 3-6) 범주화(카테고리 만들기) : 혈압 단계 분류
# 정상 :SBP < 120 그리고 DBP < 80
# 주의(상승) : SBP 120~129 그리고 DBP < 80
# 고혈압1기 : SBP 130 ~139 또는 DBP 80~89
# 고혈압2기 : SBP >= 140 또는 DBP >= 90

def bp_category(sbp, dbp):
    if sbp  >= 140 or dbp  >= 90:
        return '고혈압2기'
    
    elif 130 <= sbp <= 139 or 80 <= dbp <= 89:
        return '고혈압1기'

    elif 120 <= sbp <= 129 and dbp < 80:
        return '주의(상승)'
    
    elif sbp < 120 and dbp < 80:
        return '정상'

    else :
        return '기타'

# 일반 함수
# def test(row):
#   return 'hi'

# lambda : 이름 없이 간단하게 작성하는 함수(익명 함수)
# lambda 매개변수 : 반환할_표현식
# lambda row: 'hi'

# apply() : 행 단위로 함수를 적용할 때 자주 사용
df['혈압분류'] = df.apply(lambda row: bp_category(row['SBP(수축기)'], row['DBP(이완기)']), axis = 1)

print("[혈압 분류 결과]")
print(df[['SBP(수축기)', 'DBP(이완기)', '혈압분류']].head())

#3-7) 범주화 : BMI 단계 분류(간단 예시)


def bmi_category(bmi):
    if bmi < 18.5:
        return '저체중'
    elif bmi < 23.0:
        return '정상'
    elif bmi < 25.0:
        return '과체중'
    else:
        return '비만'

df['BMI분류'] = df['BMI'].apply(bmi_category)

print(df['BMI분류'].head())