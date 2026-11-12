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

# 4) 분석

# 4) 최고/최저 : BMI가 가장 높은/낮은 사람

max_bmi_row = df.loc[df['BMI'].idxmax()] # 최댓값 위치 반환
min_bmi_row = df.loc[df['BMI'].idxmin()] # 최솟값 위치 반환

print("[BMI 최고]")
print(max_bmi_row[['환자ID', '성별', '나이', '키_cm', '몸무게_kg', 'BMI', 'BMI분류']])

print("[BMI 최저]")
print(min_bmi_row[['환자ID', '성별', '나이', '키_cm', '몸무게_kg', 'BMI', 'BMI분류']])

# 4-2) 혈압 분류별 평균 지표
# groupby() : 데이터를 같은 기준으로 묶는기능 
bp_group = df.groupby('혈압분류')[['SBP(수축기)', 'DBP(이완기)', '맥박압_pp', '평균동맥압_MAP', 'BMI', '운동빈도(주)']].mean()
print(bp_group)

# 4-3) 성별 평균 비교

gender_mean = df.groupby('성별')[['SBP(수축기)', 'DBP(이완기)', '맥박압_pp', '평균동맥압_MAP', 'BMI', '운동빈도(주)']].mean()
print(gender_mean)

# 4-4) 나이대(연령대) 만들기 _ 그룹 평균
# pd.cut() : 숫자를 구간으로 나눠 범주형 컬럼 생성

bins = [0, 29, 39, 49, 59, 120]
labels = ["20대", "30대", "40대", "50대", "60대 이상"]
df['나이대'] = pd.cut(df['나이'], bins = bins, labels = labels, right = True)

age_group = df.groupby('나이대')[['SBP(수축기)', 'DBP(이완기)', '맥박압_pp', '평균동맥압_MAP', 'BMI', '운동빈도(주)']].mean()
print(age_group)

# 4-5) 상관관계(수치형 컬럼)
# corr() : 상관계수 (-1 ~ 1)
# - 1에 가까울수록 같이 오르는 경향(양의 상관)
# - -1에 가까울수록 반대로 움직이는 경향(음의 상관)
# - 0에 가까우면 관계가 거의 없다(무상관에 가까움)

corr_cols = ['나이', '키_cm', '몸무게_kg', 'SBP(수축기)', 'DBP(이완기)', '공복혈당', '운동빈도(주)', 'BMI', '맥박압_pp', '평균동맥압_MAP']
corr_df = df[corr_cols].corr()
print("상관관계")
print(corr_df)
print("상관관계 - 소수점 정리")
print(corr_df.round(2)) # 소수점 2번째자리까지본다잉

# 4-6) 조건 필터링
rist_df = df[(df['혈압분류'] == '고혈압2기') | # | -> or연산자 
             (df['공복혈당'] >= 126) |
             (df['BMI'] >= 25.0)
            ].copy()

print("관심 대상 목록")
print(rist_df[['환자ID', '성별', '나이', 'SBP(수축기)', 'DBP(이완기)', '혈압분류', '공복혈당', 'BMI', 'BMI분류', '운동빈도(주)']])

# 4-7) 교차분석 : 성별 + 혈압분류
# 두개의 기준을 동시에 비교하는 분석
# 성별에 따라 혈압 분포가 어떻게 다른지 확인
# 단순 평균보다 더 분포를 볼 수 있음

# crosstab() : 두 개 이상의 범주형 변수의 관계를 표로 정리하는 함수
cross = pd.crosstab(df['성별'], df['혈압분류'])
print(cross)
