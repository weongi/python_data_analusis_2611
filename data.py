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

# 4-8) 비율 분석
# 개수보다 비율이 중요할 때가 많음
#  - 전체 중에서 각 혈압분류가 몇 %(백분율)인지 확인
bp_ratio = df["혈압분류"].value_counts(normalize=True) * 100  # value_counts() : 각 항목이 몇 번 나왔는지 세는 함수, normalize =True를 넣으면 개수가아닌 비율로 바꺼줌

print("[혈압분류 비율(%)]")
print(bp_ratio.round(1))

# 4-9) 상위 N명 분석(Top 분석)
# 특정 기준에서 상위 데이터 확인
# - BMI 높은 사람 상위 3명
# - 위험군 빠르게파악가능
top3_bmi = df.nlargest(3, 'BMI') # nlargest() : 데이터에서 값이 가장 큰 상위 N개를 뽑는 함수

print("[BMI 상위 3명]")
print(top3_bmi[['환자ID', 'BMI', 'BMI분류']])

# 4-10) 조건 + 그룹 분석
# 특정 조건을 만족하는 데이터만 따로 분석
# - 고혈압 2기 환자만 따로 모아서 성별 평균 비교

ht_df =df[df['혈압분류'] == '고혈압2기']
ht_mean = ht_df.groupby('성별')[['BMI', '공복혈당']].mean()

print(["고혈압2기 환자 - 성별 평균"])
print(ht_mean)

# 4-11) 순위(Rank) 만들기
# 데이터에 순서를 부여
# - BMI 기준 순위 자성

df['BMI순위'] = df['BMI'].rank(ascending = True) # rank() : 데이터 순위를 매기는 함수, ascending = True -> 작은 값부터 1등으로 순위를 부여함

print("[BMI 순위](환자 id 기준 순서)")
print(df[['환자ID', 'BMI', 'BMI순위']])

# BMI 순위별로 출력
print("[BMI 순위](BMI순위 기준 순서)")
df_sorted = df.sort_values('BMI순위') # 그냥 순위별로 정렬
print(df_sorted[['환자ID','BMI','BMI순위']])

# 4-12) 위험 점수 만들기
# 조건을 점수화해서 위험도를 표현
# 조건:
# - 고혈압2기
# - 공복혈당 >= 126
# - BMI >= 25
# - 각 조건마다 1점

df['위험점수'] = (
    (df['혈압분류'] == '고혈압2기').astype(int) + 
    (df['공복혈당'] >= 126).astype(int) + 
    (df['BMI'] >= 25 ).astype(int)
)

print("[위험점수]")
print(df[["환자ID", "위험점수"]]) 

# 5) 시각화

# 5-1) 혈압(SBP/DBP) 분포 히스토그램
# 히스토그램
# - 숫자 데이터가 어느 구간에 얼마나 많이 모여 있는지 보여주는 그래프
# - bin이 크면 클 수록 나뉘는 구간이 촘촘함
# - 구간 수에 따라 그래프 모양이 달라짐

plt.rcParams['font.family'] = 'NanumGothic' # Linux 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False # 음수 기호 깨짐 방지

plt.figure() # 그래프 공간 생성
plt.hist(df['SBP(수축기)'], bins=8) # 구간 설정
plt.title('수축기 혈압(SBP) 분포') # 그래프 위 제목
plt.xlabel('SBP(mmHg)') # 가로축 의미설정
plt.ylabel('빈도') # 세로축 의미 설명
plt.tight_layout() # 글자가 잘리지 않게 자동 정렬
plt.show()

# 5-2) 혈압분류별 평균 SBP 막대그래프
bp_sbp_mean = df.groupby("혈압분류")['SBP(수축기)'].mean().sort_values(ascending=False)
print(bp_sbp_mean)
plt.figure()
plt.bar(bp_sbp_mean.index, bp_sbp_mean.values)
plt.title("평균 SBP 혈압분류")
plt.xlabel("혈압분류 종류")
plt.ylabel('SBP(mmHg)평균')
plt.tight_layout() 
plt.show()

# 5-3) 산점도 : BMI vs SBP
# 산점도 : 두 개의 숫자 변수 관계를 점(dot)으로 표현한 그래프
plt.figure()
plt.scatter(df['BMI'], df['SBP(수축기)'])
plt.title('BMI vs SBP 산점도')
plt.xlabel('BMI')
plt.ylabel('SBP(mmHg)')
plt.tight_layout()
plt.show()

# 5-4) 측정일 기준 SBP 추이(날짜별 평균)
date_sbp = df.groupby('측정일')['SBP(수축기)'].mean()

plt.figure()
plt.plot(date_sbp.index, date_sbp.values) # 선 그래프 생성
plt.title('측정일별 평균 SBP')
plt.xlabel("측정일")
plt.ylabel("SBP(수축기)평균")
plt.xticks(rotation = 30) # x축 눈금 글자 30도 회전
plt.tight_layout()
plt.show()

# 5-5) BMI분류 인원 수(파이차트)
bmi_count = df['BMI분류'].value_counts()
plt.figure()
plt.pie(
    bmi_count.values,
    labels = bmi_count.index, # 인덱스를 라벨에 넣음
    autopct = '%1.1f%%'  # 백분율표시 소수 첫째자리 까지
    
    )
plt.title("BMI 종류 개수")
plt.tight_layout()
plt.show()

# 5-6) 혈압분류별 BMI 분포(박스플롯)
# 박스 플롯 : 평균이 아니라 '분포'를 보여줌 , 이상치 확인가능

df.boxplot(column='BMI', by='혈압분류') # BMI값을 분석하여 혈압분류를 기준으로 그룹을 나눔
plt.title("혈압분류별 BMI")
plt.suptitle("") # 자동으로 생성되는 상단 제목 제거
plt.xlabel("혈압분류")
plt.ylabel("BMI")
plt.tight_layout()
plt.show()

# 5-7) 성별 BMI
plt.figure()
df[df['성별'] == 'M']['BMI'].hist(alpha = 0.5, bins=8) # 히스토그램 으로 보는데 투명도는 0.5로 보겠다잉
df[df['성별'] == 'F']['BMI'].hist(alpha = 0.5, bins=8) # 히스토그램 으로 보는데 투명도는 0.5로 보겠다잉

plt.title('BMI 성별')
plt.xlabel('BMI')
plt.ylabel('성별')
plt.legend() # 범례 표시
plt.tight_layout()
plt.show()

# 5-8) 상관 관계 히트맵
# 히트맵
# - 값의 크기를 색으로 표현하는 그래프
# - 값이 클수록 진한 색
# - 값이 작을수록 연한 색

corr_cols = ['나이', 
             '키_cm', 
             '몸무게_kg', 
             'SBP(수축기)', 
             'DBP(이완기)',
               '공복혈당', 
               'BMI'
             ]
corr = df[corr_cols].corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr_cols)), corr_cols, rotation = 45)
plt.yticks(range(len(corr_cols)), corr_cols, rotation = 45)
plt.title("상관 관계 히트맵")
plt.tight_layout()
plt.show()

# 6-1) 혈압분류 비율
bp_counts = df['혈압분류'].value_counts() # 혈압분류 개수 세기
bp_ratio = (bp_counts / len(df)) * 100 # 비율 계싼

print('[혈압분류 분포(개수)]')
print(bp_ratio.round(1))

# 6-2) 위험 신호 : 공복혈당 평균과 126 이상 비율
mean_glu = df['공복혈당'].mean()
high_glu_ratio = (df['공복혈당'] >= 126).mean() * 100

print(f"[공복혈당 평균 : {mean_glu: .2f} mg/dL]")
print(f"[공복혈당 126 이상 비율 : {high_glu_ratio: .1f}%]")

# 6-3) 운동빈도와 주요 지표 관계
corr_exercise_sbp = df['운동빈도(주)'].corr(df['SBP(수축기)'])
corr_exercise_bmi = df['운동빈도(주)'].corr(df['BMI'])

print(f"[운동빈도 vs SBP 상관계수] : {corr_exercise_sbp: .2f}")
print(f"[운동빈도 vs BMI 상관계수] : {corr_exercise_bmi: .2f}")

# 6-4) 전략/제안 3줄
stratege = [] 

ht2_ratio = (df['혈압분류'] == '고혈압2기').mean() * 100

if ht2_ratio >= 30:
    stratege.append('고혈압2기 비율이 높은 편이라, 생활습관(운동/체중/염분)관련 교육 및 추적 관리가 필요합니다.')

else:
    stratege.append('고혈압2기 비율이 과도하게 높진 않지만,'\
                    '고혈압1기 이상에 대한 예방 안내가 유효합니다.')

# BMI 비만 비율 코멘트
obesity_ratio = (df['BMI분류'] == "비만").mean() * 100
stratege.append(f"BMI 기준 비만 비율이 {obesity_ratio: .1f}%로 확인되어, 체중관리(식습관+운동) 프로그램을 고려할 수 있습니다.")

# 혈당 고위험 비율 코멘트
if high_glu_ratio > 20:
    stratege.append('공복혈당 126 이상 비율이 꽤 있어, 혈당 관련 추가 검사/상담 연계를 고려할 수 있습니다.')
else:
    stratege.append('공복혈당 고위험 비율이 높진 않지만, 식습관/활동량 관련 기본 가이드를 제공하면 좋습니다')

print('[제안 3줄]')
for s in stratege:
    print(f' - {s}')

# 6-5) 위험 점수 기반 위험군 요약
# - 위험점수 2 이상 -> 고위험군
high_risk = df[df['위험점수'] >= 2]
high_risk_ratio = (len(high_risk) / len(df)) * 100

print(f"[고위험군 비율(위험점수 2 이상)] : {high_risk_ratio: .1f}")
print("[고위험군 목록]")
print(high_risk[['환자ID', 'BMI', '공복혈당', '혈압분류', '위험점수']])

# 6-6) 최상위 위험 대상 Top3
# 가장 위험한 사람 추출
# - 위험점수 기준
# - 동일하면 BMI 높은 순으로

top_risk = df.sort_values(['위험점수', 'BMI'], ascending = True).head(3)
print(["최상위 위험 대상 Top 3"])
print(top_risk[['환자ID', '위험점수', 'BMI', '공복혈당', '혈압분류']])

# 데이터 전체를 하나의 결과로 요약
# 기준
# - 고위험군 비율
# - 혈당 위험 비율


if(high_risk_ratio > 30) or (high_glu_ratio > 30):
    health_level = '위험'
elif(high_risk_ratio > 15) or (high_glu_ratio > 15):
    health_level = '주의'
else:
    health_level = '양호'

print(f"[전체 건강 상태 평가] => {health_level}")