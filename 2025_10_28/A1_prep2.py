import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

curr_path = os.getcwd()
parent_path = os.path.dirname(curr_path)
DATA_DIR  = os.path.join(parent_path, "data")
OUT_DIR   = os.path.join(curr_path, "output")
Asample = os.path.join(DATA_DIR, "A_sample.csv")

Asample = pd.read_csv(Asample)
Asample_cols = Asample.columns
# print(Asample_cols)
# print(Asample.head(5))

# 데이터프레임 설정
df = Asample

# ---- 데이터 로드 ----
A1_cols = ['A1-1', 'A1-2', 'A1-3', 'A1-4']
A1 = df[A1_cols]

# # ---- 전처리 함수 정의 ----
# def A1_parse_and_split(row):
#     """A1-1 기준으로 왼쪽(1), 오른쪽(2)에 해당하는 A1-2, A1-3, A1-4 리스트 추출"""
#     # 각 항목을 문자열→리스트로 변환
#     a1_1 = np.array([int(x) for x in str(row['A1-1']).split(',')])
#     a1_2 = np.array([int(x) for x in str(row['A1-2']).split(',')])
#     a1_3 = np.array([int(x) for x in str(row['A1-3']).split(',')])
#     a1_4 = np.array([float(x) for x in str(row['A1-4']).split(',')])

#     # 왼쪽(1)과 오른쪽(2) 위치 인덱스
#     idx_left = np.where(a1_1 == 1)[0]
#     idx_right = np.where(a1_1 == 2)[0]

#     left_A1_3 = a1_3[idx_left].tolist()
#     right_A1_3 = a1_3[idx_right].tolist()

#     slow_rt   = a1_4[a1_2 == 1].tolist()
#     normal_rt = a1_4[a1_2 == 2].tolist()
#     fast_rt   = a1_4[a1_2 == 3].tolist()

#     return pd.Series({
#         'left_A1_3': left_A1_3,
#         'right_A1_3': right_A1_3,
#         'slow_rt' : slow_rt,
#         'normal_rt' : normal_rt,
#         'fast_rt' : fast_rt
#     })
    
def A1_parse_and_split(row):
    """
    A1-1(방향:1=왼,2=오) × A1-2(속도:1=느림,2=중간,3=빠름)
    총 6가지 조건에 해당하는 A1-3(반응)·A1-4(반응시간)을 리스트로 추출
    """
    # 문자열 → numpy array 변환
    a1_1 = np.array([int(x) for x in str(row['A1-1']).split(',')])
    a1_2 = np.array([int(x) for x in str(row['A1-2']).split(',')])
    a1_3 = np.array([int(x) for x in str(row['A1-3']).split(',')])
    a1_4 = np.array([float(x) for x in str(row['A1-4']).split(',')])

    # 6개 조합에 대한 인덱스 필터
    def idx(direction, speed):
        return np.where((a1_1 == direction) & (a1_2 == speed))[0]

    # 왼쪽(1)
    idx_L_S, idx_L_N, idx_L_F = idx(1, 1), idx(1, 2), idx(1, 3)
    # 오른쪽(2)
    idx_R_S, idx_R_N, idx_R_F = idx(2, 1), idx(2, 2), idx(2, 3)
    
    idx_left = np.where(a1_1 == 1)[0]
    idx_right = np.where(a1_1 == 2)[0]
    
    left_slow_rt = a1_4[idx_L_S].tolist()
    left_normal_rt = a1_4[idx_L_N].tolist()
    left_fast_rt = a1_4[idx_L_F].tolist()
    
    right_slow_rt = a1_4[idx_R_S].tolist()
    right_normal_rt = a1_4[idx_R_N].tolist()
    right_fast_rt = a1_4[idx_R_F].tolist()
    
    left_res = a1_3[idx_left].tolist()
    right_res = a1_3[idx_right].tolist()
    
    res = pd.Series({
        'left_slow_rt':   left_slow_rt,
        'left_normal_rt': left_normal_rt,
        'left_fast_rt':   left_fast_rt,
        'right_slow_rt':   right_slow_rt,
        'right_normal_rt': right_normal_rt,
        'right_fast_rt':   right_fast_rt,
        'left_res':   left_res,
        'right_res': right_res
    })   

    # 각 조건별 리스트 추출
    return res
    

# ---- 행별로 적용 ----
A1_split_df = A1.apply(A1_parse_and_split, axis=1)

# ---- 결과 병합 ----
A1_split = pd.concat([A1_split_df], axis=1)

# print(A1_split.head(3))
# print(A1_split.columns)
for col in A1_split.columns:
    print(A1_split[col][0])

print(type(A1_split['left_slow_rt']))
# print(A1_split['left_slow_rt'])
print(A1_split['left_slow_rt'].shape)
print(type(A1_split['left_slow_rt'][0]))
# scaler = StandardScaler()
# left_slow_rt_scaled = scaler.fit_transform(A1_split['left_slow_rt'])
# print(left_slow_rt_scaled)



def A1_features(df_split, method = "zscore"):
    """A1-3 기준 비정상 반응 피처 생성"""
    df_feat = pd.DataFrame()
    df_feat['left_A1_3_sum'] = df_split['left_res'].apply(lambda x: sum(x))
    df_feat['right_A1_3_sum'] = df_split['right_res'].apply(lambda x: sum(x))
    df_feat['A1_3_total_abnormal'] = (
        df_feat['left_A1_3_sum'] + df_feat['right_A1_3_sum']
    )
    df_feat['A1_3_abnormal_ratio'] = df_feat['A1_3_total_abnormal'] / 18

    return df_feat


# 예시: A1_split이 parse_and_split 결과라고 가정
A1_feat_ab = A1_features(A1_split)
print(A1_feat_ab.columns)
print(A1_feat_ab.head())




##### 여기부터 이어서 할 것 : A1_features함수 통합하기

def summarize_rt_features(df, cols=None, mode='mu_sigma'):
    """
    cols: 리스트 컬럼 이름들
    mode: 'mu' | 'mu_sigma' | 'mu_range'
    """
    if cols is None:
        cols = ['left_slow_rt','left_normal_rt','left_fast_rt',
                'right_slow_rt','right_normal_rt','right_fast_rt']

    rows = []
    for _, row in df.iterrows():
        summary = {}
        for col in cols:
            arr = np.array(row[col], dtype=float)
            if len(arr) == 0:
                summary[col + '_summary'] = np.nan
                continue
            mu = np.mean(arr)
            if mode == 'mu':
                val = mu
            elif mode == 'mu_sigma':
                sigma = np.std(arr)
                val = mu / sigma if sigma != 0 else np.nan
            elif mode == 'mu_range':
                r = np.max(arr) - np.min(arr)
                val = mu / r if r != 0 else np.nan
            summary[col + '_summary'] = val
        rows.append(summary)
    return pd.DataFrame(rows, index=df.index)