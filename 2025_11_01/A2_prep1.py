import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from typing import List, Optional

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
# df_cols = df.columns
# print(df_cols)
# print(df.head(3))


A1_cols = ['A1-1', 'A1-2', 'A1-3', 'A1-4']
A2_cols = ['A2-1', 'A2-2', 'A2-3', 'A2-4']
A3_cols = ['A3-1', 'A3-2', 'A3-3', 'A3-4', 'A3-5', 'A3-6', 'A3-7']


A1 = df[A1_cols]
A2 = df[A2_cols]
A3 = df[A3_cols]





class Preprocess:
    def __init__(self, dataframe):
        self.df = dataframe
        # 자주 쓰는 컬럼 세트
        self.A1_cols = ['A1-1', 'A1-2', 'A1-3', 'A1-4']
        self.A2_cols = ['A2-1', 'A2-2', 'A2-3', 'A2-4']
        self.A3_cols = ['A3-1', 'A3-2', 'A3-3', 'A3-4', 'A3-5', 'A3-6', 'A3-7']
        # 존재 확인(없으면 KeyError)
        _ = self.df[self.A1_cols]
        _ = self.df[self.A2_cols]
        _ = self.df[self.A3_cols]


        pass
    
    # ---------- ---------- ---------- 공용 유틸 ---------- ---------- ----------

    # 콤마 문자열 → np.array 변환 유틸 (공백/누락/타입 안전)
    @staticmethod
    def _to_array(val, dtype=float):
        if pd.isna(val):
            return np.array([], dtype=dtype)
        s = str(val).strip()
        if s == "":
            return np.array([], dtype=dtype)
        toks = [t.strip() for t in s.split(',') if t.strip() != ""]
        try:
            return np.array(toks, dtype=dtype)
        except Exception:
            # 숫자로 안 바뀌는 값이 섞여 있으면 안전하게 변환 시도
            casted = []
            for t in toks:
                try:
                    casted.append(dtype(t))
                except Exception:
                    # 무효값은 NaN 처리
                    casted.append(np.nan)
            return np.array(casted, dtype=float)

    @staticmethod
    # 요약 함수: 리스트(또는 배열) -> 스칼라
    def _summarize_list(x, mode='mu'):
        """
        x: list-like of numbers
        mode: 'mu' | 'mu_sigma' | 'mu_range'
        """
        if x is None:
            return np.nan
        arr = np.asarray(list(x), dtype=float)  # list/tuple/np.array 모두 허용
        # if arr.size == 0 or np.all(~np.isfinite(arr)):
        #     return np.nan
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan

        mu = arr.mean()
        c = mu ** 2
        if mode == 'mu':
            return float(mu)
        elif mode == 'mu_sigma':
            sigma = arr.std(ddof=0)
            sigma2 = arr.var(ddof=0)
            # return float(mu / sd) if sd > 0 else np.nan
            return np.log1p(c / (sigma2 + c))
            # 또는 return mu / (sigma + 0.05 * mu)
        elif mode == 'mu_range':
            r = arr.max() - arr.min()
            return float(mu / r) if r > 0 else np.nan
        else:
            raise ValueError("mode must be one of {'mu','mu_sigma','mu_range'}")

    # ---------- ---------- ---------- ---------- ---------- ---------- ----------

    # ---------- ---------- ---------- A1: 요약(스칼라 피처) ---------- ---------- ----------
    # --- 핵심: A1 한 행(row)을 6셀 리스트+좌/우 반응 리스트로 분해 ---
    def A1_parse_and_split_row(self, row: pd.Series) -> pd.Series:
        """
        A1-1(방향:1=왼,2=오) × A1-2(속도:1=느림,2=중간,3=빠름)
        총 6가지 조건에 해당하는 A1-3(반응)·A1-4(반응시간)을 리스트로 추출
        """
        # 명시적으로 변수에 할당
        a1_1 = self._to_array(row['A1-1'], dtype=int)    # 방향: 1=왼, 2=오
        a1_2 = self._to_array(row['A1-2'], dtype=int)    # 속도: 1=느림,2=중간,3=빠름
        a1_3 = self._to_array(row['A1-3'], dtype=int)    # 반응: 0=N(정상),1=Y(비정상)
        a1_4 = self._to_array(row['A1-4'], dtype=float)  # 반응시간(혹은 편차)

        # # 길이 불일치 방지: 가장 긴 길이에 맞추어 잘라내기/패딩(여기선 잘라내기)
        # n = int(max(map(len, (a1_1, a1_2, a1_3, a1_4))) if any(len(x)>0 for x in (a1_1,a1_2,a1_3,a1_4)) else 0)
        # def _cut(v, n):
        #     return v[:n] if v.size >= n else np.pad(v, (0, n - v.size), constant_values=np.nan)
        # a1_1, a1_2, a1_3, a1_4 = _cut(a1_1,n), _cut(a1_2,n), _cut(a1_3,n), _cut(a1_4,n)

        # 인덱스 헬퍼 (방향·속도 동시조건)
        def idx(direction, speed):
            return np.where((a1_1 == direction) & (a1_2 == speed))[0]

        # 좌/우×속도 = 6셀 인덱스
        idx_L_S, idx_L_N, idx_L_F = idx(1,1), idx(1,2), idx(1,3)
        idx_R_S, idx_R_N, idx_R_F = idx(2,1), idx(2,2), idx(2,3)

        # 좌/우 전체 인덱스(반응 리스트용)
        idx_left, idx_right  = np.where(a1_1 == 1)[0], np.where(a1_1 == 2)[0]

        res =  pd.Series({
            # 6셀 RT 리스트
            'left_slow_rt':    a1_4[idx_L_S].tolist(),
            'left_normal_rt':  a1_4[idx_L_N].tolist(),
            'left_fast_rt':    a1_4[idx_L_F].tolist(),
            'right_slow_rt':   a1_4[idx_R_S].tolist(),
            'right_normal_rt': a1_4[idx_R_N].tolist(),
            'right_fast_rt':   a1_4[idx_R_F].tolist(),
            # 좌/우 반응(비정상 여부 집계용)
            'left_res':  a1_3[idx_left].tolist(),
            'right_res': a1_3[idx_right].tolist()
        })
        return res
    
    # DataFrame 전체에 적용하여 A1 분해 결과 반환
    def A1_parse_and_split(self) -> pd.DataFrame:
        return self.df[self.A1_cols].apply(self.A1_parse_and_split_row, axis=1)    


    def A1_features(self,
        df_split: Optional[pd.DataFrame] = None,
        cols: Optional[List[str]] = None,
        mode: str = 'mu_sigma',
        abnormal_denominator: int = 18,
        ) -> pd.DataFrame:
        """
        - df_split이 None이면 내부에서 A1_parse_and_split() 먼저 실행
        - cols: 요약할 RT 리스트 컬럼(기본 6셀)
        - mode: 'mu' | 'mu_sigma' | 'mu_range'
        - abnormal_denominator: 비정상 비율 분모(보통 18 trials)
        """
        if df_split is None:
            df_split = self.A1_parse_and_split()

        if cols is None:
            cols = ['left_slow_rt','left_normal_rt','left_fast_rt',
                'right_slow_rt','right_normal_rt','right_fast_rt']

        out = pd.DataFrame(index=df_split.index)

        # (1) 비정상 반응 집계
        out['left_A1_3_sum']  = df_split['left_res'].apply(lambda x: int(sum(x)) if x is not None else np.nan)
        out['right_A1_3_sum'] = df_split['right_res'].apply(lambda x: int(sum(x)) if x is not None else np.nan)
        out['A1_3_abnormal_ratio'] = (out['left_A1_3_sum'] + out['right_A1_3_sum']) / abnormal_denominator

        # (2) 6셀 RT 리스트 요약 (각 리스트 -> 스칼라 1개)
        for c in cols:
            out[c + f'_{mode}'] = df_split[c].apply(self._summarize_list, mode=mode)

        return out    
    # ---------- ---------- ---------- ---------- ---------- ---------- ----------
    
    # --- 핵심: A2 한 행(row)을 Condition1 x Condition2 별로 반응, 반응시간 리스트로 분해 ---
    def A2_parse_and_split_row(self, row: pd.Series) -> pd.Series:
        """
        A2-1(방향:1=왼,2=오) × A2-2(속도:1=느림,2=중간,3=빠름)
        총 6가지 조건에 해당하는 A2-3(반응)·A2-4(반응시간)을 리스트로 추출
        """
        # 명시적으로 변수에 할당
        print("A2_test --- --- ---")
        a2_1 = self._to_array(row['A2-1'], dtype=int)    # 방향: 1=왼, 2=오
        a2_2 = self._to_array(row['A2-2'], dtype=int)    # 속도: 1=느림,2=중간,3=빠름
        a2_3 = self._to_array(row['A2-3'], dtype=int)    # 반응: 0=N(정상),1=Y(비정상)
        a2_4 = self._to_array(row['A2-4'], dtype=float)  # 반응시간(혹은 편차)
        print("A2_test --- --- --- done")    


        # 인덱스 헬퍼 (속도1·속도2 동시조건)
        def idx(speed1, speed2):
            return np.where((a2_1 == speed1) & (a2_2 == speed2))[0]

        # A2-1 × A2-2 = 9셀 인덱스
        idx_S_S, idx_S_N, idx_S_F = idx(1,1), idx(1,2), idx(1,3)
        idx_N_S, idx_N_N, idx_N_F = idx(2,1), idx(2,2), idx(2,3)
        idx_F_S, idx_F_N, idx_F_F = idx(3,1), idx(3,2), idx(3,3)

        # --- consistency 인덱스 (일치/불일치) ---
        idx_match    = np.where(a2_1 == a2_2)[0]   # (1,1), (2,2), (3,3)
        idx_mismatch = np.where(a2_1 != a2_2)[0]   # 나머지 6개 조합

        res =  pd.Series({
            # 9셀 RT 리스트
            'slow_slow_rt':    a2_4[idx_S_S].tolist(),
            'slow_normal_rt':  a2_4[idx_S_N].tolist(),
            'slow_fast_rt':    a2_4[idx_S_F].tolist(),
            'normal_slow_rt':   a2_4[idx_N_S].tolist(),
            'normal_normal_rt': a2_4[idx_N_N].tolist(),
            'normal_fast_rt':   a2_4[idx_N_F].tolist(),
            'fast_slow_rt':   a2_4[idx_F_S].tolist(),
            'fast_normal_rt': a2_4[idx_F_N].tolist(),
            'fast_fast_rt':   a2_4[idx_F_F].tolist(),

            # --- consistency 기반 리스트/집계 ---
            'match_rt':          a2_4[idx_match].tolist(),      # 일치 trial의 RT들
            'mismatch_rt':       a2_4[idx_mismatch].tolist(),   # 불일치 trial의 RT들
            'match_res_sum':     int(np.nansum(a2_3[idx_match])) if idx_match.size>0 else 0,
            'mismatch_res_sum':  int(np.nansum(a2_3[idx_mismatch])) if idx_mismatch.size>0 else 0,

            
            # 전체 반응 (비정상 여부 집계용)
            'res':  sum(a2_3)
        })
        return res
    
    # DataFrame 전체에 적용하여 A2 분해 결과 반환
    def A2_parse_and_split(self) -> pd.DataFrame:
        return self.df[self.A2_cols].apply(self.A2_parse_and_split_row, axis=1) 
    
       
    def A2_features(self,
        df_split: Optional[pd.DataFrame] = None,
        cols: Optional[List[str]] = None,
        mode: str = 'mu_sigma',
        abnormal_denominator: int = 18,
        ) -> pd.DataFrame:
        """
        - df_split이 None이면 내부에서 A2_parse_and_split() 먼저 실행
        - cols: 요약할 RT 리스트 컬럼(기본 9셀)
        - mode: 'mu' | 'mu_sigma' | 'mu_range'
        - abnormal_denominator: 비정상 비율 분모(보통 18 trials)
        """
        if df_split is None:
            df_split = self.A2_parse_and_split()

        if cols is None:
            cols = ['slow_slow_rt', 'slow_normal_rt', 'slow_fast_rt',
            'normal_slow_rt', 'normal_normal_rt', 'normal_fast_rt',
            'fast_slow_rt', 'fast_normal_rt', 'fast_fast_rt']

        out = pd.DataFrame(index=df_split.index)

        # (1) 비정상 반응 집계
        out['A2_3_sum']  = df_split['res']
        out['A2_3_match_res_sum'] = df_split['match_res_sum']
        out['A2_3_mismatch_res_sum'] = df_split['mismatch_res_sum']
        out['A2_3_abnormal_ratio'] = out['A2_3_sum'] / abnormal_denominator

        # (2) 6셀 RT 리스트 요약 (각 리스트 -> 스칼라 1개)
        for c in cols:
            out[c + f'_{mode}'] = df_split[c].apply(self._summarize_list, mode=mode)

        return out    
    # ---------- ---------- ---------- ---------- ---------- ---------- ----------







pp = Preprocess(Asample)           # Asample: pd.DataFrame
A1_split = pp.A1_parse_and_split() # 6개 리스트 + 좌/우 반응 리스트
A1_feat  = pp.A1_features(A1_split, mode='mu_sigma')  # 혹은 mode='mu','mu_range'

A2_split =pp.A2_parse_and_split()
A2_feat = pp.A2_features(A2_split, mode = 'mu_sigma')

print(A1_split.head())
print(A1_feat.head())

print(A2.head())
print(A2['A2-3'])

print(A2_split.head())
print(A2_feat.head())


print(A3.head())