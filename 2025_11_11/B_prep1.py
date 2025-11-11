import numpy as np
import pandas as pd
import os, re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from typing import List, Optional
from functools import lru_cache

curr_path = os.getcwd()
parent_path = os.path.dirname(curr_path)
DATA_DIR  = os.path.join(parent_path, "data")
OUT_DIR   = os.path.join(curr_path, "output")
Bsample = os.path.join(DATA_DIR, "B_sample.csv")

Bsample = pd.read_csv(Bsample)
Bsample_cols = Bsample.columns
# print(Asample_cols)
# print(Asample.head(5))

# 데이터프레임 설정
df = Bsample

# ---- 데이터 로드 ----
df_cols = df.columns
print(df_cols)
print(df.head(3))




class PreprocessB:
    def __init__(self, dataframe):
        self.df = dataframe
        # 자주 쓰는 컬럼 세트
        self.B1_cols = ['B1-1', 'B1-2', 'B1-3']
        self.B2_cols = ['B2-1', 'B2-2', 'B2-3']
        self.B3_cols = ['B3-1', 'B3-2']
        self.B4_cols = ['B4-1', 'B4-2']
        self.B5_cols = ['B5-1', 'B5-2']
        # 존재 확인(없으면 KeyError)
        _ = self.df[self.B1_cols]
        _ = self.df[self.B2_cols]
        _ = self.df[self.B3_cols]
        _ = self.df[self.B4_cols]
        _ = self.df[self.B5_cols]


        pass
    
    # ---------- ---------- ---------- 공용 유틸 ---------- ---------- ----------



    # ---- 빠른 파서: 콤마구분 문자열 -> np.array (약간의 불량치 방어 포함)


    @staticmethod
    @lru_cache(maxsize=200_000)
    def _parse_cached(colname, raw):
        return np.fromstring(raw, sep=',', dtype=np.float32)

    # 인스턴스 메서드로!
    def _to_array_fast(self, colname, val, dtype=float):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.empty(0, dtype=dtype)
        s = str(val).strip()
        if not s:
            return np.empty(0, dtype=dtype)
        s = s.replace(',,', ',')
        try:
            arr = self._parse_cached(colname, s)
            return arr.astype(dtype, copy=False)
        except Exception:
            toks = [t for t in s.split(',') if t.strip() != ""]
            if not toks:
                return np.empty(0, dtype=dtype)
            arr = np.array(toks, dtype=float if dtype is not int else float)
            if dtype is int:
                mask = np.isfinite(arr)
                arr = arr[mask].astype(int, copy=False)
            else:
                arr = arr[np.isfinite(arr)].astype(dtype, copy=False)
            return arr

    # ---- 빠른 요약자: 리스트/array -> 스칼라 (mu / mu_sigma / mu_range)
    @staticmethod
    def _summarize_array(arr, mode='mu_sigma'):
        if arr is None:
            return np.nan
        x = np.asarray(arr, dtype=np.float32)  # float32로도 충분 (메모리/대역폭↓)
        if x.size == 0:
            return np.nan
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan

        n = x.size
        # 1패스(정확히는 2개 합계 연산)로 통계
        s1 = x.sum(dtype=np.float64)
        s2 = (x * x).sum(dtype=np.float64)
        mu = s1 / n
        # 분산(모분산): E[X^2] - (E[X])^2
        sigma2 = (s2 / n) - (mu * mu)

        if mode == 'mu':
            return float(mu)
        elif mode == 'mu_sigma':
            c = mu * mu
            # 네거티브 제로/수치 잡음 방지
            sigma2 = max(float(sigma2), 0.0)
            return float(np.log1p(c / (sigma2 + c + 1e-12)))
        elif mode == 'mu_sigma_norm':
            c = mu * mu
            # 네거티브 제로/수치 잡음 방지
            sigma2 = max(float(sigma2), 0.0)
            return float(c / (sigma2 + c + 1e-12))
        elif mode == 'mu_range':
            r = float(x.max() - x.min())
            return float(mu / r) if r > 0 else np.nan
        else:
            raise ValueError("mode must be one of {'mu','mu_sigma','mu_range'}")

    # ---------- ---------- ---------- ---------- ---------- ---------- ----------

    # ---------- ---------- ---------- B1: 요약(스칼라 피처) ---------- ---------- ----------
    # ---------- ---------- ---------- ---------- ---------- ---------- ----------
    
    # ---------- B1: 행동반응 측정 검사 ----------

    def B1_features_fast(self, mode: str = 'mu_sigma') -> pd.DataFrame:
        """
        B1 검사 전처리 (고속):

        컬럼:
        - B1-1: 과제1 응답 (1=correct, 2=incorrect)  -> 독립 과제
        - B1-2: 응답시간 (RT)  ※ 0도 그대로 사용, omission 아님
        - B1-3: 과제2 응답 (1~4, change/non-change & 정오답) -> 독립 과제

        피처:

        (A) 전체 RT
          - B1_n_trials
          - B1_rt_all_mean
          - B1_rt_all_{mode}

        (C) B1-3 값(1~4)에 따른 RT
          - B1_rt_t2_k_mean, B1_rt_t2_k_{mode}   (k=1,2,3,4)

        (E) 반응별 총합 카운트 (선형종속 피처 제외)
          - B1_1_1_sum           = count(B1-1 == 1)      (B1-1==2는 제외)
          - B1_3_1_sum, B1_3_2_sum, B1_3_3_sum           (B1-3==4는 제외)

        ※ RT 관련 피처에서, 해당 조합 trial이 하나도 없으면 NaN 대신 -1로 채움.
        """

        to_arr = self._to_array_fast
        summarize = self._summarize_array

        # 결과 버퍼들
        rt_all_mean, rt_all_summ = [], []

        # B1-3 (1~4)에 따른 RT
        rt_t2_mean = {k: [] for k in (1, 2, 3, 4)}
        rt_t2_summ = {k: [] for k in (1, 2, 3, 4)}

        # 정답/오답 그룹 RT
        rt_t2_corr_mean, rt_t2_inc_mean = [], []
        rt_t2_corr_summ, rt_t2_inc_summ = [], []
        rt_t2_corr_minus_inc_mean = []
        rt_t2_corr_minus_inc_summ = []

        # 반응별 총합
        b1_1_1_sum_list = []          # B1-1 == 1
        b1_1_2_sum_list = []          # B1-1 == 2
        b1_3_1_sum_list = []          # B1-3 == 1
        b1_3_2_sum_list = []          # B1-3 == 2
        b1_3_3_sum_list = []          # B1-3 == 3
        b1_3_4_sum_list = []          # B1-3 == 4

        def _mean(arr_like):
            x = np.asarray(arr_like, dtype=np.float32)
            x = x[np.isfinite(x)]
            return float(x.mean()) if x.size > 0 else np.nan

        # 메인 루프
        for b1_1_s, b1_2_s, b1_3_s in self.df[self.B1_cols].itertuples(index=False, name=None):
            # 파싱
            resp1 = to_arr('B1-1', b1_1_s, dtype=np.int8)
            rt    = to_arr('B1-2', b1_2_s, dtype=np.float32)
            resp2 = to_arr('B1-3', b1_3_s, dtype=np.int8)

            if not (resp1.size and rt.size and resp2.size):
                # 결측 로우: trials=0, 나머지 전부 NaN / 0
                rt_all_mean.append(np.nan)
                rt_all_summ.append(np.nan)

                for k in (1, 2, 3, 4):
                    rt_t2_mean[k].append(np.nan)
                    rt_t2_summ[k].append(np.nan)

                rt_t2_corr_mean.append(np.nan)
                rt_t2_inc_mean.append(np.nan)
                rt_t2_corr_summ.append(np.nan)
                rt_t2_inc_summ.append(np.nan)
                rt_t2_corr_minus_inc_mean.append(np.nan)
                rt_t2_corr_minus_inc_summ.append(np.nan)

                b1_1_1_sum_list.append(0)
                b1_1_2_sum_list.append(0)
                b1_3_1_sum_list.append(0)
                b1_3_2_sum_list.append(0)
                b1_3_3_sum_list.append(0)
                b1_3_4_sum_list.append(0)
                continue

            # 길이 맞추기
            n = min(resp1.size, rt.size, resp2.size)
            resp1 = resp1[:n]
            rt    = rt[:n]
            resp2 = resp2[:n]


            # === (E) 반응별 총합 카운트 ===
            # B1-1 == 1
            cnt_1_1 = int((resp1 == 1).sum())
            cnt_1_2 = int((resp1 == 2).sum())
            b1_1_1_sum_list.append(cnt_1_1)
            b1_1_2_sum_list.append(cnt_1_2)

            # B1-3 == 1,2,3  (4는 생략)
            cnt_3_1 = int((resp2 == 1).sum())
            cnt_3_2 = int((resp2 == 2).sum())
            cnt_3_3 = int((resp2 == 3).sum())
            cnt_3_4 = int((resp2 == 4).sum())
            b1_3_1_sum_list.append(cnt_3_1)
            b1_3_2_sum_list.append(cnt_3_2)
            b1_3_3_sum_list.append(cnt_3_3)
            b1_3_4_sum_list.append(cnt_3_4)

            # === (A) 전체 RT 요약 (0도 그대로 사용) ===
            rt_all_mean.append(_mean(rt))
            rt_all_summ.append(summarize(rt, mode=mode))


            # === (C) B1-3 값(1~4)별 RT ===
            for k in (1, 2, 3, 4):
                mask = (resp2 == k)
                r = rt[mask]
                rt_t2_mean[k].append(_mean(r))
                rt_t2_summ[k].append(summarize(r, mode=mode))


            # === (D) 과제2 정답(1,3) vs 오답(2,4) RT ===
            m_corr = (resp2 == 1) | (resp2 == 3)
            m_inc  = (resp2 == 2) | (resp2 == 4)

            rt_corr = rt[m_corr]
            rt_inc  = rt[m_inc]

            corr_mean = _mean(rt_corr)
            inc_mean  = _mean(rt_inc)
            rt_t2_corr_mean.append(corr_mean)
            rt_t2_inc_mean.append(inc_mean)

            corr_summ = summarize(rt_corr, mode=mode)
            inc_summ  = summarize(rt_inc,  mode=mode)
            rt_t2_corr_summ.append(corr_summ)
            rt_t2_inc_summ.append(inc_summ)

            # 차이 피처 (둘 중 하나라도 NaN이면 NaN 유지)
            if np.isnan(corr_mean) or np.isnan(inc_mean):
                rt_t2_corr_minus_inc_mean.append(np.nan)
            else:
                rt_t2_corr_minus_inc_mean.append(corr_mean - inc_mean)

            if np.isnan(corr_summ) or np.isnan(inc_summ):
                rt_t2_corr_minus_inc_summ.append(np.nan)
            else:
                rt_t2_corr_minus_inc_summ.append(corr_summ - inc_summ)

        # DataFrame로 한 번에 정리
        data = {
            'B1_rt_all_mean': rt_all_mean,
            f'B1_rt_all_{mode}': rt_all_summ,

            'B1_1_1_sum': b1_1_1_sum_list,
            'B1_1_2_sum': b1_1_2_sum_list,
            'B1_3_1_sum': b1_3_1_sum_list,
            'B1_3_2_sum': b1_3_2_sum_list,
            'B1_3_3_sum': b1_3_3_sum_list,
            'B1_3_4_sum': b1_3_4_sum_list,
            
            'B1_rt_t2_corr_mean': rt_t2_corr_mean,
            'B1_rt_t2_inc_mean':  rt_t2_inc_mean,
            f'B1_rt_t2_corr_{mode}': rt_t2_corr_summ,
            f'B1_rt_t2_inc_{mode}':  rt_t2_inc_summ,
            'B1_rt_t2_corr_minus_inc_mean': rt_t2_corr_minus_inc_mean,
            f'B1_rt_t2_corr_minus_inc_{mode}': rt_t2_corr_minus_inc_summ,
        }

        # B1-3 값(1~4)별
        for k in (1, 2, 3, 4):
            data[f'B1_rt_t2_{k}_mean'] = rt_t2_mean[k]
            data[f'B1_rt_t2_{k}_{mode}'] = rt_t2_summ[k]

        out = pd.DataFrame(data, index=self.df.index)

        # === RT 관련 피처 NaN → -1 치환 (조합이 한 번도 없었던 경우) ===
        # out 생성 후 맨 마지막에
        rt_cols = [c for c in out.columns 
                if c.startswith('B1_rt_') and 'minus' not in c]

        out[rt_cols] = out[rt_cols].fillna(-1.0)

        return out


    # ---------- B1: 행동반응 측정 검사 ----------

    def B2_features_fast(self, mode: str = 'mu_sigma') -> pd.DataFrame:
        """
        B2 검사 전처리 (고속):

        컬럼:
        - B2-1: 과제1 응답 (1=correct, 2=incorrect)  -> 독립 과제
        - B2-2: 응답시간 (RT)  ※ 0도 그대로 사용, omission 아님
        - B2-3: 과제2 응답 (1~4, change/non-change & 정오답) -> 독립 과제

        피처:

        (A) 전체 RT
          - B2_rt_all_mean
          - B2_rt_all_{mode}

        (C) B2-3 값(1~4)에 따른 RT
          - B2_rt_t2_k_mean, B2_rt_t2_k_{mode}   (k=1,2,3,4)

        (E) 반응별 총합 카운트 (선형종속 피처 제외)
          - B2_1_1_sum           = count(B2-1 == 1)      (B2-1==2는 제외)
          - B2_3_1_sum, B2_3_2_sum, B2_3_3_sum           (B2-3==4는 제외)

        ※ RT 관련 피처에서, 해당 조합 trial이 하나도 없으면 NaN 대신 -1로 채움.
        """

        to_arr = self._to_array_fast
        summarize = self._summarize_array

        # 결과 버퍼들
        rt_all_mean, rt_all_summ = [], []

        # B1-3 (1~4)에 따른 RT
        rt_t2_mean = {k: [] for k in (1, 2, 3, 4)}
        rt_t2_summ = {k: [] for k in (1, 2, 3, 4)}

        # 정답/오답 그룹 RT
        rt_t2_corr_mean, rt_t2_inc_mean = [], []
        rt_t2_corr_summ, rt_t2_inc_summ = [], []
        rt_t2_corr_minus_inc_mean = []
        rt_t2_corr_minus_inc_summ = []

        # 반응별 총합
        b2_1_1_sum_list = []          # B2-1 == 1
        b2_1_2_sum_list = []          # B2-1 == 2
        b2_3_1_sum_list = []          # B2-3 == 1
        b2_3_2_sum_list = []          # B2-3 == 2
        b2_3_3_sum_list = []          # B2-3 == 3
        b2_3_4_sum_list = []          # B2-3 == 4

        def _mean(arr_like):
            x = np.asarray(arr_like, dtype=np.float32)
            x = x[np.isfinite(x)]
            return float(x.mean()) if x.size > 0 else np.nan

        # 메인 루프
        for b2_1_s, b2_2_s, b2_3_s in self.df[self.B2_cols].itertuples(index=False, name=None):
            # 파싱
            resp1 = to_arr('B2-1', b2_1_s, dtype=np.int8)
            rt    = to_arr('B2-2', b2_2_s, dtype=np.float32)
            resp2 = to_arr('B2-3', b2_3_s, dtype=np.int8)

            if not (resp1.size and rt.size and resp2.size):
                # 결측 로우: trials=0, 나머지 전부 NaN / 0
                rt_all_mean.append(np.nan)
                rt_all_summ.append(np.nan)

                for k in (1, 2, 3, 4):
                    rt_t2_mean[k].append(np.nan)
                    rt_t2_summ[k].append(np.nan)

                rt_t2_corr_mean.append(np.nan)
                rt_t2_inc_mean.append(np.nan)
                rt_t2_corr_summ.append(np.nan)
                rt_t2_inc_summ.append(np.nan)
                rt_t2_corr_minus_inc_mean.append(np.nan)
                rt_t2_corr_minus_inc_summ.append(np.nan)

                b2_1_1_sum_list.append(0)
                b2_1_2_sum_list.append(0)
                b2_3_1_sum_list.append(0)
                b2_3_2_sum_list.append(0)
                b2_3_3_sum_list.append(0)
                b2_3_4_sum_list.append(0)
                continue

            # 길이 맞추기
            n = min(resp1.size, rt.size, resp2.size)
            resp1 = resp1[:n]
            rt    = rt[:n]
            resp2 = resp2[:n]


            # === (E) 반응별 총합 카운트 ===
            # B1-1 == 1
            cnt_1_1 = int((resp1 == 1).sum())
            cnt_1_2 = int((resp1 == 2).sum())
            b2_1_1_sum_list.append(cnt_1_1)
            b2_1_2_sum_list.append(cnt_1_2)

            # B1-3 == 1,2,3  (4는 생략)
            cnt_3_1 = int((resp2 == 1).sum())
            cnt_3_2 = int((resp2 == 2).sum())
            cnt_3_3 = int((resp2 == 3).sum())
            cnt_3_4 = int((resp2 == 4).sum())
            b2_3_1_sum_list.append(cnt_3_1)
            b2_3_2_sum_list.append(cnt_3_2)
            b2_3_3_sum_list.append(cnt_3_3)
            b2_3_4_sum_list.append(cnt_3_4)

            # === (A) 전체 RT 요약 (0도 그대로 사용) ===
            rt_all_mean.append(_mean(rt))
            rt_all_summ.append(summarize(rt, mode=mode))


            # === (C) B1-3 값(1~4)별 RT ===
            for k in (1, 2, 3, 4):
                mask = (resp2 == k)
                r = rt[mask]
                rt_t2_mean[k].append(_mean(r))
                rt_t2_summ[k].append(summarize(r, mode=mode))


            # === (D) 과제2 정답(1,3) vs 오답(2,4) RT ===
            m_corr = (resp2 == 1) | (resp2 == 3)
            m_inc  = (resp2 == 2) | (resp2 == 4)

            rt_corr = rt[m_corr]
            rt_inc  = rt[m_inc]

            corr_mean = _mean(rt_corr)
            inc_mean  = _mean(rt_inc)
            rt_t2_corr_mean.append(corr_mean)
            rt_t2_inc_mean.append(inc_mean)

            corr_summ = summarize(rt_corr, mode=mode)
            inc_summ  = summarize(rt_inc,  mode=mode)
            rt_t2_corr_summ.append(corr_summ)
            rt_t2_inc_summ.append(inc_summ)

            # 차이 피처 (둘 중 하나라도 NaN이면 NaN 유지)
            if np.isnan(corr_mean) or np.isnan(inc_mean):
                rt_t2_corr_minus_inc_mean.append(np.nan)
            else:
                rt_t2_corr_minus_inc_mean.append(corr_mean - inc_mean)

            if np.isnan(corr_summ) or np.isnan(inc_summ):
                rt_t2_corr_minus_inc_summ.append(np.nan)
            else:
                rt_t2_corr_minus_inc_summ.append(corr_summ - inc_summ)

        # DataFrame로 한 번에 정리
        data = {
            'B2_rt_all_mean': rt_all_mean,
            f'B2_rt_all_{mode}': rt_all_summ,
            'B2_1_1_sum': b2_1_1_sum_list,
            'B2_1_2_sum': b2_1_2_sum_list,
            'B2_3_1_sum': b2_3_1_sum_list,
            'B2_3_2_sum': b2_3_2_sum_list,
            'B2_3_3_sum': b2_3_3_sum_list,
            'B2_3_4_sum': b2_3_4_sum_list,
            
            'B1_rt_t2_corr_mean': rt_t2_corr_mean,
            'B1_rt_t2_inc_mean':  rt_t2_inc_mean,
            f'B1_rt_t2_corr_{mode}': rt_t2_corr_summ,
            f'B1_rt_t2_inc_{mode}':  rt_t2_inc_summ,
            'B1_rt_t2_corr_minus_inc_mean': rt_t2_corr_minus_inc_mean,
            f'B1_rt_t2_corr_minus_inc_{mode}': rt_t2_corr_minus_inc_summ,
        }

        # B1-3 값(1~4)별
        for k in (1, 2, 3, 4):
            data[f'B2_rt_t2_{k}_mean'] = rt_t2_mean[k]
            data[f'B2_rt_t2_{k}_{mode}'] = rt_t2_summ[k]

        out = pd.DataFrame(data, index=self.df.index)

        # === RT 관련 피처 NaN → -1 치환 (조합이 한 번도 없었던 경우) ===
        # out 생성 후 맨 마지막에
        rt_cols = [c for c in out.columns 
                if c.startswith('B2_rt_') and 'minus' not in c]

        out[rt_cols] = out[rt_cols].fillna(-1.0)

        return out


    # ---------- ---------- ---------- B3: 시각-운동 협응 속도 ---------- ---------- ----------

    def B3_features_fast(self, mode: str = 'mu_sigma') -> pd.DataFrame:
        """
        B3 검사 전처리 (고속):

        - 15 trials
        - B3-1: 응답 (1=correct, 2=incorrect)
        - B3-2: 반응시간 (0도 유효값으로 사용, omission 아님)

        생성 피처:
        - B3_n_correct, B3_n_incorrect, B3_correct_rate

        - B3_rt_all_mean
        - B3_rt_all_{mode}

        - B3_rt_correct_mean
        - B3_rt_correct_{mode}

        - B3_rt_incorrect_mean
        - B3_rt_incorrect_{mode}

        - B3_rt_err_minus_corr  (= incorrect_mean - correct_mean)
        """

        to_arr = self._to_array_fast
        summarize = self._summarize_array


        n_corr_list, n_inc_list = [], []
        corr_rate_list = []

        rt_all_mean, rt_all_summ = [], []
        rt_corr_mean, rt_corr_summ = [], []
        rt_inc_mean,  rt_inc_summ  = [], []

        rt_err_minus_corr = []

        def _mean(arr_like):
            x = np.asarray(arr_like, dtype=np.float32)
            x = x[np.isfinite(x)]
            return float(x.mean()) if x.size > 0 else np.nan

        for b3_1_s, b3_2_s in self.df[self.B3_cols].itertuples(index=False, name=None):
            resp = to_arr('B3-1', b3_1_s, dtype=np.int8)
            rt   = to_arr('B3-2', b3_2_s, dtype=np.float32)

            if not (resp.size and rt.size):
                # 완전 결측 row
                n_corr_list.append(0); n_inc_list.append(0)
                corr_rate_list.append(np.nan)

                rt_all_mean.append(np.nan); rt_all_summ.append(np.nan)
                rt_corr_mean.append(np.nan); rt_corr_summ.append(np.nan)
                rt_inc_mean.append(np.nan);  rt_inc_summ.append(np.nan)
                rt_err_minus_corr.append(np.nan)
                continue

            # 길이 맞추기
            n = min(resp.size, rt.size)
            resp = resp[:n]
            rt   = rt[:n]


            # 마스크
            is_corr = (resp == 1)
            is_inc  = (resp == 2)

            n_corr = int(is_corr.sum())
            n_inc  = int(is_inc.sum())

            n_corr_list.append(n_corr)
            n_inc_list.append(n_inc)

            corr_rate = (n_corr / n) if n > 0 else np.nan
            corr_rate_list.append(corr_rate)

            # RT 요약 (0도 그대로 사용)
            rt_all_mean.append(_mean(rt))
            rt_all_summ.append(summarize(rt, mode=mode))

            rt_c = rt[is_corr]
            rt_i = rt[is_inc]

            m_c = _mean(rt_c)
            s_c = summarize(rt_c, mode=mode)
            m_i = _mean(rt_i)
            s_i = summarize(rt_i, mode=mode)

            rt_corr_mean.append(m_c)
            rt_corr_summ.append(s_c)
            rt_inc_mean.append(m_i)
            rt_inc_summ.append(s_i)

            # 오답-정답 RT 차이
            if np.isnan(m_c) or np.isnan(m_i):
                rt_err_minus_corr.append(np.nan)
            else:
                rt_err_minus_corr.append(m_i - m_c)

        out = pd.DataFrame({
            'B3_n_correct': n_corr_list,
            'B3_n_incorrect': n_inc_list,
            'B3_correct_rate': corr_rate_list,

            'B3_rt_all_mean': rt_all_mean,
            f'B3_rt_all_{mode}': rt_all_summ,

            'B3_rt_correct_mean': rt_corr_mean,
            f'B3_rt_correct_{mode}': rt_corr_summ,

            'B3_rt_incorrect_mean': rt_inc_mean,
            f'B3_rt_incorrect_{mode}': rt_inc_summ,

            'B3_rt_err_minus_corr': rt_err_minus_corr,
        }, index=self.df.index)
        
        # === RT 관련 피처 NaN → -1 치환 (조합이 한 번도 없었던 경우) ===
        # out 생성 후 맨 마지막에
        rt_cols = [c for c in out.columns 
                if c.startswith('B3_rt_') and 'minus' not in c]

        out[rt_cols] = out[rt_cols].fillna(-1.0)

        return out


    # ---------- ---------- ---------- B4: 선택적 주의력 ---------- ---------- ----------

    def B4_features_fast(self, mode: str = 'mu_sigma') -> pd.DataFrame:
        """
        B4 검사 전처리 (고속):

        - 60 trials (congruent 30 / incongruent 30)
        - B4-1: 응답
            1 = congruent correct
            2 = congruent incorrect
            3,5 = incongruent correct
            4,6 = incongruent incorrect
        - B4-2: 반응시간 (0도 그대로 사용)
        """

        to_arr = self._to_array_fast
        summarize = self._summarize_array

        rt_all_mean, rt_all_summ = [], []

        # 조건별
        cg_corr_mean, cg_corr_summ = [], []
        cg_inc_mean,  cg_inc_summ  = [], []
        icg_corr_mean, icg_corr_summ = [], []
        icg_inc_mean,  icg_inc_summ  = [], []

        # 세부 3,4,5,6
        rt_mean = {k: [] for k in (3, 4, 5, 6)}
        rt_summ = {k: [] for k in (3, 4, 5, 6)}

        # 차이 피처
        icg_minus_cg_mean, icg_minus_cg_summ = [], []

        def _mean(x):
            x = np.asarray(x, dtype=np.float32)
            x = x[np.isfinite(x)]
            return float(x.mean()) if x.size > 0 else np.nan

        for b4_1_s, b4_2_s in self.df[self.B4_cols].itertuples(index=False, name=None):
            resp = to_arr('B4-1', b4_1_s, dtype=np.int8)
            rt   = to_arr('B4-2', b4_2_s, dtype=np.float32)

            if not (resp.size and rt.size):
                rt_all_mean.append(np.nan); rt_all_summ.append(np.nan)
                cg_corr_mean.append(np.nan); cg_corr_summ.append(np.nan)
                cg_inc_mean.append(np.nan); cg_inc_summ.append(np.nan)
                icg_corr_mean.append(np.nan); icg_corr_summ.append(np.nan)
                icg_inc_mean.append(np.nan); icg_inc_summ.append(np.nan)
                icg_minus_cg_mean.append(np.nan); icg_minus_cg_summ.append(np.nan)
                for k in (3,4,5,6):
                    rt_mean[k].append(np.nan)
                    rt_summ[k].append(np.nan)
                continue

            n = min(resp.size, rt.size)
            resp = resp[:n]; rt = rt[:n]

            # 전체 RT
            rt_all_mean.append(_mean(rt))
            rt_all_summ.append(summarize(rt, mode=mode))

            # congruent correct/incorrect
            cg_corr = rt[resp == 1]
            cg_inc  = rt[resp == 2]

            # incongruent correct (3,5)
            icg_corr = rt[(resp == 3) | (resp == 5)]
            # incongruent incorrect (4,6)
            icg_inc  = rt[(resp == 4) | (resp == 6)]

            cg_corr_mean.append(_mean(cg_corr))
            cg_corr_summ.append(summarize(cg_corr, mode=mode))
            cg_inc_mean.append(_mean(cg_inc))
            cg_inc_summ.append(summarize(cg_inc, mode=mode))

            icg_corr_mean.append(_mean(icg_corr))
            icg_corr_summ.append(summarize(icg_corr, mode=mode))
            icg_inc_mean.append(_mean(icg_inc))
            icg_inc_summ.append(summarize(icg_inc, mode=mode))

            # 세부 응답 3,4,5,6
            for k in (3,4,5,6):
                mask = (resp == k)
                r = rt[mask]
                rt_mean[k].append(_mean(r))
                rt_summ[k].append(summarize(r, mode=mode))

            # incongruent - congruent 차이
            m_diff = np.nan
            s_diff = np.nan
            if not np.isnan(cg_corr_mean[-1]) and not np.isnan(icg_corr_mean[-1]):
                m_diff = icg_corr_mean[-1] - cg_corr_mean[-1]
            if not np.isnan(cg_corr_summ[-1]) and not np.isnan(icg_corr_summ[-1]):
                s_diff = icg_corr_summ[-1] - cg_corr_summ[-1]
            icg_minus_cg_mean.append(m_diff)
            icg_minus_cg_summ.append(s_diff)

        # DataFrame 정리
        data = {
            'B4_rt_all_mean': rt_all_mean,
            f'B4_rt_all_{mode}': rt_all_summ,

            'B4_rt_cg_corr_mean': cg_corr_mean,
            f'B4_rt_cg_corr_{mode}': cg_corr_summ,
            'B4_rt_cg_incorr_mean': cg_inc_mean,
            f'B4_rt_cg_incorr_{mode}': cg_inc_summ,

            'B4_rt_icg_corr_mean': icg_corr_mean,
            f'B4_rt_icg_corr_{mode}': icg_corr_summ,
            'B4_rt_icg_incorr_mean': icg_inc_mean,
            f'B4_rt_icg_incorr_{mode}': icg_inc_summ,

            'B4_icg_minus_cg_mean': icg_minus_cg_mean,
            f'B4_icg_minus_cg_{mode}': icg_minus_cg_summ,
        }

        for k in (3,4,5,6):
            data[f'B4_rt_{k}_mean'] = rt_mean[k]
            data[f'B4_rt_{k}_{mode}'] = rt_summ[k]

        out = pd.DataFrame(data, index=self.df.index)
        
        # === RT 관련 피처 NaN → -1 치환 (조합이 한 번도 없었던 경우) ===
        # out 생성 후 맨 마지막에
        rt_cols = [c for c in out.columns 
                if c.startswith('B4_rt_') and 'minus' not in c]
        out[rt_cols] = out[rt_cols].fillna(-1.0)
        
        return out


    # ---------- ---------- ---------- B5: 공간 판단 속도 ---------- ---------- ----------

    def B5_features_fast(self, mode: str = 'mu_sigma') -> pd.DataFrame:
        """
        B5 검사 전처리 (고속):

        - 15 trials
        - B5-1: 응답 (1=correct, 2=incorrect)
        - B5-2: 반응시간 (0도 유효값으로 사용, omission 아님)

        생성 피처:
        - B5_n_correct, B5_n_incorrect, B5_correct_rate

        - B5_rt_all_mean
        - B5_rt_all_{mode}

        - B5_rt_correct_mean
        - B5_rt_correct_{mode}

        - B5_rt_incorrect_mean
        - B5_rt_incorrect_{mode}

        - B5_rt_err_minus_corr  (= incorrect_mean - correct_mean)
        """

        to_arr = self._to_array_fast
        summarize = self._summarize_array


        n_corr_list, n_inc_list = [], []
        corr_rate_list = []

        rt_all_mean, rt_all_summ = [], []
        rt_corr_mean, rt_corr_summ = [], []
        rt_inc_mean,  rt_inc_summ  = [], []

        rt_err_minus_corr = []

        def _mean(arr_like):
            x = np.asarray(arr_like, dtype=np.float32)
            x = x[np.isfinite(x)]
            return float(x.mean()) if x.size > 0 else np.nan

        for b5_1_s, b5_2_s in self.df[self.B5_cols].itertuples(index=False, name=None):
            resp = to_arr('B5-1', b5_1_s, dtype=np.int8)
            rt   = to_arr('B5-2', b5_2_s, dtype=np.float32)

            if not (resp.size and rt.size):
                # 완전 결측 row
                n_corr_list.append(0); n_inc_list.append(0)
                corr_rate_list.append(np.nan)

                rt_all_mean.append(np.nan); rt_all_summ.append(np.nan)
                rt_corr_mean.append(np.nan); rt_corr_summ.append(np.nan)
                rt_inc_mean.append(np.nan);  rt_inc_summ.append(np.nan)
                rt_err_minus_corr.append(np.nan)
                continue

            # 길이 맞추기
            n = min(resp.size, rt.size)
            resp = resp[:n]
            rt   = rt[:n]


            # 마스크
            is_corr = (resp == 1)
            is_inc  = (resp == 2)

            n_corr = int(is_corr.sum())
            n_inc  = int(is_inc.sum())

            n_corr_list.append(n_corr)
            n_inc_list.append(n_inc)

            corr_rate = (n_corr / n) if n > 0 else np.nan
            corr_rate_list.append(corr_rate)

            # RT 요약 (0도 그대로 사용)
            rt_all_mean.append(_mean(rt))
            rt_all_summ.append(summarize(rt, mode=mode))

            rt_c = rt[is_corr]
            rt_i = rt[is_inc]

            m_c = _mean(rt_c)
            s_c = summarize(rt_c, mode=mode)
            m_i = _mean(rt_i)
            s_i = summarize(rt_i, mode=mode)

            rt_corr_mean.append(m_c)
            rt_corr_summ.append(s_c)
            rt_inc_mean.append(m_i)
            rt_inc_summ.append(s_i)

            # 오답-정답 RT 차이
            if np.isnan(m_c) or np.isnan(m_i):
                rt_err_minus_corr.append(np.nan)
            else:
                rt_err_minus_corr.append(m_i - m_c)

        out = pd.DataFrame({
            'B5_n_correct': n_corr_list,
            'B5_n_incorrect': n_inc_list,
            'B5_correct_rate': corr_rate_list,

            'B5_rt_all_mean': rt_all_mean,
            f'B5_rt_all_{mode}': rt_all_summ,

            'B5_rt_correct_mean': rt_corr_mean,
            f'B5_rt_correct_{mode}': rt_corr_summ,

            'B5_rt_incorrect_mean': rt_inc_mean,
            f'B5_rt_incorrect_{mode}': rt_inc_summ,

            'B5_rt_err_minus_corr': rt_err_minus_corr,
        }, index=self.df.index)
        
        # === RT 관련 피처 NaN → -1 치환 (조합이 한 번도 없었던 경우) ===
        # out 생성 후 맨 마지막에
        rt_cols = [c for c in out.columns 
                if c.startswith('B5_rt_') and 'minus' not in c]
        out[rt_cols] = out[rt_cols].fillna(-1.0)

        return out
    
    
    # ---------- 공용: 이진 정오답 검사(B6/B7/B8 공통) ----------
    def _binary_correct_features(self, colname: str, prefix: str) -> pd.DataFrame:
        """
        colname: 'B6-1' 처럼 1=correct / 2=incorrect 응답이 콤마로 들어있는 컬럼
        prefix : 'B6', 'B7', 'B8' 처럼 피처 이름 앞에 붙일 접두사
        """
        to_arr = self._to_array_fast

        n_trials_list = []
        n_corr_list   = []
        n_inc_list    = []
        corr_ratio_list = []
        inc_ratio_list  = []
        diff_list       = []

        for (resp_s,) in self.df[[colname]].itertuples(index=False, name=None):
            resp = to_arr(colname, resp_s, dtype=np.int8)

            if resp.size == 0:
                n_trials_list.append(0)
                n_corr_list.append(0)
                n_inc_list.append(0)
                corr_ratio_list.append(np.nan)
                inc_ratio_list.append(np.nan)
                diff_list.append(np.nan)
                continue

            n_trials = int(resp.size)
            n_corr = int((resp == 1).sum())
            n_inc  = int((resp == 2).sum())
            n_trials_list.append(n_trials)
            n_corr_list.append(n_corr)
            n_inc_list.append(n_inc)

            total = n_corr + n_inc
            if total > 0:
                corr_ratio = n_corr / total
                inc_ratio  = n_inc  / total
                diff       = corr_ratio - inc_ratio
            else:
                corr_ratio = np.nan
                inc_ratio  = np.nan
                diff       = np.nan

            corr_ratio_list.append(corr_ratio)
            inc_ratio_list.append(inc_ratio)
            diff_list.append(diff)

        out = pd.DataFrame({
            f'{prefix}_n_correct':       n_corr_list,
            f'{prefix}_n_incorrect':     n_inc_list,
            f'{prefix}_correct_ratio':   corr_ratio_list,
            f'{prefix}_incorrect_ratio': inc_ratio_list,
            f'{prefix}_corr_minus_inc':  diff_list,
        }, index=self.df.index)

        return out

    # ---------- B6/B7/B8 개별 래퍼 ----------
    def B6_features_fast(self, mode: str = 'mu_sigma') -> pd.DataFrame:
        """B6 (시각적 기억력 1) : 1/2 정오답 기반 피처"""
        return self._binary_correct_features(colname='B6', prefix='B6')

    def B7_features_fast(self, mode: str = 'mu_sigma') -> pd.DataFrame:
        """B7 (시각적 기억력 2) : 1/2 정오답 기반 피처"""
        return self._binary_correct_features(colname='B7', prefix='B7')

    def B8_features_fast(self, mode: str = 'mu_sigma') -> pd.DataFrame:
        """B8 (주의 지속능력) : 1/2 정오답 기반 피처"""
        return self._binary_correct_features(colname='B8', prefix='B8')    
    
    

# --- Age 파싱 함수 ---
def parse_age_to_midpoint(x):
    """
    '30a' -> 32, '30b' -> 37 규칙 적용.
    NaN이나 형식이 다르면 np.nan 반환.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    m = re.fullmatch(r'(\d+)\s*([ab])', s)
    if m:
        base = int(m.group(1))
        tag = m.group(2)
        offset = 2 if tag == 'a' else 7  # a:+2, b:+7
        return base + offset
    # 숫자만 들어온 경우(예: '33')는 그대로 정수 변환 시도
    if re.fullmatch(r'\d+', s):
        return int(s)
    return np.nan


B_test = Bsample[:5]
# print(A_test)
# print(A_test.shape)


# --- A_age DataFrame 생성 ---
B_age = pd.DataFrame({
    'B_age': B_test['Age'].apply(parse_age_to_midpoint).astype('Int64')  # 결측 허용 정수 타입
})





pp = PreprocessB(B_test)

B1 = pp.B1_features_fast(mode='mu_sigma')
B2 = pp.B2_features_fast(mode='mu_sigma')
B3 = pp.B3_features_fast(mode='mu_sigma')
B4 = pp.B4_features_fast(mode='mu_sigma')
B5 = pp.B5_features_fast(mode='mu_sigma')
B6 = pp.B6_features_fast(mode='mu_sigma')
B7 = pp.B7_features_fast(mode='mu_sigma')
B8 = pp.B8_features_fast(mode='mu_sigma')

# print(B1.shape)
# print(B1.head(5))
# print(B1.columns)
B1_cols = B1.columns
B2_cols = B2.columns
B3_cols = B3.columns
B4_cols = B4.columns
B5_cols = B5.columns
B6_cols = B6.columns
B7_cols = B7.columns
B8_cols = B8.columns

print(len(B1_cols), len(B2_cols), len(B3_cols), len(B4_cols), len(B5_cols), len(B6_cols), len(B7_cols), len(B8_cols))

    
for col in B6_cols:
    col_0, col_1 = B6[col][0], B6[col][1]
    print(col, f'{col_0}', f'{col_1}')    


for col in B7_cols:
    col_0, col_1 = B7[col][0], B7[col][1]
    print(col, f'{col_0}', f'{col_1}')  
    
for col in B8_cols:
    col_0, col_1 = B8[col][0], B8[col][1]
    print(col, f'{col_0}', f'{col_1}')  
    
    
    
    
    
# # feature DataFrame들을 리스트로 묶기
# feat_list = [A1_feat, A2_feat, A3_feat, A4_feat, A5_feat]
# feat_names = ['A1', 'A2', 'A3', 'A4', 'A5']

# # A1~A5 컬럼명 일괄 변경
# for i, (df, name) in enumerate(zip(feat_list, feat_names), start=1):
#     new_cols = [f'{name}-{j+1}' for j in range(df.shape[1])]
#     df.columns = new_cols

# # A6_7_feat은 수동 처리
# A6_7_feat.columns = ['A6-1', 'A7-1']

# print(A1_feat.columns)
# print(A2_feat.columns)
# print(A3_feat.columns)
# print(A4_feat.columns)
# print(A5_feat.columns)
# print(A6_7_feat.columns)
# print(A8_9_feat.columns)

# print(type(A1_feat))
# print(type(A2_feat))
# print(type(A3_feat))
# print(type(A4_feat))
# print(type(A5_feat))
# print(type(A6_7_feat))


