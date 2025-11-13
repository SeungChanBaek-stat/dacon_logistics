import numpy as np
import pandas as pd
import os, re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from typing import List, Optional
from functools import lru_cache

# curr_path = os.getcwd()
# parent_path = os.path.dirname(curr_path)
# DATA_DIR  = os.path.join(parent_path, "data")
# OUT_DIR   = os.path.join(curr_path, "output")
# Asample = os.path.join(DATA_DIR, "A_sample.csv")

# Asample = pd.read_csv(Asample)
# Asample_cols = Asample.columns
# # print(Asample_cols)
# # print(Asample.head(5))

# # 데이터프레임 설정
# df = Asample


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

class PreprocessA:
    def __init__(self, dataframe):
        self.df = dataframe
        # 자주 쓰는 컬럼 세트
        self.A1_cols = ['A1-1', 'A1-2', 'A1-3', 'A1-4']
        self.A2_cols = ['A2-1', 'A2-2', 'A2-3', 'A2-4']
        self.A3_cols = ['A3-1', 'A3-2', 'A3-3', 'A3-4', 'A3-5', 'A3-6', 'A3-7']
        self.A4_cols = ['A4-1','A4-2','A4-3','A4-4','A4-5']
        self.A5_cols = ['A5-1','A5-2','A5-3']
        self.A6_7_cols = ['A6-1', 'A7-1']
        # 존재 확인(없으면 KeyError)
        _ = self.df[self.A1_cols]
        _ = self.df[self.A2_cols]
        _ = self.df[self.A3_cols]
        _ = self.df[self.A4_cols]
        _ = self.df[self.A5_cols]
        _ = self.df[self.A6_7_cols]


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
        elif mode == 'mu_range':
            r = float(x.max() - x.min())
            return float(mu / r) if r > 0 else np.nan
        else:
            raise ValueError("mode must be one of {'mu','mu_sigma','mu_range'}")

    # ---------- ---------- ---------- ---------- ---------- ---------- ----------

    # ---------- ---------- ---------- A1: 요약(스칼라 피처) ---------- ---------- ----------
    # ---------- ---------- ---------- ---------- ---------- ---------- ----------

    # ---- A1: 파싱+분해+요약을 '한 번'에 끝내는 고속 피처라이저
    def A1_features_fast(self, mode='mu_sigma', abnormal_denominator=18):
        """
        기존 A1_parse_and_split() + A1_features()를 합친 고속 버전.
        반환 컬럼 이름은 기존 A1_features와 동일한 접미사 유지:
        left_slow_rt_{mode}, ..., right_fast_rt_{mode}, A1_3_abnormal_ratio, ...
        """
        # 결과 누적 버퍼(리스트) – 마지막에 DataFrame으로 캐스팅
        left_sum, right_sum, abn_ratio = [], [], []

        l_s, l_n, l_f = [], [], []
        r_s, r_n, r_f = [], [], []

        # 로컬 바인딩으로 속도 미세 개선
        to_arr = self._to_array_fast
        summarize = self._summarize_array

        # 판다스 apply 대신 itertuples() 사용 → 오버헤드 감소
        for row in self.df[self.A1_cols].itertuples(index=False, name=None):
            a1_1 = to_arr('A1-1', row[0], dtype=np.int8)     # 방향 1/2
            a1_2 = to_arr('A1-2', row[1], dtype=np.int8)     # 속도 1/2/3
            a1_3 = to_arr('A1-3', row[2], dtype=np.int8)     # 반응 0/1
            a1_4 = to_arr('A1-4', row[3], dtype=np.float32)   # RT

            # 길이 안 맞아도 공통 인덱스에서만 보겠다는 전략: 최소 길이로 컷
            if not (a1_1.size and a1_2.size and a1_3.size and a1_4.size):
                # 전부 NaN/0 처리
                left_sum.append(0); right_sum.append(0); abn_ratio.append(np.nan)
                for buf in (l_s,l_n,l_f,r_s,r_n,r_f):
                    buf.append(np.nan)
                continue

            n = min(a1_1.size, a1_2.size, a1_3.size, a1_4.size)
            a1_1 = a1_1[:n]; a1_2 = a1_2[:n]; a1_3 = a1_3[:n]; a1_4 = a1_4[:n]

            # 마스크 (인덱스 배열 만들지 말고 바로 불리언 마스크로 슬라이싱)
            mL = (a1_1 == 1); mR = (a1_1 == 2)
            mS = (a1_2 == 1); mN = (a1_2 == 2); mF = (a1_2 == 3)

            # 비정상합(=1의 개수) – 좌/우
            l_sum = int(a1_3[mL].sum()) if mL.any() else 0
            r_sum = int(a1_3[mR].sum()) if mR.any() else 0

            left_sum.append(l_sum)
            right_sum.append(r_sum)
            abn_ratio.append((l_sum + r_sum) / abnormal_denominator)

            # 6셀 RT 요약
            l_s.append(summarize(a1_4[mL & mS], mode))
            l_n.append(summarize(a1_4[mL & mN], mode))
            l_f.append(summarize(a1_4[mL & mF], mode))

            r_s.append(summarize(a1_4[mR & mS], mode))
            r_n.append(summarize(a1_4[mR & mN], mode))
            r_f.append(summarize(a1_4[mR & mF], mode))

        # 한 번에 DF로
        out = pd.DataFrame({
            'left_A1_3_sum': left_sum,
            'right_A1_3_sum': right_sum,
            'A1_3_abnormal_ratio': abn_ratio,
            f'left_slow_rt_{mode}':   l_s,
            f'left_normal_rt_{mode}': l_n,
            f'left_fast_rt_{mode}':   l_f,
            f'right_slow_rt_{mode}':  r_s,
            f'right_normal_rt_{mode}':r_n,
            f'right_fast_rt_{mode}':  r_f,
        }, index=self.df.index)

        return out
    
    def A2_features_fast(self, mode='mu_sigma', abnormal_denominator=18.0):
        """
        A2-1(speed1: 1/2/3) × A2-2(speed2: 1/2/3) 9셀 + match/mismatch 요약을
        한 번의 패스에서 계산하는 고속 버전.

        반환 컬럼:
        - A2_3_abnormal_ratio
        - match_rt_{mode}, mismatch_rt_{mode}
        - slow_slow_rt_{mode}, slow_normal_rt_{mode}, slow_fast_rt_{mode},
            normal_slow_rt_{mode}, normal_normal_rt_{mode}, normal_fast_rt_{mode},
            fast_slow_rt_{mode}, fast_normal_rt_{mode}, fast_fast_rt_{mode}
        """
        # 결과 버퍼
        abn_ratio = []

        match_rt, mismatch_rt = [], []

        ss, sn, sf = [], [], []
        ns, nn, nf = [], [], []
        fs, fn, ff = [], [], []

        # 유틸(있으면 fast/array 버전, 없으면 기존 것 사용)
        to_arr = self._to_array_fast if hasattr(self, "_to_array_fast") else self._to_array
        summarize = getattr(self, "_summarize_array", None)
        if summarize is None:
            def summarize(arr, m=mode):
                return self._summarize_list(arr, mode=m)

        # 메인 루프: apply(axis=1) 대신 itertuples → 오버헤드 감소
        for row in self.df[self.A2_cols].itertuples(index=False, name=None):
            a2_1 = to_arr('A2-1', row[0], dtype=np.int8)     # condition1 (speed1)
            a2_2 = to_arr('A2-2', row[1], dtype=np.int8)     # condition2 (speed2)
            a2_3 = to_arr('A2-3', row[2], dtype=np.int8)     # response (0/1)
            a2_4 = to_arr('A2-4', row[3], dtype=np.float32)   # RT

            if not (a2_1.size and a2_2.size and a2_3.size and a2_4.size):
                abn_ratio.append(np.nan)
                for buf in (match_rt, mismatch_rt,
                            ss, sn, sf, ns, nn, nf, fs, fn, ff):
                    buf.append(np.nan)
                continue

            # 최소 길이로 정렬(길이 불일치 안전)
            n = min(a2_1.size, a2_2.size, a2_3.size, a2_4.size)
            a2_1 = a2_1[:n]; a2_2 = a2_2[:n]; a2_3 = a2_3[:n]; a2_4 = a2_4[:n]

            # 마스크
            m1, m2, m3 = (a2_1 == 1), (a2_1 == 2), (a2_1 == 3)
            n1, n2, n3 = (a2_2 == 1), (a2_2 == 2), (a2_2 == 3)

            # abnormal ratio (A2-3의 1의 합 / 18)
            abn = int(a2_3.sum())
            abn_ratio.append(abn / abnormal_denominator)

            # 9셀 RT 요약
            ss.append(summarize(a2_4[m1 & n1], mode))
            sn.append(summarize(a2_4[m1 & n2], mode))
            sf.append(summarize(a2_4[m1 & n3], mode))

            ns.append(summarize(a2_4[m2 & n1], mode))
            nn.append(summarize(a2_4[m2 & n2], mode))
            nf.append(summarize(a2_4[m2 & n3], mode))

            fs.append(summarize(a2_4[m3 & n1], mode))
            fn.append(summarize(a2_4[m3 & n2], mode))
            ff.append(summarize(a2_4[m3 & n3], mode))

            # match / mismatch
            eq = (a2_1 == a2_2)
            match_rt.append(summarize(a2_4[eq], mode))
            mismatch_rt.append(summarize(a2_4[~eq], mode))

        # DataFrame으로 한 번에 반환
        out = pd.DataFrame({
            'A2_3_abnormal_ratio': abn_ratio,

            f'match_rt_{mode}':    match_rt,
            f'mismatch_rt_{mode}': mismatch_rt,

            f'slow_slow_rt_{mode}':     ss,
            f'slow_normal_rt_{mode}':   sn,
            f'slow_fast_rt_{mode}':     sf,

            f'normal_slow_rt_{mode}':   ns,
            f'normal_normal_rt_{mode}': nn,
            f'normal_fast_rt_{mode}':   nf,

            f'fast_slow_rt_{mode}':     fs,
            f'fast_normal_rt_{mode}':   fn,
            f'fast_fast_rt_{mode}':     ff,
        }, index=self.df.index)

        return out
    # ---------- ---------- ---------- ---------- ---------- ---------- ----------
    
    # ================== Preprocess methods ==================
    # assume you already have: self._to_array(val, dtype) in your class


    def A3_features_fast(self, mode='mu_sigma'):
        """
        A3 전체를 고속 처리:
        - apply(axis=1) 사용 X
        - 중간 df_split 생성 X
        - itertuples 한 번 순회로 파싱/집계/요약 완료
        반환 컬럼(기존 의미 유지):
        correct_rate, abnormal_rate,
        validity_benefit_acc, validity_benefit_rt,
        size_cost_acc, size_cost_rt,
        lr_bias_acc, lr_bias_rt,
        rt_correct
        """
        to_arr = self._to_array_fast if hasattr(self, "_to_array_fast") else self._to_array
        summarize = getattr(self, "_summarize_array", None)
        if summarize is None:
            summarize = lambda arr, m=mode: self._summarize_list(arr, mode=m)

        # 결과 버퍼(리스트에 누적 후 한 번에 DataFrame 화 → 판다스 오버헤드 최소화)
        correct_rate, abnormal_rate = [], []
        validity_benefit_acc, validity_benefit_rt = [], []
        size_cost_acc, size_cost_rt = [], []
        lr_bias_acc, lr_bias_rt = [], []
        rt_correct = []

        # 보조
        def _safe_div(a, b):
            return (a / b) if (b and b > 0) else np.nan

        def _mean(arr_like):
            x = np.asarray(arr_like, dtype=float)
            x = x[np.isfinite(x)]
            return float(x.mean()) if x.size > 0 else np.nan

        # 인덱스: A3 관련 컬럼만 슬라이스하여 튜플로 읽기 → 파싱 최소화
        for (a3_1_s, a3_2_s, a3_3_s, a3_4_s, a3_5_s, a3_6_s, a3_7_s) in self.df[self.A3_cols].itertuples(index=False, name=None):
            a3_1 = to_arr('A3-1', a3_1_s, dtype=np.int8)    # size: 1 small / 2 big
            a3_3 = to_arr('A3-3', a3_3_s, dtype=np.int8)    # dir: 1 left / 2 right
            a3_5 = to_arr('A3-5', a3_5_s, dtype=np.int8)    # resp1: 1 vc, 2 vi, 3 ic, 4 ii
            a3_6 = to_arr('A3-6', a3_6_s, dtype=np.int8)    # resp2: 0 omission, 1 response (optional)
            a3_7 = to_arr('A3-7', a3_7_s, dtype=np.float32)  # RT

            # 필수 4개 컬럼 길이 동기화 (안전하게 최소 길이로 자름)
            if not (a3_1.size and a3_3.size and a3_5.size and a3_7.size):
                # 결측 로우
                correct_rate.append(np.nan); abnormal_rate.append(np.nan)
                validity_benefit_acc.append(np.nan); validity_benefit_rt.append(np.nan)
                size_cost_acc.append(np.nan); size_cost_rt.append(np.nan)
                lr_bias_acc.append(np.nan); lr_bias_rt.append(np.nan)
                rt_correct.append(np.nan)
                continue

            n = min(a3_1.size, a3_3.size, a3_5.size, a3_7.size)
            a3_1 = a3_1[:n]; a3_3 = a3_3[:n]; a3_5 = a3_5[:n]; a3_7 = a3_7[:n]
            if a3_6.size >= n:
                a3_6 = a3_6[:n]  # 있으면 동기화

            # 마스크 구성
            is_valid   = (a3_5 == 1) | (a3_5 == 2)
            is_invalid = (a3_5 == 3) | (a3_5 == 4)
            is_correct = (a3_5 == 1) | (a3_5 == 3)
            is_incorr  = (a3_5 == 2) | (a3_5 == 4)

            is_small = (a3_1 == 1)
            is_big   = (a3_1 == 2)
            is_left  = (a3_3 == 1)
            is_right = (a3_3 == 2)

            # 카운트
            n_trials   = n
            n_correct  = int(is_correct.sum())
            n_valid    = int(is_valid.sum())
            n_invalid  = int(is_invalid.sum())
            n_small    = int(is_small.sum())
            n_big      = int(is_big.sum())
            n_left     = int(is_left.sum())
            n_right    = int(is_right.sum())

            # abnormal
            if isinstance(a3_6, np.ndarray) and a3_6.size == n:
                n_abnormal = int((a3_6 == 1).sum())
                abnormal_rate.append(_safe_div(n_abnormal, n_trials))
            else:
                abnormal_rate.append(np.nan)

            correct_rate.append(_safe_div(n_correct, n_trials))

            # 조건별 정답 카운트
            n_valid_correct   = int((is_valid   & is_correct).sum())
            n_invalid_correct = int((is_invalid & is_correct).sum())
            n_small_correct   = int((is_small   & is_correct).sum())
            n_big_correct     = int((is_big     & is_correct).sum())
            n_left_correct    = int((is_left    & is_correct).sum())
            n_right_correct   = int((is_right   & is_correct).sum())

            # 정확도 효과
            acc_valid   = _safe_div(n_valid_correct,   n_valid)
            acc_invalid = _safe_div(n_invalid_correct, n_invalid)
            validity_benefit_acc.append(acc_valid - acc_invalid)

            acc_small = _safe_div(n_small_correct, n_small)
            acc_big   = _safe_div(n_big_correct,   n_big)
            size_cost_acc.append(acc_big - acc_small)

            acc_left  = _safe_div(n_left_correct,  n_left)
            acc_right = _safe_div(n_right_correct, n_right)
            lr_bias_acc.append(acc_right - acc_left)

            # RT 효과(정답 trial만)
            rt_valid_correct   = a3_7[is_valid   & is_correct]
            rt_invalid_correct = a3_7[is_invalid & is_correct]
            rt_small_correct   = a3_7[is_small   & is_correct]
            rt_big_correct     = a3_7[is_big     & is_correct]
            rt_left_correct    = a3_7[is_left    & is_correct]
            rt_right_correct   = a3_7[is_right   & is_correct]
            rt_all_correct     = a3_7[is_correct]

            validity_benefit_rt.append(_mean(rt_valid_correct) - _mean(rt_invalid_correct))
            size_cost_rt.append(_mean(rt_big_correct) - _mean(rt_small_correct))
            lr_bias_rt.append(_mean(rt_right_correct) - _mean(rt_left_correct))

            # RT 일관성/중심 요약
            rt_correct.append(summarize(rt_all_correct, mode))

        # 결과 DF
        out = pd.DataFrame({
            'correct_rate': correct_rate,
            'abnormal_rate': abnormal_rate,
            'validity_benefit_acc': validity_benefit_acc,
            'validity_benefit_rt':  validity_benefit_rt,
            'size_cost_acc': size_cost_acc,
            'size_cost_rt':  size_cost_rt,
            'lr_bias_acc':   lr_bias_acc,
            'lr_bias_rt':    lr_bias_rt,
            'rt_correct':    rt_correct,
        }, index=self.df.index)

        return out



    def A4_features_fast(self, mode='mu_sigma'):
        """
        A4 (선택적 주의 검사) 고속 전처리 버전.
        - apply(axis=1) 없이 itertuples 한 번 순회로 처리.
        - 파싱 + 피처 집계를 한 번에 수행.
        """
        to_arr = self._to_array_fast if hasattr(self, "_to_array_fast") else self._to_array
        summarize = getattr(self, "_summarize_array", None)
        if summarize is None:
            summarize = lambda arr, m=mode: self._summarize_list(arr, mode=m)

        # 결과 버퍼
        n_incorrect, n_abnormal = [], []
        con_red_corr_ratio, incon_red_corr_ratio = [], []
        con_green_corr_ratio, incon_green_corr_ratio = [], []
        rt_con_red_corr, rt_incon_red_corr = [], []
        rt_con_green_corr, rt_incon_green_corr = [], []
        rt_congruent_corr, rt_incongruent_corr, rt_correct_all = [], [], []

        for (a4_1_s, a4_2_s, a4_3_s, a4_4_s, a4_5_s) in self.df[self.A4_cols].itertuples(index=False, name=None):
            # 파싱
            a4_1 = to_arr('A4-1', a4_1_s, dtype=np.int8)    # congruency
            a4_2 = to_arr('A4-2', a4_2_s, dtype=np.int8)    # color
            a4_3 = to_arr('A4-3', a4_3_s, dtype=np.int8)    # response1
            a4_4 = to_arr('A4-4', a4_4_s, dtype=np.int8)    # response2 (abnormal)
            a4_5 = to_arr('A4-5', a4_5_s, dtype=np.float32)  # RT

            if not (a4_1.size and a4_2.size and a4_3.size and a4_5.size):
                # 결측 로우 방어
                n_incorrect.append(np.nan); n_abnormal.append(np.nan)
                con_red_corr_ratio.append(np.nan); incon_red_corr_ratio.append(np.nan)
                con_green_corr_ratio.append(np.nan); incon_green_corr_ratio.append(np.nan)
                for lst in (rt_con_red_corr, rt_incon_red_corr,
                            rt_con_green_corr, rt_incon_green_corr,
                            rt_congruent_corr, rt_incongruent_corr,
                            rt_correct_all):
                    lst.append(np.nan)
                continue

            n = min(a4_1.size, a4_2.size, a4_3.size, a4_4.size, a4_5.size)
            a4_1, a4_2, a4_3, a4_4, a4_5 = a4_1[:n], a4_2[:n], a4_3[:n], a4_4[:n], a4_5[:n]

            # 마스크
            is_con    = (a4_1 == 1)
            is_incon  = (a4_1 == 2)
            is_red    = (a4_2 == 1)
            is_green  = (a4_2 == 2)
            is_corr   = (a4_3 == 1)
            is_incorr = (a4_3 == 2)
            is_abn    = (a4_4 == 1)

            # RT 리스트 (정답만)
            r_con_red_corr     = a4_5[is_con   & is_red   & is_corr]
            r_incon_red_corr   = a4_5[is_incon & is_red   & is_corr]
            r_con_green_corr   = a4_5[is_con   & is_green & is_corr]
            r_incon_green_corr = a4_5[is_incon & is_green & is_corr]
            r_congruent_corr   = a4_5[is_con   & is_corr]
            r_incongruent_corr = a4_5[is_incon & is_corr]
            r_correct_all      = a4_5[is_corr]

            # 카운트
            n_corr     = int(is_corr.sum())
            n_inc      = int(is_incorr.sum())
            n_abn_cnt  = int(is_abn.sum())
            n_incorrect.append(n_inc)
            n_abnormal.append(n_abn_cnt)

            # 각 조건별 정확도 계산
            def _safe_ratio(num, den):
                return (num / den) if den > 0 else np.nan

            n_con_red     = int((is_con & is_red).sum())
            n_incon_red   = int((is_incon & is_red).sum())
            n_con_green   = int((is_con & is_green).sum())
            n_incon_green = int((is_incon & is_green).sum())

            n_con_red_corr     = int((is_con & is_red   & is_corr).sum())
            n_incon_red_corr   = int((is_incon & is_red & is_corr).sum())
            n_con_green_corr   = int((is_con & is_green & is_corr).sum())
            n_incon_green_corr = int((is_incon & is_green & is_corr).sum())

            con_red_corr_ratio.append(_safe_ratio(n_con_red_corr, n_con_red))
            incon_red_corr_ratio.append(_safe_ratio(n_incon_red_corr, n_incon_red))
            con_green_corr_ratio.append(_safe_ratio(n_con_green_corr, n_con_green))
            incon_green_corr_ratio.append(_safe_ratio(n_incon_green_corr, n_incon_green))

            # RT 요약값 저장
            rt_con_red_corr.append(summarize(r_con_red_corr, mode))
            rt_incon_red_corr.append(summarize(r_incon_red_corr, mode))
            rt_con_green_corr.append(summarize(r_con_green_corr, mode))
            rt_incon_green_corr.append(summarize(r_incon_green_corr, mode))
            rt_congruent_corr.append(summarize(r_congruent_corr, mode))
            rt_incongruent_corr.append(summarize(r_incongruent_corr, mode))
            rt_correct_all.append(summarize(r_correct_all, mode))

        # 최종 DataFrame 구성
        out = pd.DataFrame({
            'n_incorrect': n_incorrect,
            # 'n_abnormal': n_abnormal,
            'con_red_corr_ratio': con_red_corr_ratio,
            'incon_red_corr_ratio': incon_red_corr_ratio,
            'con_green_corr_ratio': con_green_corr_ratio,
            'incon_green_corr_ratio': incon_green_corr_ratio,
            f'rt_con_red_corr_{mode}': rt_con_red_corr,
            f'rt_incon_red_corr_{mode}': rt_incon_red_corr,
            f'rt_con_green_corr_{mode}': rt_con_green_corr,
            f'rt_incon_green_corr_{mode}': rt_incon_green_corr,
            f'rt_congruent_corr_{mode}': rt_congruent_corr,
            f'rt_incongruent_corr_{mode}': rt_incongruent_corr,
            f'rt_correct_all_{mode}': rt_correct_all
        }, index=self.df.index)

        return out



    def A5_features_fast(self) -> pd.DataFrame:
        """
        A5 (변화 탐지) 고속 전처리:
        - itertuples() 한 번 순회
        - _to_array_fast로 파싱
        - 조건별/전체 정확도 및 요약 피처 계산
        반환 컬럼:
        n_trials, n_non/n_pos/n_color/n_shape,
        n_non_corr/n_pos_corr/n_color_corr/n_shape_corr,
        n_correct, n_incorrect, n_abnormal,
        abnormal_ratio, n_corr_ratio,
        n_non_corr_ratio, n_pos_corr_ratio, n_color_corr_ratio, n_shape_corr_ratio,
        change_acc_mean, change_vs_non_diff
        """
        to_arr = self._to_array_fast if hasattr(self, "_to_array_fast") else self._to_array

        # 결과 버퍼
        n_trials = []
        n_non = []; n_pos = []; n_color = []; n_shape = []
        n_non_corr = []; n_pos_corr = []; n_color_corr = []; n_shape_corr = []
        n_correct = []; n_incorrect = []; n_abnormal = []

        def _safe_ratio(num, den):
            return (num / den) if den and den > 0 else np.nan

        for a5_1_s, a5_2_s, a5_3_s in self.df[self.A5_cols].itertuples(index=False, name=None):
            # 파싱
            a5_1 = to_arr('A5-1', a5_1_s, dtype=np.int8)   # condition: 1/2/3/4
            a5_2 = to_arr('A5-2', a5_2_s, dtype=np.int8)   # resp1: 1=correct, 2=incorrect
            a5_3 = to_arr('A5-3', a5_3_s, dtype=np.int8)   # resp2: 0=N, 1=Y (abnormal)

            if not (a5_1.size and a5_2.size):
                # 완전 결측 방어
                n_trials.append(np.nan)
                n_non.append(np.nan); n_pos.append(np.nan); n_color.append(np.nan); n_shape.append(np.nan)
                n_non_corr.append(np.nan); n_pos_corr.append(np.nan); n_color_corr.append(np.nan); n_shape_corr.append(np.nan)
                n_correct.append(np.nan); n_incorrect.append(np.nan); n_abnormal.append(np.nan)
                continue

            # 길이 맞추기 (안전)
            n = min(a5_1.size, a5_2.size, a5_3.size if a5_3.size else a5_2.size)
            a5_1 = a5_1[:n]; a5_2 = a5_2[:n]; a5_3 = a5_3[:n] if a5_3.size else np.zeros(n, dtype=np.int8)
            n_trials.append(n)

            # 마스크
            is_non   = (a5_1 == 1)
            is_pos   = (a5_1 == 2)
            is_color = (a5_1 == 3)
            is_shape = (a5_1 == 4)
            is_corr  = (a5_2 == 1)
            is_inc   = (a5_2 == 2)
            is_abn   = (a5_3 == 1)

            # 카운트
            n_non_i, n_pos_i, n_color_i, n_shape_i = int(is_non.sum()), int(is_pos.sum()), int(is_color.sum()), int(is_shape.sum())
            n_non.append(n_non_i); n_pos.append(n_pos_i); n_color.append(n_color_i); n_shape.append(n_shape_i)

            n_non_corr_i   = int((is_non   & is_corr).sum())
            n_pos_corr_i   = int((is_pos   & is_corr).sum())
            n_color_corr_i = int((is_color & is_corr).sum())
            n_shape_corr_i = int((is_shape & is_corr).sum())
            n_non_corr.append(n_non_corr_i); n_pos_corr.append(n_pos_corr_i)
            n_color_corr.append(n_color_corr_i); n_shape_corr.append(n_shape_corr_i)

            n_correct_i = int(is_corr.sum());  n_incorrect_i = int(is_inc.sum());  n_abn_i = int(is_abn.sum())
            n_correct.append(n_correct_i); n_incorrect.append(n_incorrect_i); n_abnormal.append(n_abn_i)

        # DataFrame 구성
        out = pd.DataFrame({
            ### 전처리시에 전부 같게 나오는 값들
            'n_trials': n_trials,
            'n_non': n_non, 'n_pos': n_pos, 'n_color': n_color, 'n_shape': n_shape,
            ## 이 값들은 괜찮다
            'n_non_corr': n_non_corr, 'n_pos_corr': n_pos_corr, 'n_color_corr': n_color_corr, 'n_shape_corr': n_shape_corr,
            'n_correct': n_correct, 'n_incorrect': n_incorrect, 'n_abnormal': n_abnormal,
        }, index=self.df.index)

        # 파생 비율들
        out['abnormal_ratio']   = out['n_abnormal'] / out['n_trials']
        out['n_corr_ratio']     = out['n_correct']  / (out['n_correct'] + out['n_incorrect'])

        out['n_non_corr_ratio']   = out['n_non_corr']   / out['n_non']
        out['n_pos_corr_ratio']   = out['n_pos_corr']   / out['n_pos']
        out['n_color_corr_ratio'] = out['n_color_corr'] / out['n_color']
        out['n_shape_corr_ratio'] = out['n_shape_corr'] / out['n_shape']

        out['change_acc_mean']   = out[['n_pos_corr_ratio','n_color_corr_ratio','n_shape_corr_ratio']].mean(axis=1)
        out['change_vs_non_diff'] = out['change_acc_mean'] - out['n_non_corr_ratio']

        out = out.drop(columns=['n_trials', 'n_non', 'n_pos', 'n_color', 'n_shape'])
        return out
    

    def A6_7_features_fast(self) -> pd.DataFrame:
        """
        A6(14문항), A7(18문항) 고속 전처리:
        - to_numeric 벡터화 변환 + 클리핑(분모 초과/음수 방어)
        - 정답수, 오답수, 정답률 산출
        반환:
        a6_correct, a6_incorrect, a6_ratio,
        a7_correct, a7_incorrect, a7_ratio
        (옵션) combined_mean_ratio, any_out_of_range
        """
        a6 = pd.to_numeric(self.df['A6-1'], errors='coerce')
        a7 = pd.to_numeric(self.df['A7-1'], errors='coerce')

        # 범위 방어(음수/초과값을 합리적으로 잘라냄)
        a6c = a6.clip(lower=0, upper=14)
        a7c = a7.clip(lower=0, upper=18)

        out = pd.DataFrame(index=self.df.index)
        out['a6_ratio']     = a6c / 14.0
        out['a7_ratio']     = a7c / 18.0


        return out


    def A8_9_features(self):
        cols = ['A8-1', 'A8-2',
                'A9-1', 'A9-2', 'A9-3', 'A9-4', 'A9-5']
        existing = [c for c in cols if c in self.df.columns]
        out = self.df[existing].copy()

        # 수치형 변환 (문자/결측 방어)
        out = out.apply(pd.to_numeric, errors='coerce')

        out_scaled = pd.DataFrame(out, columns=out.columns, index=out.index)

        return out_scaled
    
    
    



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
            # 'B1_rt_t2_inc_mean':  rt_t2_inc_mean,
            f'B1_rt_t2_corr_{mode}': rt_t2_corr_summ,
            # f'B1_rt_t2_inc_{mode}':  rt_t2_inc_summ,
            ## 이러한 마이너스 term은 nan값들을 많이 만들어내고, 합성데이터 생성도 잘되지않는것 같다. 그러니 걍 삭제하자
            # 'B1_rt_t2_corr_minus_inc_mean': rt_t2_corr_minus_inc_mean,
            # f'B1_rt_t2_corr_minus_inc_{mode}': rt_t2_corr_minus_inc_summ,
        }

        # B1-3 값(1~4)별 # 4는 제외 : 의미없어보임
        for k in (1, 2, 3):
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
            
            'B2_rt_t2_corr_mean': rt_t2_corr_mean,
            # 'B2_rt_t2_inc_mean':  rt_t2_inc_mean,
            f'B2_rt_t2_corr_{mode}': rt_t2_corr_summ,
            # f'B2_rt_t2_inc_{mode}':  rt_t2_inc_summ,
            ## 이러한 마이너스 term은 nan값들을 많이 만들어내고, 합성데이터 생성도 잘되지않는것 같다. 그러니 걍 삭제하자
            # 'B1_rt_t2_corr_minus_inc_mean': rt_t2_corr_minus_inc_mean,
            # f'B1_rt_t2_corr_minus_inc_{mode}': rt_t2_corr_minus_inc_summ,
        }

        # B1-3 값(1~4)별 # 4는 의미없어보인다
        for k in (1, 2, 3):
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
            # 'B3_n_incorrect': n_inc_list,
            'B3_correct_rate': corr_rate_list,

            'B3_rt_all_mean': rt_all_mean,
            f'B3_rt_all_{mode}': rt_all_summ,

            'B3_rt_correct_mean': rt_corr_mean,
            f'B3_rt_correct_{mode}': rt_corr_summ,

            # 'B3_rt_incorrect_mean': rt_inc_mean,
            # f'B3_rt_incorrect_{mode}': rt_inc_summ,

            ## 이러한 마이너스 term은 nan값들을 많이 만들어내고, 합성데이터 생성도 잘되지않는것 같다. 그러니 걍 삭제하자
            # 'B3_rt_err_minus_corr': rt_err_minus_corr,
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

            ## 이러한 마이너스 term은 nan값들을 많이 만들어내고, 합성데이터 생성도 잘되지않는것 같다. 그러니 걍 삭제하자
            # 'B4_icg_minus_cg_mean': icg_minus_cg_mean,
            # f'B4_icg_minus_cg_{mode}': icg_minus_cg_summ,
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

            ## 이러한 마이너스 term은 nan값들을 많이 만들어내고, 합성데이터 생성도 잘되지않는것 같다. 그러니 걍 삭제하자
            # 'B5_rt_err_minus_corr': rt_err_minus_corr,
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
            ## 이러한 마이너스 term은 nan값들을 많이 만들어내고, 합성데이터 생성도 잘되지않는것 같다. 그러니 걍 삭제하자
            # f'{prefix}_corr_minus_inc':  diff_list,
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