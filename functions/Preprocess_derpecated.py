import numpy as np
import pandas as pd
import os, re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from typing import List, Optional

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
            return np.log1p(c / (sigma2 + c + 1e-12))
            # 또는 return mu / (sigma + 0.05 * mu)
        elif mode == 'mu_range':
            r = arr.max() - arr.min()
            return float(mu / r) if r > 0 else np.nan
        else:
            raise ValueError("mode must be one of {'mu','mu_sigma','mu_range'}")

    # ---- 빠른 파서: 콤마구분 문자열 -> np.array (약간의 불량치 방어 포함)
    @staticmethod
    def _to_array_fast(val, dtype=float):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.empty(0, dtype=dtype)
        s = str(val).strip()
        if not s:
            return np.empty(0, dtype=dtype)
        # 공백/연속콤마 방어
        s = s.replace(',,', ',')
        try:
            arr = np.fromstring(s, sep=',', dtype=dtype)
        except Exception:
            # 극단적 불량치가 섞이면 마지막 방어
            toks = [t for t in s.split(',') if t.strip() != ""]
            if not toks:
                return np.empty(0, dtype=dtype)
            arr = np.array(toks, dtype=float if dtype is not int else float)
            if dtype is int:
                # int로 캐스팅 가능한 값만
                mask = np.isfinite(arr)
                arr = arr[mask].astype(int, copy=False)
            else:
                arr = arr[np.isfinite(arr)]
                arr = arr.astype(dtype, copy=False)
        return arr

    # ---- 빠른 요약자: 리스트/array -> 스칼라 (mu / mu_sigma / mu_range)
    @staticmethod
    def _summarize_array(arr, mode='mu_sigma'):
        if arr is None:
            return np.nan
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.nan
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan

        mu = arr.mean()
        if mode == 'mu':
            return float(mu)
        elif mode == 'mu_sigma':
            # 제안하신 log(1 + mu^2/(sigma^2+mu^2)) 안정화 버전
            sigma2 = arr.var(ddof=0)
            c = mu * mu
            return float(np.log1p(c / (sigma2 + c + 1e-12)))
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
            a1_1 = to_arr(row[0], dtype=int)     # 방향 1/2
            a1_2 = to_arr(row[1], dtype=int)     # 속도 1/2/3
            a1_3 = to_arr(row[2], dtype=int)     # 반응 0/1
            a1_4 = to_arr(row[3], dtype=float)   # RT

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

    # --- 핵심: A2 한 행(row)을 Condition1 x Condition2 별로 반응, 반응시간 리스트로 분해 ---
    def A2_parse_and_split_row(self, row: pd.Series) -> pd.Series:
        """
        A2-1(방향:1=왼,2=오) × A2-2(속도:1=느림,2=중간,3=빠름)
        총 6가지 조건에 해당하는 A2-3(반응)·A2-4(반응시간)을 리스트로 추출
        """
        # 명시적으로 변수에 할당
        # print("A2_test --- --- ---")
        a2_1 = self._to_array(row['A2-1'], dtype=int)    # 방향: 1=왼, 2=오
        a2_2 = self._to_array(row['A2-2'], dtype=int)    # 속도: 1=느림,2=중간,3=빠름
        a2_3 = self._to_array(row['A2-3'], dtype=int)    # 반응: 0=N(정상),1=Y(비정상)
        a2_4 = self._to_array(row['A2-4'], dtype=float)  # 반응시간(혹은 편차)
        # print("A2_test --- --- --- done")    


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
            cols = ['match_rt', 'mismatch_rt',
                    'slow_slow_rt', 'slow_normal_rt', 'slow_fast_rt',
                    'normal_slow_rt', 'normal_normal_rt', 'normal_fast_rt',
                    'fast_slow_rt', 'fast_normal_rt', 'fast_fast_rt']

        out = pd.DataFrame(index=df_split.index)

        # (1) 비정상 반응 집계
        print(df_split['match_res_sum'])
        print(df_split['mismatch_res_sum'])
        
        out['A2_3_sum']  = df_split['res']
        # out['A2_3_match_res_sum'] = df_split['match_res_sum']
        # out['A2_3_mismatch_res_sum'] = df_split['mismatch_res_sum']
        out['A2_3_abnormal_ratio'] = out['A2_3_sum'] / abnormal_denominator

        # (2) 6셀 RT 리스트 요약 (각 리스트 -> 스칼라 1개)
        for c in cols:
            out[c + f'_{mode}'] = df_split[c].apply(self._summarize_list, mode=mode)

        return out    
    # ---------- ---------- ---------- ---------- ---------- ---------- ----------




    def A3_parse_and_split_row(self, row: pd.Series) -> pd.Series:
        """
        A3-1(size: 1=small, 2=big), A3-2(pos 1~8), A3-3(dir: 1=left,2=right),
        A3-4(pos 1~8), A3-5(resp1: 1=valid correct,2=valid incorrect,3=invalid correct,4=invalid incorrect),
        A3-6(resp2: 0=omission,1=commission/response), A3-7(RT)
        """
        a3_1 = self._to_array(row['A3-1'], dtype=int)
        a3_2 = self._to_array(row['A3-2'], dtype=int)  # 위치 (여기선 사용 X, 확장용)
        a3_3 = self._to_array(row['A3-3'], dtype=int)  # 방향(왼/오)
        a3_4 = self._to_array(row['A3-4'], dtype=int)  # 위치 (여기선 사용 X, 확장용)
        a3_5 = self._to_array(row['A3-5'], dtype=int)  # 정답 유형(1~4)
        a3_6 = self._to_array(row['A3-6'], dtype=int)  # 반응 유무(0/1) - 없으면 길이 0일 수 있음
        a3_7 = self._to_array(row['A3-7'], dtype=float)  # RT

        n = len(a3_7)
        # if n == 0:
        #     return pd.Series({
        #         'rt_valid_correct': [], 'rt_invalid_correct': [],
        #         'rt_small_correct': [], 'rt_big_correct': [],
        #         'rt_left_correct': [],  'rt_right_correct': [],
        #         'rt_correct_all':   [],
        #         # 카운트
        #         'n_trials': 0, 'n_correct': 0, 'n_incorrect': 0,
        #         'n_valid': 0, 'n_invalid': 0,
        #         'n_valid_correct': 0, 'n_invalid_correct': 0,
        #         'n_small': 0, 'n_big': 0,
        #         'n_small_correct': 0, 'n_big_correct': 0,
        #         'n_left': 0, 'n_right': 0,
        #         'n_left_correct': 0, 'n_right_correct': 0,
        #         'n_omission': np.nan,  # A3-6이 비어있다면 NaN
        #     })

        # 마스크
        is_valid   = np.isin(a3_5, [1, 2])
        is_invalid = np.isin(a3_5, [3, 4])
        is_correct = np.isin(a3_5, [1, 3])
        # is_incorrect = ~is_correct
        is_incorrect = np.isin(a3_5, [2, 4])

        is_small = (a3_1 == 1)
        is_big   = (a3_1 == 2)

        is_left  = (a3_3 == 1)
        is_right = (a3_3 == 2)

        # RT 리스트 (정답만 사용)
        rt_valid_correct   = a3_7[is_valid   & is_correct].tolist()
        rt_invalid_correct = a3_7[is_invalid & is_correct].tolist()
        rt_small_correct   = a3_7[is_small   & is_correct].tolist()
        rt_big_correct     = a3_7[is_big     & is_correct].tolist()
        rt_left_correct    = a3_7[is_left    & is_correct].tolist()
        rt_right_correct   = a3_7[is_right   & is_correct].tolist()
        rt_correct_all     = a3_7[is_correct].tolist()
        rt_incorrect_all = a3_7[is_incorrect].tolist()

        # print("rt_correct_all = ", rt_correct_all)
        # print("rt_incorrect_all = ", rt_incorrect_all)

        # 카운트
        n_valid  = int(is_valid.sum())
        n_invalid = int(is_invalid.sum())
        n_small  = int(is_small.sum())
        n_big    = int(is_big.sum())
        n_left   = int(is_left.sum())
        n_right  = int(is_right.sum())

        n_valid_correct   = int((is_valid   & is_correct).sum())
        n_invalid_correct = int((is_invalid & is_correct).sum())
        n_small_correct   = int((is_small   & is_correct).sum())
        n_big_correct     = int((is_big     & is_correct).sum())
        n_left_correct    = int((is_left    & is_correct).sum())
        n_right_correct   = int((is_right   & is_correct).sum())

        n_correct   = int(is_correct.sum())
        n_incorrect = int(is_incorrect.sum())
        
        # print("n_correct = ", n_correct)
        # print("n_incorrect = ", n_incorrect)

        # omission (A3-6이 제공되는 경우에만)
        n_abnormal = (int((a3_6 == 1).sum()) if len(a3_6) == n else np.nan)
        


        return pd.Series({
            'rt_valid_correct': rt_valid_correct,
            'rt_invalid_correct': rt_invalid_correct,
            'rt_small_correct': rt_small_correct,
            'rt_big_correct': rt_big_correct,
            'rt_left_correct': rt_left_correct,
            'rt_right_correct': rt_right_correct,
            'rt_correct_all': rt_correct_all,
            'rt_incorrect_all': rt_incorrect_all,
            # 카운트
            'n_trials': n,
            'n_correct': n_correct,
            'n_incorrect': n_incorrect,
            'n_valid': n_valid, 'n_invalid': n_invalid,
            'n_valid_correct': n_valid_correct, 'n_invalid_correct': n_invalid_correct,
            'n_small': n_small, 'n_big': n_big,
            'n_small_correct': n_small_correct, 'n_big_correct': n_big_correct,
            'n_left': n_left, 'n_right': n_right,
            'n_left_correct': n_left_correct, 'n_right_correct': n_right_correct,
            'n_abnormal': n_abnormal,
        })


    def A3_parse_and_split(self) -> pd.DataFrame:
        return self.df[self.A3_cols].apply(self.A3_parse_and_split_row, axis=1)

    def A3_features(self, df_split: pd.DataFrame, mode: str = 'mu_sigma') -> pd.DataFrame:
        out = pd.DataFrame(index=df_split.index)

        def _safe_mean(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            return float(arr.mean()) if arr.size > 0 else np.nan

        def _safe_acc(n_correct, n_total):
            return (n_correct / n_total) if (n_total and n_total > 0) else np.nan

        # (정확도/오류)
        out['correct_rate'] = df_split.eval('n_correct / n_trials')
        out['abnormal_rate'] = df_split['n_abnormal'] / df_split['n_trials']  # A3-6 없으면 NaN

        # (유효성 효과)
        acc_valid   = df_split.apply(lambda r: _safe_acc(r['n_valid_correct'], r['n_valid']), axis=1)
        acc_invalid = df_split.apply(lambda r: _safe_acc(r['n_invalid_correct'], r['n_invalid']), axis=1)
        out['validity_benefit_acc'] = acc_valid - acc_invalid

        mean_rt_valid   = df_split['rt_valid_correct'  ].apply(_safe_mean)
        mean_rt_invalid = df_split['rt_invalid_correct'].apply(_safe_mean)
        out['validity_benefit_rt'] = mean_rt_valid - mean_rt_invalid

        # (크기 효과)
        acc_small = df_split.apply(lambda r: _safe_acc(r['n_small_correct'], r['n_small']), axis=1)
        acc_big   = df_split.apply(lambda r: _safe_acc(r['n_big_correct'],   r['n_big']),   axis=1)
        out['size_cost_acc'] = acc_big - acc_small

        mean_rt_small = df_split['rt_small_correct'].apply(_safe_mean)
        mean_rt_big   = df_split['rt_big_correct'  ].apply(_safe_mean)
        out['size_cost_rt'] = mean_rt_big - mean_rt_small

        # (좌/우 편향)
        acc_left  = df_split.apply(lambda r: _safe_acc(r['n_left_correct'],  r['n_left']),  axis=1)
        acc_right = df_split.apply(lambda r: _safe_acc(r['n_right_correct'], r['n_right']), axis=1)
        out['lr_bias_acc'] = acc_right - acc_left

        mean_rt_left  = df_split['rt_left_correct' ].apply(_safe_mean)
        mean_rt_right = df_split['rt_right_correct'].apply(_safe_mean)
        out['lr_bias_rt'] = mean_rt_right - mean_rt_left

        # (RT 일관성/중심)
        out['rt_correct']    = df_split['rt_correct_all'].apply(self._summarize_list, mode=mode)
        # out['rt_incorrect']    = df_split['rt_incorrect_all'].apply(self._summarize_list, mode=mode)
        # mean_rt_correct   = df_split['rt_correct_all'].apply(_safe_mean)
        # mean_rt_incorrect = df_split['rt_incorrect_all'].apply(_safe_mean)
        # out['error_rt_gap'] = mean_rt_incorrect - mean_rt_correct
        # out['error_rt_mean'] = mean_rt_incorrect
        # out['robust_rt_center'] = df_split['rt_correct_all'].apply(
        #     lambda xs: _safe_mean([np.median(np.asarray(xs, dtype=float))])  # median as center
        # )

        return out
    
    # ================== Preprocess methods ==================
    # assume you already have: self._to_array(val, dtype) in your class

    def A4_parse_and_split_row(self, row: pd.Series) -> pd.Series:
        """
        A4-1: 1=congruent, 2=incongruent
        A4-2: 1=red, 2=green
        A4-3: 1=correct,  2=incorrect
        A4-4: 0=N(normal), 1=Y(abnormal)
        A4-5: response time
        """
        a4_1 = self._to_array(row['A4-1'], dtype=int)
        a4_2 = self._to_array(row['A4-2'], dtype=int)
        a4_3 = self._to_array(row['A4-3'], dtype=int)
        a4_4 = self._to_array(row['A4-4'], dtype=int)
        a4_5 = self._to_array(row['A4-5'], dtype=float)

        n = len(a4_5)

        is_con    = (a4_1 == 1)
        is_incon  = (a4_1 == 2)
        is_red    = (a4_2 == 1)
        is_green  = (a4_2 == 2)
        is_corr   = (a4_3 == 1)
        is_incorr    = (a4_3 == 2)
        is_abn    = (a4_4 == 1)

        # # RT lists (all trials)
        # rt_con_red_all    = a4_5[is_con   & is_red].tolist()
        # rt_incon_red_all  = a4_5[is_incon & is_red].tolist()
        # rt_con_green_all  = a4_5[is_con   & is_green].tolist()
        # rt_incon_green_all= a4_5[is_incon & is_green].tolist()

        # RT lists (correct-only)
        rt_con_red_corr    = a4_5[is_con   & is_red   & is_corr].tolist()
        rt_incon_red_corr  = a4_5[is_incon & is_red   & is_corr].tolist()
        rt_con_green_corr  = a4_5[is_con   & is_green & is_corr].tolist()
        rt_incon_green_corr= a4_5[is_incon & is_green & is_corr].tolist()
        # 색상 통합(일치/불일치별)
        rt_congruent_corr    = a4_5[is_con   & is_corr].tolist()
        rt_incongruent_corr  = a4_5[is_incon & is_corr].tolist()
        # 전체 정답 RT
        rt_correct_all       = a4_5[is_corr].tolist()
        
        # Response1 정답 마스크(정확도 계산용; 1=correct, 0=incorrect)
        res_con_red      = (a4_3[is_con   & is_red]   == 1).astype(int).tolist()
        res_incon_red    = (a4_3[is_incon & is_red]   == 1).astype(int).tolist()
        res_con_green    = (a4_3[is_con   & is_green] == 1).astype(int).tolist()
        res_incon_green  = (a4_3[is_incon & is_green] == 1).astype(int).tolist()
        
        # counts for accs and error stats
        # n_con, n_incon     = int(is_con.sum()), int(is_incon.sum())
        # n_red, n_green     = int(is_red.sum()), int(is_green.sum())
        n_corr, n_incorr   = int(is_corr.sum()), int(is_incorr.sum())
        n_abnormal         = int(is_abn.sum())
        n_con_red         = int((is_con & is_red).sum())
        n_incon_red       = int((is_incon & is_red).sum())
        n_con_green         = int((is_con & is_green).sum())
        n_incon_green       = int((is_incon & is_green).sum())
        n_con_red_corr         = int((is_con & is_red & is_corr).sum())
        n_incon_red_corr       = int((is_incon & is_red & is_corr).sum())
        n_con_green_corr         = int((is_con & is_green & is_corr).sum())
        n_incon_green_corr       = int((is_incon & is_green & is_corr).sum())

        return pd.Series({
            # RT lists (cell-level, correct-only)
            'rt_con_red_corr':     rt_con_red_corr,
            'rt_incon_red_corr':   rt_incon_red_corr,
            'rt_con_green_corr':   rt_con_green_corr,
            'rt_incon_green_corr': rt_incon_green_corr,
            # aggregated RT lists
            'rt_congruent_corr':   rt_congruent_corr,
            'rt_incongruent_corr': rt_incongruent_corr,
            'rt_correct_all':      rt_correct_all,

            # correctness masks per cell (for accuracy via mean)
            'res_con_red':     res_con_red,
            'res_incon_red':   res_incon_red,
            'res_con_green':   res_con_green,
            'res_incon_green': res_incon_green,

            # counts
            'n_trials': n,
            'n_correct': n_corr,
            'n_incorrect': n_incorr,
            'n_abnormal': n_abnormal,
            'n_con_red_corr':     n_con_red_corr,
            'n_incon_red_corr':   n_incon_red_corr,
            'n_con_green_corr':   n_con_green_corr,
            'n_incon_green_corr': n_incon_green_corr,
            'con_red_corr_ratio':     (n_con_red_corr / n_con_red),
            'incon_red_corr_ratio':   (n_incon_red_corr / n_incon_red),
            'con_green_corr_ratio':   (n_con_green_corr / n_con_green),
            'incon_green_corr_ratio': (n_incon_green_corr / n_incon_green)
        })


    def A4_parse_and_split(self) -> pd.DataFrame:
        return self.df[self.A4_cols].apply(self.A4_parse_and_split_row, axis=1)


    def A4_features(self, df_split: pd.DataFrame,
                    cols: Optional[List[str]] = None,
                    mode: str = 'mu_sigma',                    
                    include_color_rt: bool = True) -> pd.DataFrame:
        """
        Core 8–10 features for A4 (selective attention):
        - acc_congruent, acc_incongruent, acc_diff
        - rt_congruent_mu, rt_incongruent_mu, rt_diff, rt_ratio
        - n_incorrect, n_abnormal
        - (optional) rt_red_mu, rt_green_mu
        - (bonus) rt_stability (correct trials)
        """
        if cols is None:
            cols1 = ['n_con_red_corr', 'n_incon_red_corr', 'n_con_green_corr', 'n_incon_green_corr',
                     'con_red_corr_ratio', 'incon_red_corr_ratio', 'con_green_corr_ratio', 'incon_green_corr_ratio']
            cols2 = ['rt_con_red_corr', 'rt_incon_red_corr', 'rt_con_green_corr',
                    'rt_incon_green_corr', 'rt_congruent_corr',
                    'rt_incongruent_corr', 'rt_correct_all']

        out = pd.DataFrame(index=df_split.index)

        # errors / abnormal
        out['n_incorrect'] = df_split['n_incorrect']
        out['n_abnormal']  = df_split['n_abnormal']

        # (2) 6셀 RT 리스트 요약 (각 리스트 -> 스칼라 1개)
        for c in cols1:
            out[c] = df_split[c]
        for c in cols2:
            out[c + f'_{mode}'] = df_split[c].apply(self._summarize_list, mode=mode)
            
        
            
        return out    
    
    
    def A5_parse_and_split_row(self, row: pd.Series) -> pd.Series:
        """
        A5-1: Condition (1=non-change, 2=pos-change, 3=color-change, 4=shape-change)
        A5-2: Response1 (1=correct, 2=incorrect)
        A5-3: Response2 (0=N(normal), 1=Y(abnormal))
        """
        a5_1 = self._to_array(row['A5-1'], dtype=int)
        a5_2 = self._to_array(row['A5-2'], dtype=int)
        a5_3 = self._to_array(row['A5-3'], dtype=int)

        n = len(a5_1)  # 36 예정

        is_non   = (a5_1 == 1)
        is_pos   = (a5_1 == 2)
        is_color = (a5_1 == 3)
        is_shape = (a5_1 == 4)

        is_corr  = (a5_2 == 1)
        is_inc   = (a5_2 == 2)
        is_abn   = (a5_3 == 1)

        # 카운트
        n_non   = int(is_non.sum())
        n_pos   = int(is_pos.sum())
        n_color = int(is_color.sum())
        n_shape = int(is_shape.sum())
        
        n_non_corr   = int((a5_2[is_non]   == 1).sum())
        n_pos_corr   = int((a5_2[is_pos]   == 1).sum())
        n_color_corr = int((a5_2[is_color] == 1).sum())
        n_shape_corr = int((a5_2[is_shape] == 1).sum())

        n_corr   = int(is_corr.sum())
        n_inc    = int(is_inc.sum())
        n_abnorm = int(is_abn.sum())

        return pd.Series({
            # counts
            'n_trials': n,
            'n_non': n_non, 'n_pos': n_pos, 'n_color': n_color, 'n_shape': n_shape,
            'n_non_corr': n_non_corr, 'n_pos_corr': n_pos_corr, 'n_color_corr': n_color_corr, 'n_shape_corr': n_shape_corr,
            'n_correct': n_corr, 'n_incorrect': n_inc, 'n_abnormal': n_abnorm,
        })


    def A5_parse_and_split(self) -> pd.DataFrame:
        return self.df[self.A5_cols].apply(self.A5_parse_and_split_row, axis=1)    

    def A5_features(self, df_split: pd.DataFrame,
                    cols: Optional[List[str]] = None,
                    mode: str = 'mu_sigma') -> pd.DataFrame:

        out = pd.DataFrame(index=df_split.index)

        # errors / abnormal
        out['abnormal_ratio'] = (df_split['n_abnormal'] / df_split['n_trials'])
        out['n_abnormal']  = df_split['n_abnormal']
        out['n_corr_ratio'] = df_split['n_correct'] / (df_split['n_correct'] + df_split['n_incorrect'])
        out['n_non_corr_ratio'] = df_split['n_non_corr'] / df_split['n_non']
        out['n_pos_corr_ratio'] = df_split['n_pos_corr'] / df_split['n_pos']
        out['n_color_corr_ratio'] = df_split['n_color_corr'] / df_split['n_color']
        out['n_shape_corr_ratio'] = df_split['n_shape_corr'] / df_split['n_shape']
        out['n_non_corr'] = df_split['n_non_corr']
        out['n_pos_corr'] = df_split['n_pos_corr']
        out['n_color_corr'] = df_split['n_color_corr']
        out['n_shape_corr'] = df_split['n_shape_corr'] 
        out['change_acc_mean'] = out[['n_pos_corr_ratio','n_color_corr_ratio','n_shape_corr_ratio']].mean(axis=1)
        out['change_vs_non_diff'] = out['change_acc_mean'] - out['n_non_corr_ratio']
    
        
            
        return out    
    
    
    def A6_7_parse_and_split_row(self, row: pd.Series) -> pd.Series:
        """
        
        """
        a6_1 = int(row['A6-1'])
        a7_1 = int(row['A7-1'])
        
        return pd.Series({
            # counts
            'a6_1': a6_1, 'a7_1': a7_1
        })


    def A6_7_parse_and_split(self) -> pd.DataFrame:
        return self.df[self.A6_7_cols].apply(self.A6_7_parse_and_split_row, axis=1)    

    def A6_7_features(self, df_split: pd.DataFrame,
                    cols: Optional[List[str]] = None,
                    mode: str = 'mu_sigma') -> pd.DataFrame:

        out = pd.DataFrame(index=df_split.index)

        # errors / abnormal
        out['a6_1_ratio'] = df_split['a6_1'] / 14.0
        out['a7_1_ratio'] = df_split['a7_1'] / 18.0
    
        
            
        return out 
    
    def A8_9_features(self):
        cols = ['A8-1', 'A8-2',
                'A9-1', 'A9-2', 'A9-3', 'A9-4', 'A9-5']
        existing = [c for c in cols if c in self.df.columns]
        out = self.df[existing].copy()

        # 수치형 변환 (문자/결측 방어)
        out = out.apply(pd.to_numeric, errors='coerce')

        # 정규화
        scaler = StandardScaler()
        out_scaled = pd.DataFrame(scaler.fit_transform(out), columns=out.columns, index=out.index)

        return out_scaled