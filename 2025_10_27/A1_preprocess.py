"""
A1 Featurizer v2 — 운수종사자 인지검사 A1(행동반응 측정)

업데이트 포인트(요구사항 반영):
  - RT를 예측 편차로 해석: 조기(음수) / 지연(양수) / 정시(0)
  - rt_abs(절댓값), rt_sign(부호) 도출 → early/late ratio, 편향 지표 생성
  - 속도·방향·교호(2x3)별 요약 + 동역학(학습/피로/스위치 코스트)
  - 허용편차(유효범위)가 미지이므로 2가지 정규화 지표 제공
      1) 캡 기반 정확도: allowed_dev_ms가 주어지면 `prediction_accuracy_cap`
      2) 로버스트 정확도: 개인 내부 척도(MAD/IQR)로 정규화한 `prediction_accuracy_robust`

입력 형식: 한 행(row)에 다음 4개 문자열/숫자 컬럼이 존재한다고 가정
  - A1_1: '1,2,1,...'  (좌/우: 1=left, 2=right) — 18 trials
  - A1_2: '1,2,3,...'  (속도: 1=slow, 2=normal, 3=fast) — 각 6회
  - A1_3: '0,1,0,...'  (Response: 1=유효반응, 0=무반응/오반응)
  - A1_4: '23,43,-27,...' (ResponseTime=목표지점 대비 편차, ms 가정; 음수=조기, 양수=지연)

산출: 조건별 요약(좌/우, 속도, 2x3 교호), 분포·편향·동역학 요약, 품질지표 등

주의:
- 스케일링은 전표준화(학습셋 기준)로 별도로 하세요. 여기서는 feature 생성만 수행.
- A1_3의 의미가 환경에 따라 다를 수 있으므로, 1=유효반응 가정. 필요 시 매핑만 바꾸면 됨.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class A1Config:
    trials: int = 18
    dir_left: int = 1
    dir_right: int = 2
    sp_slow: int = 1
    sp_normal: int = 2
    sp_fast: int = 3
    # 품질관리/정규화 옵션
    allowed_dev_ms: Optional[float] = None  # 예: 200(ms). None이면 cap 기반 정확도 미계산
    rt_cap_ms: Optional[int] = 10000        # 이 값을 초과하면 invalid로 간주(옵션)
    trimmed_prop: float = 0.10               # 절단평균 비율(양끝 10%)
    small_eps: float = 1e-9                  # 0 나눗셈 방지용

# --------- 유틸리티 ---------

def _parse_commaseq(x) -> np.ndarray:
    """문자열 '1,2,3' -> np.array([1,2,3]). 리스트/숫자/NaN도 허용."""
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return np.array([], dtype=float)
        return np.array([float(t.strip()) for t in x.split(',') if t.strip()!=''], dtype=float)
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return np.asarray(x, dtype=float)
    if pd.isna(x):
        return np.array([], dtype=float)
    return np.array([float(x)], dtype=float)


def _safe_quantiles(v: np.ndarray, qs: List[float]) -> Dict[str, float]:
    if v.size == 0:
        return {f"q{int(q*100)}": np.nan for q in qs}
    out = np.quantile(v, qs)
    return {f"q{int(q*100)}": float(val) for q, val in zip(qs, out)}


def _trimmed_mean(v: np.ndarray, p: float) -> float:
    if v.size == 0:
        return np.nan
    if not (0.0 <= p < 0.5):
        return float(np.mean(v))
    v_sorted = np.sort(v)
    k = int(np.floor(p * v.size))
    if 2*k >= v.size:
        return float(np.mean(v))
    return float(np.mean(v_sorted[k: v.size - k]))


def _cv(mean: float, std: float) -> float:
    if mean == 0 or np.isnan(mean) or np.isnan(std):
        return np.nan
    return float(std / (mean if mean!=0 else 1.0))


def _lag1_corr(v: np.ndarray) -> float:
    if v.size < 2:
        return np.nan
    return float(np.corrcoef(v[:-1], v[1:])[0,1])


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2 or np.all(np.isnan(y)):
        return np.nan
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan
    slope, _ = np.polyfit(x[mask], y[mask], deg=1)
    return float(slope)


def _mad(v: np.ndarray) -> float:
    if v.size == 0:
        return np.nan
    med = np.median(v)
    return float(np.median(np.abs(v - med)))

# --------- 코어 계산 ---------

def _cell_mask(a1_1: np.ndarray, a1_2: np.ndarray, cfg: A1Config, d: int, s: int) -> np.ndarray:
    return (a1_1 == d) & (a1_2 == s)


def _valid_rt_mask(rt_abs: np.ndarray, cfg: A1Config) -> np.ndarray:
    if cfg.rt_cap_ms is None:
        return rt_abs > 0
    return (rt_abs > 0) & (rt_abs <= cfg.rt_cap_ms)


def _post_error_slowing(rt_abs: np.ndarray, resp: np.ndarray) -> float:
    """
    오답 다음 trial의 RT_abs 평균 - 정답 다음 trial의 RT_abs 평균
    resp[t]=1이면 t trial이 유효, t+1의 rt_abs를 비교.
    """
    n = rt_abs.size
    if n < 2 or resp.size != n:
        return np.nan
    next_rt = rt_abs[1:]
    was_correct = (resp[:-1] == 1)
    if np.all(was_correct==False) or np.all(was_correct==True):
        return np.nan
    mean_after_correct = np.mean(next_rt[was_correct]) if np.any(was_correct) else np.nan
    mean_after_error   = np.mean(next_rt[~was_correct]) if np.any(~was_correct) else np.nan
    if np.isnan(mean_after_correct) or np.isnan(mean_after_error):
        return np.nan
    return float(mean_after_error - mean_after_correct)


def _switch_cost(rt_abs: np.ndarray, a1_1: np.ndarray) -> float:
    """좌/우 전환 시(rt_abs)의 증가량: switch trial mean - stay trial mean (t>=2)."""
    n = rt_abs.size
    if n < 2:
        return np.nan
    switched = (a1_1[1:] != a1_1[:-1])
    if np.all(switched==False) or np.all(switched==True):
        return np.nan
    mean_sw = np.mean(rt_abs[1:][switched]) if np.any(switched) else np.nan
    mean_st = np.mean(rt_abs[1:][~switched]) if np.any(~switched) else np.nan
    if np.isnan(mean_sw) or np.isnan(mean_st):
        return np.nan
    return float(mean_sw - mean_st)

# --------- 공개 API ---------

def featurize_A1(
    df: pd.DataFrame,
    col_a1_1: str = "A1_1",
    col_a1_2: str = "A1_2",
    col_a1_3: str = "A1_3",
    col_a1_4: str = "A1_4",
    cfg: Optional[A1Config] = None,
) -> pd.DataFrame:
    """A1 전처리/피처 생성. 입력 df와 동일한 인덱스의 피처 DataFrame 반환.

    반환 컬럼(핵심 예시):
      - 전역/분포: mean_absRT_trim, std_absRT, cv_absRT, q10/q50/q90_absRT, iqr_absRT,
                   early_ratio, late_ratio, mean_rt_sign, acc_overall
      - 방향/속도: mean_absRT_left/right, acc_left/right, bias_direction,
                   mean_absRT_slow/normal/fast, acc_*, speed_bias_delta,
      - 2x3 셀: cell_absRT_mean_{L/R}_{S/N/F}, cell_acc_mean_{L/R}_{S/N/F},
               var_across_6cells_absRT
      - 동역학: absRT_slope_over_trials, post_error_slowing_abs, switch_cost_abs,
               fatigue_delta_abs (후반-전반)
      - 품질: valid_trial_ratio, n_missing_trials, n_invalid_rt
      - 정확도(정규화 대안): prediction_accuracy_cap(옵션), prediction_accuracy_robust
    """
    if cfg is None:
        cfg = A1Config()

    feats = []

    for idx, row in df.iterrows():
        a1_1 = _parse_commaseq(row.get(col_a1_1))  # direction
        a1_2 = _parse_commaseq(row.get(col_a1_2))  # speed
        a1_3 = _parse_commaseq(row.get(col_a1_3))  # response 0/1
        a1_4 = _parse_commaseq(row.get(col_a1_4))  # RT (편차, 부호 의미 포함)

        # 길이 맞추기
        n_trials = int(max(map(len, (a1_1, a1_2, a1_3, a1_4))) if any(len(x)>0 for x in (a1_1,a1_2,a1_3,a1_4)) else 0)
        def _pad(v, n):
            if v.size >= n:
                return v[:n]
            if v.size == 0:
                return np.full(n, np.nan)
            w = np.full(n, np.nan)
            w[:v.size] = v
            return w
        a1_1 = _pad(a1_1, n_trials)
        a1_2 = _pad(a1_2, n_trials)
        a1_3 = _pad(a1_3, n_trials)
        a1_4 = _pad(a1_4, n_trials)

        # RT 분해
        rt = a1_4
        rt_abs = np.abs(rt)
        rt_sign = np.sign(rt)  # -1=조기, +1=지연, 0=정시(근처)

        # 유효/무효 trial 정의
        valid_rt_mask = _valid_rt_mask(rt_abs, cfg)
        # 반응 자체의 유효성(대회 정의 상 A1_3=1이면 유효로 가정)
        resp_valid = (a1_3 == 1)
        # 최종 유효 trial: RT 유효 & 반응 유효
        valid_trial_mask = valid_rt_mask & resp_valid & ~np.isnan(rt)

        n_invalid_rt = int(np.sum(~valid_rt_mask & ~np.isnan(rt_abs)))
        n_missing_trials = int(np.sum(np.isnan(a1_1) | np.isnan(a1_2) | np.isnan(a1_3) | np.isnan(a1_4)))
        valid_trial_ratio = float(np.mean(valid_trial_mask)) if n_trials>0 else np.nan
        acc_overall = float(np.nanmean(a1_3)) if n_trials>0 else np.nan

        # 전역 분포 요약 (절댓값 기준)
        v_abs = rt_abs[valid_trial_mask]
        qd_abs = _safe_quantiles(v_abs, [0.10, 0.50, 0.90]) if v_abs.size>0 else {"q10":np.nan, "q50":np.nan, "q90":np.nan}
        mean_absRT_trim = _trimmed_mean(v_abs, cfg.trimmed_prop) if v_abs.size>0 else np.nan
        std_absRT = float(np.std(v_abs)) if v_abs.size>0 else np.nan
        cv_absRT = _cv(mean_absRT_trim if not np.isnan(mean_absRT_trim) else (float(np.mean(v_abs)) if v_abs.size>0 else np.nan), std_absRT)
        # IQR
        if v_abs.size>0:
            q25, q75 = np.quantile(v_abs, [0.25, 0.75])
            iqr_abs = float(q75 - q25)
        else:
            iqr_abs = np.nan

        # 조기/지연 비율 및 평균 부호
        sgn = rt_sign[valid_trial_mask]
        early_ratio = float(np.mean(sgn < 0)) if sgn.size>0 else np.nan
        late_ratio  = float(np.mean(sgn > 0)) if sgn.size>0 else np.nan
        mean_rt_sign = float(np.mean(sgn)) if sgn.size>0 else np.nan

        # 정규화된 정확도 대안 1: cap 기반 (허용편차 제공 시)
        if cfg.allowed_dev_ms is not None and v_abs.size>0:
            norm_err_cap = np.clip(v_abs / (cfg.allowed_dev_ms + cfg.small_eps), 0.0, 1.0)
            prediction_accuracy_cap = float(1.0 - np.mean(norm_err_cap))
        else:
            prediction_accuracy_cap = np.nan

        # 정규화된 정확도 대안 2: 로버스트 기반(개인 내부 척도)
        if v_abs.size>0:
            mad = _mad(v_abs)
            denom = mad if (mad > 0 and not np.isnan(mad)) else (np.median(v_abs) + cfg.small_eps)
            norm_err_robust = v_abs / (denom + cfg.small_eps)
            # 길항 효과 방지 위해 soft clipping
            norm_err_robust = np.log1p(norm_err_robust) / np.log(2.0)  # 1 -> 1, 3 -> ~2
            prediction_accuracy_robust = float(1.0 - np.mean(np.clip(norm_err_robust/3.0, 0.0, 1.0)))
        else:
            prediction_accuracy_robust = np.nan

        # 방향/속도별 집계 helper (절댓값/부호/정확도)
        def _agg_mask(mask: np.ndarray) -> Tuple[float, float, float, int]:
            idxs = np.where(mask & valid_trial_mask)[0]
            if idxs.size == 0:
                return np.nan, np.nan, np.nan, 0
            vals = rt_abs[idxs]
            svals = rt_sign[idxs]
            accs = a1_3[idxs]
            return float(np.mean(vals)), float(np.mean(svals)), float(np.mean(accs)), int(idxs.size)

        # 방향 요약
        mean_absRT_left,  mean_sign_left,  acc_left,  nL = _agg_mask(a1_1 == cfg.dir_left)
        mean_absRT_right, mean_sign_right, acc_right, nR = _agg_mask(a1_1 == cfg.dir_right)
        bias_direction = (mean_sign_right - mean_sign_left) if (not np.isnan(mean_sign_right) and not np.isnan(mean_sign_left)) else np.nan

        # 속도 요약
        mean_absRT_slow,   mean_sign_slow,   acc_slow,   nS = _agg_mask(a1_2 == cfg.sp_slow)
        mean_absRT_normal, mean_sign_normal, acc_normal, nN = _agg_mask(a1_2 == cfg.sp_normal)
        mean_absRT_fast,   mean_sign_fast,   acc_fast,   nF = _agg_mask(a1_2 == cfg.sp_fast)
        speed_bias_delta = (mean_sign_fast - mean_sign_slow) if (not np.isnan(mean_sign_fast) and not np.isnan(mean_sign_slow)) else np.nan

        # 2x3 셀: 절댓값평균/정확도평균
        cell_abs_means, cell_acc_means = {}, {}
        cell_labels = [
            (cfg.dir_left,  cfg.sp_slow,   "L_S"),
            (cfg.dir_left,  cfg.sp_normal, "L_N"),
            (cfg.dir_left,  cfg.sp_fast,   "L_F"),
            (cfg.dir_right, cfg.sp_slow,   "R_S"),
            (cfg.dir_right, cfg.sp_normal, "R_N"),
            (cfg.dir_right, cfg.sp_fast,   "R_F"),
        ]
        for d, s, tag in cell_labels:
            m = _cell_mask(a1_1, a1_2, cfg, d, s)
            mean_abs, _, accm, _ = _agg_mask(m)
            cell_abs_means[tag] = mean_abs
            cell_acc_means[tag] = accm
        abs_means = np.array([cell_abs_means[k] for k in ["L_S","L_N","L_F","R_S","R_N","R_F"]], dtype=float)
        var_across_6cells_absRT = float(np.nanvar(abs_means)) if np.any(~np.isnan(abs_means)) else np.nan

        # 동역학: 학습/피로/스위치/포스트-에러
        idxs = np.arange(1, n_trials+1, dtype=float)
        absRT_slope_over_trials = _ols_slope(idxs, rt_abs)
        # 전반/후반 평균 차이
        if n_trials >= 4:
            first_half = rt_abs[: n_trials//2]
            last_half  = rt_abs[n_trials//2 :]
            fatigue_delta_abs = float(np.nanmean(last_half) - np.nanmean(first_half))
        else:
            fatigue_delta_abs = np.nan
        post_error_slowing_abs = _post_error_slowing(rt_abs, a1_3)
        switch_cost_abs = _switch_cost(rt_abs, a1_1)

        row_feat = {
            # 전역/분포·편향
            "mean_absRT_trim": mean_absRT_trim,
            "std_absRT": std_absRT,
            "cv_absRT": cv_absRT,
            "q10_absRT": qd_abs["q10"],
            "q50_absRT": qd_abs["q50"],
            "q90_absRT": qd_abs["q90"],
            "iqr_absRT": iqr_abs,
            "early_ratio": early_ratio,
            "late_ratio": late_ratio,
            "mean_rt_sign": mean_rt_sign,
            "acc_overall": acc_overall,
            # 방향
            "mean_absRT_left": mean_absRT_left,
            "mean_absRT_right": mean_absRT_right,
            "acc_left": acc_left,
            "acc_right": acc_right,
            "bias_direction": bias_direction,
            # 속도
            "mean_absRT_slow": mean_absRT_slow,
            "mean_absRT_normal": mean_absRT_normal,
            "mean_absRT_fast": mean_absRT_fast,
            "acc_slow": acc_slow,
            "acc_normal": acc_normal,
            "acc_fast": acc_fast,
            "speed_bias_delta": speed_bias_delta,
            # 2x3 셀
            **{f"cell_absRT_mean_{k}": v for k, v in cell_abs_means.items()},
            **{f"cell_acc_mean_{k}": v for k, v in cell_acc_means.items()},
            "var_across_6cells_absRT": var_across_6cells_absRT,
            # 동역학
            "absRT_slope_over_trials": absRT_slope_over_trials,
            "fatigue_delta_abs": fatigue_delta_abs,
            "post_error_slowing_abs": post_error_slowing_abs,
            "switch_cost_abs": switch_cost_abs,
            # 품질
            "valid_trial_ratio": valid_trial_ratio,
            "n_missing_trials": n_missing_trials,
            "n_invalid_rt": n_invalid_rt,
            # 정확도(정규화 대안)
            "prediction_accuracy_cap": prediction_accuracy_cap,
            "prediction_accuracy_robust": prediction_accuracy_robust,
        }
        feats.append(row_feat)

    return pd.DataFrame(feats, index=df.index)


# --------- 사용 예시 ---------
if __name__ == "__main__":
    # 데모: 사용자 제공 예시 1행
    demo = pd.DataFrame({
        "A1_1": ["1,2,1,1,1,2,1,1,1,2,2,2,2,1,2,1,1,2"],
        "A1_2": ["1,1,2,2,2,1,2,3,3,3,2,3,2,3,1,2,2,2"],
        "A1_3": ["0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"],
        "A1_4": ["23,43,22,33,10,15,-27,35,38,-42,16,19,20,30,55,39,43,22"],
    })

    # 구성: 허용편차를 모르는 상황이므로 robust 지표를 우선 사용
    cfg = A1Config(allowed_dev_ms=None)
    feats = featurize_A1(demo, cfg=cfg)
    pd.set_option('display.max_columns', None)
    print(feats.head(1).T)