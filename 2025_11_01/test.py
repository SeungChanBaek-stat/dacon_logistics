## test
import numpy as np

x = [16.0, -33.0]
x = np.array(x)

print(x)
print(type(x))

mean_x = sum(x) / 2


var_x = [(16 - mean_x) ** 2, (-33 - mean_x) ** 2]
var_x = sum(var_x) / 2
mu_sigma = mean_x / np.sqrt(var_x)

def _summarize_list(x, mode='mu'):
    """
    x: list-like of numbers
    mode: 'mu' | 'mu_sigma' | 'mu_range'
    """
    if x is None:
        return np.nan
    arr = np.asarray(list(x), dtype=float)  # list/tuple/np.array 모두 허용
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return np.nan
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan

    mu = arr.mean()
    if mode == 'mu':
        return float(mu)
    elif mode == 'mu_sigma':
        sd = arr.std(ddof=0)
        return float(mu / sd) if sd > 0 else np.nan
    elif mode == 'mu_range':
        r = arr.max() - arr.min()
        return float(mu / r) if r > 0 else np.nan
    else:
        raise ValueError("mode must be one of {'mu','mu_sigma','mu_range'}")



print(mu_sigma)

print(_summarize_list([16.0, -33.0], mode = "mu_sigma"))