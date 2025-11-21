import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


BASE_ORIG_PATH  = r"E:\Data\sdata_Score_CompleteYear_1016\result_v2\总数据表_latest_去除新增城市.xlsx"
BASE_FILLED_PATH = r"E:\Data\sdata_Score_CompleteYear_1016\result_v2\总数据表_补齐版.xlsx"
OUTPUT_CLEANED   = r"E:\Data\sdata_Score_CompleteYear_1016\result_v2\总数据表_补齐版_清洗后_去平台版.xlsx"
OUTPUT_REPORT    = r"E:\Data\sdata_Score_CompleteYear_1016\result_v2\总数据表_补齐版_清洗报告_去平台版.xlsx"

KEY_CODE = "行政区划代码"
KEY_YEAR = "年份"
ID_COLS  = ["省份", "地区", KEY_CODE, KEY_YEAR]


L_MIN = 4         # 平台段最小长度（连续≥4年）
EPS_REL = 1e-4    # 相邻年相对变化阈值


def is_percent_col(col: str) -> bool:
    return any(p in col for p in ["(%)", "百分比", "比重", "率"])

def canonical_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c not in ID_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out[KEY_CODE] = out[KEY_CODE].astype(str).str.strip()
    return out

def detect_flat_runs(values: np.ndarray) -> list:
    """检测平台段：连续≥L_MIN年，变化率<阈值"""
    v = np.asarray(values, dtype=float)
    dif = np.abs(np.diff(v))
    base = np.nanmedian(np.abs(v)) if np.isfinite(np.nanmedian(np.abs(v))) else 1.0
    thr = EPS_REL * max(base, 1.0)
    flat_step = dif < thr
    runs = []
    n = len(v)
    i = 0
    while i < n - 1:
        if flat_step[i]:
            j = i
            while j < n - 2 and flat_step[j + 1]:
                j += 1
            if (j + 1 - i + 1) >= L_MIN:
                runs.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return runs

# 读取数据
print("读取原始与补齐数据中...")
orig_df   = pd.read_excel(BASE_ORIG_PATH)
filled_df = pd.read_excel(BASE_FILLED_PATH)
orig_df   = canonical_numeric(orig_df)
filled_df = canonical_numeric(filled_df)

common_cols = [c for c in filled_df.columns if c in orig_df.columns]
filled_df = filled_df[common_cols]
orig_df   = orig_df[common_cols]

merge_key = [KEY_CODE, KEY_YEAR]
orig_sorted   = orig_df.sort_values(merge_key).reset_index(drop=True)
filled_sorted = filled_df.sort_values(merge_key).reset_index(drop=True)
assert len(orig_sorted) == len(filled_sorted), "行数不匹配，请检查补齐表！"

original_non_nan = {
    c: orig_sorted[c].notna().values for c in common_cols if c not in ID_COLS
}

# 主清洗流程
clean_df = filled_sorted.copy()
value_cols = [c for c in common_cols if c not in ID_COLS]
stats = []

print("开始清洗（插值 + 裁剪 + 平滑 + 去平台）...")
for col in tqdm(value_cols):
    is_pct = is_percent_col(col)

    for code, grp_idx in clean_df.groupby(KEY_CODE).groups.items():
        idx = list(grp_idx)
        s = clean_df.loc[idx, col].astype("float64").values
        y = clean_df.loc[idx, KEY_YEAR].values
        can_change = ~original_non_nan[col][idx]

        # 1) 线性插值
        ser = pd.Series(s, index=y)
        ser = ser.interpolate(method="linear", limit_direction="both")
        fill_mask = can_change & np.isnan(s) & ser.notna().values
        s[fill_mask] = ser.values[fill_mask]

        # 2) 边界裁剪
        if is_pct:
            s[can_change] = np.clip(s[can_change], 0, 100)
        else:
            s[can_change & (s < 0)] = 0

        # 3) 平滑处理
        ser2 = pd.Series(s, index=y)
        smooth_vals = ser2.rolling(window=3, center=True, min_periods=1).mean().values
        s[can_change] = smooth_vals[can_change]

        # 4) 去平台
        runs = detect_flat_runs(s)
        if runs:
            for (i0, i1) in runs:
                seg_mask = np.zeros_like(s, dtype=bool)
                seg_mask[i0:i1+1] = True
                if can_change[seg_mask].sum() == 0:
                    continue
                left_val = s[i0 - 1] if i0 - 1 >= 0 else s[i0]
                right_val = s[i1 + 1] if i1 + 1 < len(s) else s[i1]
                left_year = y[i0 - 1] if i0 - 1 >= 0 else y[i0]
                right_year = y[i1 + 1] if i1 + 1 < len(s) else y[i1]
                for k in range(i0, i1 + 1):
                    if can_change[k]:
                        ratio = (y[k] - left_year) / max(right_year - left_year, 1)
                        s[k] = left_val + ratio * (right_val - left_val)
                # 再次裁剪
                if is_pct:
                    s = np.clip(s, 0, 100)
                else:
                    s[s < 0] = 0

        clean_df.loc[idx, col] = s

    stats.append({
        "指标列": col,
        "是否百分比列": "是" if is_pct else "否"
    })

# 输出
front = ["省份", "地区", KEY_CODE, KEY_YEAR]
clean_df = clean_df[front + [c for c in clean_df.columns if c not in front]]

os.makedirs(os.path.dirname(OUTPUT_CLEANED), exist_ok=True)
clean_df.to_excel(OUTPUT_CLEANED, index=False, engine="openpyxl")
pd.DataFrame(stats).to_excel(OUTPUT_REPORT, index=False, engine="openpyxl")

print(f"\n清洗与去平台完成，输出：{OUTPUT_CLEANED}")
print(f"报告已生成：{OUTPUT_REPORT}")
