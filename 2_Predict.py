import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


INPUT_PATH  = r"E:\Data\sdata_Score_CompleteYear_1016\总数据表_latest.xlsx"
MODEL_ROOT  = r"E:\Data\sdata_Score_CompleteYear_1016\Model_multi_indicators"
OUTPUT_ROOT = r"E:\Data\sdata_Score_CompleteYear_1016\Predicted_multi_indicators"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def canonical(s: str) -> str:
    """规范化字符串"""
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("_", "").replace(" ", "")
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    return s

def find_matching_column(folder_name: str, columns) -> str | None:
    """根据文件夹名模糊匹配列名"""
    cand = canonical(folder_name)
    for col in columns:
        if canonical(col) == cand:
            return col
    for col in columns:
        if canonical(col) in cand or cand in canonical(col):
            return col
    return None

def ensure_2d(x):
    x = np.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x

df = pd.read_excel(INPUT_PATH)
df = df.dropna(subset=["省份", "地区", "年份", "行政区划代码"]).copy()
df["行政区划代码"] = df["行政区划代码"].astype(str).str.strip()
for c in df.columns:
    if c not in ["省份", "地区", "年份", "行政区划代码"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

print(f"数据加载完成，共 {len(df)} 条记录，{df['省份'].nunique()} 个省份，{df['地区'].nunique()} 个地级市。")

for ind_dir in tqdm(sorted(os.listdir(MODEL_ROOT)), desc="Predicting all indicators"):
    indicator_dir = os.path.join(MODEL_ROOT, ind_dir)
    if not os.path.isdir(indicator_dir):
        continue

    best_path = os.path.join(indicator_dir, "best_models_all_province_and_municipality.xlsx")
    if not os.path.exists(best_path):
        print(f"{ind_dir} 缺少模型汇总表，跳过。")
        continue

    # 匹配主表列名
    target_col = find_matching_column(ind_dir, df.columns)
    print(f"\n正在预测指标：《{ind_dir}》")
    print(f"匹配到列名: {target_col}")
    if not target_col:
        print("未匹配到列名，跳过。")
        continue

    # 当前指标数据
    work = df[["年份", "省份", "地区", "行政区划代码", target_col]].copy()
    best_df = pd.read_excel(best_path)

    # 收集预测结果
    pred_rows = []

    # 分省预测
    for _, row in best_df.iterrows():
        prov = str(row["省份"]).strip()
        model_file = str(row["模型文件"]).strip()
        model_path = os.path.join(indicator_dir, model_file)
        if not os.path.exists(model_path):
            continue

        sub = work[work["省份"] == prov].copy()
        if sub.empty:
            continue

        miss_mask = sub[target_col].isna().values
        if not miss_mask.any():
            continue

        # 载入模型
        payload = joblib.load(model_path)
        if isinstance(payload, tuple) and len(payload) >= 7:
            model, enc, year_min, year_max, enc_cols, use_log1p_flag, saved_target = payload
        else:
            model = payload
            enc_cols = ["地区"]
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            enc.fit(sub[enc_cols])
            year_min, year_max = sub["年份"].min(), sub["年份"].max()
            use_log1p_flag = True

        # 特征构造
        sub["year_norm"] = (sub["年份"] - year_min) / max(1e-9, (year_max - year_min))
        X_reg = enc.transform(sub[enc_cols])
        X_all = np.column_stack([X_reg, sub["year_norm"].values.reshape(-1, 1)])
        preds = model.predict(X_all)
        if use_log1p_flag:
            preds = np.expm1(preds)

        # 保存预测结果（仅缺失行）
        sub_pred = sub.loc[miss_mask, ["行政区划代码", "年份", "省份", "地区"]].copy()
        sub_pred[f"{target_col}_pred"] = preds[miss_mask]
        pred_rows.append(sub_pred)

        # 回填缺失
        sub.loc[miss_mask, target_col] = preds[miss_mask]
        work.loc[sub.index, target_col] = sub[target_col].values

        print(f"{prov} 已补 {miss_mask.sum()} 条缺失")

    # 输出
    indicator_out_dir = os.path.join(OUTPUT_ROOT, ind_dir)
    os.makedirs(indicator_out_dir, exist_ok=True)

    if pred_rows:
        # 保存预测结果表（仅缺失补齐部分）
        pred_df = pd.concat(pred_rows, ignore_index=True)
        pred_out_path = os.path.join(indicator_out_dir, f"{target_col}_预测结果.xlsx")
        pred_df.to_excel(pred_out_path, index=False, engine="openpyxl")
        print(f"预测结果保存：{pred_out_path}")

        # 保存补齐后的总表（仅本指标列被更新）
        filled_out_path = os.path.join(indicator_out_dir, f"{target_col}_已补齐.xlsx")
        filled_df = df.copy()
        key_cols = ["行政区划代码", "年份"]
        merged = filled_df.merge(pred_df[key_cols + [f"{target_col}_pred"]], on=key_cols, how="left")
        need_fill = merged[target_col].isna() & merged[f"{target_col}_pred"].notna()
        merged.loc[need_fill, target_col] = merged.loc[need_fill, f"{target_col}_pred"]
        merged.drop(columns=[f"{target_col}_pred"], inplace=True)
        merged.to_excel(filled_out_path, index=False, engine="openpyxl")
        print(f"补齐后总表保存：{filled_out_path}")
    else:
        print(f"指标《{target_col}》无缺失或模型缺失，未输出。")

print("\n所有指标预测与填充完成！")
