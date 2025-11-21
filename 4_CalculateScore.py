import os
import numpy as np
import pandas as pd


DATA_PATH = r"E:\Data\sdata_Score_CompleteYear_1016\result_v2\总数据表_补齐版_清洗后_去平台版.xlsx"
META_PATH = r"E:\Data\sdata_Score_CompleteYear_1016\SDG_indicators_summary_formatted.xlsx"
OUTPUT_DIR = r"E:\Data\sdata_Score_CompleteYear_1016\CalculateScore\score\Test3"

COL_CODE   = "行政区划代码"
COL_REGION = "地区"
COL_YEAR   = "年份"

YEAR_START = 1990
YEAR_END   = 2022
YEAR_LIST  = list(range(YEAR_START, YEAR_END + 1))

DIR_COL_CANDIDATES = ["正负向", "正负", "方向", "指标方向", "正/负"]

UNKNOWN_DIR_POLICY = "skip"

# 其他参数
EPS = 1e-12
DROP_ZERO_VAR = True


def detect_dir_col(meta_df: pd.DataFrame) -> str:
    for c in DIR_COL_CANDIDATES:
        if c in meta_df.columns:
            return c
    for c in meta_df.columns:
        if any(k in str(c) for k in ["正", "负", "向"]):
            return c
    raise ValueError("指标说明表未找到“正负向”列，请检查。")


def load_meta(path: str) -> pd.DataFrame:
    meta = pd.read_excel(path)
    if "SDG" not in meta.columns or "指标名称" not in meta.columns:
        raise ValueError("指标说明表必须包含列：SDG, 指标名称, 正负向")

    # 处理合并单元格：向下填充 SDG
    meta["SDG"] = meta["SDG"].ffill()

    dir_col = detect_dir_col(meta)
    df = meta[["SDG", "指标名称", dir_col]].copy()

    # 规范“正负向”
    map_dir = {
        "正": "正", "正向": "正", "pos": "正", "positive": "正", "+": "正",
        "负": "负", "负向": "负", "neg": "负", "negative": "负", "-": "负"
    }
    df["方向"] = df[dir_col].astype(str).str.strip().map(map_dir)

    # 清洗指标名
    def norm(s):
        return str(s).replace("（","(").replace("）",")").strip()
    df["指标名称"] = df["指标名称"].map(norm)
    df["SDG"] = df["SDG"].astype(str).str.strip()

    return df[["SDG", "指标名称", "方向"]]


def entropy_weight(df_norm: pd.DataFrame) -> pd.Series:
    """ df_norm：行=城市，列=指标（已标准化到[0,1]） """
    # p_ij
    col_sum = df_norm.sum(axis=0).replace(0, EPS)
    p = df_norm.div(col_sum, axis=1).clip(lower=EPS)
    k = 1.0 / np.log(len(df_norm))
    e = -k * (p * np.log(p)).sum(axis=0)
    d = 1 - e
    if d.sum() <= EPS:
        # 极端情况：全部无信息，均分
        return pd.Series(1.0 / len(d), index=d.index)
    return d / d.sum()


def minmax_pos(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(1.0, index=s.index)
    return (s - s.min()) / rng


def minmax_neg(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(1.0, index=s.index)
    return (s.max() - s) / rng


def interpolate_by_city_time(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """对每个城市在时间轴上做线性插值（limit_direction='both'）"""
    def _interp(g):
        g = g.sort_values(COL_YEAR)
        g[cols] = g[cols].apply(lambda x: pd.to_numeric(x, errors="coerce").interpolate(
            method="linear", limit_direction="both"))
        return g
    return df.groupby(COL_CODE, group_keys=False).apply(_interp)


def normalize_year_block(g: pd.DataFrame, pos_cols: list, neg_cols: list):
    """对某一年的城市样本做正/负向 Min–Max 标准化；返回标准化矩阵及该年被剔除的零方差列"""
    zero_var_cols = []
    df_norm = pd.DataFrame(index=g.index)

    for c in pos_cols:
        if c in g.columns:
            s = pd.to_numeric(g[c], errors="coerce")
            if s.max() - s.min() < EPS:
                zero_var_cols.append(c)
                continue
            df_norm[c] = minmax_pos(s)

    for c in neg_cols:
        if c in g.columns:
            s = pd.to_numeric(g[c], errors="coerce")
            if s.max() - s.min() < EPS:
                zero_var_cols.append(c)
                continue
            df_norm[c] = minmax_neg(s)

    if DROP_ZERO_VAR and zero_var_cols:
        df_norm = df_norm.drop(columns=[c for c in zero_var_cols if c in df_norm.columns], errors="ignore")
    return df_norm, zero_var_cols


def calculate_scores_for_sdg(sdg_name: str, meta_df: pd.DataFrame, raw_df: pd.DataFrame):
    """返回该 SDG 的结果表（不含 composite_score）"""
    sub = meta_df[meta_df["SDG"] == sdg_name].copy()

    known = sub[sub["方向"].isin(["正", "负"])].drop_duplicates(subset=["指标名称", "方向"])
    unknown = sub[~sub["方向"].isin(["正", "负"])]

    if not unknown.empty:
        msg = f"[{sdg_name}] 有 {len(unknown)} 个未知方向指标：{list(unknown['指标名称'])}"
        if UNKNOWN_DIR_POLICY == "skip":
            print(msg + " -> 已跳过")
        elif UNKNOWN_DIR_POLICY == "treat_pos":
            print(msg + " -> 按正向处理")
            unknown["方向"] = "正"
            known = pd.concat([known, unknown], ignore_index=True)
        elif UNKNOWN_DIR_POLICY == "treat_neg":
            print(msg + " -> 按负向处理")
            unknown["方向"] = "负"
            known = pd.concat([known, unknown], ignore_index=True)

    pos_cols = [c for c in known.loc[known["方向"] == "正", "指标名称"] if c in raw_df.columns]
    neg_cols = [c for c in known.loc[known["方向"] == "负", "指标名称"] if c in raw_df.columns]
    use_cols = pos_cols + neg_cols

    missing = [c for c in known["指标名称"].tolist() if c not in raw_df.columns]
    if missing:
        print(f"[{sdg_name}] 警告：总数据表缺少列（已忽略）：{missing}")

    if not use_cols:
        print(f"[{sdg_name}] 无可用指标，跳过。")
        return None

    # 仅保留必要列
    df = raw_df[[COL_CODE, COL_REGION, COL_YEAR] + use_cols].copy()

    # 线性插值（按城市）
    df = interpolate_by_city_time(df, use_cols)

    # 建立结果骨架
    codes = df[[COL_CODE, COL_REGION]].drop_duplicates().sort_values([COL_CODE, COL_REGION])
    result = codes.copy()
    # 每年的得分列
    for y in YEAR_LIST:
        result[str(y)] = np.nan

    # 逐年计算熵权得分
    for y, g in df.groupby(COL_YEAR):
        if y < YEAR_START or y > YEAR_END:
            continue
        g = g.set_index([COL_CODE, COL_REGION])
        g_norm, zero_var_cols = normalize_year_block(g[use_cols], pos_cols, neg_cols)
        if g_norm.shape[1] == 0:
            # 该年所有指标零方差或缺失
            continue
        w = entropy_weight(g_norm)
        score = (g_norm * w).sum(axis=1)
        # 回填到结果
        score_col = str(y)
        result = result.merge(score.reset_index().rename(columns={0: score_col, score.name: score_col}),
                              on=[COL_CODE, COL_REGION], how="left", suffixes=("", "_tmp"))
        if score_col + "_tmp" in result.columns:
            result[score_col] = result[score_col + "_tmp"]
            result.drop(columns=[score_col + "_tmp"], inplace=True)

    # 排序列顺序：code, region, 1990..2022
    ordered_cols = [COL_CODE, COL_REGION] + [str(y) for y in YEAR_LIST]
    result = result.reindex(columns=ordered_cols)

    # 所有年份得分 ×100
    result.iloc[:, 2:] = result.iloc[:, 2:] * 100

    # 重命名成英文字段名
    #result = result.rename(columns={COL_CODE: "code", COL_REGION: "region"})

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读入数据
    raw = pd.read_excel(DATA_PATH)
    raw.columns = [str(c).replace("（", "(").replace("）", ")").strip() for c in raw.columns]
    meta = load_meta(META_PATH)

    # 规范年份为整数
    raw[COL_YEAR] = pd.to_numeric(raw[COL_YEAR], errors="coerce").astype("Int64")

    # SDG 列表（保证顺序 SDG 1..17）
    sdg_list = sorted(meta["SDG"].dropna().unique(), key=lambda x: int(str(x).split()[-1]))

    for sdg in sdg_list:
        res = calculate_scores_for_sdg(sdg, meta, raw)
        if res is None:
            continue
        sdg_idx = int(str(sdg).split()[-1])
        out_path = os.path.join(OUTPUT_DIR, f"score_{sdg_idx}.xlsx")
        res.to_excel(out_path, index=False)
        print(f"输出：{out_path}")

    print("全部SDG年度得分已生成（不含综合均值列）。")


if __name__ == "__main__":
    main()
