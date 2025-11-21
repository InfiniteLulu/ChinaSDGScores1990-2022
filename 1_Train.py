import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor


INPUT_PATH = r"E:\Data\sdata_Score_CompleteYear_1016\总数据表_latest_去除新增城市.xlsx"

INDICATOR_LIST = [
    "城镇职工基本养老保险参保人数(人)", "城镇基本医疗保险参保人数(人)", "失业保险参保人数(人)",
    "人均地区生产总值(元)", "职工平均工资(元)", "社会消费品零售总额(万元)",
    "地方财政一般预算内收入(万元)", "地方财政一般预算内支出(万元)", "城乡居民储蓄年末余额(万元)",
    "第一产业增加值(万元)", "第一产业增加值占GDP比重(%)", "农林牧渔业从业人员数(万人)",
    "医院、卫生院数(个)", "医院、卫生院床位数(张)", "医生数(人)", "生活垃圾无害化处理率(%)",
    "教育业从业人员数(万人)", "教育支出(万元)", "普通高等学校学校数(所)", "中等职业教育学校数(所)",
    "普通高等学校专任教师数(人)", "中等职业教育学校专任教师数(人)", "每百人公共图书馆藏书(册、件)",
    "水利、环境和公共设施管理业从业人员数(万人)", "工业废水排放量(万吨)", "地区生产总值增长率(%)",
    "年末单位从业人员数(万人)", "年末城镇登记失业人员数(人)", "在岗职工平均人数(万人)",
    "第二产业增加值(万元)", "第二产业增加值占GDP比重(%)", "建筑业从业人员数(万人)",
    "公路客运量(万人)", "民用航空货邮运量(吨)", "工业二氧化硫排放量(吨)",
    "工业烟粉尘排放量(吨)", "污水处理厂集中处理率(%)", "港、澳、台商投资企业数(个)",
    "外商投资企业数(个)", "国际互联网用户数(户)", "本年征用土地面积_平方公里",
    "本年征用土地面积_耕地_平方公里", "文盲综合值", "高等教育综合值", "人均日生活用水量_升",
    "供水普及率_百分比", "污水处理率_百分比", "燃气普及率_百分比", "每万人拥有公共交通车辆_标台",
    "人均道路面积_平方米", "人均公园绿地面积_平方米", "建成区绿地率_百分比",
    "建成区绿化覆盖率_百分比", "建成区排水管道密度_公里每平方公里", "用水人口_万人",
    "供水总量_居民家庭用水_万立方米", "供水总量_免费供水量_生活用水_万立方米",
    "全社会用电量(亿千瓦时)", "城市建设用地面积_道路交通设施用地_平方公里",
    "城市建设用地面积_公共管理与公共服务用地_平方公里", "城市建设用地面积_公共设施用地_平方公里",
    "城市建设用地面积_特殊用地_平方公里", "城市建设用地面积_绿地与广场用地_平方公里",
    "城市建设用地面积_物流仓储用地_平方公里", "环境产出废水排放量（万吨）",
    "环境产出SO2排放量（万吨）", "环境产出烟粉尘排放量（万吨）", "环境污染指数",
    "环保处罚统计_案件数目（篇）", "公园个数_门票免费_个", "AQI指数_年平均", "碳排放量",
]


INDICATORS_LOG1P = {name: True for name in INDICATOR_LIST}
for name in [
    "第一产业增加值占GDP比重(%)", "地区生产总值增长率(%)", "生活垃圾无害化处理率(%)",
    "污水处理厂集中处理率(%)", "供水普及率_百分比", "污水处理率_百分比", "燃气普及率_百分比",
    "建成区绿地率_百分比", "建成区绿化覆盖率_百分比", "建成区排水管道密度_公里每平方公里",
    "文盲综合值", "高等教育综合值", "每百人公共图书馆藏书(册、件)", "AQI指数_年平均", "环境污染指数",
]:
    if name in INDICATOR_LIST:
        INDICATORS_LOG1P[name] = False

OUTPUT_ROOT = r"E:\Data\sdata_Score_CompleteYear_1016\Model_multi_indicators"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 直辖市邻省联合
MUNI_GROUPS = {
    "北京市": ["北京市", "天津市", "河北省"],
    "天津市": ["天津市", "北京市", "河北省"],
    "上海市": ["上海市", "江苏省", "浙江省"],
    "重庆市": ["重庆市", "四川省"],
}
MUNICIPALITIES = set(MUNI_GROUPS.keys())

# 扩展模型与参数空间
MODELS = {
    "RandomForest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        [
            Integer(50, 1000, name="n_estimators"),
            Integer(2, 30, name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Real(0.1, 0.999, name="max_features"),
            Integer(1, 20, name="min_samples_leaf"),
            Categorical([True, False], name="bootstrap"),
            Real(0.0, 0.5, name="min_weight_fraction_leaf"),
        ],
    ),
    "SVR": (
        SVR(),
        [
            Real(1e-3, 1e+4, name="C", prior="log-uniform"),
            Real(1e-5, 1.0, name="epsilon", prior="log-uniform"),
            Categorical(["linear", "poly", "rbf", "sigmoid"], name="kernel"),
            Integer(1, 5, name="degree"),
            Real(1e-5, 1.0, name="gamma", prior="log-uniform"),
        ],
    ),
    "KNeighbors": (
        KNeighborsRegressor(),
        [
            Integer(1, 30, name="n_neighbors"),
            Categorical(["uniform", "distance"], name="weights"),
            Integer(1, 2, name="p"),
            Categorical(["auto", "ball_tree", "kd_tree", "brute"], name="algorithm"),
            Integer(10, 50, name="leaf_size"),
        ],
    ),
    "DecisionTree": (
        DecisionTreeRegressor(random_state=42),
        [
            Integer(2, 30, name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 20, name="min_samples_leaf"),
            Categorical([None, "sqrt", "log2"], name="max_features"),
            Integer(0, 200, name="max_leaf_nodes"),
        ],
    ),
    "XGBRegressor": (
        XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist"),
        [
            Integer(50, 1000, name="n_estimators"),
            Real(0.01, 0.5, name="learning_rate", prior="log-uniform"),
            Integer(2, 20, name="max_depth"),
            Real(0.5, 1.0, name="subsample", prior="uniform"),
            Real(0.5, 1.0, name="colsample_bytree", prior="uniform"),
            Real(0.0, 10.0, name="gamma"),
            Real(0.0, 10.0, name="reg_lambda"),
            Real(0.0, 10.0, name="reg_alpha"),
        ],
    ),
    "LGBMRegressor": (
        LGBMRegressor(random_state=42, n_jobs=-1),
        [
            Integer(50, 1000, name="n_estimators"),
            Real(0.01, 0.5, name="learning_rate", prior="log-uniform"),
            Integer(2, 20, name="max_depth"),
            Real(0.5, 1.0, name="subsample", prior="uniform"),
            Real(0.5, 1.0, name="colsample_bytree", prior="uniform"),
            Integer(20, 255, name="num_leaves"),
            Real(0.0, 10.0, name="reg_lambda"),
            Real(0.0, 10.0, name="reg_alpha"),
        ],
    ),
    "MLPRegressor": (
        MLPRegressor(random_state=42, max_iter=600),
        [
            Integer(50, 400, name="hidden_layer_sizes"),
            Real(1e-5, 1e-2, name="alpha", prior="log-uniform"),
            Categorical(["identity", "logistic", "tanh", "relu"], name="activation"),
            Real(1e-3, 1e-1, name="learning_rate_init", prior="log-uniform"),
            Categorical(["constant", "invscaling", "adaptive"], name="learning_rate"),
            Integer(300, 800, name="max_iter"),
        ],
    ),
}


def slugify(s: str) -> str:
    """把中文指标名转成安全的文件夹名：保留中文、字母数字，其他转下划线"""
    s = s.strip()
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^\w\u4e00-\u9fff\(\)（）%-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.replace("(", "_").replace(")", "_").replace("（", "_").replace("）", "_").replace("%", "百分比").replace("．", "_").replace("、", "_")


def cv_eval(model, X, y, n_splits=3, use_log=True):
    """时间序列交叉验证：返回 mean R2（对 y 或 log1p(y) 空间）、以及原空间 MSE"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    r2s, mses_raw = [], []
    for tr, va in tscv.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        model.fit(Xtr, ytr)
        p = model.predict(Xva)
        r2s.append(r2_score(yva, p))
        # MSE 用原空间
        yv_raw = np.expm1(yva) if use_log else yva
        pv_raw = np.expm1(p) if use_log else p
        mses_raw.append(mean_squared_error(yv_raw, pv_raw))
    return float(np.mean(r2s)), float(np.mean(mses_raw))


def optimize_model(name, model, space, X_fit, y_fit, use_log=True, n_calls=25):
    """贝叶斯优化，目标最大化 R2（这里返回负R2给 gp_minimize）"""
    # 特例：KNN 的 n_neighbors 不能超过样本数-1
    if name == "KNeighbors":
        n_max = max(2, min(30, len(y_fit) - 1))
        space = [
            Integer(1, n_max, name="n_neighbors"),
            Categorical(["uniform", "distance"], name="weights"),
            Integer(1, 2, name="p"),
            Categorical(["auto", "ball_tree", "kd_tree", "brute"], name="algorithm"),
            Integer(10, 50, name="leaf_size"),
        ]

    @use_named_args(space)
    def objective(**params):
        model.set_params(**params)
        r2_cv, _ = cv_eval(model, X_fit, y_fit, n_splits=3, use_log=use_log)
        return -r2_cv

    try:
        res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
        best_params = {p.name: v for p, v in zip(space, res.x)}
        model.set_params(**best_params)
        r2_cv, mse_cv_raw = cv_eval(model, X_fit, y_fit, n_splits=3, use_log=use_log)
        return best_params, r2_cv, mse_cv_raw
    except Exception as e:
        print(f"{name} 调参失败：{e}")
        return None, -1e9, 1e18

df = pd.read_excel(INPUT_PATH)
df = df.dropna(subset=["省份", "地区", "年份"])

for col in INDICATOR_LIST:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        print(f"警告：原表中未找到列：{col}")

all_provinces = df["省份"].unique()

for target_col in INDICATOR_LIST:
    if target_col not in df.columns:
        continue

    indicator_dir = os.path.join(OUTPUT_ROOT, slugify(target_col))
    os.makedirs(indicator_dir, exist_ok=True)

    print(f"指标：{target_col} 输出目录：{indicator_dir}")

    USE_LOG1P = INDICATORS_LOG1P.get(target_col, True)  # 默认 log1p

    results_rows = []

    for prov in tqdm(all_provinces, desc=f"Training [{target_col}]"):
        # 直辖市：邻省联合；其他省：本省 pooled
        if prov in MUNICIPALITIES:
            group = MUNI_GROUPS[prov]
            sub = df[df["省份"].isin(group)].copy()
            tag = f"{prov}_邻省组"
            enc_cols = ["省份", "地区"]
        else:
            sub = df[df["省份"] == prov].copy()
            tag = prov
            enc_cols = ["地区"]

        sub = sub.dropna(subset=["年份"])  # 年份不能空
        # 训练用：目标必须非空
        train_df = sub.dropna(subset=[target_col]).sort_values("年份").reset_index(drop=True)
        if len(train_df) < 8:
            print(f"{tag}: 样本太少（{len(train_df)}），跳过。")
            continue

        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc.fit(train_df[enc_cols])

        # 年份归一化范围
        year_min, year_max = train_df["年份"].min(), train_df["年份"].max()
        year_norm = (train_df["年份"] - year_min) / max(1e-9, (year_max - year_min))

        # 组特征
        X_reg = enc.transform(train_df[enc_cols])
        X = np.column_stack([X_reg, year_norm.values.reshape(-1, 1)])
        y_raw = train_df[target_col].values
        y = np.log1p(y_raw) if USE_LOG1P else y_raw

        # 留出法最后两年（如果年份过少就只留最后一年）
        uniq_years = np.sort(train_df["年份"].unique())
        hold_years = uniq_years[-2:] if len(uniq_years) >= 4 else uniq_years[-1:]
        m_hold = train_df["年份"].isin(hold_years).values
        X_fit, y_fit = X[~m_hold], y[~m_hold]
        X_hold, y_hold = X[m_hold], y[m_hold]

        if len(y_fit) < 5:  # 再保护一下
            print(f"{tag}: 可用于CV的样本太少（{len(y_fit)}），跳过。")
            continue

        # 搜索最优模型
        best = {"name": None, "r2_cv": -1e9, "mse_cv_raw": 1e18, "params": None}
        for name, (model, space) in MODELS.items():
            params, r2_cv, mse_cv_raw = optimize_model(
                name, model, space, X_fit, y_fit, use_log=USE_LOG1P, n_calls=25
            )
            if r2_cv > best["r2_cv"]:
                best.update({"name": name, "r2_cv": r2_cv, "mse_cv_raw": mse_cv_raw, "params": params})

        if not best["name"]:
            print(f"{tag}: 所有模型都调参失败，跳过。")
            continue

        # 用最优模型做最终验证，并保存
        final_model = None
        try:
            final_model = MODELS[best["name"]][0].set_params(**best["params"])
            final_model.fit(X_fit, y_fit)
            pred_hold = final_model.predict(X_hold)
            r2_hold = r2_score(y_hold, pred_hold)
            mse_hold_raw = mean_squared_error(
                np.expm1(y_hold) if USE_LOG1P else y_hold,
                np.expm1(pred_hold) if USE_LOG1P else pred_hold
            )
        except Exception as e:
            print(f"{tag}: 最终训练失败：{e}")
            continue

        # 持久化：保存 模型 + 编码器 + 年份范围 + 关键元信息
        model_fname = f"{slugify(tag)}_{best['name']}.pkl"
        joblib.dump(
            (final_model, enc, year_min, year_max, enc_cols, USE_LOG1P, target_col),
            os.path.join(indicator_dir, model_fname)
        )

        results_rows.append({
            "指标": target_col,
            "省份": prov,
            "联合省份": ",".join(MUNI_GROUPS[prov]) if prov in MUNICIPALITIES else prov,
            "最佳模型": best["name"],
            "R2_CV": round(best["r2_cv"], 4),
            "MSE_CV(raw)": round(best["mse_cv_raw"], 4),
            "R2_Holdout": round(float(r2_hold), 4),
            "MSE_Holdout(raw)": round(float(mse_hold_raw), 4),
            "训练样本数": int(len(y_fit)),
            "留出样本数": int(len(y_hold)),
            "模型文件": model_fname,
            "年份范围": f"{int(year_min)}-{int(year_max)}",
        })
        print(f"[{target_col}] {tag} → {best['name']} | R²(cv)={best['r2_cv']:.3f} | R²(hold)={r2_hold:.3f}")

    # 指标级汇总表
    if results_rows:
        res_df = pd.DataFrame(results_rows)
        out_xlsx = os.path.join(indicator_dir, "best_models_all_province_and_municipality.xlsx")
        res_df.to_excel(out_xlsx, index=False, engine="openpyxl")
        print(f"指标《{target_col}》训练完成，汇总已保存：{out_xlsx}")
    else:
        print(f"指标《{target_col}》无可用省份输出（样本过少或训练失败）。")

print("\n 全部指标训练流程结束。模型与汇总已按“每指标文件夹”分别保存。")
