import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="中考英语词表可视化（Python/Streamlit）", layout="wide")

METRICS = [
    "tf_passage", "tf_item", "tf_total", "df", "num_passages", "coverage",
    "idf", "tfidf", "dispersion", "general_score", "passage_frac",
    "passage_priority_score", "passage_df"
]

def coerce_number_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def load_and_prepare(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # 统一列名去空格
    df.columns = [c.strip() for c in df.columns]

    # word 列兜底
    if "word" not in df.columns:
        for alt in ["Word", "WORD", "lemma", "Lemma"]:
            if alt in df.columns:
                df["word"] = df[alt].astype(str)
                break
    if "word" not in df.columns:
        raise ValueError("CSV 必须包含列：word（或 Lemma/Word）。")

    if "词汇等级by课标" not in df.columns:
        raise ValueError("CSV 必须包含列：词汇等级by课标。")

    # 指标转数值
    for m in METRICS:
        if m in df.columns:
            df[m] = coerce_number_series(df[m])
        else:
            df[m] = 0.0  # 缺列也能运行

    # 等级转 int
    df["词汇等级by课标"] = pd.to_numeric(df["词汇等级by课标"], errors="coerce").fillna(0).astype(int)

    # word 清理
    df["word"] = df["word"].astype(str).str.strip()
    df = df[df["word"] != ""]
    return df

st.title("中考英语词表可视化（Python/Streamlit）")
st.markdown(
    "上传脚本生成的 CSV（推荐：**vocab_full_metrics.csv**）。"
    "使用左侧筛选与上方 Tabs 切换指标，两个图会同步更新。"
)

uploaded = st.file_uploader("选择 CSV 文件（UTF-8/utf-8-sig）", type=["csv"])

# 侧边栏筛选
with st.sidebar:
    st.header("筛选")
    kb_level = st.selectbox("词汇等级by课标", options=["全部", 3, 2, 0], index=0, help="3=三级（包含但不在二级）、2=二级、0=不在考纲")
    top_n = st.slider("Top N", min_value=10, max_value=300, value=50, step=10)

if uploaded is not None:
    try:
        df = load_and_prepare(uploaded)
    except Exception as e:
        st.error(f"CSV 读取/校验失败：{e}")
        st.stop()

    # 应用筛选
    if kb_level != "全部":
        df_f = df[df["词汇等级by课标"] == int(kb_level)].copy()
    else:
        df_f = df.copy()

    st.caption(f"已加载 {len(df)} 行；当前筛选后 {len(df_f)} 行。")

    # 构建指标 Tabs
    tabs = st.tabs(METRICS)

    for i, metric in enumerate(METRICS):
        with tabs[i]:
            st.subheader(f"指标：{metric}")
            if metric not in df_f.columns:
                st.warning(f"列 {metric} 不在 CSV 中，默认按 0 处理。")

            # 取 Top N
            tmp = df_f[["word", metric]].copy()
            tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce").fillna(0.0)
            tmp = tmp.sort_values(metric, ascending=False).head(top_n)

            col1, col2 = st.columns(2)

            # 柱状图
            with col1:
                st.markdown("**排名图（降序）**")
                if tmp.empty:
                    st.info("无数据")
                else:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(tmp["word"], tmp[metric])
                    ax.set_xticklabels(tmp["word"], rotation=60, ha="right", fontsize=8)
                    ax.set_ylabel(metric)
                    ax.set_xlabel("word")
                    ax.grid(True, linestyle="--", linewidth=0.5, axis="y")
                    st.pyplot(fig, clear_figure=True)

            # 词云
            with col2:
                st.markdown("**词云（与左图同条件）**")
                if tmp.empty:
                    st.info("无数据")
                else:
                    # 用 itertuples + getattr 安全取值（修复你的报错）
                    freq = {}
                    for row in tmp.itertuples(index=False, name="Row"):
                        w = str(getattr(row, "word"))
                        v = float(getattr(row, metric))
                        if not np.isfinite(v) or v <= 0:
                            v = 1.0
                        freq[w] = v
                    wc = WordCloud(width=800, height=500, background_color="white")
                    wc.generate_from_frequencies(freq)
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.imshow(wc, interpolation="bilinear")
                    ax2.axis("off")
                    st.pyplot(fig2, clear_figure=True)

    # 数据预览
    with st.expander("查看原始数据（前 200 行）"):
        st.dataframe(df.head(200))
else:
    st.info("请先上传 CSV。")

st.markdown("""
---
**字段对照（与脚本一致）**：
- 指标 Tabs：`tf_passage`, `tf_item`, `tf_total`, `df`, `num_passages`, `coverage`,
  `idf`, `tfidf`, `dispersion`, `general_score`, `passage_frac`, `passage_priority_score`, `passage_df`
- 筛选：`词汇等级by课标`（3 / 2 / 0 / 全部）
""")
