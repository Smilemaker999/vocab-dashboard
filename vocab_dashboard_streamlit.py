# vocab_dashboard_streamlit.py
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from wordcloud import WordCloud

# ---------------- 基本设置 ----------------
st.set_page_config(page_title="中考英语词表可视化", layout="wide")

METRICS = [
    "tf_passage","tf_item","tf_total","df","num_passages","coverage",
    "idf","tfidf","dispersion","general_score","passage_frac",
    "passage_priority_score","passage_df"
]
METRIC_DESC_ZH = {
    "tf_passage":"正文高频词汇",
    "tf_item":"题目高频词汇",
    "tf_total":"正文+题目高频词汇",
    "df":"在多少语篇中出现（含正文+题目）",
    "num_passages":"参与统计的总语篇数",
    "coverage":"单词的语篇出现概率（出现语篇/总语篇）",
    "idf":"逆文档出现概率（越大说明语篇中出现概率低）",
    "tfidf":"平衡总出现频次和语篇稀有度（中间值均衡性较好）",
    "dispersion":"年份出现均匀程度（接近0只在一年一个地区出现，接近1每年每地区都出现）",
    "general_score":"综合高频词（出现总次数多，出现在单篇文章的概率大，且出现的年份多）",
    "passage_frac":"正文贡献占比（正文频次在总频次中的加权占比）",
    "passage_priority_score":"正文加权综合高频词（更多考虑正文贡献后的综合评分）",
    "passage_df":"在多少语篇中出现（仅含正文）"
}
TAB_TITLES = {
    k:(f"🪐 {k}｜{METRIC_DESC_ZH[k]}" if k=="general_score" else f"{k}｜{METRIC_DESC_ZH[k]}")
    for k in METRICS
}

# —— 详细解释（面向非编程同学）——  # ★ MOD（新增：长解释面板使用）
METRIC_LONG_DESC = {
    "tf_passage": """
**tf_passage（正文高频词汇）**  
表示这个词在“文章正文”里出现的总次数（同一篇里多次出现也会累计）。  
**怎么用：** 找“阅读材料里反复出现”的词。  
**如何解读：** tf_passage 高但 coverage/df 低 ⇒ 可能集中在少数文章里；偏“篇内高频”而非“通用词”。
""",
    "tf_item": """
**tf_item（题目高频词汇）**  
统计的是“题干+选项”，但**每道题只算 1 次**（binary），避免同题重复刷高。  
**怎么用：** 找命题常用提示词/设问词/选项词。  
**如何解读：** tf_item 高而 tf_passage 低 ⇒ 更偏“作答词”，未必是“阅读通用词”。
""",
    "tf_total": """
**tf_total（正文+题目高频词汇）** = tf_passage + tf_item  
**怎么用：** 作为“出现强度”的总指标，粗排常见词。  
**如何解读：** 与 coverage/df 搭配看更稳妥：tf_total 高 + 覆盖广 ⇒ 更通用。
""",
    "df": """
**df（文档频率）**  
一个词在多少篇不同语篇中出现（把该篇的正文与所有题合并看，只要出现一次就记入）。  
**怎么用：** 直观看覆盖面。  
**如何解读：** df 高 ⇒ 更常见、更泛用；df 低 ⇒ 可能是“主题词/话题词”。
""",
    "num_passages": """
**num_passages（总语篇数）**  
参与统计的文章总数，用于做分母（如 coverage）。  
**如何解读：** 本身不是排序指标，是理解其它比例指标的“总盘子”。
""",
    "coverage": """
**coverage（覆盖率）** = df / num_passages  
表示覆盖了多少比例的文章。  
**怎么用：** 选“通用词”（越接近 1 越通用）。  
**如何解读：** coverage 高但 tf_total 低 ⇒ “处处见，但次数少”；反之 ⇒ “少数文章里很多”。
""",
    "idf": """
**idf（逆文档频率）** = log((num_passages+1)/(df+1)) + 1  
覆盖面越小，idf 越大（越稀有）。  
**怎么用：** 用在 tfidf 中平衡“常见 vs 稀有”。  
**如何解读：** 单看 idf 越大越稀有，不适合“通用词”筛选。
""",
    "tfidf": """
**tfidf**  
= tf_total × idf。它会给“在少数文章里频繁出现”的词更高分。  
**怎么用：** 剔除极端值。它像一个智能过滤器，帮我们自动排除那些"太普通"和"太特殊"的词汇。  
**如何解读：** 低值区代表无意义高频词（龙套词），高值区代表过度专业词（偶发词），中间值为核心词汇。
""",
    "dispersion": """
**dispersion（分布均匀度）**  
按（地区, 年份）单元统计出现分布，计算变异系数 CV 并取 1/(1+CV)。越接近 1 越均匀。  
**怎么用：** 过滤只在某一年/某地区冒头的偏门词。  
**如何解读：** 常与 coverage 搭配：覆盖广 + 均匀 ⇒ 更稳定。
""",
    "general_score": """
**general_score（综合高频词）**  
= (coverage^β) × (归一化 tf_total^α) × dispersion（默认 β=2 强调覆盖，α=1 兼顾频次）。  
**怎么用：** 作为“通用词表”的主排序，越高越通用。  
**如何解读：** 同时考虑“次数多、覆盖广、分布均匀”，适合挑“教学必备词”。
""",
    "passage_frac": """
**passage_frac（正文贡献占比）**  
在正文与题目端分别加权后（正文权重大于题目），该值表示“总得分里正文占比”。  
**怎么用：** 想让词表更贴近“真实阅读”时，可设下限。  
**如何解读：** 值越大越偏正文。
""",
    "passage_priority_score": """
**passage_priority_score（正文加权综合分）**  
在 general_score 上再乘以 passage_frac^γ（默认 γ=1），进一步偏向正文贡献高的词。  
**怎么用：** 做“通用且更偏正文”的排序。
""",
    "passage_df": """
**passage_df（仅正文覆盖语篇数）**  
只统计正文本覆盖，不看题目端。  
**怎么用：** 过滤“主要出现在题目端”的词。
"""
}

# 课标着色（图3）
KB_LEVEL_COLOR = {3:"#d62728", 2:"#1f77b4", 0:"#7f7f7f"}  # 红/蓝/灰
# CEFR 着色（图4）：0 灰，1..6 同色系加深
def color_for_cefr(n):
    try:
        n = int(n)
    except:
        n = 0
    if n == 0:
        return "#7f7f7f"
    shades = ["#c7c1f0","#a89ee9","#8a7be2","#6a5acd","#4f3fb4","#392a99"]
    return shades[max(1,min(6,n))-1]

# ---------------- 工具函数 ----------------
def coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def load_and_prepare(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    if "word" not in df.columns:
        for alt in ["Word","WORD","lemma","Lemma"]:
            if alt in df.columns:
                df["word"] = df[alt].astype(str)
                break
    if "word" not in df.columns:
        raise ValueError("CSV 必须包含列：word（或 Lemma/Word）。")
    if "词汇等级by课标" not in df.columns:
        raise ValueError("CSV 必须包含列：词汇等级by课标。")
    if "CEFR_numeric" not in df.columns:
        df["CEFR_numeric"] = 0
    if "CEFR_level" not in df.columns:
        df["CEFR_level"] = ""

    for m in METRICS:
        if m in df.columns: df[m] = coerce_num(df[m])
        else: df[m] = 0.0

    df["词汇等级by课标"] = pd.to_numeric(df["词汇等级by课标"], errors="coerce").fillna(0).astype(int)
    df["CEFR_numeric"] = pd.to_numeric(df["CEFR_numeric"], errors="coerce").fillna(0).astype(int)
    df["CEFR_level"] = df["CEFR_level"].astype(str)
    df["word"] = df["word"].astype(str).str.strip()
    df = df[df["word"]!=""]
    return df

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0); return buf.getvalue()

def df_to_excel_or_csv_bytes(df: pd.DataFrame, sheet_name="selection"):
    try:
        import xlsxwriter  # noqa
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            df.to_excel(w, index=False, sheet_name=sheet_name)
        buf.seek(0); return buf.getvalue(), ".xlsx"
    except Exception:
        pass
    try:
        import openpyxl  # noqa
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name=sheet_name)
        buf.seek(0); return buf.getvalue(), ".xlsx"
    except Exception:
        s = io.StringIO()
        df.to_csv(s, index=False, encoding="utf-8-sig")
        return s.getvalue().encode("utf-8-sig"), ".csv"

# ---------------- 样式（Tabs 字号） ----------------
st.markdown("<style>.stTabs [role='tab']{font-size:14px!important;}</style>", unsafe_allow_html=True)

# ---------------- 页头 ----------------
st.title("中考英语词表可视化")
st.markdown("上传脚本生成的 CSV（推荐：**vocab_full_metrics.csv**）。左侧筛选 + 上方 Tabs 切换指标。")

uploaded = st.file_uploader("选择 CSV 文件（UTF-8/utf-8-sig）", type=["csv"])

# ---------------- 状态默认值 ----------------
DEFAULTS = {
    "kb_levels":[0,2,3],              # 课标多选
    "cefr_levels":[0,1,2,3,4,5,6],    # CEFR 多选
    "top_n":50,
    "sort_order":"降序",
    "mode":"Top N",
    "range_from":1,
    "range_to":100
}
for k,v in DEFAULTS.items():
    st.session_state.setdefault(k,v)
st.session_state.setdefault("_rows_max",300)
st.session_state.setdefault("_last_filter_signature",None)

# ---- 回调：Top N/区间控件同步（确保滑动立即生效） ----
def _sync_top_n_from_slider():
    st.session_state.top_n = int(st.session_state.top_n_slider)
def _sync_top_n_from_number():
    v = int(st.session_state.top_n_number)
    v = max(1, min(v, st.session_state._rows_max))
    st.session_state.top_n = v
def _sync_range_from_slider():
    a,b = st.session_state.range_slider
    a,b = int(min(a,b)), int(max(a,b))
    st.session_state.range_from, st.session_state.range_to = a,b
def _sync_range_from_numbers():
    a = max(1, int(st.session_state.range_from_num))
    b = max(1, int(st.session_state.range_to_num))
    a,b = int(min(a,b)), int(max(a,b))
    a = min(a, st.session_state._rows_max)
    b = min(b, st.session_state._rows_max)
    st.session_state.range_from, st.session_state.range_to = a,b

# ---------------- 主流程 ----------------
if uploaded is None:
    st.info("请先上传 CSV。")
    st.stop()

try:
    df = load_and_prepare(uploaded)
except Exception as e:
    st.error(f"CSV 读取/校验失败：{e}")
    st.stop()

# 侧栏筛选
with st.sidebar:
    st.header("筛选")
    kb_opts = [0,2,3]
    st.session_state.kb_levels = st.multiselect(
        "词汇等级by课标（多选）",
        options=kb_opts,
        default=st.session_state.kb_levels if set(st.session_state.kb_levels).issubset(kb_opts) else kb_opts
    )
    cefr_opts = [0,1,2,3,4,5,6]
    st.session_state.cefr_levels = st.multiselect(
        "词汇等级by CEFR（多选）",
        options=cefr_opts,
        default=st.session_state.cefr_levels if set(st.session_state.cefr_levels).issubset(cefr_opts) else cefr_opts,
        help=("CEFR_numeric：1=A1，2=A2，3=B1，4=B2，5=C1，6=C2，0=未指定")
    )

# 应用筛选
df_f = df[
    df["词汇等级by课标"].isin(st.session_state.kb_levels) &
    df["CEFR_numeric"].isin(st.session_state.cefr_levels)
].copy()

current_rows = len(df_f)
st.caption(f"已加载 {len(df)} 行；当前筛选后 {current_rows} 行。")

# 冻结上限（筛选变化时更新）
sig = (tuple(sorted(st.session_state.kb_levels)), tuple(sorted(st.session_state.cefr_levels)))
if st.session_state._last_filter_signature != sig:
    st.session_state._rows_max = max(10, current_rows)
    st.session_state._last_filter_signature = sig
dynamic_max = max(10, int(st.session_state._rows_max))

# 侧栏 TopN/区间、排序
with st.sidebar:
    st.session_state.mode = st.radio(
        "选择模式", ["Top N","区间"],
        index=0 if st.session_state.mode=="Top N" else 1, horizontal=True
    )
    if st.session_state.mode == "Top N":
        c1,c2 = st.columns([3,1])
        with c1:
            st.slider(
                "Top N（滑块）",
                min_value=1, max_value=dynamic_max, step=1,
                value=int(min(st.session_state.top_n, dynamic_max)),
                key="top_n_slider",
                on_change=_sync_top_n_from_slider,  # 滑动即生效
            )
        with c2:
            st.number_input(
                "输入", min_value=1, max_value=dynamic_max, step=1,
                value=int(min(st.session_state.top_n, dynamic_max)),
                key="top_n_number",
                on_change=_sync_top_n_from_number,
            )
            st.button("应用", use_container_width=True, on_click=_sync_top_n_from_number)
    else:
        st.caption("按当前指标排序后，选择要查看的**区间**（滑块或直接输入数字）")
        st.slider(
            "区间（第 i - 第 j 个）",
            min_value=1, max_value=max(dynamic_max,2), step=1,
            value=(int(st.session_state.range_from), int(st.session_state.range_to)),
            key="range_slider", on_change=_sync_range_from_slider,
        )
        col_l,col_r = st.columns(2)
        with col_l:
            st.number_input(
                "i", min_value=1, max_value=dynamic_max, step=1,
                value=int(st.session_state.range_from),
                key="range_from_num", on_change=_sync_range_from_numbers,
            )
        with col_r:
            st.number_input(
                "j", min_value=1, max_value=dynamic_max, step=1,
                value=int(st.session_state.range_to),
                key="range_to_num", on_change=_sync_range_from_numbers,
            )
        st.button("应用", use_container_width=True, on_click=_sync_range_from_numbers)

    st.session_state.sort_order = st.radio(
        "排序方向", ["降序", "升序"],
        index=0 if st.session_state.sort_order=="降序" else 1, horizontal=True
    )

# ---------------- 辅助：排序与切片 ----------------
def build_base(df_f: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    cols = ["word","pos","词汇等级by课标","CEFR_numeric","CEFR_level",metric]
    base = df_f[cols].copy()
    base[metric] = pd.to_numeric(base[metric], errors="coerce").fillna(0.0)
    base["CEFR_numeric"] = pd.to_numeric(base["CEFR_numeric"], errors="coerce").fillna(0).astype(int)
    base["CEFR_level"] = base["CEFR_level"].astype(str)
    return base.sort_values(metric, ascending=ascending, kind="mergesort")

def slice_df(base: pd.DataFrame, metric: str):
    if st.session_state.mode == "Top N":
        N = int(min(st.session_state.top_n, len(base)))
        return base.head(N), f"Mode=Top N, N={N}"
    total = len(base)
    a = max(1, min(st.session_state.range_from, total))
    b = max(1, min(st.session_state.range_to, total))
    if a>b: a,b = b,a
    return base.iloc[a-1:b], f"Mode=Range, From={a}, To={b}"

# ---------------- Tabs & Charts ----------------
tabs = st.tabs([TAB_TITLES[m] for m in METRICS])

for tab, metric in zip(tabs, METRICS):
    with tab:
        st.subheader(f"排序指标：{metric}")
        st.caption(METRIC_DESC_ZH.get(metric,""))
        # ★ MOD：可隐藏的长解释
        with st.expander("展开查看该指标的解释（面向非编程同学）", expanded=False):
            st.markdown(METRIC_LONG_DESC.get(metric, ""))

        ascending = (st.session_state.sort_order == "升序")
        base = build_base(df_f, metric, ascending)
        show_df, note_slice = slice_df(base, metric)

        # 1) 基础排名图 + 2) 词云
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("**指标排名图（基础）**")
            if show_df.empty:
                st.info("无数据")
            else:
                fig, ax = plt.subplots(figsize=(8,6))
                ax.bar(show_df["word"], show_df[metric])
                ax.set_xticklabels(show_df["word"], rotation=60, ha="right", fontsize=8)
                ax.set_ylabel(metric); ax.set_xlabel("word")
                ax.grid(True, linestyle="--", linewidth=0.5, axis="y")
                st.pyplot(fig, clear_figure=False)
                st.download_button("下载 PNG（基础排名图）", data=fig_to_png_bytes(fig),
                                   file_name=f"{metric}_rank_basic.png", mime="image/png", use_container_width=True)
                plt.close(fig)

        with col2:
            st.markdown("**词云（与左图同条件）**")
            if show_df.empty:
                st.info("无数据")
            else:
                freq = {}
                for row in show_df.itertuples(index=False, name="Row"):
                    w = str(getattr(row,"word"))
                    v = float(getattr(row,metric))
                    if not np.isfinite(v) or v <= 0: v = 1.0
                    freq[w] = v
                wc = WordCloud(width=800, height=500, background_color="white")
                wc.generate_from_frequencies(freq)
                fig2, ax2 = plt.subplots(figsize=(8,6))
                ax2.imshow(wc, interpolation="bilinear"); ax2.axis("off")
                st.pyplot(fig2, clear_figure=False)
                st.download_button("下载 PNG（词云）", data=fig_to_png_bytes(fig2),
                                   file_name=f"{metric}_wordcloud.png", mime="image/png", use_container_width=True)
                plt.close(fig2)

        # 3) by 新课标词汇等级（仅图例英文，其它中文）
        st.markdown("**指标排名图（by 新课标词汇等级）**")
        if show_df.empty:
            st.info("无数据")
        else:
            colors3 = [KB_LEVEL_COLOR.get(int(v), "#7f7f7f") for v in show_df["词汇等级by课标"].tolist()]
            fig3, ax3 = plt.subplots(figsize=(6.5,4.5))
            ax3.bar(show_df["word"], show_df[metric], color=colors3)
            ax3.set_xticklabels(show_df["word"], rotation=60, ha="right", fontsize=8)
            ax3.set_ylabel(metric); ax3.set_xlabel("word")
            ax3.grid(True, linestyle="--", linewidth=0.5, axis="y")
            legend3 = [
                Patch(facecolor=KB_LEVEL_COLOR[3], label="3 = Level 3 (exclude Level 2; added in L3)"),
                Patch(facecolor=KB_LEVEL_COLOR[2], label="2 = Level 2"),
                Patch(facecolor=KB_LEVEL_COLOR[0], label="0 = Not in curriculum"),
            ]
            ax3.legend(handles=legend3, title=None, loc="upper right")
            st.pyplot(fig3, clear_figure=False)
            st.download_button("下载 PNG（按课标等级着色）", data=fig_to_png_bytes(fig3),
                               file_name=f"{metric}_rank_by_kb.png", mime="image/png", use_container_width=True)
            plt.close(fig3)

        # 4) by CEFR（仅图例英文；纵轴=英文变量；动态自适应）  # ★ MOD：动态 Y 轴、无 0-8 死限
        st.markdown("**指标排名图（by CEFR词汇等级）**")
        if show_df.empty:
            st.info("无数据")
        else:
            cefr_vals = pd.to_numeric(show_df["CEFR_numeric"], errors="coerce").fillna(0).astype(int).tolist()
            colors4 = [color_for_cefr(v) for v in cefr_vals]
            fig4, ax4 = plt.subplots(figsize=(6.5,4.5))
            ax4.bar(show_df["word"], show_df[metric], color=colors4)
            ax4.set_xticklabels(show_df["word"], rotation=60, ha="right", fontsize=8)
            ax4.set_ylabel(metric)  # 英文变量名
            ax4.set_xlabel("word")
            ax4.grid(True, linestyle="--", linewidth=0.5, axis="y")

            # —— 动态 Y 轴：按数据自适应（0~1 指标单独限制到 ≤1）
            y_series = pd.to_numeric(show_df[metric], errors="coerce").fillna(0.0)
            y_max = float(y_series.max()) if len(y_series) else 0.0
            bounded01 = {"coverage","dispersion","passage_frac"}
            if metric in bounded01:
                upper = min(1.0, max(0.2, y_max * 1.10))
                if upper <= 0: upper = 1.0
                ax4.set_ylim(0, upper)
            else:
                upper = y_max * 1.10 if y_max > 0 else 1.0
                ax4.set_ylim(0, upper)

            legend4 = [
                Patch(facecolor="#7f7f7f", label="0 = Unspecified"),
                Patch(facecolor=color_for_cefr(1), label="1 = A1"),
                Patch(facecolor=color_for_cefr(2), label="2 = A2"),
                Patch(facecolor=color_for_cefr(3), label="3 = B1"),
                Patch(facecolor=color_for_cefr(4), label="4 = B2"),
                Patch(facecolor=color_for_cefr(5), label="5 = C1"),
                Patch(facecolor=color_for_cefr(6), label="6 = C2"),
            ]
            ax4.legend(handles=legend4, title=None, loc="upper right", ncol=2)
            st.pyplot(fig4, clear_figure=False)
            st.download_button("下载 PNG（按 CEFR 着色）", data=fig_to_png_bytes(fig4),
                               file_name=f"{metric}_rank_by_cefr.png", mime="image/png", use_container_width=True)
            plt.close(fig4)

        # 5) 双坐标轴：两个纵轴名称使用当前 feature 英文名（避免乱码）  # ★ MOD
        st.markdown("**指标排名图（双坐标轴）**")
        if show_df.empty:
            st.info("无数据")
        else:
            x = show_df["word"]
            y_left  = pd.to_numeric(show_df[metric], errors="coerce").fillna(0.0).values
            y_right = pd.to_numeric(show_df["CEFR_numeric"], errors="coerce").fillna(0.0).values

            fig5, axL = plt.subplots(figsize=(6.8,4.6))
            bars = axL.bar(x, y_left, alpha=0.75)
            axL.set_ylabel(f"{metric} (left)")   # 英文
            axL.set_xlabel("word")
            axL.grid(True, linestyle="--", linewidth=0.5, axis="y")
            axL.tick_params(axis='x', rotation=60, labelsize=8)

            axR = axL.twinx()
            axR.plot(x, y_right, marker="o", linewidth=1.0, alpha=0.9, color="#FA8072")  # salmon pink
            axR.scatter(x, y_right, s=10, color="#FA8072", zorder=3)
            axR.set_ylabel("CEFR_numeric (right)")  # 英文
            axR.set_ylim(-0.2, max(10, y_right.max() + 1))
            axR.set_yticks(range(0, int(max(10, y_right.max() + 1)) + 1))

            legend_dual = [
                Patch(facecolor=bars.patches[0].get_facecolor(), label=f"{metric} (left)"),
                Patch(facecolor="#FA8072", label="CEFR_numeric (right)"),
            ]
            axL.legend(handles=legend_dual, loc="upper right")
            st.pyplot(fig5, clear_figure=False)
            st.download_button("下载 PNG（双坐标轴）", data=fig_to_png_bytes(fig5),
                               file_name=f"{metric}_dual_axis.png", mime="image/png", use_container_width=True)
            plt.close(fig5)

        # 6) 新增：课标 × CEFR 堆叠柱图（横轴课标，内部按 CEFR 分色；柱内比例、柱顶总数）  # ★ MOD
        st.markdown("**等级分布（课标 × CEFR）**")
        if show_df.empty:
            st.info("无数据")
        else:
            data = show_df[["词汇等级by课标","CEFR_numeric"]].copy()
            data["词汇等级by课标"] = pd.to_numeric(data["词汇等级by课标"], errors="coerce").fillna(0).astype(int)
            data["CEFR_numeric"]   = pd.to_numeric(data["CEFR_numeric"], errors="coerce").fillna(0).astype(int)
            data = data.rename(columns=lambda c: c.strip())  # 以防空格等问题

            kb_order   = [0,2,3]
            cefr_order = [0,1,2,3,4,5,6]

            # —— 关键修复：MultiIndex 起名字，避免 KeyError（names=...）  # ★ MOD（修复 KeyError）
            ct = (
                data.groupby(["词汇等级by课标", "CEFR_numeric"])
                    .size()
                    .reindex(
                        pd.MultiIndex.from_product(
                            [kb_order, cefr_order],
                            names=["词汇等级by课标", "CEFR_numeric"]  # 起名很关键
                        ),
                        fill_value=0
                    )
                    .rename("count")
                    .reset_index()
            )

            totals = ct.groupby("词汇等级by课标")["count"].sum().reindex(kb_order, fill_value=0)

            fig6, ax6 = plt.subplots(figsize=(7.2, 4.8))
            bottoms = np.zeros(len(kb_order), dtype=float)

            for c in cefr_order:
                seg = ct[ct["CEFR_numeric"] == c]["count"].values.reshape(len(kb_order))
                ax6.bar(
                    [str(k) for k in kb_order], seg,
                    bottom=bottoms,
                    color=color_for_cefr(c),
                    label=f"{c} = " + ("Unspecified" if c == 0 else ["A1","A2","B1","B2","C1","C2"][c-1])
                )
                # 段内显示比例（占该柱总数，<8%不显示避免拥挤）
                for i, v in enumerate(seg):
                    total = totals.iloc[i]
                    if total > 0:
                        ratio = v / total
                        if ratio >= 0.08 and v > 0:
                            ax6.text(
                                i, bottoms[i] + v * 0.5, f"{ratio*100:.0f}%",
                                ha="center", va="center", color="white", fontsize=9, fontweight="bold"
                            )
                bottoms += seg

            # 柱顶总数
            for i, total in enumerate(totals.values):
                ax6.text(i, total + max(1, totals.max() * 0.02), f"{int(total)}",
                         ha="center", va="bottom", fontsize=10)

            ax6.set_xlabel("Curriculum Level (0 / 2 / 3)")
            ax6.set_ylabel("Count")  # 英文纵轴，避免乱码
            ax6.set_xticks(range(len(kb_order)))
            ax6.set_xticklabels([str(k) for k in kb_order])
            ax6.grid(True, axis="y", linestyle="--", linewidth=0.5)
            ax6.legend(title="CEFR_numeric", ncol=4, loc="upper right")
            st.pyplot(fig6, clear_figure=False)
            st.download_button(
                "下载 PNG（课标×CEFR 分布）",
                data=fig_to_png_bytes(fig6),
                file_name=f"{metric}_kb_cefr_distribution.png",
                mime="image/png",
                use_container_width=True
            )
            plt.close(fig6)

        # 结果表（导出包含 CEFR_level 文本列）
        st.markdown("**筛选结果预览（与上图同步）**")
        note = (
            f"Note: Metric={metric}, Order={'ASC' if ascending else 'DESC'}, "
            f"KB Levels={sorted(st.session_state.kb_levels)}, "
            f"CEFR={sorted(st.session_state.cefr_levels)}, {note_slice}"
        )
        st.caption(note)

        cols_order = [
            "word","pos","CEFR_level","CEFR_numeric","词汇等级by课标",
            "tf_passage","tf_item","tf_total","df","num_passages","coverage",
            "idf","tfidf","dispersion","general_score","passage_frac",
            "passage_priority_score","passage_df"
        ]
        cols_exist = [c for c in cols_order if c in show_df.columns]
        preview = show_df[cols_exist].reset_index(drop=True)
        if "CEFR_level" in preview.columns:
            preview["CEFR_level"] = preview["CEFR_level"].astype(str)

        st.dataframe(preview, use_container_width=True, height=360)

        ts = time.strftime("%Y%m%d-%H%M%S")
        data_bytes, ext = df_to_excel_or_csv_bytes(preview, sheet_name="selection")
        st.download_button(
            "导出（Excel 优先，失败则 CSV）",
            data=data_bytes,
            file_name=f"vocab_selection_{metric}_{ts}{ext}",
            mime="application/octet-stream",
            use_container_width=True
        )

# ---------------- 页脚 ----------------
st.markdown(
    "<div style='text-align:center;color:#888;margin-top:12px;'>Copyright © 3Q English</div>",
    unsafe_allow_html=True
)
