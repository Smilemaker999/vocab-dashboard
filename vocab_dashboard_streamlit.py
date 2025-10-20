# vocab_dashboard_streamlit_v3_9.py
import io
import time
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from wordcloud import WordCloud

# ---------------- Page config ----------------
st.set_page_config(page_title="中考英语词表可视化", layout="wide")

# ---------------- Metrics & labels ----------------
METRICS = [
    "tf_passage", "tf_item", "tf_total", "df", "num_passages", "coverage",
    "idf", "tfidf", "dispersion", "general_score", "passage_frac",
    "passage_priority_score", "passage_df"
]

METRIC_DESC_ZH = {
    "tf_passage": "正文高频词汇",
    "tf_item": "题目高频词汇",
    "tf_total": "正文+题目高频词汇",
    "df": "在多少语篇中出现（含正文+题目）",
    "num_passages": "参与统计的总语篇数",
    "coverage": "单词的语篇出现概率（出现语篇/总语篇）",
    "idf": "逆文档出现概率（越大说明语篇中出现概率低）",
    "tfidf": "平衡总出现频次和语篇稀有度（中间值均衡性较好）",
    "dispersion": "年份出现均匀程度（接近0只在一年一个地区出现，接近1每年每地区都出现）",
    "general_score": "综合高频词（出现总次数多，出现在单篇文章的概率大，且出现的年份多）",
    "passage_frac": "正文贡献占比（正文频次在总频次中的加权占比）",
    "passage_priority_score": "正文加权综合高频词（更多考虑正文贡献后的综合评分）",
    "passage_df": "在多少语篇中出现（仅含正文）",
}
TAB_TITLES = {k: (f"🪐 {k}｜{METRIC_DESC_ZH[k]}" if k=="general_score" else f"{k}｜{METRIC_DESC_ZH[k]}") for k in METRICS}

# —— 详细解释（面向非编程同学）——
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

# 难度颜色
LEVEL_COLOR = {3: "#d62728", 2: "#1f77b4", 0: "#7f7f7f"}  # red / blue / gray

# ---------------- Utils ----------------
def coerce_number_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def load_and_prepare(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    if "word" not in df.columns:
        for alt in ["Word", "WORD", "lemma", "Lemma"]:
            if alt in df.columns:
                df["word"] = df[alt].astype(str)
                break
    if "word" not in df.columns:
        raise ValueError("CSV 必须包含列：word（或 Lemma/Word）。")
    if "词汇等级by课标" not in df.columns:
        raise ValueError("CSV 必须包含列：词汇等级by课标。")
    for m in METRICS:
        if m in df.columns:
            df[m] = coerce_number_series(df[m])
        else:
            df[m] = 0.0
    df["词汇等级by课标"] = pd.to_numeric(df["词汇等级by课标"], errors="coerce").fillna(0).astype(int)
    df["word"] = df["word"].astype(str).str.strip()
    df = df[df["word"] != ""]
    return df

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf.getvalue()

def df_to_excel_or_csv_bytes(df: pd.DataFrame, sheet_name="selection"):
    """优先导出为 .xlsx（需要 xlsxwriter/openpyxl），否则 CSV。"""
    try:
        import xlsxwriter  # noqa: F401
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        buf.seek(0)
        return buf.getvalue(), ".xlsx"
    except Exception:
        pass
    try:
        import openpyxl  # noqa: F401
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        buf.seek(0)
        return buf.getvalue(), ".xlsx"
    except Exception:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
        return csv_buf.getvalue().encode("utf-8-sig"), ".csv"

# ---------------- Style: bigger tab font ----------------
st.markdown("""
<style>
/* 放大 Tabs 标签字号（兼容不同版本结构） */
.stTabs [role="tab"] { font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.title("中考英语词表可视化")
st.markdown(
    "上传脚本生成的 CSV（推荐：**vocab_full_metrics.csv**）。"
    "使用左侧筛选与上方 Tabs 切换指标，三张图和结果表会同步更新。"
)

uploaded = st.file_uploader("选择 CSV 文件（UTF-8/utf-8-sig）", type=["csv"])

# ---------------- Defaults & frozen max ----------------
DEFAULTS = {
    "kb_level": "全部",
    "top_n": 50,
    "sort_order": "降序",
    "mode": "Top N",              # Top N | 区间
    "range_from": 1,              # 排序后第 i 个（1-based）
    "range_to": 100,              # 排序后第 j 个（1-based）
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.session_state.setdefault("_rows_max", 300)
st.session_state.setdefault("_rows_count", 300)
st.session_state.setdefault("_last_filter_signature", None)

# ---------- Callbacks ----------
def _sync_top_n_from_slider():
    st.session_state.top_n = int(st.session_state.top_n_slider)

def _sync_top_n_from_number():
    v = int(st.session_state.top_n_number)
    v = max(1, min(v, st.session_state._rows_max))
    st.session_state.top_n = v

def _sync_range_from_slider():
    a, b = st.session_state.range_slider
    a, b = int(min(a, b)), int(max(a, b))
    st.session_state.range_from = a
    st.session_state.range_to = b

def _sync_range_from_numbers():
    a = int(st.session_state.range_from_num)
    b = int(st.session_state.range_to_num)
    a, b = max(1, a), max(1, b)
    a, b = int(min(a, b)), int(max(a, b))
    a = min(a, st.session_state._rows_max)
    b = min(b, st.session_state._rows_max)
    st.session_state.range_from = a
    st.session_state.range_to = b

# ---------------- Main flow ----------------
if uploaded is None:
    st.info("请先上传 CSV。")
    st.stop()

# 读 CSV
try:
    df = load_and_prepare(uploaded)
except Exception as e:
    st.error(f"CSV 读取/校验失败：{e}")
    st.stop()

# 侧栏：先选课标等级，拿到行数后“冻结”其余控件上限
with st.sidebar:
    st.header("筛选")
    kb_level_val = st.selectbox(
        "词汇等级by课标",
        options=["全部", 3, 2, 0],
        index=["全部", 3, 2, 0].index(st.session_state.kb_level),
        help="3=三级（二级不包含，三级增加）、2=二级、0=不在课标中",
        key="kb_level_widget",
    )
    st.session_state.kb_level = kb_level_val

# 应用等级过滤
if st.session_state.kb_level != "全部":
    df_f = df[df["词汇等级by课标"] == int(st.session_state.kb_level)].copy()
else:
    df_f = df.copy()

current_rows = int(len(df_f))
st.caption(f"已加载 {len(df)} 行；当前筛选后 {current_rows} 行。")

# 冻结上限：仅在过滤变更时更新
filter_signature = (st.session_state.kb_level,)
if st.session_state._last_filter_signature != filter_signature:
    st.session_state._rows_max = max(10, current_rows)
    st.session_state._rows_count = current_rows
    st.session_state._last_filter_signature = filter_signature
dynamic_max = max(10, int(st.session_state._rows_max))

# 继续渲染其余侧栏控件（使用冻结后的 dynamic_max）
with st.sidebar:
    st.session_state.mode = st.radio(
        "选择模式", options=["Top N", "区间"],
        index=0 if st.session_state.mode == "Top N" else 1,
        horizontal=True,
        key="mode_widget",
    )

    if st.session_state.mode == "Top N":
        # Top N：滑块 + 数字输入（on_change 同步）
        c1, c2 = st.columns([3, 1])
        with c1:
            st.slider(
                "Top N（滑块）",
                min_value=1, max_value=dynamic_max, step=1,
                value=int(min(st.session_state.top_n, dynamic_max)),
                key="top_n_slider",
                on_change=_sync_top_n_from_slider,
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
        # 区间：滑块 + 两个数字输入（on_change 同步）
        st.slider(
            "区间（第 i - 第 j 个）",
            min_value=1, max_value=max(dynamic_max, 2),
            value=(int(st.session_state.range_from), int(st.session_state.range_to)),
            step=1,
            key="range_slider",
            on_change=_sync_range_from_slider,
        )
        col_l, col_r = st.columns(2)
        with col_l:
            st.number_input(
                "i", min_value=1, max_value=dynamic_max, step=1,
                value=int(st.session_state.range_from),
                key="range_from_num",
                on_change=_sync_range_from_numbers,
            )
        with col_r:
            st.number_input(
                "j", min_value=1, max_value=dynamic_max, step=1,
                value=int(st.session_state.range_to),
                key="range_to_num",
                on_change=_sync_range_from_numbers,
            )
        st.button("应用", use_container_width=True, on_click=_sync_range_from_numbers)

    st.session_state.sort_order = st.radio(
        "排序方向", options=["降序", "升序"],
        index=0 if st.session_state.sort_order == "降序" else 1,
        horizontal=True,
        key="sort_order_widget",
    )

# ---------------- Tabs & Charts ----------------
tabs = st.tabs([TAB_TITLES[m] for m in METRICS])

for tab, metric in zip(tabs, METRICS):
    with tab:
        desc = METRIC_DESC_ZH.get(metric, "")
        st.subheader(f"指标：{metric}")
        if desc:
            st.caption(desc)

        # —— 可隐藏的详细解释 —— #
        with st.expander("查看该指标的详细解释（点击展开/隐藏）", expanded=False):
            st.markdown(METRIC_LONG_DESC.get(metric, "暂无补充说明。"))

        # 排序基表
        base = df_f[["word", "pos", "词汇等级by课标", metric]].copy()
        base[metric] = pd.to_numeric(base[metric], errors="coerce").fillna(0.0)
        ascending = (st.session_state.sort_order == "升序")
        base = base.sort_values(metric, ascending=ascending, kind="mergesort")  # 稳定排序

        # 采样：Top N / 区间
        if st.session_state.mode == "Top N":
            N = int(min(st.session_state.top_n, len(base)))
            show_df = base.head(N)
            note_slice = f"Mode=Top N, N={N}"
        else:
            total_rows = len(base)
            a = max(1, min(st.session_state.range_from, total_rows))
            b = max(1, min(st.session_state.range_to,   total_rows))
            if a > b: a, b = b, a
            idx_from, idx_to = a - 1, b - 1
            show_df = base.iloc[idx_from: idx_to + 1]
            note_slice = f"Mode=Range, From={a}, To={b}"

        # ---------------- 1) 指标排名图（基础） ----------------
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**指标排名图（基础）**")
            if show_df.empty:
                st.info("无数据")
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(show_df["word"], show_df[metric])
                ax.set_xticklabels(show_df["word"], rotation=60, ha="right", fontsize=8)
                ax.set_ylabel(metric)
                ax.set_xlabel("word")
                ax.grid(True, linestyle="--", linewidth=0.5, axis="y")
                st.pyplot(fig, clear_figure=False)
                st.download_button(
                    "下载 PNG（基础排名图）",
                    data=fig_to_png_bytes(fig),
                    file_name=f"{metric}_rank_basic.png",
                    mime="image/png",
                    use_container_width=True
                )
                plt.close(fig)

        # ---------------- 2) 词云（同尺寸） ----------------
        with col2:
            st.markdown("**词云（与左图同条件）**")
            if show_df.empty:
                st.info("无数据")
            else:
                freq = {}
                for row in show_df.itertuples(index=False, name="Row"):
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
                st.pyplot(fig2, clear_figure=False)
                st.download_button(
                    "下载 PNG（词云）",
                    data=fig_to_png_bytes(fig2),
                    file_name=f"{metric}_wordcloud.png",
                    mime="image/png",
                    use_container_width=True
                )
                plt.close(fig2)

        # ---------------- 3) 指标排名图（by 难度等级） ----------------
        st.markdown("**指标排名图（by 难度等级）**")
        if show_df.empty:
            st.info("无数据")
        else:
            colors = [LEVEL_COLOR.get(int(level), "#7f7f7f") for level in show_df["词汇等级by课标"].tolist()]
            fig3, ax3 = plt.subplots(figsize=(6.5, 4.5))  # 小一点
            ax3.bar(show_df["word"], show_df[metric], color=colors)
            ax3.set_xticklabels(show_df["word"], rotation=60, ha="right", fontsize=8)
            ax3.set_ylabel(metric)
            ax3.set_xlabel("word")
            ax3.grid(True, linestyle="--", linewidth=0.5, axis="y")
            legend_elements = [
                Patch(facecolor=LEVEL_COLOR[3], label="3 = Level 3 (not including Level 2, newly added in Level 3)"),
                Patch(facecolor=LEVEL_COLOR[2], label="2 = Level 2"),
                Patch(facecolor=LEVEL_COLOR[0], label="0 = Not included in the curriculum standard"),
            ]
            ax3.legend(handles=legend_elements, title=None, loc="upper right")
            st.pyplot(fig3, clear_figure=False)
            st.download_button(
                "下载 PNG（按难度着色）",
                data=fig_to_png_bytes(fig3),
                file_name=f"{metric}_rank_by_level.png",
                mime="image/png",
                use_container_width=True
            )
            plt.close(fig3)

        # ---------------- 4) 结果表 & 导出 ----------------
        st.markdown("**筛选结果预览（与上图同步）**")
        note = (
            f"Note: Metric={metric}, Order={'ASC' if ascending else 'DESC'}, "
            f"KB Level={st.session_state.kb_level}, {note_slice}"
        )
        st.caption(note)

        cols_order = [
            "word", "pos", "词汇等级by课标",
            "tf_passage", "tf_item", "tf_total", "df", "num_passages", "coverage",
            "idf", "tfidf", "dispersion", "general_score", "passage_frac",
            "passage_priority_score", "passage_df"
        ]
        cols_exist = [c for c in cols_order if c in show_df.columns]
        preview = show_df[cols_exist].reset_index(drop=True)

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

# ---------------- Footer ----------------
st.markdown(
    "<div style='text-align:center;color:#888;margin-top:12px;'>Copyright © 3Q English</div>",
    unsafe_allow_html=True
)
