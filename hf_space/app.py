"""日本株予測ダッシュボード - HF Spaces版（表示専用）"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="日本株予測システム", layout="wide")
st.title("📈 日本株 AI予測ダッシュボード")


def load_predictions():
    """latest.jsonから予測結果を読み込み"""
    path = DATA_DIR / "latest.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_chart_data():
    """chart_data.jsonからチャートデータを読み込み"""
    path = DATA_DIR / "chart_data.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


data = load_predictions()
chart_data = load_chart_data()

if data is None:
    st.error("予測データがまだありません。バッチ処理の完了をお待ちください。")
    st.stop()

predictions = data["predictions"]
if not predictions:
    st.warning("予測結果が0件です。")
    st.stop()

# --- サイドバー ---
st.sidebar.header("設定")

display_names = {p["display"]: p["ticker"] for p in predictions}
selected_display = st.sidebar.selectbox("銘柄を選択", list(display_names.keys()))
selected_ticker = display_names[selected_display]
selected_pred = next(p for p in predictions if p["ticker"] == selected_ticker)

chart_days = st.sidebar.slider("チャート表示日数", 30, 500, 180)

st.sidebar.markdown("---")
st.sidebar.subheader("確信度フィルタ")
confidence_threshold = st.sidebar.slider(
    "最低確信度 (%)", 0, 100, 0, step=5,
    help="設定値以上の確信度を持つ銘柄のみサマリーに表示します",
)
sort_by_confidence = st.sidebar.checkbox("確信度順にソート", value=True)

# 更新情報
st.sidebar.markdown("---")
st.sidebar.caption(f"最終更新: {data['generated_at'][:16]}")
st.sidebar.caption(f"CV AUC: {data['cv_auc']:.4f}")

# --- 株価チャート ---
st.subheader(f"{selected_pred['name']}（{selected_ticker}）株価チャート")

if chart_data and selected_ticker in chart_data:
    ticker_chart = chart_data[selected_ticker]
    df_chart = pd.DataFrame.from_dict(ticker_chart, orient="index")
    df_chart.index = pd.to_datetime(df_chart.index)
    df_chart = df_chart.sort_index().tail(chart_days)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.18, 0.18, 0.18],
        subplot_titles=("株価 & ボリンジャーバンド", "MACD", "RSI", "出来高"),
    )

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df_chart.index, open=df_chart["Open"], high=df_chart["High"],
        low=df_chart["Low"], close=df_chart["Close"], name="株価",
    ), row=1, col=1)

    # ボリンジャーバンド
    for col, name, color in [
        ("bb_high", "BB上限", "rgba(173,216,230,0.3)"),
        ("bb_mid", "BB中央", "rgba(100,149,237,0.5)"),
        ("bb_low", "BB下限", "rgba(173,216,230,0.3)"),
    ]:
        if col in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart[col], name=name,
                line=dict(width=1, color=color),
            ), row=1, col=1)

    # SMA
    sma_colors = ["orange", "red", "purple"]
    sma_cols = [c for c in df_chart.columns if c.startswith("sma_")]
    for i, col in enumerate(sorted(sma_cols)):
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart[col],
            name=col.upper(), line=dict(width=1, color=sma_colors[i % len(sma_colors)]),
        ), row=1, col=1)

    # MACD
    if "macd" in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["macd"], name="MACD",
            line=dict(color="blue", width=1),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["macd_signal"], name="Signal",
            line=dict(color="red", width=1),
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=df_chart.index, y=df_chart["macd_diff"], name="MACD Diff",
            marker_color=np.where(df_chart["macd_diff"] >= 0, "green", "red"),
        ), row=2, col=1)

    # RSI
    if "rsi" in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["rsi"], name="RSI",
            line=dict(color="purple", width=1),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # 出来高
    if "Volume" in df_chart.columns:
        fig.add_trace(go.Bar(
            x=df_chart.index, y=df_chart["Volume"], name="出来高",
            marker_color="rgba(100,149,237,0.5)",
        ), row=4, col=1)

    fig.update_layout(
        height=900, xaxis_rangeslider_visible=False,
        showlegend=False, margin=dict(l=50, r=50, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("チャートデータがありません。")


# --- AI予測 ---
st.subheader(f"🤖 AI予測 - {selected_pred['name']}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("基準日", selected_pred["date"])
with col2:
    st.metric("終値", f"¥{selected_pred['close']:,.0f}")
with col3:
    use_alpha = selected_pred["use_alpha"]
    horizon = selected_pred["horizon"]
    if use_alpha:
        color = "🟢" if selected_pred["prediction"] == "市場超過" else "🔴"
        label = f"{horizon}営業日後 対N225"
    else:
        color = "🟢" if selected_pred["prediction"] != "下落" else "🔴"
        label = f"{horizon}営業日後予測"
    st.metric(
        label,
        f"{color} {selected_pred['prediction']}",
        f"確信度: {selected_pred['confidence']:.1f}%",
    )
with col4:
    st.metric("モデル", selected_pred["model_type"])

if use_alpha:
    st.caption(f"市場超過確率: {selected_pred['prediction_proba']:.1%}（日経225対比アルファ予測）")
else:
    st.caption(f"上昇確率: {selected_pred['prediction_proba']:.1%}")

# SHAP分析
st.subheader("📊 予測要因分析 (SHAP)")

shap_data = selected_pred.get("shap_top15", [])
if shap_data:
    shap_df = pd.DataFrame(shap_data)

    fig_shap = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=np.where(shap_df["shap_value"] >= 0, "#2ecc71", "#e74c3c"),
        text=shap_df["feature_value"].apply(lambda v: f"{v:.4f}"),
        textposition="auto",
    ))
    shap_title = ("予測への寄与度 TOP15（→市場超過方向 / ←市場未満方向）"
                  if use_alpha else
                  "予測への寄与度 TOP15（→上昇方向 / ←下落方向）")
    fig_shap.update_layout(
        title=shap_title,
        height=500, yaxis=dict(autorange="reversed"),
        margin=dict(l=150, r=50, t=50, b=30),
    )
    st.plotly_chart(fig_shap, use_container_width=True)


# --- 全銘柄サマリー ---
st.subheader("📋 全銘柄予測サマリー")

prob_col = "市場超過確率" if data["use_alpha"] else "上昇確率"
pred_col = f"{data['horizon']}日後 対N225" if data["use_alpha"] else f"{data['horizon']}日後予測"

summary_rows = []
for p in predictions:
    summary_rows.append({
        "銘柄": p["display"],
        "終値": f"¥{p['close']:,.0f}",
        pred_col: p["prediction"],
        prob_col: f"{p['prediction_proba']:.1%}",
        "確信度": f"{p['confidence']:.1f}%",
        "_確信度値": p["confidence"],
        "モデル": p["model_type"],
    })

df_pred = pd.DataFrame(summary_rows)
total_count = len(df_pred)

# フィルタ
df_pred = df_pred[df_pred["_確信度値"] >= confidence_threshold]
if sort_by_confidence:
    df_pred = df_pred.sort_values("_確信度値", ascending=False)
df_display_pred = df_pred.drop(columns=["_確信度値"])

st.caption(
    f"確信度 {confidence_threshold}% 以上: "
    f"{len(df_display_pred)}銘柄 / 全{total_count}銘柄"
)

if len(df_display_pred) > 0:
    st.dataframe(df_display_pred, use_container_width=True, hide_index=True)
else:
    st.warning("該当する銘柄がありません。サイドバーで閾値を下げてください。")

# エラー表示
if data.get("errors"):
    with st.expander(f"⚠️ {len(data['errors'])}銘柄で予測取得に失敗"):
        for err in data["errors"]:
            st.text(f"{err['ticker']}: {err['error']}")
