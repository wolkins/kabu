"""日本株予測ダッシュボード - Streamlit"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import lightgbm as lgb
from pathlib import Path

from src.data.fetch import fetch_all, load_raw
from src.features.builder import build_features, build_all_features, get_feature_columns
from src.models.train import train_model, train_cross_sectional, predict_latest, MODEL_DIR
from src.utils.config import load_config, ticker_list, ticker_name, ticker_display

st.set_page_config(page_title="日本株予測システム", layout="wide")
st.title("📈 日本株 AI予測ダッシュボード")

config = load_config()
tickers = ticker_list(config)

# 表示名リスト（セレクトボックス用）
display_names = {ticker_display(config, t): t for t in tickers}
default_key = ticker_display(config, config["dashboard"]["default_ticker"])


# --- サイドバー ---
st.sidebar.header("設定")
selected_display = st.sidebar.selectbox(
    "銘柄を選択",
    list(display_names.keys()),
    index=list(display_names.keys()).index(default_key),
)
selected_ticker = display_names[selected_display]
company_name = ticker_name(config, selected_ticker)

chart_days = st.sidebar.slider("チャート表示日数", 30, 500, config["dashboard"]["chart_days"])

st.sidebar.markdown("---")
st.sidebar.subheader("確信度フィルタ")
confidence_threshold = st.sidebar.slider(
    "最低確信度 (%)", 0, 100, 0, step=5,
    help="設定値以上の確信度を持つ銘柄のみサマリーに表示します",
)
sort_by_confidence = st.sidebar.checkbox("確信度順にソート", value=True)

# データ取得ボタン
if st.sidebar.button("🔄 データ更新", use_container_width=True):
    with st.spinner("Yahoo Financeからデータ取得中..."):
        fetch_all(config)
    st.sidebar.success("データ更新完了!")

# モデル学習ボタン
if st.sidebar.button("🧠 個別モデル学習", use_container_width=True):
    with st.spinner(f"{company_name} のモデルを学習中..."):
        result = train_model(selected_ticker, config)
    st.sidebar.success(
        f"学習完了! 平均AUC: {result['scores']['auc'].mean():.4f}"
    )

if st.sidebar.button("🧠 全銘柄一括学習", use_container_width=True):
    with st.spinner("クロスセクション学習中（全銘柄）..."):
        result = train_cross_sectional(config)
    st.sidebar.success(
        f"一括学習完了! 平均AUC: {result['scores']['auc'].mean():.4f}"
    )

if st.sidebar.button("🔬 Optuna最適化学習", use_container_width=True):
    with st.spinner("Optunaでハイパーパラメータ最適化中（50試行）..."):
        result = train_cross_sectional(config, use_optuna=True, n_trials=50)
    st.sidebar.success(
        f"最適化学習完了! 平均AUC: {result['scores']['auc'].mean():.4f}"
    )


# --- メインコンテンツ ---
try:
    df = build_features(selected_ticker, config)
except FileNotFoundError:
    st.warning("データがありません。サイドバーの「データ更新」ボタンを押してください。")
    st.stop()

df_display = df.tail(chart_days)

# 株価チャート + テクニカル指標
st.subheader(f"{company_name}（{selected_ticker}）株価チャート")

fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.45, 0.18, 0.18, 0.18],
    subplot_titles=("株価 & ボリンジャーバンド", "MACD", "RSI", "出来高"),
)

# 株価ローソク足
fig.add_trace(go.Candlestick(
    x=df_display.index, open=df_display["Open"], high=df_display["High"],
    low=df_display["Low"], close=df_display["Close"], name="株価",
), row=1, col=1)

# ボリンジャーバンド
for col, name, color in [
    ("bb_high", "BB上限", "rgba(173,216,230,0.3)"),
    ("bb_mid", "BB中央", "rgba(100,149,237,0.5)"),
    ("bb_low", "BB下限", "rgba(173,216,230,0.3)"),
]:
    fig.add_trace(go.Scatter(
        x=df_display.index, y=df_display[col], name=name,
        line=dict(width=1, color=color),
    ), row=1, col=1)

# 移動平均線
colors = ["orange", "red", "purple"]
for i, w in enumerate(config["features"]["sma_windows"]):
    fig.add_trace(go.Scatter(
        x=df_display.index, y=df_display[f"sma_{w}"],
        name=f"SMA{w}", line=dict(width=1, color=colors[i]),
    ), row=1, col=1)

# MACD
fig.add_trace(go.Scatter(
    x=df_display.index, y=df_display["macd"], name="MACD",
    line=dict(color="blue", width=1),
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=df_display.index, y=df_display["macd_signal"], name="Signal",
    line=dict(color="red", width=1),
), row=2, col=1)
fig.add_trace(go.Bar(
    x=df_display.index, y=df_display["macd_diff"], name="MACD Diff",
    marker_color=np.where(df_display["macd_diff"] >= 0, "green", "red"),
), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=df_display.index, y=df_display["rsi"], name="RSI",
    line=dict(color="purple", width=1),
), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# 出来高
fig.add_trace(go.Bar(
    x=df_display.index, y=df_display["Volume"], name="出来高",
    marker_color="rgba(100,149,237,0.5)",
), row=4, col=1)

fig.update_layout(
    height=900, xaxis_rangeslider_visible=False,
    showlegend=False, margin=dict(l=50, r=50, t=30, b=30),
)
st.plotly_chart(fig, use_container_width=True)


# --- AI予測セクション ---
st.subheader(f"🤖 AI予測 - {company_name}")

safe_name = selected_ticker.replace("^", "IDX_").replace("=", "_")
cross_path = MODEL_DIR / "cross_sectional_lgbm.txt"
single_path = MODEL_DIR / f"{safe_name}_lgbm.txt"

if cross_path.exists() or single_path.exists():
    result = predict_latest(selected_ticker, config)
    horizon = result.get("horizon", config["model"]["target_horizon"])

    use_alpha = result.get("use_alpha", False)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("基準日", str(result["date"].date()))
    with col2:
        st.metric("終値", f"¥{result['close']:,.0f}")
    with col3:
        if use_alpha:
            color = "🟢" if result["prediction"] == "市場超過" else "🔴"
            label = f"{horizon}営業日後 対N225"
        else:
            color = "🟢" if result["prediction"] != "下落" else "🔴"
            label = f"{horizon}営業日後予測"
        st.metric(
            label,
            f"{color} {result['prediction']}",
            f"確信度: {abs(result['prediction_proba'] - 0.5) * 200:.1f}%",
        )
    with col4:
        st.metric("モデル", result.get("model_type", "個別"))

    if use_alpha:
        st.caption(f"市場超過確率: {result['prediction_proba']:.1%}（日経225対比アルファ予測）")
    else:
        st.caption(f"上昇確率: {result['prediction_proba']:.1%}")

    # SHAP分析
    st.subheader("📊 予測要因分析 (SHAP)")

    shap_vals = result["shap_values"]
    if isinstance(shap_vals, list):
        shap_arr = shap_vals[1]
    else:
        shap_arr = shap_vals

    shap_df = pd.DataFrame({
        "feature": result["feature_names"],
        "shap_value": shap_arr.flatten(),
        "feature_value": [result["feature_values"][f] for f in result["feature_names"]],
    })
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False).head(15)

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

else:
    st.info("モデルが未学習です。サイドバーの「モデル学習」ボタンを押してください。")


# --- 全銘柄一括予測 ---
st.subheader("📋 全銘柄予測サマリー")


@st.cache_data(show_spinner="全銘柄の予測を計算中...")
def _get_all_predictions(_tickers, _config):
    """全銘柄の予測結果を取得（キャッシュ付き、特徴量は1回だけビルド）"""
    # 全銘柄の特徴量を1回だけ構築（25回→1回に削減）
    try:
        df_all = build_all_features(_config)
    except Exception:
        df_all = None

    results = []
    errors = []
    for ticker in _tickers:
        try:
            r = predict_latest(ticker, _config, df_all_cache=df_all)
            alpha_mode = r.get("use_alpha", False)
            pred_col = f"{r['horizon']}日後 対N225" if alpha_mode else f"{r['horizon']}日後予測"
            prob_col = "市場超過確率" if alpha_mode else "上昇確率"
            conf_value = abs(r["prediction_proba"] - 0.5) * 200
            results.append({
                "銘柄": ticker_display(_config, ticker),
                "終値": f"¥{r['close']:,.0f}",
                pred_col: r["prediction"],
                prob_col: f"{r['prediction_proba']:.1%}",
                "確信度": f"{conf_value:.1f}%",
                "_確信度値": conf_value,
                "モデル": r.get("model_type", "-"),
            })
        except Exception as e:
            errors.append(f"{ticker}: {e}")
    return results, errors


predictions, pred_errors = _get_all_predictions(tuple(tickers), config)

if pred_errors:
    with st.expander(f"⚠️ {len(pred_errors)}銘柄で予測取得に失敗"):
        for err in pred_errors:
            st.text(err)

if predictions:
    df_pred = pd.DataFrame(predictions)
    total_count = len(df_pred)

    # 確信度フィルタリング
    df_pred = df_pred[df_pred["_確信度値"] >= confidence_threshold]

    # 確信度順ソート
    if sort_by_confidence:
        df_pred = df_pred.sort_values("_確信度値", ascending=False)

    # 内部用カラムを除去して表示
    df_display_pred = df_pred.drop(columns=["_確信度値"])

    st.caption(
        f"確信度 {confidence_threshold}% 以上: "
        f"{len(df_display_pred)}銘柄 / 全{total_count}銘柄"
    )

    if len(df_display_pred) > 0:
        st.dataframe(df_display_pred, use_container_width=True, hide_index=True)
    else:
        st.warning("該当する銘柄がありません。サイドバーで閾値を下げてください。")
else:
    st.info("いずれかの銘柄でモデル学習を実行してください。")
