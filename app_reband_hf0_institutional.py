import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import warnings

import altair as alt
from arch import arch_model
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

# ==========================================
# 0. 頁面設定與機構實盤規格 (REBAND_HF0_200_175 單一檔案完整部署版)
# ==========================================
st.set_page_config(
    page_title="REBAND_HF0_200_175 機構實盤部署與單一帳戶下單系統",
    layout="wide",
)
warnings.simplefilter(action="ignore")
alt.data_transformers.disable_max_rows()

st.markdown(
    """
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-label {font-size: 14px; color: #6c757d; margin-bottom: 0; font-weight: 600;}
    .metric-value {font-size: 24px; font-weight: bold; color: #212529 !important; margin: 5px 0;}
    .metric-sub {font-size: 12px; color: #6c757d; margin-bottom: 0;}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 1. 凍結策略核心常數 (REBAND_HF0_200_175 & O 對照)
# ==========================================
MAPPING = {"UPRO": "SPY", "EURL": "VGK", "EDC": "EEM"}
REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}
TRADE_ETFS_3X = ["UPRO", "EURL", "EDC"]
PROTO_ETFS_1X = ["SPY", "VGK", "EEM"]
SAFE_POOL = ["GLD", "TLT"]
ALL_EXEC_ASSETS = ["UPRO", "EURL", "EDC", "GLD", "TLT"]

SMA_MONTHS = 6
GARCH_WINDOW = 504
QUANTILE_WINDOW = 252
PHASE_COUNT = 5
MOM_SEGMENTS = [(0, 3), (3, 7), (7, 12)]
DEFENSE_HORIZONS = [6, 12]

# REBAND_HF0_200_175 專屬凍結常數
THETA_VOL_PCT = 41.0          # 常態波動率縮放門檻 41%
VOL_SCALING_POWER = 2.0       # 平方反比縮放 a = min(1, (41/sigma)^2)
RHO_EXIT = 2.50               # 初次/再次危機退出門檻 rho >= 2.5
RHO_REENTRY = 1.25            # 危機解除復歸門檻 rho <= 1.25
PEAK_SIGMA_DROP = 0.40        # 退出期波動率自峰值回落 40% (即 <= 0.60 * peak) 復歸
TROUGH_SIGMA_RISE = 0.25      # 復歸後波動率較活動期低點升 25% 且 rho >= 2.5 才再次退出
BAND_TRIG_TH = 0.0200         # 200 bps (2.00%) 觸發門檻 (Vanguard, 2024)
BAND_DEST_TH = 0.0175         # 175 bps (1.75%) 拉回目標邊界 (Vanguard, 2024)

# 原版 O 對照常數
O_EXIT_Q = 0.975
O_ENTRY_Q = 0.900

EXPECTED_DATA_START = pd.Timestamp("2005-03-11")
ANCHOR_TOLERANCE_DAYS = 7

with st.sidebar:
    st.header("🔧 單一帳戶下單設定")
    try:
        tz_tw = pytz.timezone("Asia/Taipei")
        now_tw = datetime.now(tz_tw)
        st.info(f"🇹🇼 台灣時間: {now_tw.strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        st.error(f"時區錯誤: {e}")

    ACCOUNT_NAV = st.number_input(
        "💰 單一帳戶總淨值 (USD)", min_value=1000.0, value=100000.0, step=5000.0
    )
    ORDER_MODE = st.radio(
        "🧭 單一帳戶庫存對齊模式",
        options=[
            "自動同步模式 (平日日常續抱使用)",
            "手動輸入目前持倉比例 (首次建倉或入金對齊)",
        ],
        index=0,
        help="若你已經照策略持有，選『自動同步模式』即可，系統會自動用五相位各自的昨日收盤權重判斷是否需要交易；若你是今天剛要把資金投入或手動調整過部位，請選『手動輸入目前持倉比例』。",
    )
    FEE_BPS = st.selectbox("💸 單邊交易成本假設 (bps)", options=[10, 20, 40], index=1)
    FORCE_EOM = st.checkbox(
        "⚠️ 執行月底強制結算",
        value=False,
        help="僅在『當月最後一個交易日收盤後』勾選，強制系統立刻以最新收盤價結算次月訊號。",
    )
    if st.button("🗑️ 強制清除快取 (重抓最新報價)"):
        st.cache_data.clear()
        st.rerun()


# ==========================================
# 2. 數據下載 (同時抓取 Open 與 Close 以精確模擬隔夜與日內漂移)
# ==========================================
def get_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    period_idx = df.index.to_period("M")
    month_end_dates = df.index.to_series().groupby(period_idx).max()
    return df.loc[month_end_dates]


@st.cache_data(ttl=300, show_spinner="正在從 Yahoo Finance 下載最新全球 ETF 報價...")
def get_live_market_bundle():
    tickers = TRADE_ETFS_3X + PROTO_ETFS_1X + SAFE_POOL
    try:
        raw = yf.download(
            tickers,
            start=EXPECTED_DATA_START.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )
        if raw.empty:
            return pd.DataFrame(), pd.DataFrame()
        closes = pd.DataFrame(index=raw.index)
        opens = pd.DataFrame(index=raw.index)
        for t in tickers:
            if t in raw.columns:
                t_df = raw[t]
                if "Close" in t_df.columns:
                    closes[t] = t_df["Close"]
                elif "Price" in t_df.columns:
                    closes[t] = t_df["Price"]
                if "Open" in t_df.columns:
                    opens[t] = t_df["Open"]
                elif t in closes.columns:
                    opens[t] = closes[t]
        if closes.index.tz is not None:
            closes.index = closes.index.tz_localize(None)
            opens.index = opens.index.tz_localize(None)
        closes = closes.ffill()
        opens = opens.ffill()
        return closes, opens
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame(), pd.DataFrame()


# ==========================================
# 3. 月度動能排名 (1x 原型)、SMA6 與防禦資產選擇
# ==========================================
def calculate_monthly_signals_1x(closes: pd.DataFrame):
    """嚴格使用 1x 原型 ETF (SPY, VGK, EEM) 計算全歷史與當月動能、SMA6 與 GLD/TLT 防禦選擇。"""
    prices_1x = closes[PROTO_ETFS_1X].dropna(how="all")
    monthly_1x = get_monthly_data(prices_1x)
    daily_vol_252 = prices_1x.pct_change().rolling(252).std(ddof=1) * np.sqrt(252.0)
    monthly_vol_252 = daily_vol_252.reindex(monthly_1x.index)

    # 1. 非重疊三段動能 (0,3], (3,7], (7,12]
    seg_scores = []
    seg_rets = {}
    for start_m, end_m in MOM_SEGMENTS:
        if start_m == 0:
            r_seg = (monthly_1x / monthly_1x.shift(end_m)) - 1.0
        else:
            r_seg = (monthly_1x.shift(start_m) / monthly_1x.shift(end_m)) - 1.0
        seg_rets[(start_m, end_m)] = r_seg
        seg_scores.append(r_seg / (monthly_vol_252 + 1e-6))

    total_score_df = (seg_scores[0] + seg_scores[1] + seg_scores[2]) / 3.0

    # 2. 月度 SMA6 (嚴格大於)
    sma6_mean_df = monthly_1x.rolling(SMA_MONTHS).mean()
    sma6_state_df = (monthly_1x > sma6_mean_df).astype(float)

    # 3. 避險資產 6M + 12M 平均報酬
    safe_monthly = get_monthly_data(closes[SAFE_POOL].dropna(how="all"))
    safe_ret6 = (safe_monthly / safe_monthly.shift(6)) - 1.0
    safe_ret12 = (safe_monthly / safe_monthly.shift(12)) - 1.0
    safe_comp = (safe_ret6 + safe_ret12) / 2.0

    # 決定目前生效的月底鎖定日 ref_date
    last_date = closes.index[-1]
    tz_tw = pytz.timezone("Asia/Taipei")
    now_tw = datetime.now(tz_tw)
    last_data_period = last_date.to_period("M")
    current_tw_period = pd.Period(now_tw.strftime("%Y-%m"), freq="M")

    if FORCE_EOM or (last_data_period < current_tw_period):
        ref_date = monthly_1x.index[-1]
    else:
        prev_m = monthly_1x[monthly_1x.index.to_period("M") < last_data_period]
        ref_date = prev_m.index[-1] if not prev_m.empty else monthly_1x.index[-1]

    # 組裝當月進攻排名表
    sel_rows = []
    for proto_t in PROTO_ETFS_1X:
        trade_3x = REVERSE_MAPPING[proto_t]
        sel_rows.append({
            "Trade_ETF_3x": trade_3x,
            "Signal_Proto_1x": proto_t,
            "Close_1x": float(monthly_1x.loc[ref_date, proto_t]),
            "SMA6_1x": float(sma6_mean_df.loc[ref_date, proto_t]),
            "SMA6_State": int(sma6_state_df.loc[ref_date, proto_t]),
            "Vol_252d_Ann": float(monthly_vol_252.loc[ref_date, proto_t]),
            "Ret_0to3M": float(seg_rets[(0, 3)].loc[ref_date, proto_t]),
            "Ret_3to7M": float(seg_rets[(3, 7)].loc[ref_date, proto_t]),
            "Ret_7to12M": float(seg_rets[(7, 12)].loc[ref_date, proto_t]),
            "Total_Score": float(total_score_df.loc[ref_date, proto_t]),
        })
    sel_df = pd.DataFrame(sel_rows).set_index("Trade_ETF_3x").sort_values("Total_Score", ascending=False)

    safe_df = pd.DataFrame({
        "Ticker": SAFE_POOL,
        "Ret_6M": [float(safe_ret6.loc[ref_date, t]) for t in SAFE_POOL],
        "Ret_12M": [float(safe_ret12.loc[ref_date, t]) for t in SAFE_POOL],
        "Comp_Ret": [float(safe_comp.loc[ref_date, t]) for t in SAFE_POOL],
    }).set_index("Ticker")
    safe_win = safe_df["Comp_Ret"].idxmax()

    # 建立日頻對齊的月度訊號序列 (供 5 相位 200/175 狀態機回放使用)
    # 若未勾選 FORCE_EOM，當月尚未結束的那筆月底列不生效於當月日頻序列
    effective_monthly_idx = monthly_1x.index if (FORCE_EOM or last_data_period < current_tw_period) else monthly_1x.index[:-1]
    leader_1x_monthly = total_score_df.loc[effective_monthly_idx].idxmax(axis=1)
    leader_3x_monthly = leader_1x_monthly.map(REVERSE_MAPPING)
    safe_win_monthly = safe_comp.loc[effective_monthly_idx].idxmax(axis=1)

    daily_leader_3x = leader_3x_monthly.reindex(closes.index).ffill()
    daily_safe_win = safe_win_monthly.reindex(closes.index).ffill()
    daily_sma6_by_3x = sma6_state_df.loc[effective_monthly_idx].rename(columns=REVERSE_MAPPING).reindex(closes.index).ffill()

    return {
        "ref_date": ref_date,
        "sel_df": sel_df,
        "safe_df": safe_df,
        "safe_win": safe_win,
        "daily_leader_3x": daily_leader_3x,
        "daily_safe_win": daily_safe_win,
        "daily_sma6_by_3x": daily_sma6_by_3x,
    }


# ==========================================
# 4. 五相位 GARCH + Q 狀態機 + a 縮放器 + 200/175 護欄自動回放引擎
# ==========================================
def run_q_state_machine_1d(vol_arr: np.ndarray, med_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(vol_arr)
    q_out = np.ones(n, dtype=float)
    a_out = np.ones(n, dtype=float)
    state = 1
    peak = 0.0
    trough = -1.0
    for t in range(n):
        vt = vol_arr[t]
        mt = med_arr[t]
        if np.isfinite(vt) and vt > 0.0:
            a_out[t] = min(1.0, (THETA_VOL_PCT / vt) ** VOL_SCALING_POWER)
        else:
            a_out[t] = 1.0
        if not (np.isfinite(vt) and np.isfinite(mt) and mt > 0.0):
            q_out[t] = float(state)
            continue
        rho = vt / mt
        if state == 1:
            if trough > 0.0 and vt < trough:
                trough = vt
            can_exit = (
                (rho >= RHO_EXIT)
                if (trough <= 0.0)
                else (rho >= RHO_EXIT and vt >= trough * (1.0 + TROUGH_SIGMA_RISE))
            )
            if can_exit:
                state = 0
                peak = vt
        else:
            if vt > peak:
                peak = vt
            if (rho <= RHO_REENTRY) or (vt <= peak * (1.0 - PEAK_SIGMA_DROP)):
                state = 1
                trough = vt
        q_out[t] = float(state)
    return q_out, a_out


@st.cache_data(ttl=600, show_spinner="正在執行 5 相位 Student-t GARCH(1,1) 與 REBAND 200/175 狀態自動回放...")
def run_5phase_live_reband_replay(closes: pd.DataFrame, opens: pd.DataFrame, force_eom_flag: bool):
    """自動回放近期 5 相位 (Phase 0..4) 的 GARCH、Q、a 與 200/175 bps 實際持倉漂移狀態，
    讓使用者打開網頁時無需手記昨日 5 相位權重，系統即自動算出今日開盤每個相位的 Action (0/1/2)！
    """
    m_sig = calculate_monthly_signals_1x(closes)
    # 取最近 900 個交易日進行 GARCH 504d 滾動擬合 + 252d 中位數 + 近 140 日 REBAND 狀態精確回放
    tail_n = GARCH_WINDOW + QUANTILE_WINDOW + 140
    sub_closes = closes.tail(tail_n).copy()
    sub_opens = opens.reindex(sub_closes.index).copy()
    abs_indices = closes.index.get_indexer(sub_closes.index)
    dates_sub = sub_closes.index

    # 對 3 檔 1x 原型 (SPY, VGK, EEM) 計算 5 相位 GARCH 預測波動率
    vol_5ph = {t3x: np.full((PHASE_COUNT, len(dates_sub)), np.nan) for t3x in TRADE_ETFS_3X}
    q_5ph = {t3x: np.ones((PHASE_COUNT, len(dates_sub)), dtype=float) for t3x in TRADE_ETFS_3X}
    a_5ph = {t3x: np.ones((PHASE_COUNT, len(dates_sub)), dtype=float) for t3x in TRADE_ETFS_3X}
    med_5ph = {t3x: np.full((PHASE_COUNT, len(dates_sub)), np.nan) for t3x in TRADE_ETFS_3X}
    g_o_5ph = {t3x: np.ones((PHASE_COUNT, len(dates_sub)), dtype=float) for t3x in TRADE_ETFS_3X}

    for t3x, proto_1x in MAPPING.items():
        ret_pct = closes[proto_1x].pct_change() * 100.0
        ret_sub = ret_pct.reindex(dates_sub).to_numpy(dtype=float)
        last_vol = [np.nan] * PHASE_COUNT
        for idx_t in range(GARCH_WINDOW, len(dates_sub)):
            abs_pos = int(abs_indices[idx_t])
            due_ph = (abs_pos - GARCH_WINDOW) % PHASE_COUNT
            train = ret_sub[idx_t - GARCH_WINDOW + 1 : idx_t + 1]
            train = train[np.isfinite(train)]
            if len(train) >= 250:
                try:
                    am = arch_model(train, vol="Garch", p=1, q=1, dist="t", rescale=False)
                    res = am.fit(disp="off", show_warning=False)
                    fc = res.forecast(horizon=1, reindex=False)
                    last_vol[due_ph] = float(np.sqrt(fc.variance.iloc[-1].values[0]) * np.sqrt(252.0))
                except Exception:
                    pass
            for ph in range(PHASE_COUNT):
                vol_5ph[t3x][ph, idx_t] = last_vol[ph]

        for ph in range(PHASE_COUNT):
            s_v = pd.Series(vol_5ph[t3x][ph], index=dates_sub).ffill()
            s_med = s_v.rolling(QUANTILE_WINDOW, min_periods=60).median().shift(1)
            s_q975 = s_v.rolling(QUANTILE_WINDOW, min_periods=60).quantile(O_EXIT_Q).shift(1)
            s_q900 = s_v.rolling(QUANTILE_WINDOW, min_periods=60).quantile(O_ENTRY_Q).shift(1)
            v_arr = s_v.to_numpy(dtype=float)
            m_arr = s_med.to_numpy(dtype=float)
            q_arr, a_arr = run_q_state_machine_1d(v_arr, m_arr)
            vol_5ph[t3x][ph] = v_arr
            med_5ph[t3x][ph] = m_arr
            q_5ph[t3x][ph] = q_arr
            a_5ph[t3x][ph] = a_arr

            g_state = 1.0
            for k_i in range(len(v_arr)):
                if np.isfinite(v_arr[k_i]) and np.isfinite(s_q975.iloc[k_i]):
                    if v_arr[k_i] > s_q975.iloc[k_i]:
                        g_state = 0.0
                    elif v_arr[k_i] < s_q900.iloc[k_i]:
                        g_state = 1.0
                g_o_5ph[t3x][ph, k_i] = g_state

    # 從最後 120 個交易日起，自動回放 5 相位的 REBAND 200/175 實際持倉漂移至最新收盤日
    replay_start = max(GARCH_WINDOW + QUANTILE_WINDOW, len(dates_sub) - 120)
    r_on_df = (sub_opens[ALL_EXEC_ASSETS] / sub_closes[ALL_EXEC_ASSETS].shift(1)) - 1.0
    r_id_df = (sub_closes[ALL_EXEC_ASSETS] / sub_opens[ALL_EXEC_ASSETS]) - 1.0

    # 我們要算出：「截至最新一個收盤日 t_last 收盤後，每個相位 p 的收盤實際持倉權重 prior_close_w[p]」
    # 以及「根據最新收盤日 t_last 訊號，在『下一個開盤日 t_next』每個相位 p 的 Base Target、Action 與 Planned Weight」！
    phase_latest_records = []
    for ph in range(PHASE_COUNT):
        holdings = {k: 0.0 for k in ALL_EXEC_ASSETS}
        prev_lead = None
        prev_s = None
        prev_q = None
        prev_safe = None
        prev_base_w = None
        prev_b = None

        # 先回放歷史交易日 (t_sig = idx - 1 決定 idx 開盤動作，收盤後漂移成 holdings)
        for idx in range(replay_start, len(dates_sub)):
            sig_dt = dates_sub[idx - 1]
            lead_3x = str(m_sig["daily_leader_3x"].loc[sig_dt])
            safe_1x = str(m_sig["daily_safe_win"].loc[sig_dt])
            s_val = float(m_sig["daily_sma6_by_3x"].loc[sig_dt, lead_3x])
            q_val = float(q_5ph[lead_3x][ph, idx - 1])
            a_val = float(a_5ph[lead_3x][ph, idx - 1])
            b_val = q_val * (0.5 * s_val + 0.5 * a_val)

            base_w = {k: 0.0 for k in ALL_EXEC_ASSETS}
            base_w[lead_3x] = b_val
            base_w[safe_1x] = 1.0 - b_val

            if prev_lead is None:
                mand = True
            else:
                supp_chg = any((base_w[k] > 1e-9) != (prev_base_w[k] > 1e-9) for k in ALL_EXEC_ASSETS)
                mand = (
                    (lead_3x != prev_lead)
                    or (s_val != prev_s)
                    or (q_val != prev_q)
                    or (safe_1x != prev_safe)
                    or supp_chg
                    or (b_val < prev_b - 1e-12)
                )

            d_max = max(abs(holdings[k] - base_w[k]) for k in ALL_EXEC_ASSETS)
            if mand:
                plan_w = dict(base_w)
            elif d_max >= BAND_TRIG_TH and d_max > 0:
                ratio = BAND_DEST_TH / d_max
                plan_w = {k: base_w[k] + ratio * (holdings[k] - base_w[k]) for k in ALL_EXEC_ASSETS}
            else:
                plan_w = dict(holdings)

            # 模擬 idx 當日實際價格漂移 (開盤若有交易則以 plan_w 進入日內；若 Skip 則經歷隔夜+日內完整日報酬漂移)
            if mand or d_max >= BAND_TRIG_TH:
                end_vals = {k: plan_w[k] * (1.0 + float(r_id_df.iloc[idx][k])) for k in ALL_EXEC_ASSETS}
            else:
                end_vals = {
                    k: holdings[k] * (1.0 + float(r_on_df.iloc[idx][k])) * (1.0 + float(r_id_df.iloc[idx][k]))
                    for k in ALL_EXEC_ASSETS
                }
            tot_v = sum(end_vals.values())
            holdings = {k: (end_vals[k] / tot_v if tot_v > 0 else base_w[k]) for k in ALL_EXEC_ASSETS}

            prev_lead, prev_s, prev_q, prev_safe, prev_base_w, prev_b = (
                lead_3x, s_val, q_val, safe_1x, dict(base_w), b_val
            )

        # 現在來到「最新收盤日 idx_now = len(dates_sub) - 1」：計算「下一個交易日開盤」的決策！
        idx_now = len(dates_sub) - 1
        winner_now = m_sig["sel_df"].index[0]
        safe_now = m_sig["safe_win"]
        s_now = float(m_sig["sel_df"].loc[winner_now, "SMA6_State"])
        q_now = float(q_5ph[winner_now][ph, idx_now])
        a_now = float(a_5ph[winner_now][ph, idx_now])
        sig_now = float(vol_5ph[winner_now][ph, idx_now])
        med_now = float(med_5ph[winner_now][ph, idx_now])
        rho_now = sig_now / med_now if (np.isfinite(med_now) and med_now > 0) else np.nan
        g_o_now = float(g_o_5ph[winner_now][ph, idx_now])

        b_now = q_now * (0.5 * s_now + 0.5 * a_now)
        b_o_now = 0.5 * s_now + 0.5 * g_o_now
        base_w_now = {k: 0.0 for k in ALL_EXEC_ASSETS}
        base_w_now[winner_now] = b_now
        base_w_now[safe_now] = 1.0 - b_now

        reasons = []
        if winner_now != prev_lead:
            reasons.append("RANK換股")
        if s_now != prev_s:
            reasons.append("SMA6翻轉")
        if q_now != prev_q:
            reasons.append("Q危機閘門翻轉")
        if safe_now != prev_safe:
            reasons.append("避險標的換倉")
        if any((base_w_now[k] > 1e-9) != (prev_base_w[k] > 1e-9) for k in ALL_EXEC_ASSETS):
            reasons.append("持倉集合改變")
        if b_now < prev_b - 1e-12:
            reasons.append("股票預算調降(減碼)")

        mand_now = len(reasons) > 0
        d_max_now = max(abs(holdings[k] - base_w_now[k]) for k in ALL_EXEC_ASSETS)

        if mand_now:
            act_now = 2
            plan_w_now = dict(base_w_now)
        elif d_max_now >= BAND_TRIG_TH and d_max_now > 0:
            act_now = 1
            ratio = BAND_DEST_TH / d_max_now
            plan_w_now = {k: base_w_now[k] + ratio * (holdings[k] - base_w_now[k]) for k in ALL_EXEC_ASSETS}
        else:
            act_now = 0
            plan_w_now = dict(holdings)

        phase_latest_records.append({
            "Phase": ph,
            "Trade_ETF_3x": winner_now,
            "Safe_ETF_1x": safe_now,
            "Sigma_Ann_Pct": round(sig_now, 2),
            "Med252_Pct": round(med_now, 2),
            "Rho_Ratio": round(rho_now, 3),
            "Q_Gate": int(q_now),
            "SMA6": int(s_now),
            "a_Scalar": round(a_now, 4),
            "Base_Stock_Weight": round(b_now, 4),
            "Base_Safe_Weight": round(1.0 - b_now, 4),
            "O_Stock_Weight": round(b_o_now, 4),
            "Prior_Close_Weights": holdings,
            "Base_Target_Weights": base_w_now,
            "Planned_Open_Weights": plan_w_now,
            "Decision_Distance_bps": round(d_max_now * 10000.0, 1),
            "Action_Code": act_now,
            "Mandatory_Reasons": ", ".join(reasons) if reasons else "無",
        })

    return m_sig, phase_latest_records


# ==========================================
# 5. 主畫面渲染：一目了然的單一帳戶下單指令與五相位明細
# ==========================================
st.title("🛡️ REBAND_HF0_200_175 機構實盤部署與單一帳戶下單系統")
st.caption(
    "自動執行：1x 原型三段動能 (0-3/3-7/7-12M) + SMA6 + 5 相位 Student-t GARCH(1,1) "
    "[Q 峰值回落 40% 復歸 / θ=41% 平方縮放 a] + Vanguard 200/175 bps 成本護欄自動回放"
)

closes_df, opens_df = get_live_market_bundle()

if not closes_df.empty:
    m_sig, phase_records = run_5phase_live_reband_replay(closes_df, opens_df, FORCE_EOM)
    sel_df = m_sig["sel_df"]
    safe_df = m_sig["safe_df"]
    ref_date = m_sig["ref_date"]
    winner_3x = sel_df.index[0]
    safe_win = m_sig["safe_win"]

    avg_base_stock = float(np.mean([r["Base_Stock_Weight"] for r in phase_records]))
    avg_o_stock = float(np.mean([r["O_Stock_Weight"] for r in phase_records]))
    active_q = sum(1 for r in phase_records if r["Q_Gate"] == 1)
    s_now_int = int(phase_records[0]["SMA6"])

    st.info(
        f"📅 **月度訊號鎖定日**: `{ref_date.strftime('%Y-%m-%d')}` ｜ "
        f"📈 **最新日頻收盤數據**: `{closes_df.index[-1].strftime('%Y-%m-%d')}` ｜ "
        f"🎯 **下一交易日開盤執行標的**: 進攻 **`{winner_3x}`** (訊號源 `{MAPPING[winner_3x]}`) ＋ 防禦 **`{safe_win}`**"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏆 本月進攻贏家 (3x)", f"{winner_3x} ({MAPPING[winner_3x]})")
    with c2:
        st.metric("🎯 五相位平均基礎股票預算", f"{avg_base_stock*100:.1f}%", f"對照舊版 O: {avg_o_stock*100:.1f}%")
    with c3:
        st.metric("⚡ Q 危機閘門安全相位數", f"{active_q} / 5 相位", f"SMA6: {'多頭 (1)' if s_now_int==1 else '空頭 (0)'}")
    with c4:
        st.metric("🛡️ 本月勝出避險資產", safe_win, f"6+12M 平均報酬: {safe_df.loc[safe_win, 'Comp_Ret']:.1%}")

    st.divider()

    # 計算五相位匯總後的「昨日收盤權重」與「今日開盤計畫權重」
    agg_prior_w = {k: float(np.mean([r["Prior_Close_Weights"][k] for r in phase_records])) for k in ALL_EXEC_ASSETS}
    agg_base_w = {k: float(np.mean([r["Base_Target_Weights"][k] for r in phase_records])) for k in ALL_EXEC_ASSETS}
    agg_plan_w = {k: float(np.mean([r["Planned_Open_Weights"][k] for r in phase_records])) for k in ALL_EXEC_ASSETS}

    if ORDER_MODE.startswith("手動"):
        st.markdown("#### ✍️ 手動輸入你目前單一帳戶的各 ETF 持倉比例 (%)")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            u_w = st.number_input("目前 UPRO (%)", 0.0, 100.0, round(agg_prior_w["UPRO"] * 100.0, 1), 1.0) / 100.0
        with mc2:
            e_w = st.number_input("目前 EURL (%)", 0.0, 100.0, round(agg_prior_w["EURL"] * 100.0, 1), 1.0) / 100.0
        with mc3:
            d_w = st.number_input("目前 EDC (%)", 0.0, 100.0, round(agg_prior_w["EDC"] * 100.0, 1), 1.0) / 100.0
        with mc4:
            g_w = st.number_input("目前 GLD (%)", 0.0, 100.0, round(agg_prior_w["GLD"] * 100.0, 1), 1.0) / 100.0
        with mc5:
            t_w = st.number_input("目前 TLT (%)", 0.0, 100.0, round(agg_prior_w["TLT"] * 100.0, 1), 1.0) / 100.0
        user_curr_w = {"UPRO": u_w, "EURL": e_w, "EDC": d_w, "GLD": g_w, "TLT": t_w}
        # 手動對齊模式下，目標即為五相位匯總之計畫權重 agg_plan_w
        target_account_w = agg_plan_w
    else:
        user_curr_w = agg_prior_w
        target_account_w = agg_plan_w

    # 檢查 5 個相位今日是否有任何一個相位觸發 Action != 0
    phases_trading = [r["Phase"] for r in phase_records if r["Action_Code"] != 0]

    latest_prices = {t: float(closes_df[t].iloc[-1]) for t in ALL_EXEC_ASSETS}
    order_rows = []
    for t in ALL_EXEC_ASSETS:
        w_curr = user_curr_w[t]
        w_targ = target_account_w[t]
        w_diff = w_targ - w_curr
        if ORDER_MODE.startswith("自動") and len(phases_trading) == 0:
            w_diff = 0.0
            w_targ = w_curr
        usd_curr = w_curr * ACCOUNT_NAV
        usd_targ = w_targ * ACCOUNT_NAV
        usd_diff = usd_targ - usd_curr
        px = latest_prices[t]
        shares_diff = usd_diff / px if px > 0 else 0.0
        if abs(w_diff) < 1e-4:
            cmd = "HOLD (不交易 / 抱牢原股數)"
        elif w_diff > 0:
            cmd = "BUY (開盤買進)"
        else:
            cmd = "SELL (開盤賣出)"
        order_rows.append({
            "ETF 代號": t,
            "昨日收盤權重 (%)": round(w_curr * 100.0, 2),
            "今日五相位匯總基礎目標 (%)": round(agg_base_w[t] * 100.0, 2),
            "今日五相位匯總計畫權重 (%)": round(w_targ * 100.0, 2),
            "本次調倉幅度 (%)": round(w_diff * 100.0, 2),
            "目前持倉市值 (USD)": round(usd_curr, 1),
            "目標持倉市值 (USD)": round(usd_targ, 1),
            "建議買賣金額 (USD)": round(usd_diff, 1),
            "最新參考收盤價": round(px, 2),
            "建議買賣股數 (碎股)": round(shares_diff, 2),
            "建議買賣整數股": int(round(shares_diff)),
            "下單動作": cmd,
        })

    active_orders = [r for r in order_rows if not r["下單動作"].startswith("HOLD")]
    if len(active_orders) == 0:
        st.success(
            f"✅ **今日開盤結論：0 筆委託單（全部 5 個相位均為 `Action = 0` 不交易，請抱牢現有 `{winner_3x}` / `{safe_win}` 股數不動，省下交易成本！）**"
        )
    else:
        order_summary = " ｜ ".join(
            f"**{r['下單動作']} `{r['ETF 代號']}`**: `${abs(r['建議買賣金額 (USD)']):,.0f} USD` (約 `{abs(r['建議買賣股數 (碎股)']):.2f}` 股 / 整數 `{abs(r['建議買賣整數股'])}` 股)"
            for r in active_orders
        )
        st.warning(f"🔔 **今日開盤前請送出以下單一帳戶匯總委託單 (觸發相位: Phase {phases_trading})**：\n\n{order_summary}")

    t_order, t_phase, t_rank, t_safe, t_price = st.tabs([
        "💳 單一帳戶匯總下單表",
        "🔬 五相位 GARCH 與 200/175 護欄狀態表",
        "📊 1x 原型動能選股排名",
        "🛡️ 避險資產輪動 (GLD vs TLT)",
        "🗄️ 最新 ETF 收盤價",
    ])

    with t_order:
        st.dataframe(pd.DataFrame(order_rows).set_index("ETF 代號"), use_container_width=True)

    with t_phase:
        disp_rows = []
        for r in phase_records:
            act_map = {
                0: "0: 不交易 (偏離 < 200bps，保留原股數)",
                1: "1: 緩衝調倉 (偏離 >= 200bps，拉回至 175bps)",
                2: f"2: 強制精確覆蓋 ({r['Mandatory_Reasons']})",
            }
            disp_rows.append({
                "相位 (Phase)": f"Phase {r['Phase']} (佔 20% 資金)",
                "進攻標的": r["Trade_ETF_3x"],
                "預測波動率 σ (%)": r["Sigma_Ann_Pct"],
                "252日中位數 (%)": r["Med252_Pct"],
                "警報比 ρ": r["Rho_Ratio"],
                "Q 危機閘門": r["Q_Gate"],
                "SMA6 趨勢": r["SMA6"],
                "a 縮放 (θ=41%)": r["a_Scalar"],
                f"基礎目標 {winner_3x} (%)": round(r["Base_Stock_Weight"] * 100.0, 2),
                f"昨收實際 {winner_3x} (%)": round(r["Prior_Close_Weights"][winner_3x] * 100.0, 2),
                "最大偏離 d_max (bps)": r["Decision_Distance_bps"],
                "今日開盤動作 (Action)": act_map[r["Action_Code"]],
                f"今日計畫 {winner_3x} (%)": round(r["Planned_Open_Weights"][winner_3x] * 100.0, 2),
                f"今日計畫 {safe_win} (%)": round(r["Planned_Open_Weights"][safe_win] * 100.0, 2),
            })
        st.dataframe(pd.DataFrame(disp_rows).set_index("相位 (Phase)"), use_container_width=True)

    with t_rank:
        st.dataframe(sel_df, use_container_width=True)

    with t_safe:
        st.dataframe(safe_df.style.format("{:.2%}"), use_container_width=True)

    with t_price:
        st.dataframe(closes_df.tail(10).style.format("{:.2f}"), use_container_width=True)
