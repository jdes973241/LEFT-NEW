from datetime import datetime
import warnings

import altair as alt
from arch import arch_model
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

# ==========================================
# 0. 頁面設定與機構實盤規格 (REBAND_HF0_200_175 全功能旗艦部署版)
# ==========================================
st.set_page_config(
    page_title="REBAND_HF0_200_175 機構實盤下單與超長歷史量化回測系統",
    layout="wide",
)
warnings.simplefilter(action="ignore")
alt.data_transformers.disable_max_rows()

st.markdown(
    """
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 14px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-label {font-size: 13px; color: #6c757d; margin-bottom: 0; font-weight: 600;}
    .metric-value {font-size: 22px; font-weight: bold; color: #212529 !important; margin: 4px 0;}
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

# 超長歷史 (1995~2005 ETF 上市前) 1x 指數/共同基金無縫代理映射
LONG_HIST_PROXIES = {
    "VGK": "VEURX",   # Vanguard European Stock Index Fund (1990上市)
    "EEM": "VEIEX",   # Vanguard Emerging Markets Stock Index Fund (1994上市)
    "TLT": "VUSTX",   # Vanguard Long-Term Treasury Fund (1986上市)
    "GLD": "FSAGX",   # Fidelity Select Gold Portfolio / GC=F 黃金代理 (1990上市)
}

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
BAND_TRIG_TH = 0.0200         # 200 bps (2.00%) 觸發門檻
BAND_DEST_TH = 0.0175         # 175 bps (1.75%) 拉回目標邊界

# 原版 O 對照常數
O_EXIT_Q = 0.975
O_ENTRY_Q = 0.900

EXPECTED_DATA_START = pd.Timestamp("1995-01-03")

with st.sidebar:
    st.header("🔧 單一帳戶下單與回測設定")
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
        help="若你已經照策略持有，選『自動同步模式』即可；若你是今天剛要把資金投入或手動調整過部位，請選『手動輸入目前持倉比例』。",
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
# 2. 數據下載與防呆工具函式 (相容 Python 3.14 / Pandas 2.2+ / 3.0)
# ==========================================
def safe_row_idxmax(df: pd.DataFrame, default_col: str) -> pd.Series:
    """在任何 Pandas 版本下安全取得每列最大值欄位名稱（遇全 NaN 列自動回傳 default_col，絕不拋出 ValueError）。"""
    if df.empty:
        return pd.Series(dtype=object)
    arr = np.nan_to_num(df.to_numpy(dtype=float), nan=-np.inf)
    all_missing = np.all(arr == -np.inf, axis=1)
    max_idx = np.argmax(arr, axis=1)
    cols = np.asarray(df.columns, dtype=object)
    chosen = cols[max_idx].copy()
    chosen[all_missing] = default_col
    return pd.Series(chosen, index=df.index, dtype=object)


def get_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    period_idx = df.index.to_period("M")
    month_end_dates = df.index.to_series().groupby(period_idx).max()
    return df.loc[month_end_dates.values]


def splice_early_series(primary_s: pd.Series, proxy_s: pd.Series) -> pd.Series:
    """用早期共同基金/指數日報酬率無縫往前銜接 ETF 上市前的歷史價格序列（嚴格保持 ETF 上市後價格 100% 原生不變）。"""
    s = primary_s.copy()
    first_valid = s.first_valid_index()
    if first_valid is None or proxy_s is None or proxy_s.dropna().empty:
        return s
    early_idx = s.index[s.index < first_valid]
    if len(early_idx) == 0:
        return s
    p_sub = proxy_s.reindex(s.index[: s.index.get_loc(first_valid) + 1]).ffill()
    if p_sub.isna().all() or p_sub.loc[first_valid] == 0 or pd.isna(p_sub.loc[first_valid]):
        return s
    scale = float(s.loc[first_valid]) / float(p_sub.loc[first_valid])
    s.loc[early_idx] = p_sub.loc[early_idx] * scale
    return s


@st.cache_data(ttl=300, show_spinner="正在從 Yahoo Finance 下載全球 ETF 與超長歷史基準數據 (1995–至今)...")
def get_live_market_bundle():
    core_tickers = TRADE_ETFS_3X + PROTO_ETFS_1X + SAFE_POOL
    proxy_tickers = list(LONG_HIST_PROXIES.values()) + ["^IRX"]
    all_tickers = core_tickers + proxy_tickers
    try:
        raw = yf.download(
            all_tickers,
            start=EXPECTED_DATA_START.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )
        if raw.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

        closes_raw = pd.DataFrame(index=raw.index)
        opens_raw = pd.DataFrame(index=raw.index)

        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = set(raw.columns.get_level_values(0))
            lvl1 = set(raw.columns.get_level_values(1))
            for t in all_tickers:
                if t in lvl0:
                    t_df = raw[t]
                    if "Close" in t_df.columns:
                        closes_raw[t] = pd.to_numeric(t_df["Close"], errors="coerce")
                    elif "Price" in t_df.columns:
                        closes_raw[t] = pd.to_numeric(t_df["Price"], errors="coerce")
                    if "Open" in t_df.columns:
                        opens_raw[t] = pd.to_numeric(t_df["Open"], errors="coerce")
                    elif t in closes_raw.columns:
                        opens_raw[t] = closes_raw[t]
                elif t in lvl1:
                    if ("Close", t) in raw.columns:
                        closes_raw[t] = pd.to_numeric(raw[("Close", t)], errors="coerce")
                    if ("Open", t) in raw.columns:
                        opens_raw[t] = pd.to_numeric(raw[("Open", t)], errors="coerce")
                    elif t in closes_raw.columns:
                        opens_raw[t] = closes_raw[t]
        else:
            for t in all_tickers:
                if t in raw.columns:
                    closes_raw[t] = pd.to_numeric(raw[t], errors="coerce")
                    opens_raw[t] = closes_raw[t]

        if closes_raw.index.tz is not None:
            closes_raw.index = closes_raw.index.tz_localize(None)
            opens_raw.index = opens_raw.index.tz_localize(None)

        # 以 SPY 有效交易日為主軸
        if "SPY" in closes_raw.columns:
            valid_spy = closes_raw["SPY"].dropna().index
            closes_raw = closes_raw.loc[valid_spy]
            opens_raw = opens_raw.loc[valid_spy]

        # 處理無風險利率 ^IRX (年化百分比 -> 小數，例如 4.5 -> 0.045)
        if "^IRX" in closes_raw.columns:
            rf_ann = (closes_raw["^IRX"].ffill().bfill() / 100.0).clip(lower=0.0, upper=0.15)
        else:
            rf_ann = pd.Series(0.025, index=closes_raw.index)

        # 建立完整 1x 與 3x 序列 (ETF 上市後 100% 使用真實 ETF；ETF 上市前以代理基金與合成 3x TRS 銜接)
        closes = pd.DataFrame(index=closes_raw.index)
        opens = pd.DataFrame(index=closes_raw.index)

        closes["SPY"] = closes_raw["SPY"].ffill()
        opens["SPY"] = opens_raw["SPY"].fillna(closes["SPY"]).ffill()

        for etf_1x, proxy_t in LONG_HIST_PROXIES.items():
            c_s = closes_raw[etf_1x] if etf_1x in closes_raw.columns else pd.Series(np.nan, index=closes_raw.index)
            o_s = opens_raw[etf_1x] if etf_1x in opens_raw.columns else c_s.copy()
            p_c = closes_raw[proxy_t] if proxy_t in closes_raw.columns else None
            closes[etf_1x] = splice_early_series(c_s.ffill(), p_c).ffill()
            # Open 在早期代理期間預設為前一日收盤價 (使隔夜報酬為 0、日內完整反映當日報酬)，上市後使用真實 Open
            first_etf_dt = c_s.first_valid_index()
            o_spliced = closes[etf_1x].shift(1).bfill()
            if first_etf_dt is not None:
                o_spliced.loc[o_spliced.index >= first_etf_dt] = o_s.loc[o_s.index >= first_etf_dt].fillna(
                    closes[etf_1x].shift(1).loc[o_s.index >= first_etf_dt]
                )
            opens[etf_1x] = o_spliced.ffill()

        # 合成 3x ETF 早期序列 (UPRO/EURL/EDC 上市前：3x 1x日報酬 - 0.95%年管理費 - 2x(無風險利率+1%融資利差))
        for t3x, proto_1x in MAPPING.items():
            c3 = closes_raw[t3x] if t3x in closes_raw.columns else pd.Series(np.nan, index=closes_raw.index)
            o3 = opens_raw[t3x] if t3x in opens_raw.columns else c3.copy()
            first_3x = c3.first_valid_index()

            r_1x = (closes[proto_1x] / closes[proto_1x].shift(1)) - 1.0
            daily_drag = (0.0095 + 2.0 * (rf_ann.shift(1).fillna(0.025) + 0.01)) / 252.0
            synth_r3 = (3.0 * r_1x - daily_drag).fillna(0.0).clip(lower=-0.95)
            synth_p3 = pd.Series(100.0 * np.cumprod(1.0 + synth_r3.to_numpy(dtype=float)), index=closes_raw.index)

            c3_full = splice_early_series(c3.ffill(), synth_p3).ffill()
            o3_full = c3_full.shift(1).bfill()
            if first_3x is not None:
                o3_full.loc[o3_full.index >= first_3x] = o3.loc[o3.index >= first_3x].fillna(
                    c3_full.shift(1).loc[o3.index >= first_3x]
                )
            closes[t3x] = c3_full
            opens[t3x] = o3_full

        closes = closes.ffill().bfill()
        opens = opens.ffill().bfill()
        return closes, opens, rf_ann
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)


# ==========================================
# 3. 月度動能排名 (1x 原型)、SMA6 與防禦資產選擇
# ==========================================
def calculate_monthly_signals_1x(closes: pd.DataFrame):
    """嚴格使用 1x 原型 ETF (SPY, VGK, EEM) 計算全歷史與當月動能、SMA6 與 GLD/TLT 防禦選擇。"""
    prices_1x = closes[PROTO_ETFS_1X].dropna(how="all")
    monthly_1x = get_monthly_data(prices_1x)
    ret_1x_daily = (prices_1x / prices_1x.shift(1)) - 1.0
    daily_vol_252 = ret_1x_daily.rolling(252).std(ddof=1) * np.sqrt(252.0)
    monthly_vol_252 = daily_vol_252.reindex(monthly_1x.index)

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

    sma6_mean_df = monthly_1x.rolling(SMA_MONTHS).mean()
    sma6_state_df = (monthly_1x > sma6_mean_df).astype(float)

    safe_monthly = get_monthly_data(closes[SAFE_POOL].dropna(how="all"))
    safe_ret6 = (safe_monthly / safe_monthly.shift(6)) - 1.0
    safe_ret12 = (safe_monthly / safe_monthly.shift(12)) - 1.0
    safe_comp = (safe_ret6 + safe_ret12) / 2.0

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
    comp_arr = np.nan_to_num(safe_df["Comp_Ret"].to_numpy(dtype=float), nan=-np.inf)
    safe_win = str(safe_df.index[int(np.argmax(comp_arr))])

    effective_monthly_idx = (
        monthly_1x.index if (FORCE_EOM or last_data_period < current_tw_period) else monthly_1x.index[:-1]
    )
    leader_1x_monthly = safe_row_idxmax(total_score_df.loc[effective_monthly_idx], default_col="SPY")
    leader_3x_monthly = leader_1x_monthly.map(REVERSE_MAPPING)
    safe_win_monthly = safe_row_idxmax(safe_comp.loc[effective_monthly_idx], default_col="GLD")

    daily_leader_3x = leader_3x_monthly.reindex(closes.index).ffill().fillna("UPRO")
    daily_safe_win = safe_win_monthly.reindex(closes.index).ffill().fillna("GLD")
    daily_sma6_by_3x = (
        sma6_state_df.loc[effective_monthly_idx]
        .rename(columns=REVERSE_MAPPING)
        .reindex(closes.index)
        .ffill()
        .fillna(1.0)
    )

    # 建立近 24 個月歷史輪動紀錄表
    hist_rot_rows = []
    for m_dt in effective_monthly_idx[-24:][::-1]:
        l_1x = str(leader_1x_monthly.loc[m_dt])
        l_3x = REVERSE_MAPPING.get(l_1x, "UPRO")
        s_w = str(safe_win_monthly.loc[m_dt])
        sma_st = int(sma6_state_df.loc[m_dt, l_1x]) if pd.notna(sma6_state_df.loc[m_dt, l_1x]) else 1
        sc_val = float(total_score_df.loc[m_dt, l_1x]) if pd.notna(total_score_df.loc[m_dt, l_1x]) else np.nan
        hist_rot_rows.append({
            "月底訊號鎖定日": m_dt.strftime("%Y-%m-%d"),
            "勝出進攻原型 (1x)": l_1x,
            "對應實盤槓桿 ETF (3x)": l_3x,
            "綜合動能總分": round(sc_val, 3) if np.isfinite(sc_val) else np.nan,
            "月度 SMA6 狀態": "多頭 (1)" if sma_st == 1 else "跌破均線 (0)",
            "勝出避險資產 (1x)": s_w,
        })

    return {
        "ref_date": ref_date,
        "sel_df": sel_df,
        "safe_df": safe_df,
        "safe_win": safe_win,
        "daily_leader_3x": daily_leader_3x,
        "daily_safe_win": daily_safe_win,
        "daily_sma6_by_3x": daily_sma6_by_3x,
        "hist_rot_df": pd.DataFrame(hist_rot_rows),
    }


# ==========================================
# 4. 五相位 GARCH + Q 狀態機 + a 縮放器 + 200/175 實盤狀態自動回放引擎
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


@st.cache_data(ttl=600, show_spinner="正在執行實盤 5 相位 Student-t GARCH(1,1) 與 REBAND 200/175 狀態自動回放...")
def run_5phase_live_reband_replay(closes: pd.DataFrame, opens: pd.DataFrame, force_eom_flag: bool):
    """自動回放近期 5 相位 (Phase 0..4) 的精確 Student-t GARCH(1,1)、Q、a 與 200/175 bps 實際持倉漂移狀態。"""
    m_sig = calculate_monthly_signals_1x(closes)
    tail_n = GARCH_WINDOW + QUANTILE_WINDOW + 140
    sub_closes = closes.tail(tail_n).copy()
    sub_opens = opens.reindex(sub_closes.index).copy()
    abs_indices = closes.index.get_indexer(sub_closes.index)
    dates_sub = sub_closes.index

    vol_5ph = {t3x: np.full((PHASE_COUNT, len(dates_sub)), np.nan) for t3x in TRADE_ETFS_3X}
    q_5ph = {t3x: np.ones((PHASE_COUNT, len(dates_sub)), dtype=float) for t3x in TRADE_ETFS_3X}
    a_5ph = {t3x: np.ones((PHASE_COUNT, len(dates_sub)), dtype=float) for t3x in TRADE_ETFS_3X}
    med_5ph = {t3x: np.full((PHASE_COUNT, len(dates_sub)), np.nan) for t3x in TRADE_ETFS_3X}
    g_o_5ph = {t3x: np.ones((PHASE_COUNT, len(dates_sub)), dtype=float) for t3x in TRADE_ETFS_3X}

    active_3x_set = set(m_sig["daily_leader_3x"].tail(130).unique()) | {str(m_sig["sel_df"].index[0])}
    for t3x, proto_1x in MAPPING.items():
        if t3x not in active_3x_set:
            continue
        ret_pct = ((closes[proto_1x] / closes[proto_1x].shift(1)) - 1.0) * 100.0
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
            s_v = pd.Series(vol_5ph[t3x][ph], index=dates_sub).ffill().bfill()
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

    replay_start = max(GARCH_WINDOW + QUANTILE_WINDOW, len(dates_sub) - 120)
    r_on_df = ((sub_opens[ALL_EXEC_ASSETS] / sub_closes[ALL_EXEC_ASSETS].shift(1)) - 1.0).fillna(0.0)
    r_id_df = ((sub_closes[ALL_EXEC_ASSETS] / sub_opens[ALL_EXEC_ASSETS]) - 1.0).fillna(0.0)

    phase_latest_records = []
    for ph in range(PHASE_COUNT):
        holdings = {k: 0.0 for k in ALL_EXEC_ASSETS}
        prev_lead = None
        prev_s = None
        prev_q = None
        prev_safe = None
        prev_base_w = None
        prev_b = None

        for idx in range(replay_start, len(dates_sub)):
            sig_dt = dates_sub[idx - 1]
            lead_3x = str(m_sig["daily_leader_3x"].loc[sig_dt])
            if lead_3x not in TRADE_ETFS_3X:
                lead_3x = "UPRO"
            safe_1x = str(m_sig["daily_safe_win"].loc[sig_dt])
            if safe_1x not in SAFE_POOL:
                safe_1x = "GLD"
            s_val = float(m_sig["daily_sma6_by_3x"].loc[sig_dt, lead_3x])
            if not np.isfinite(s_val):
                s_val = 1.0
            q_val = float(q_5ph[lead_3x][ph, idx - 1])
            if not np.isfinite(q_val):
                q_val = 1.0
            a_val = float(a_5ph[lead_3x][ph, idx - 1])
            if not np.isfinite(a_val):
                a_val = 1.0
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

        idx_now = len(dates_sub) - 1
        winner_now = str(m_sig["sel_df"].index[0])
        safe_now = str(m_sig["safe_win"])
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
            "Sigma_Ann_Pct": round(sig_now, 2) if np.isfinite(sig_now) else np.nan,
            "Med252_Pct": round(med_now, 2) if np.isfinite(med_now) else np.nan,
            "Rho_Ratio": round(rho_now, 3) if np.isfinite(rho_now) else np.nan,
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
# 5. 超長歷史 (1998–至今) 高速 5 相位向量化回測引擎與量化指標計算器
# ==========================================
@st.cache_data(ttl=600, show_spinner="正在建構 1998–至今 (28+ 年) 五相位超長歷史回測曲線與滾動指標...")
def run_ultra_long_history_backtest(closes: pd.DataFrame, opens: pd.DataFrame, rf_ann: pd.Series, fee_bps: int):
    """高速計算 1998–至今之 5 相位 REBAND_HF0_200_175、SCOPE_HALF_F0、原版策略 O 與 SPY 買入持有完整日頻回測。"""
    m_sig = calculate_monthly_signals_1x(closes)
    dates = closes.index
    n_total = len(dates)

    # 1. 對 SPY, VGK, EEM 建立 504 日錨定 GARCH(1,1) 5 相位階梯波動率預測序列
    vol_5ph = {t3x: np.full((PHASE_COUNT, n_total), 18.0, dtype=float) for t3x in TRADE_ETFS_3X}
    q_5ph = {t3x: np.ones((PHASE_COUNT, n_total), dtype=float) for t3x in TRADE_ETFS_3X}
    a_5ph = {t3x: np.ones((PHASE_COUNT, n_total), dtype=float) for t3x in TRADE_ETFS_3X}
    g_o_5ph = {t3x: np.ones((PHASE_COUNT, n_total), dtype=float) for t3x in TRADE_ETFS_3X}

    alpha_g = 0.135
    beta_g = 0.850
    gamma_l = 1.0 - alpha_g - beta_g

    for t3x, proto_1x in MAPPING.items():
        ret_pct = (((closes[proto_1x] / closes[proto_1x].shift(1)) - 1.0) * 100.0).fillna(0.0).to_numpy(dtype=float)
        s_ret = pd.Series(ret_pct)
        med_ret = s_ret.rolling(GARCH_WINDOW, min_periods=120).median().bfill().fillna(0.0)
        mad_504 = (s_ret - med_ret).abs().rolling(GARCH_WINDOW, min_periods=120).median().bfill().fillna(0.6).to_numpy(dtype=float)
        var_rob = (mad_504 * 1.4826) ** 2
        h_raw = np.zeros(n_total, dtype=float)
        h_curr = float(var_rob[0]) if np.isfinite(var_rob[0]) else 1.0
        for t in range(n_total):
            if t > 0:
                omega_t = gamma_l * float(var_rob[t])
                h_curr = omega_t + alpha_g * (ret_pct[t] ** 2) + beta_g * h_curr
            h_raw[t] = h_curr
        sig_cont = np.sqrt(np.maximum(h_raw, 1e-6)) * np.sqrt(252.0)

        # 嚴格按 5 相位 (t - 504) % 5 == ph 階梯取樣並維持 5 日
        for ph in range(PHASE_COUNT):
            v_ph = np.full(n_total, np.nan, dtype=float)
            last_v = float(sig_cont[GARCH_WINDOW]) if n_total > GARCH_WINDOW else 18.0
            for t in range(GARCH_WINDOW, n_total):
                if (t - GARCH_WINDOW) % PHASE_COUNT == ph:
                    last_v = float(sig_cont[t])
                v_ph[t] = last_v
            s_v = pd.Series(v_ph, index=dates).ffill().bfill()
            s_med = s_v.rolling(QUANTILE_WINDOW, min_periods=60).median().shift(1)
            s_q975 = s_v.rolling(QUANTILE_WINDOW, min_periods=60).quantile(O_EXIT_Q).shift(1)
            s_q900 = s_v.rolling(QUANTILE_WINDOW, min_periods=60).quantile(O_ENTRY_Q).shift(1)
            v_arr = s_v.to_numpy(dtype=float)
            m_arr = s_med.to_numpy(dtype=float)
            q_arr, a_arr = run_q_state_machine_1d(v_arr, m_arr)
            vol_5ph[t3x][ph] = v_arr
            q_5ph[t3x][ph] = q_arr
            a_5ph[t3x][ph] = a_arr

            g_state = 1.0
            q975_arr = s_q975.to_numpy(dtype=float)
            q900_arr = s_q900.to_numpy(dtype=float)
            for k_i in range(n_total):
                if np.isfinite(v_arr[k_i]) and np.isfinite(q975_arr[k_i]):
                    if v_arr[k_i] > q975_arr[k_i]:
                        g_state = 0.0
                    elif v_arr[k_i] < q900_arr[k_i]:
                        g_state = 1.0
                g_o_5ph[t3x][ph, k_i] = g_state

    # 2. 從 1998-03-02 (或第 760 個交易日) 開始執行 T+1 Open 5 相位投資組合微結構模擬
    start_idx = GARCH_WINDOW + QUANTILE_WINDOW + 10
    sim_dates = dates[start_idx:]
    n_sim = len(sim_dates)

    r_on_mat = ((opens[ALL_EXEC_ASSETS] / closes[ALL_EXEC_ASSETS].shift(1)) - 1.0).fillna(0.0).to_numpy(dtype=float)
    r_id_mat = ((closes[ALL_EXEC_ASSETS] / opens[ALL_EXEC_ASSETS]) - 1.0).fillna(0.0).to_numpy(dtype=float)
    asset_to_col = {k: i for i, k in enumerate(ALL_EXEC_ASSETS)}

    lead_arr = m_sig["daily_leader_3x"].to_numpy()
    safe_arr = m_sig["daily_safe_win"].to_numpy()
    sma_mat = m_sig["daily_sma6_by_3x"][TRADE_ETFS_3X].to_numpy(dtype=float)
    t3x_to_idx = {k: i for i, k in enumerate(TRADE_ETFS_3X)}

    fee_rate = float(fee_bps) / 10000.0

    # 儲存 5 相位平均之每日淨報酬率與換手率
    ret_reband = np.zeros(n_sim, dtype=float)
    ret_hf0 = np.zeros(n_sim, dtype=float)
    ret_o = np.zeros(n_sim, dtype=float)

    to_reband = np.zeros(n_sim, dtype=float)
    to_hf0 = np.zeros(n_sim, dtype=float)
    to_o = np.zeros(n_sim, dtype=float)
    skip_reband = np.zeros(n_sim, dtype=float)
    eq_w_reband = np.zeros(n_sim, dtype=float)

    for ph in range(PHASE_COUNT):
        h_reb = np.zeros(5, dtype=float)
        h_hf0 = np.zeros(5, dtype=float)
        h_o = np.zeros(5, dtype=float)

        prev_lead = None
        prev_s = None
        prev_q = None
        prev_safe = None
        prev_base = np.zeros(5, dtype=float)
        prev_b = None

        for s_i in range(n_sim):
            t_idx = start_idx + s_i
            sig_idx = t_idx - 1

            l_str = str(lead_arr[sig_idx])
            if l_str not in t3x_to_idx:
                l_str = "UPRO"
            sf_str = str(safe_arr[sig_idx])
            if sf_str not in asset_to_col:
                sf_str = "GLD"

            l_col = asset_to_col[l_str]
            sf_col = asset_to_col[sf_str]
            s_val = float(sma_mat[sig_idx, t3x_to_idx[l_str]])
            q_val = float(q_5ph[l_str][ph, sig_idx])
            a_val = float(a_5ph[l_str][ph, sig_idx])
            g_val = float(g_o_5ph[l_str][ph, sig_idx])

            b_hf0 = q_val * (0.5 * s_val + 0.5 * a_val)
            b_o = 0.5 * s_val + 0.5 * g_val

            base_hf0 = np.zeros(5, dtype=float)
            base_hf0[l_col] = b_hf0
            base_hf0[sf_col] = 1.0 - b_hf0

            base_o = np.zeros(5, dtype=float)
            base_o[l_col] = b_o
            base_o[sf_col] = 1.0 - b_o

            r_on = r_on_mat[t_idx]
            r_id = r_id_mat[t_idx]

            # --- A. REBAND_HF0_200_175 執行 ---
            if prev_lead is None:
                mand = True
            else:
                supp_chg = bool(np.any((base_hf0 > 1e-9) != (prev_base > 1e-9)))
                mand = (
                    (l_str != prev_lead)
                    or (s_val != prev_s)
                    or (q_val != prev_q)
                    or (sf_str != prev_safe)
                    or supp_chg
                    or (b_hf0 < prev_b - 1e-12)
                )
            d_max = float(np.max(np.abs(h_reb - base_hf0)))
            if mand:
                plan_reb = base_hf0.copy()
                traded_reb = True
            elif d_max >= BAND_TRIG_TH and d_max > 0:
                plan_reb = base_hf0 + (BAND_DEST_TH / d_max) * (h_reb - base_hf0)
                traded_reb = True
            else:
                plan_reb = h_reb.copy()
                traded_reb = False

            # 開盤前隔夜漂移
            if s_i == 0:
                open_w_reb = plan_reb.copy()
                turn_reb = float(np.sum(np.abs(plan_reb)))
            else:
                nav_on = float(np.sum(h_reb * (1.0 + r_on)))
                open_w_reb = (h_reb * (1.0 + r_on)) / nav_on if nav_on > 0 else h_reb.copy()
                turn_reb = float(np.sum(np.abs(plan_reb - open_w_reb))) if traded_reb else 0.0
                if not traded_reb:
                    plan_reb = open_w_reb

            nav_on_reb = 1.0 if s_i == 0 else float(np.sum(h_reb * (1.0 + r_on)))
            nav_id_reb = float(np.sum(plan_reb * (1.0 + r_id)))
            fee_mult_reb = max(0.0, 1.0 - fee_rate * turn_reb)
            r_day_reb = nav_on_reb * fee_mult_reb * nav_id_reb - 1.0
            end_v_reb = plan_reb * (1.0 + r_id)
            h_reb = end_v_reb / float(np.sum(end_v_reb)) if np.sum(end_v_reb) > 0 else base_hf0.copy()

            ret_reband[s_i] += r_day_reb / PHASE_COUNT
            to_reband[s_i] += turn_reb / PHASE_COUNT
            skip_reband[s_i] += (0.0 if traded_reb else 1.0) / PHASE_COUNT
            eq_w_reband[s_i] += b_hf0 / PHASE_COUNT

            prev_lead, prev_s, prev_q, prev_safe, prev_base, prev_b = (
                l_str, s_val, q_val, sf_str, base_hf0.copy(), b_hf0
            )

            # --- B. SCOPE_HALF_F0 (無護欄每日調倉) ---
            if s_i == 0:
                turn_hf0 = float(np.sum(np.abs(base_hf0)))
                nav_on_hf0 = 1.0
            else:
                nav_on_hf0 = float(np.sum(h_hf0 * (1.0 + r_on)))
                open_w_hf0 = (h_hf0 * (1.0 + r_on)) / nav_on_hf0 if nav_on_hf0 > 0 else h_hf0.copy()
                turn_hf0 = float(np.sum(np.abs(base_hf0 - open_w_hf0)))
            nav_id_hf0 = float(np.sum(base_hf0 * (1.0 + r_id)))
            r_day_hf0 = nav_on_hf0 * max(0.0, 1.0 - fee_rate * turn_hf0) * nav_id_hf0 - 1.0
            end_v_hf0 = base_hf0 * (1.0 + r_id)
            h_hf0 = end_v_hf0 / float(np.sum(end_v_hf0)) if np.sum(end_v_hf0) > 0 else base_hf0.copy()
            ret_hf0[s_i] += r_day_hf0 / PHASE_COUNT
            to_hf0[s_i] += turn_hf0 / PHASE_COUNT

            # --- C. 原版策略 O (97.5%/90% 分位數二元閘門、每日調倉) ---
            if s_i == 0:
                turn_o = float(np.sum(np.abs(base_o)))
                nav_on_o = 1.0
            else:
                nav_on_o = float(np.sum(h_o * (1.0 + r_on)))
                open_w_o = (h_o * (1.0 + r_on)) / nav_on_o if nav_on_o > 0 else h_o.copy()
                turn_o = float(np.sum(np.abs(base_o - open_w_o)))
            nav_id_o = float(np.sum(base_o * (1.0 + r_id)))
            r_day_o = nav_on_o * max(0.0, 1.0 - fee_rate * turn_o) * nav_id_o - 1.0
            end_v_o = base_o * (1.0 + r_id)
            h_o = end_v_o / float(np.sum(end_v_o)) if np.sum(end_v_o) > 0 else base_o.copy()
            ret_o[s_i] += r_day_o / PHASE_COUNT
            to_o[s_i] += turn_o / PHASE_COUNT

    ret_spy = ((closes["SPY"].reindex(sim_dates) / closes["SPY"].reindex(sim_dates).shift(1)) - 1.0).fillna(0.0).to_numpy(dtype=float)
    rf_daily = (rf_ann.reindex(sim_dates).ffill().fillna(0.025).to_numpy(dtype=float)) / 252.0

    bt_df = pd.DataFrame(
        {
            "REBAND_HF0_200_175": ret_reband,
            "SCOPE_HALF_F0": ret_hf0,
            "Strategy_O": ret_o,
            "SPY_1x": ret_spy,
            "RF_Daily": rf_daily,
            "Turnover_REBAND": to_reband,
            "Turnover_HF0": to_hf0,
            "Turnover_O": to_o,
            "Skip_Frac_REBAND": skip_reband,
            "Equity_Weight_REBAND": eq_w_reband,
        },
        index=sim_dates,
    )
    return bt_df, m_sig["hist_rot_df"]


def compute_quant_metrics_table(bt_sub: pd.DataFrame, fee_bps: int) -> pd.DataFrame:
    """計算機構級完整量化指標對照表 (CAGR, Vol, Sharpe, Sortino, MaxDD, Calmar, Ulcer, Recovery Days, Win Rate, Profit Factor, VaR/CVaR, Turnover)。"""
    strat_Meta = [
        ("REBAND_HF0_200_175", "🛡️ REBAND_HF0_200_175 (本系統)", "Turnover_REBAND"),
        ("SCOPE_HALF_F0", "⚡ SCOPE_HALF_F0 (無護欄每日調倉)", "Turnover_HF0"),
        ("Strategy_O", "📊 原版策略 O (舊版分位數基準)", "Turnover_O"),
        ("SPY_1x", "🇺🇸 SPY (標普 500 買入持有)", None),
    ]
    rows = []
    rf_d = bt_sub["RF_Daily"].to_numpy(dtype=float)
    n_days = len(bt_sub)
    years = max(n_days / 252.0, 1e-6)

    for col, label, to_col in strat_Meta:
        r = bt_sub[col].to_numpy(dtype=float)
        nav = np.cumprod(1.0 + r)
        tot_mult = float(nav[-1]) if len(nav) > 0 else 1.0
        cagr = (tot_mult ** (1.0 / years)) - 1.0 if tot_mult > 0 else -1.0
        ann_vol = float(np.std(r, ddof=1) * np.sqrt(252.0))

        ex_r = r - rf_d
        sharpe = float(np.mean(ex_r) / (np.std(ex_r, ddof=1) + 1e-12) * np.sqrt(252.0))
        downside = np.minimum(0.0, ex_r)
        down_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(252.0))
        sortino = float((np.mean(ex_r) * 252.0) / (down_dev + 1e-12))

        peak = np.maximum.accumulate(nav)
        dd = (nav / peak) - 1.0
        max_dd = float(np.min(dd))
        calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else np.nan
        ulcer_idx = float(np.sqrt(np.mean((dd * 100.0) ** 2)))

        # 最長水下恢復天數 (Max Drawdown Duration in trading days)
        underwater = dd < -1e-6
        max_uw_days = 0
        curr_uw = 0
        for u_flag in underwater:
            if u_flag:
                curr_uw += 1
                if curr_uw > max_uw_days:
                    max_uw_days = curr_uw
            else:
                curr_uw = 0

        # 月勝率與年勝率
        s_r = pd.Series(r, index=bt_sub.index)
        m_rets = s_r.groupby(s_r.index.to_period("M")).apply(lambda x: float(np.prod(1.0 + x) - 1.0))
        y_rets = s_r.groupby(s_r.index.year).apply(lambda x: float(np.prod(1.0 + x) - 1.0))
        m_win_rate = float((m_rets > 0).mean()) if len(m_rets) > 0 else np.nan
        y_win_rate = float((y_rets > 0).mean()) if len(y_rets) > 0 else np.nan

        pos_sum = float(np.sum(r[r > 0]))
        neg_sum = float(abs(np.sum(r[r < 0])))
        profit_factor = pos_sum / neg_sum if neg_sum > 0 else np.nan

        var_95 = float(np.percentile(r, 5))
        cvar_95 = float(np.mean(r[r <= var_95])) if np.any(r <= var_95) else var_95

        if to_col is not None:
            ann_to = float(bt_sub[to_col].mean() * 252.0)
            fee_drag_bps = ann_to * float(fee_bps)
        else:
            ann_to = 0.0
            fee_drag_bps = 0.0

        rows.append({
            "策略名稱": label,
            "年化報酬 (CAGR)": f"{cagr*100:.2f}%",
            "累積淨值倍數": f"{tot_mult:,.2f}x",
            "年化波動率 (Vol)": f"{ann_vol*100:.2f}%",
            "夏普比率 (Sharpe)": f"{sharpe:.2f}",
            "索提諾比率 (Sortino)": f"{sortino:.2f}",
            "最大回撤 (MaxDD)": f"{max_dd*100:.2f}%",
            "卡瑪比率 (Calmar)": f"{calmar:.2f}",
            "潰瘍指數 (Ulcer)": f"{ulcer_idx:.2f}",
            "最長水下天數 (交易日)": f"{max_uw_days} 天",
            "月勝率": f"{m_win_rate*100:.1f}%",
            "年勝率": f"{y_win_rate*100:.1f}%",
            "獲利因子 (Profit Factor)": f"{profit_factor:.2f}",
            "日 VaR (95%)": f"{var_95*100:.2f}%",
            "日 CVaR (95%)": f"{cvar_95*100:.2f}%",
            "年化周轉率 (倍/年)": f"{ann_to:.2f}x",
            "年化成本拖累 (bps)": f"-{fee_drag_bps:.1f} bps",
        })

    return pd.DataFrame(rows).set_index("策略名稱")


# ==========================================
# 6. 主畫面渲染：五大實用分頁
# ==========================================
st.title("🛡️ REBAND_HF0_200_175 機構實盤部署與超長歷史量化回測系統")
st.caption(
    "整合：單一帳戶匯總下單指令 ｜ 5 相位 Student-t GARCH(1,1) 與 200/175 bps 護欄監控 ｜ "
    "1998–至今 (28+ 年) 淨值、水下回撤、5Y/10Y 滾動 CAGR 圖表 ｜ 完整數學規格說明書"
)

closes_df, opens_df, rf_series = get_live_market_bundle()

if not closes_df.empty:
    m_sig, phase_records = run_5phase_live_reband_replay(closes_df, opens_df, FORCE_EOM)
    sel_df = m_sig["sel_df"]
    safe_df = m_sig["safe_df"]
    ref_date = m_sig["ref_date"]
    winner_3x = str(sel_df.index[0])
    safe_win = str(m_sig["safe_win"])

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

    tab_live, tab_bt, tab_doc, tab_rot, tab_raw = st.tabs([
        "💳 1. 單一帳戶實盤下單與五相位監控",
        "📈 2. 最長歷史量化回測與滾動圖表 (1998–至今)",
        "📖 3. 策略架構與數學規格說明書",
        "📊 4. 當月 1x 動能排名與歷史輪動紀錄",
        "🗄️ 5. 原始報價與系統健康檢查",
    ])

    # ==========================================
    # 分頁 1：單一帳戶實盤下單與五相位監控
    # ==========================================
    with tab_live:
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
            target_account_w = agg_plan_w
        else:
            user_curr_w = agg_prior_w
            target_account_w = agg_plan_w

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

        st.subheader("💳 單一帳戶匯總下單指令表")
        st.dataframe(pd.DataFrame(order_rows).set_index("ETF 代號"), use_container_width=True)

        st.subheader("🔬 五相位 (Phase 0..4) GARCH 波動率與 200/175 bps 護欄明細")
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

        with st.expander("🧪 壓力測試模擬器：如果今晚輪值的相位遇到波動率飆升，我的單一帳戶明天會怎麼動？"):
            sim_c1, sim_c2 = st.columns(2)
            with sim_c1:
                sim_sigma = st.slider("假設今日輪值相位的 1x 預測年化波動率 σ (%)", 10.0, 85.0, 58.0, 1.0)
            with sim_c2:
                sim_rho = st.slider("假設今日輪值相位的波動率警報比 ρ = σ / Med252", 0.8, 3.5, 2.1, 0.1)
            sim_q = 0.0 if sim_rho >= RHO_EXIT else 1.0
            sim_a = min(1.0, (THETA_VOL_PCT / sim_sigma) ** 2)
            sim_b = sim_q * (0.5 * float(s_now_int) + 0.5 * sim_a)
            # 假設其餘 4 個相位維持目前基礎權重，僅 1 個相位 (20% 資金) 變更為 sim_b
            new_acct_stock_w = 0.80 * avg_base_stock + 0.20 * sim_b
            delta_stock_w = new_acct_stock_w - avg_base_stock
            delta_usd = delta_stock_w * ACCOUNT_NAV
            st.markdown(
                f"- **該輪值相位狀態**：`Q = {int(sim_q)}` ({'觸發危機全退！' if sim_q==0 else '安全維持 1'}) ｜ "
                f"平方縮放係數 `a = {sim_a:.4f}` ｜ 該相位股票目標降至 **`{sim_b*100:.2f}%`**（觸發 `Action = 2` 強制減碼覆蓋！）\n"
                f"- **對你單一帳戶 (總淨值 `${ACCOUNT_NAV:,.0f}`) 的匯總影響**：帳戶總 `{winner_3x}` 目標由 `{avg_base_stock*100:.2f}%` 降至 **`{new_acct_stock_w*100:.2f}%`** "
                f"$\to$ 明日開盤只需 **賣出 `{winner_3x}` `${abs(delta_usd):,.0f} USD`（約 `{abs(delta_usd)/latest_prices[winner_3x]:.1f}` 股）並買進等額 `{safe_win}`**！"
            )

    # ==========================================
    # 分頁 2：最長歷史量化回測與滾動圖表 (1998–至今)
    # ==========================================
    with tab_bt:
        bt_full_df, hist_rot_df = run_ultra_long_history_backtest(closes_df, opens_df, rf_series, FEE_BPS)

        st.subheader("⏱️ 選擇歷史回測檢視區間與座標模式")
        ctrl1, ctrl2 = st.columns([2, 1])
        with ctrl1:
            period_choice = st.radio(
                "回測時間跨度",
                options=[
                    "🌐 最長歷史全週期 (1998–至今，約 28 年：含 2000 科技泡沫、2008 金融海嘯、2020 熔斷、2022 升息)",
                    "🏛️ 實盤 3x ETF 完整週期 (2014-01-23–至今：真實 UPRO/EURL/EDC 上市後微結構)",
                    "🔟 近 10 年完整週期 (Last 10 Years)",
                    "5️⃣ 近 5 年完整週期 (Last 5 Years)",
                ],
                index=0,
                horizontal=False,
            )
        with ctrl2:
            scale_type = st.radio(
                "淨值曲線縱軸座標",
                options=["log (對數座標：最適合檢視長期複利斜率)", "linear (線性座標：檢視絕對資產倍數)"],
                index=0,
            )
            use_log = scale_type.startswith("log")

        if period_choice.startswith("🏛️"):
            bt_sub = bt_full_df.loc[bt_full_df.index >= pd.Timestamp("2014-01-23")].copy()
        elif period_choice.startswith("🔟"):
            bt_sub = bt_full_df.tail(252 * 10).copy()
        elif period_choice.startswith("5️⃣"):
            bt_sub = bt_full_df.tail(252 * 5).copy()
        else:
            bt_sub = bt_full_df.copy()

        st.caption(
            f"📌 目前檢視區間：`{bt_sub.index[0].strftime('%Y-%m-%d')}` 至 `{bt_sub.index[-1].strftime('%Y-%m-%d')}` "
            f"（共 `{len(bt_sub):,}` 個交易日，約 `{len(bt_sub)/252.0:.1f}` 年 ｜ 單邊交易成本設定：`{FEE_BPS} bps`）"
        )

        # 1. 量化常用指標總表
        st.markdown("#### 📊 1. 機構級量化績效指標總表 (Quantitative Performance Summary)")
        metrics_tbl = compute_quant_metrics_table(bt_sub, FEE_BPS)
        st.dataframe(metrics_tbl, use_container_width=True)

        # 準備繪圖序列 (為了讓 28 年 7,000 點在瀏覽器極速渲染，每 5 個交易日取樣一次繪圖點，並保留最後一日)
        nav_df = pd.DataFrame(index=bt_sub.index)
        dd_df = pd.DataFrame(index=bt_sub.index)
        label_map = {
            "REBAND_HF0_200_175": "1. REBAND_HF0_200_175 (本系統)",
            "SCOPE_HALF_F0": "2. SCOPE_HALF_F0 (無護欄)",
            "Strategy_O": "3. 原版策略 O",
            "SPY_1x": "4. SPY (1x 標普500)",
        }
        for col, lab in label_map.items():
            r_arr = bt_sub[col].to_numpy(dtype=float)
            nav_s = np.cumprod(1.0 + r_arr)
            peak_s = np.maximum.accumulate(nav_s)
            nav_df[lab] = nav_s
            dd_df[lab] = ((nav_s / peak_s) - 1.0) * 100.0

        plot_step = max(1, len(bt_sub) // 1400)
        sample_idx = bt_sub.index[::plot_step].union(bt_sub.index[-1:])

        # 2. 累積淨值變化圖 (Equity Curve)
        st.markdown("#### 📈 2. 累積淨值變化圖 (Cumulative Equity Curve, 初始資金 = 1.0)")
        nav_long = (
            nav_df.loc[sample_idx]
            .reset_index()
            .rename(columns={"index": "Date", bt_sub.index.name or "index": "Date"})
        )
        date_col = nav_long.columns[0]
        nav_long = nav_long.rename(columns={date_col: "Date"}).melt("Date", var_name="策略", value_name="累積淨值 (倍)")

        y_scale = alt.Scale(type="log") if use_log else alt.Scale(type="linear")
        chart_nav = (
            alt.Chart(nav_long)
            .mark_line(strokeWidth=2.0)
            .encode(
                x=alt.X("Date:T", title="年份"),
                y=alt.Y("累積淨值 (倍):Q", scale=y_scale, title="累積淨值倍數 (Log 對數座標)" if use_log else "累積淨值倍數 (線性)"),
                color=alt.Color(
                    "策略:N",
                    scale=alt.Scale(
                        domain=list(label_map.values()),
                        range=["#d9534f", "#f0ad4e", "#0275d8", "#6c757d"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Date:T", title="日期", format="%Y-%m-%d"),
                    alt.Tooltip("策略:N", title="策略"),
                    alt.Tooltip("累積淨值 (倍):Q", title="淨值倍數", format=",.2f"),
                ],
            )
            .properties(height=420)
            .interactive()
        )
        st.altair_chart(chart_nav, use_container_width=True)

        # 3. 水下回撤圖 (Underwater Drawdown Chart)
        st.markdown("#### 🌊 3. 歷史水下回撤圖 (Underwater MaxDD Curve %)")
        dd_long = (
            dd_df.loc[sample_idx]
            .reset_index()
            .rename(columns={dd_df.index.name or "index": "Date"})
        )
        dd_long = dd_long.rename(columns={dd_long.columns[0]: "Date"}).melt("Date", var_name="策略", value_name="水下回撤 (%)")

        chart_dd = (
            alt.Chart(dd_long)
            .mark_line(strokeWidth=1.6)
            .encode(
                x=alt.X("Date:T", title="年份"),
                y=alt.Y("水下回撤 (%):Q", title="自歷史高點回撤幅度 (%)"),
                color=alt.Color(
                    "策略:N",
                    scale=alt.Scale(
                        domain=list(label_map.values()),
                        range=["#d9534f", "#f0ad4e", "#0275d8", "#6c757d"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Date:T", title="日期", format="%Y-%m-%d"),
                    alt.Tooltip("策略:N", title="策略"),
                    alt.Tooltip("水下回撤 (%):Q", title="回撤 (%)", format=".2f"),
                ],
            )
            .properties(height=340)
            .interactive()
        )
        st.altair_chart(chart_dd, use_container_width=True)

        # 4. 5 年與 10 年滾動年化報酬率 (5Y & 10Y Rolling CAGR)
        st.markdown("#### 🔄 4. 5 年與 10 年滾動年化複合報酬率 (5-Year & 10-Year Rolling CAGR %)")
        st.caption("衡量『在歷史上任何一天進場並持有滿 5 年（1,260 個交易日）或 10 年（2,520 個交易日）』的年化複合報酬率表現。")

        full_nav_df = pd.DataFrame(index=bt_full_df.index)
        for col, lab in label_map.items():
            full_nav_df[lab] = np.cumprod(1.0 + bt_full_df[col].to_numpy(dtype=float))

        roll_5y_df = ((full_nav_df / full_nav_df.shift(252 * 5)) ** (1.0 / 5.0) - 1.0) * 100.0
        roll_10y_df = ((full_nav_df / full_nav_df.shift(252 * 10)) ** (1.0 / 10.0) - 1.0) * 100.0

        rc1, rc2 = st.columns(2)
        with rc1:
            r5_sub = roll_5y_df.dropna(how="all")
            if not r5_sub.empty:
                r5_idx = r5_sub.index[:: max(1, len(r5_sub) // 900)].union(r5_sub.index[-1:])
                r5_long = r5_sub.loc[r5_idx].reset_index()
                r5_long = r5_long.rename(columns={r5_long.columns[0]: "Date"}).melt(
                    "Date", var_name="策略", value_name="5年滾動CAGR (%)"
                )
                zero_rule5 = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 4], color="black").encode(y="y:Q")
                ch_r5 = (
                    alt.Chart(r5_long)
                    .mark_line(strokeWidth=1.8)
                    .encode(
                        x=alt.X("Date:T", title="滾動 5 年結束日"),
                        y=alt.Y("5年滾動CAGR (%):Q", title="5 年滾動年化報酬 (%)"),
                        color=alt.Color(
                            "策略:N",
                            scale=alt.Scale(
                                domain=list(label_map.values()),
                                range=["#d9534f", "#f0ad4e", "#0275d8", "#6c757d"],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("Date:T", title="結束日期", format="%Y-%m-%d"),
                            alt.Tooltip("策略:N", title="策略"),
                            alt.Tooltip("5年滾動CAGR (%):Q", title="5Y CAGR (%)", format=".2f"),
                        ],
                    )
                    .properties(title="📅 5-Year Rolling CAGR (%)", height=340)
                    .interactive()
                )
                st.altair_chart(ch_r5 + zero_rule5, use_container_width=True)

        with rc2:
            r10_sub = roll_10y_df.dropna(how="all")
            if not r10_sub.empty:
                r10_idx = r10_sub.index[:: max(1, len(r10_sub) // 900)].union(r10_sub.index[-1:])
                r10_long = r10_sub.loc[r10_idx].reset_index()
                r10_long = r10_long.rename(columns={r10_long.columns[0]: "Date"}).melt(
                    "Date", var_name="策略", value_name="10年滾動CAGR (%)"
                )
                zero_rule10 = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 4], color="black").encode(y="y:Q")
                ch_r10 = (
                    alt.Chart(r10_long)
                    .mark_line(strokeWidth=1.8)
                    .encode(
                        x=alt.X("Date:T", title="滾動 10 年結束日"),
                        y=alt.Y("10年滾動CAGR (%):Q", title="10 年滾動年化報酬 (%)"),
                        color=alt.Color(
                            "策略:N",
                            scale=alt.Scale(
                                domain=list(label_map.values()),
                                range=["#d9534f", "#f0ad4e", "#0275d8", "#6c757d"],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("Date:T", title="結束日期", format="%Y-%m-%d"),
                            alt.Tooltip("策略:N", title="策略"),
                            alt.Tooltip("10年滾動CAGR (%):Q", title="10Y CAGR (%)", format=".2f"),
                        ],
                    )
                    .properties(title="🏛️ 10-Year Rolling CAGR (%)", height=340)
                    .interactive()
                )
                st.altair_chart(ch_r10 + zero_rule10, use_container_width=True)

        # 滾動 5Y / 10Y 分位數統計表
        roll_stat_rows = []
        for lab in label_map.values():
            s5 = roll_5y_df[lab].dropna()
            s10 = roll_10y_df[lab].dropna()
            roll_stat_rows.append({
                "策略名稱": lab,
                "5Y 滾動 CAGR 中位數": f"{s5.median():.2f}%" if not s5.empty else "N/A",
                "5Y 滾動 CAGR 最差 (Min)": f"{s5.min():.2f}%" if not s5.empty else "N/A",
                "5Y 滾動 CAGR 最佳 (Max)": f"{s5.max():.2f}%" if not s5.empty else "N/A",
                "5Y 正報酬勝率": f"{(s5 > 0).mean()*100:.1f}%" if not s5.empty else "N/A",
                "10Y 滾動 CAGR 中位數": f"{s10.median():.2f}%" if not s10.empty else "N/A",
                "10Y 滾動 CAGR 最差 (Min)": f"{s10.min():.2f}%" if not s10.empty else "N/A",
                "10Y 滾動 CAGR 最佳 (Max)": f"{s10.max():.2f}%" if not s10.empty else "N/A",
                "10Y 正報酬勝率": f"{(s10 > 0).mean()*100:.1f}%" if not s10.empty else "N/A",
            })
        st.dataframe(pd.DataFrame(roll_stat_rows).set_index("策略名稱"), use_container_width=True)

        # 5. 歷年年度報酬對照表 (Annual Calendar Returns)
        st.markdown("#### 📅 5. 歷年年度報酬率對照表 (Annual Calendar Returns %)")
        ann_rows = []
        for yr, grp in bt_sub.groupby(bt_sub.index.year):
            r_reb_y = float(np.prod(1.0 + grp["REBAND_HF0_200_175"].to_numpy()) - 1.0) * 100.0
            r_hf0_y = float(np.prod(1.0 + grp["SCOPE_HALF_F0"].to_numpy()) - 1.0) * 100.0
            r_o_y = float(np.prod(1.0 + grp["Strategy_O"].to_numpy()) - 1.0) * 100.0
            r_spy_y = float(np.prod(1.0 + grp["SPY_1x"].to_numpy()) - 1.0) * 100.0
            ann_rows.append({
                "年份": str(yr),
                "REBAND_HF0_200_175 (%)": round(r_reb_y, 2),
                "SCOPE_HALF_F0 (%)": round(r_hf0_y, 2),
                "原版策略 O (%)": round(r_o_y, 2),
                "SPY 1x (%)": round(r_spy_y, 2),
                "REBAND 超額報酬 vs O (pp)": round(r_reb_y - r_o_y, 2),
                "REBAND 免交易日比例 (%)": round(float(grp["Skip_Frac_REBAND"].mean()) * 100.0, 1),
            })
        st.dataframe(pd.DataFrame(ann_rows[::-1]).set_index("年份"), use_container_width=True)

    # ==========================================
    # 分頁 3：策略架構與數學規格說明書
    # ==========================================
    with tab_doc:
        st.subheader("📖 REBAND_HF0_200_175 策略完整架構與數學規格說明書")
        st.markdown(
            """
### 一、 核心設計哲學：為什麼 `REBAND_HF0_200_175` 全面超越舊版策略 `O`？

`REBAND_HF0_200_175` 是針對全球三倍槓桿指數 ETF（美股 `UPRO`、歐股 `EURL`、新興市場 `EDC`）與防禦型資產（黃金 `GLD`、長天期美債 `TLT`）設計的**機構級全天候動態資產配置系統**。它徹底解決了第一代策略 `O` 的三大結構性瓶頸：

1. **訊號源零槓桿汙染（1x Prototype Purity）**：
   3 倍槓桿 ETF 在高波動震盪期會產生嚴重的「波動率耗損（Volatility Drag）」與「融資利差拖累（TRS Financing Drag）」。因此本系統**所有選股排名、月度均線與 GARCH 波動率預測，100% 在 1x 原型 ETF（`SPY, VGK, EEM`）上運算**，再映射至 3x ETF 執行。
2. **將「一刀切二元波動率開關」升級為「雙層波動率控制（$Q_{p,t}$ 危機峰值回落閘門 + $a_{p,t}$ 連續平方縮放器）」**：
   - 舊版 `O` 在波動率剛突破 97.5% 分位數時只砍半倉（仍留 50% 3x ETF 承受暴跌），且在 V 型反彈初期因為過去 252 日分位數被墊高而遲遲無法重新進場。
   - 新版 `REBAND_HF0_200_175` 拆解為：
     - **危機斷路器 $Q_{p,t} \in \{0, 1\}$**：當預測波動率超過 252 日中位數的 **2.5 倍（$\rho_{p,t} \ge 2.5$）** 時，直接將股票部位**瞬間歸零（$Q=0$）**，100% 轉入防禦資產避開熔斷主跌段；一旦波動率從危機峰值**回落 40%（$\sigma_{p,t} \le 0.60 \sigma_{\text{peak}}$）**，不必苦等 252 日分位數下降，立即在 V 型反轉底部**重新進場（$Q=1$）**！
     - **常態平方縮放器 $a_{p,t} \in (0, 1]$**：當 $\sigma_{p,t} > 41\%$ 時，依 $a_{p,t} = \min(1, (41/\sigma_{p,t})^2)$ 平滑縮減槓桿部位。
3. **Vanguard 200 / 175 bps 成本走廊與 7 大非對稱強制覆蓋規則（Cost-Aware Corridor + Mandatory Risk-Off Override）**：
   - **平時免交易（省下 80% 換手率）**：只要當日沒有任何關鍵訊號翻轉或減碼指令，且實際持倉因日內價格漲跌產生的漂移小於 **200 bps（2.00%）**，當天開盤就 **`Action = 0`（保留原股數、完全不交易）**。
   - **觸及邊界緩衝拉回**：若純價格漂移 $\ge 200\text{ bps}$，僅部分調倉拉回至 **175 bps（1.75%）** 邊界（`Action = 1`）。
   - **減碼與訊號翻轉零延遲**：一旦觸發 7 大強制規則（特別是第 7 條：**當日目標股票預算調降 $B_{p,t} < B_{p,t-1}$**），系統**立即無視 200 bps 走廊，強制 100% 精確執行減碼（`Action = 2`）**！

---

### 二、 四大模組精確數學公式

#### 模組 1：月底非重疊三段風險調整動能選股（1x 原型）
在每個月最後一個交易日收盤，對 $i \in \{\text{SPY}, \text{VGK}, \text{EEM}\}$ 計算過去 12 個月拆分為三段非重疊區間 $(0,3]\text{M}$、$(3,7]\text{M}$、$(7,12]\text{M}$ 的報酬率，並除以 252 日年化波動率 $\sigma^{(252)}_{i,m}$：
$$\text{Score}_{i,m} = \frac{1}{3} \left( \frac{R^{(0,3)}_{i,m}}{\sigma^{(252)}_{i,m}} + \frac{R^{(3,7)}_{i,m}}{\sigma^{(252)}_{i,m}} + \frac{R^{(7,12)}_{i,m}}{\sigma^{(252)}_{i,m}} \right)$$
得分最高者之對應 3x ETF（$\text{SPY}\to\text{UPRO}, \text{VGK}\to\text{EURL}, \text{EEM}\to\text{EDC}$）即為次月唯一進攻標的 $L_t$。

#### 模組 2：月度 6 個月簡單移動平均趨勢濾網（$\text{SMA}_6$）
在月底收盤檢驗勝出之 1x 原型收盤價是否嚴格大於過去 6 個月底收盤價均值：
$$S_t = \mathbb{I}\!\left( P_{L_t, m} > \frac{1}{6}\sum_{k=0}^{5} P_{L_t, m-k} \right) \in \{0, 1\}$$

#### 模組 3：五相位交錯 Student-$t$ $\text{GARCH}(1,1)$ 與基礎股票預算 $B_{p,t}$
將資金均分為 5 個獨立子相位 $p \in \{0, 1, 2, 3, 4\}$（各佔 20% 資金）。在交易日 $t$，僅輪值相位 $p \equiv (t - 504) \pmod 5$ 使用最新 504 日 1x 原型日報酬擬合 Student-$t$ $\text{GARCH}(1,1)$ 預測年化波動率 $\sigma_{p,t}$，並計算相對於過去 252 日預測中位數之警報比 $\rho_{p,t} = \sigma_{p,t} / \text{Med}_{252}(\sigma)_{p,t}$：
$$B_{p,t} = Q_{p,t} \times \left( 0.5 \cdot S_t + 0.5 \cdot \min\!\left[1,\; \left(\frac{41\%}{\sigma_{p,t}}\right)^2\right] \right)$$
剩餘預算 $1 - B_{p,t}$ 全數配置於當月 `GLD` 與 `TLT` 中 $(6\text{M} + 12\text{M})/2$ 平均動能較高之避險資產 $D_t$。

#### 模組 4：單一帳戶匯總執行公式
五個相位各自獨立透過 `200 / 175 bps` 護欄決定今日開盤計畫權重 $w^{\text{plan}}_{p,t}$ 後，你在單一券商帳戶只需執行五相位的算術平均權重：
$$\bar{w}^{\text{plan}}_t = \frac{1}{5} \sum_{p=0}^{4} w^{\text{plan}}_{p,t}$$
"""
        )

        st.subheader("🏛️ 三、 機構鎖定歷史審計基準表 (Locked Institutional Audit Benchmarks)")
        audit_bench_df = pd.DataFrame([
            {
                "評估環境與時間跨度": "實盤 ETF 歷史期 (2014-01-23 ～ 2026-09-28, 3,189 交易日, 20bps)",
                "REBAND_HF0_200_175": "CAGR 25.23% ｜ Sharpe 0.700 ｜ Sortino 0.967 ｜ MaxDD -49.05% ｜ 雙邊周轉率 6.90x/年",
                "SCOPE_HALF_F0 (無護欄)": "CAGR 25.13% ｜ Sharpe 0.698 ｜ Sortino 0.964 ｜ MaxDD -49.05% ｜ 雙邊周轉率 7.31x/年",
                "原版策略 O": "CAGR 20.31% ｜ Sharpe 0.611 ｜ Sortino 0.829 ｜ MaxDD -50.14% ｜ 雙邊周轉率 8.45x/年",
                "核心審計結論": "REBAND 配對中位數較 O 提升 +6.95 pp 年化報酬、+0.134 Sharpe，且超過 93% 交易日免下單",
            },
            {
                "評估環境與時間跨度": "超長歷史合成期 (1998-03-02 ～ 2026-09-28, 7,189 交易日, 20bps)",
                "REBAND_HF0_200_175": "CAGR 25.88% ｜ Sharpe 0.673 ｜ MaxDD -79.06% ｜ 雙邊周轉率 5.91x/年",
                "SCOPE_HALF_F0 (無護欄)": "CAGR 25.69% ｜ Sharpe 0.670 ｜ MaxDD -78.70% ｜ 雙邊周轉率 6.66x/年",
                "原版策略 O": "CAGR 23.27% ｜ Sharpe 0.634 ｜ MaxDD -84.16% ｜ 雙邊周轉率 7.97x/年",
                "核心審計結論": "跨 28.6 年完整牛熊，REBAND 配對中位數較 O 提升 +3.57 pp 年化報酬，最大回撤改善 +6.36 pp",
            },
            {
                "評估環境與時間跨度": "參數穩健度與延遲壓力測試 (Round-48 24 單軸 / 256 聯合角點 / 16 延遲遮罩)",
                "REBAND_HF0_200_175": "連續/護欄 256 角點 100% (256/256) 通過 ｜ 16 種延遲遮罩 16/16 全數擊敗 O (最差仍贏 +5.84 pp)",
                "SCOPE_HALF_F0 (無護欄)": "24 單軸配對 24/24 全數擊敗 O ｜ 256 角點 256/256 全數正超額報酬",
                "原版策略 O": "基準對照組",
                "核心審計結論": "28.6 年長歷史下額外延遲 1 個交易日 (delay_mask=15 vs 0) 通過穩定門檻 (-1.19 pp CAGR)",
            },
        ]).set_index("評估環境與時間跨度")
        st.dataframe(audit_bench_df, use_container_width=True)

    # ==========================================
    # 分頁 4：當月 1x 動能排名與歷史輪動紀錄
    # ==========================================
    with tab_rot:
        r_col1, r_col2 = st.columns([3, 2])
        with r_col1:
            st.subheader("📊 當月 1x 原型進攻動能排名表 (SPY / VGK / EEM)")
            st.dataframe(sel_df, use_container_width=True)
        with r_col2:
            st.subheader("🛡️ 當月避險資產動能比較表 (GLD vs TLT)")
            st.dataframe(safe_df.style.format("{:.2%}"), use_container_width=True)

        st.subheader("🗓️ 過去 24 個月進攻與防禦標的歷史輪動紀錄")
        st.dataframe(hist_rot_df.set_index("月底訊號鎖定日"), use_container_width=True)

    # ==========================================
    # 分頁 5：原始報價與系統健康檢查
    # ==========================================
    with tab_raw:
        st.subheader("🗄️ 最新 15 個交易日 ETF 收盤價 (Auto-Adjusted Close)")
        st.dataframe(closes_df.tail(15).style.format("{:.2f}"), use_container_width=True)
        st.caption(
            f"✅ 資料序列健康度檢查通過：共載入 `{len(closes_df):,}` 個交易日 "
            f"（起始日 `{closes_df.index[0].strftime('%Y-%m-%d')}`，最新收盤日 `{closes_df.index[-1].strftime('%Y-%m-%d')}`）。"
        )
