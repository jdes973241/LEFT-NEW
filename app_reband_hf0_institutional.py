from datetime import datetime, timedelta
import io
import os
import warnings

import altair as alt
from arch import arch_model
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

# ==========================================
# 0. 頁面設定與機構實盤規格 (REBAND_QONLY_200_0 無近似旗艦部署版)
# ==========================================
st.set_page_config(
    page_title="REBAND_QONLY_200_0 機構實盤下單與無近似量化回測系統",
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
# 1. 凍結策略核心常數 (REBAND_QONLY_200_0 — 已拔除冗餘內層 a)
# ==========================================
MAPPING = {"UPRO": "SPY", "EURL": "VGK", "EDC": "EEM"}
REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}
TRADE_ETFS_3X = ["UPRO", "EURL", "EDC"]
PROTO_ETFS_1X = ["SPY", "VGK", "EEM"]
SAFE_POOL = ["GLD", "TLT"]
ALL_EXEC_ASSETS = ["UPRO", "EURL", "EDC", "GLD", "TLT"]
CORE_LIVE_ETFS = TRADE_ETFS_3X + PROTO_ETFS_1X + SAFE_POOL

# 嚴格無金礦股汙染代理：GLD 上市前(2000-08-30~2004-11-17) 改用 GC=F 黃金期貨，徹底禁用 FSAGX
CLEAN_HIST_PROXIES = {
    "VGK": "VEURX",   # Vanguard European Stock Index Fund
    "EEM": "VEIEX",   # Vanguard Emerging Markets Stock Index Fund
    "TLT": "VUSTX",   # Vanguard Long-Term Treasury Fund
    "GLD": "GC=F",    # COMEX Gold Futures (取代原版誤用之 FSAGX 金礦股)
}

SMA_MONTHS = 6
GARCH_WINDOW = 504
QUANTILE_WINDOW = 252
PHASE_COUNT = 5

# 進攻與避險統一採用四等分標準日曆季度 (0, 3, 6, 9, 12) / Vol252 零自由參數規格
MOM_SEGMENTS = [(0, 3), (3, 6), (6, 9), (9, 12)]

# REBAND_QONLY_200_0 凍結 4 參數外層危機峰值回落斷路器 Q (已拔除內層 a 之 41% 與 2.0)
RHO_EXIT = 2.50               # 初次/再次危機退出門檻 rho >= 2.50
RHO_REENTRY = 1.25            # 危機解除復歸門檻 rho <= 1.25
PEAK_SIGMA_DROP = 0.40        # 退出期波動率自峰值回落 40% (<= 0.60 * peak) 復歸
TROUGH_SIGMA_RISE = 0.25      # 復歸後波動率較活動期低點升 25% 且 rho >= 2.50 才再次退出
BAND_TRIG_TH = 0.0200         # 200 bps (2.00%) 觸發門檻
BAND_DEST_TH = 0.0000         # 0 bps (0.00%) 觸發後直接一步拉回基礎目標權重

# 公平舊版 O 對照常數
O_EXIT_Q = 0.975
O_ENTRY_Q = 0.900

EXPECTED_DATA_START = pd.Timestamp("1995-01-03")
GARCH_FIT_START_DATE = "2000-06-01"
DEFAULT_GARCH_CACHE_FILE = "reband_exact_garch_5phase_cache.npz"

# 事前固定之重大歷史危機視窗 (用於分頁 2 機構級避險三指標報表)
CRISIS_WINDOWS = [
    ("2002 網路泡沫末段", "2002-03-19", "2002-10-09"),
    ("2008 全球金融海嘯", "2007-10-09", "2009-03-09"),
    ("2011 歐債與美債降評", "2011-04-29", "2011-10-03"),
    ("2018 Q4 鷹派急跌", "2018-09-20", "2018-12-24"),
    ("2020 COVID 熔斷下跌段", "2020-02-19", "2020-03-23"),
    ("2020 熔斷+V轉完整段", "2020-02-19", "2020-08-31"),
    ("2022 暴力升息熊市", "2022-01-03", "2022-10-12"),
    ("2025 關稅市場震盪", "2025-02-19", "2025-04-08"),
]

# 顯性時區初始化（絕不放入 @st.cache_data）
TZ_TW = pytz.timezone("Asia/Taipei")
TZ_NY = pytz.timezone("America/New_York")
NOW_TW = datetime.now(TZ_TW)
NOW_NY = datetime.now(TZ_NY)

with st.sidebar:
    st.header("🔧 單一帳戶下單與回測設定")
    st.info(
        f"🇹🇼 台灣時間: {NOW_TW.strftime('%Y-%m-%d %H:%M %Z')}\n\n"
        f"🇺🇸 美東時間: {NOW_NY.strftime('%Y-%m-%d %H:%M %Z')}"
    )

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
        help="僅在『美股當月最後一個交易日 16:00 EST 收盤後』勾選，強制系統立刻以最新已收盤價結算次月訊號。",
    )
    if st.button("🗑️ 強制清除下載快取 (重抓最新報價)"):
        st.cache_data.clear()
        for k in ["exact_garch_runtime_mem", "uploaded_garch_npz_bytes"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


# ==========================================
# 2. 純數據下載 (I/O) 與 MultiIndex 扁平化引擎 (嚴格遵守快取紀律：僅純 I/O 使用 @st.cache_data)
# ==========================================
def extract_flat_close_open(raw: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """下載後立即將 yfinance MultiIndex 扁平化為單層索引之 closes 與 opens DataFrame。"""
    closes = pd.DataFrame(index=raw.index)
    opens = pd.DataFrame(index=raw.index)
    if raw is None or raw.empty:
        return closes, opens

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))
        for t in tickers:
            if t in lvl0:
                sub = raw[t]
                if "Close" in sub.columns:
                    closes[t] = pd.to_numeric(sub["Close"], errors="coerce")
                elif "Price" in sub.columns:
                    closes[t] = pd.to_numeric(sub["Price"], errors="coerce")
                if "Open" in sub.columns:
                    opens[t] = pd.to_numeric(sub["Open"], errors="coerce")
                elif t in closes.columns:
                    opens[t] = closes[t]
            elif t in lvl1:
                if ("Close", t) in raw.columns:
                    closes[t] = pd.to_numeric(raw[("Close", t)], errors="coerce")
                if ("Open", t) in raw.columns:
                    opens[t] = pd.to_numeric(raw[("Open", t)], errors="coerce")
                elif t in closes.columns:
                    opens[t] = closes[t]
    else:
        for t in tickers:
            if t in raw.columns:
                closes[t] = pd.to_numeric(raw[t], errors="coerce")
                opens[t] = closes[t]
    return closes, opens


@st.cache_data(ttl=180, show_spinner="正在從 Yahoo Finance 下載長天期歷史復權報價 (單線程)...")
def download_yf_history_io(all_tickers: tuple[str, ...], start_str: str, end_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """純 I/O：不包含任何 datetime.now() 或即時狀態判斷。"""
    raw = yf.download(
        list(all_tickers),
        start=start_str,
        end=end_str,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="ticker",
    )
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    c_df, o_df = extract_flat_close_open(raw, list(all_tickers))
    if c_df.index.tz is not None:
        c_df.index = c_df.index.tz_localize(None)
        o_df.index = o_df.index.tz_localize(None)
    return c_df, o_df


@st.cache_data(ttl=180, show_spinner="正在補齊近 5 日核心 ETF 即時日K與正規盤 15m 報價...")
def download_yf_patch_io(core_tickers: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """純 I/O：抓取近 5 日 1d 與 15m 正規盤報價，不包含任何 datetime.now() 判斷。"""
    c_5d, o_5d = pd.DataFrame(), pd.DataFrame()
    c_15m, o_15m = pd.DataFrame(), pd.DataFrame()
    try:
        raw_5d = yf.download(
            list(core_tickers),
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )
        c_5d, o_5d = extract_flat_close_open(raw_5d, list(core_tickers))
        if not c_5d.empty and c_5d.index.tz is not None:
            c_5d.index = c_5d.index.tz_localize(None)
            o_5d.index = o_5d.index.tz_localize(None)
    except Exception:
        pass

    try:
        raw_15m = yf.download(
            list(core_tickers),
            period="5d",
            interval="15m",
            prepost=False,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )
        c_15m, o_15m = extract_flat_close_open(raw_15m, list(core_tickers))
    except Exception:
        pass

    return c_5d, o_5d, c_15m, o_15m


def splice_early_series_causal(primary_s: pd.Series, proxy_s: pd.Series) -> pd.Series:
    """嚴格因果銜接：僅在 primary 上市前且 proxy 已有效之區間銜接，絕不使用 .bfill()。"""
    s = primary_s.copy()
    first_valid = s.first_valid_index()
    if first_valid is None or proxy_s is None or proxy_s.dropna().empty:
        return s

    s.loc[first_valid:] = s.loc[first_valid:].ffill()
    early_idx = s.index[s.index < first_valid]
    if len(early_idx) == 0:
        return s

    proxy_first = proxy_s.first_valid_index()
    if proxy_first is None or proxy_first >= first_valid:
        return s

    p_sub = proxy_s.reindex(s.index[: s.index.get_loc(first_valid) + 1]).copy()
    p_sub.loc[proxy_first:] = p_sub.loc[proxy_first:].ffill()

    anchor_p = p_sub.loc[first_valid]
    anchor_s = s.loc[first_valid]
    if pd.isna(anchor_p) or anchor_p == 0 or pd.isna(anchor_s):
        return s

    scale = float(anchor_s) / float(anchor_p)
    valid_early = p_sub.loc[early_idx].dropna().index
    s.loc[valid_early] = p_sub.loc[valid_early] * scale
    return s


def build_clean_live_market_bundle(now_ny: datetime) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, str, bool]:
    """
    非快取即時組裝邏輯：
    1. 呼叫純 I/O 下載歷史與近 5 日數據。
    2. 嚴格執行美東收盤防呆（美東未滿 16:00 自動剔除當日盤中未收盤 K 線）。
    3. 全程禁止 .bfill()，並以 GC=F 取代 FSAGX。
    """
    proxy_tickers = list(CLEAN_HIST_PROXIES.values()) + ["^IRX"]
    all_tickers = tuple(CORE_LIVE_ETFS + proxy_tickers)
    end_tomorrow_ny = (now_ny + timedelta(days=2)).strftime("%Y-%m-%d")

    closes_raw, opens_raw = download_yf_history_io(
        all_tickers, EXPECTED_DATA_START.strftime("%Y-%m-%d"), end_tomorrow_ny
    )
    if closes_raw.empty or "SPY" not in closes_raw.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), "下載失敗", False

    valid_spy = closes_raw["SPY"].dropna().index
    closes_raw = closes_raw.loc[valid_spy].sort_index()
    opens_raw = opens_raw.reindex(valid_spy).sort_index()

    # 無風險利率 ^IRX (僅 ffill，早於首筆有效值才設常數 2.5%，不向後偷看)
    if "^IRX" in closes_raw.columns:
        rf_ann = (closes_raw["^IRX"].ffill().fillna(2.5) / 100.0).clip(lower=0.0, upper=0.15)
    else:
        rf_ann = pd.Series(0.025, index=closes_raw.index)

    closes = pd.DataFrame(index=closes_raw.index)
    opens = pd.DataFrame(index=closes_raw.index)
    closes["SPY"] = closes_raw["SPY"].ffill()
    opens["SPY"] = opens_raw["SPY"].combine_first(closes["SPY"].shift(1)).combine_first(closes["SPY"])

    # 1x 資產嚴格因果銜接 (含 GLD <- GC=F)
    for etf_1x, proxy_t in CLEAN_HIST_PROXIES.items():
        c_s = closes_raw[etf_1x] if etf_1x in closes_raw.columns else pd.Series(np.nan, index=closes_raw.index)
        o_s = opens_raw[etf_1x] if etf_1x in opens_raw.columns else pd.Series(np.nan, index=closes_raw.index)
        p_c = closes_raw[proxy_t] if proxy_t in closes_raw.columns else None

        c_spliced = splice_early_series_causal(c_s, p_c)
        closes[etf_1x] = c_spliced

        first_etf_dt = c_s.first_valid_index()
        o_full = c_spliced.shift(1)
        if first_etf_dt is not None:
            o_live = o_s.loc[first_etf_dt:].combine_first(c_spliced.shift(1).loc[first_etf_dt:])
            o_full.loc[first_etf_dt:] = o_live
        opens[etf_1x] = o_full

    # 3x ETF 嚴格因果合成 (上市後 100% 真實 ETF Open/Close)
    for t3x, proto_1x in MAPPING.items():
        c3 = closes_raw[t3x] if t3x in closes_raw.columns else pd.Series(np.nan, index=closes_raw.index)
        o3 = opens_raw[t3x] if t3x in opens_raw.columns else pd.Series(np.nan, index=closes_raw.index)
        first_3x = c3.first_valid_index()

        r_1x = (closes[proto_1x] / closes[proto_1x].shift(1)) - 1.0
        daily_drag = (0.0095 + 2.0 * (rf_ann.shift(1).fillna(0.025) + 0.01)) / 252.0
        synth_r3 = (3.0 * r_1x - daily_drag).clip(lower=-0.95)
        valid_r_idx = synth_r3.dropna().index
        synth_p3 = pd.Series(np.nan, index=closes_raw.index)
        if len(valid_r_idx) > 0:
            synth_p3.loc[valid_r_idx] = 100.0 * np.cumprod(1.0 + synth_r3.loc[valid_r_idx].to_numpy(dtype=float))

        c3_full = splice_early_series_causal(c3, synth_p3)
        o3_full = c3_full.shift(1)
        if first_3x is not None:
            o3_live = o3.loc[first_3x:].combine_first(c3_full.shift(1).loc[first_3x:])
            o3_full.loc[first_3x:] = o3_live
        closes[t3x] = c3_full
        opens[t3x] = o3_full

    # 三階段盤後即時收盤補齊 (嚴格檢查美東收盤狀態)
    patch_note = "標準日K (1d)"
    c_5d, o_5d, c_15m, o_15m = download_yf_patch_io(tuple(CORE_LIVE_ETFS))
    today_ny_ts = pd.Timestamp(now_ny.date())
    is_us_open_intraday = (now_ny.weekday() < 5) and (now_ny.hour < 16)

    if not c_5d.empty and "SPY" in c_5d.columns:
        c_5d = c_5d.loc[c_5d["SPY"].dropna().index]
        for dt in c_5d.index:
            if is_us_open_intraday and (dt >= today_ny_ts):
                continue
            for t in CORE_LIVE_ETFS:
                if t in c_5d.columns and pd.notna(c_5d.loc[dt, t]):
                    closes.loc[dt, t] = float(c_5d.loc[dt, t])
                if t in o_5d.columns and pd.notna(o_5d.loc[dt, t]):
                    opens.loc[dt, t] = float(o_5d.loc[dt, t])
        closes = closes.sort_index()
        opens = opens.sort_index()

    if not c_15m.empty and "SPY" in c_15m.columns:
        if c_15m.index.tz is None:
            c_15m.index = c_15m.index.tz_localize("UTC").tz_convert(TZ_NY)
            o_15m.index = o_15m.index.tz_localize("UTC").tz_convert(TZ_NY)
        else:
            c_15m.index = c_15m.index.tz_convert(TZ_NY)
            o_15m.index = o_15m.index.tz_convert(TZ_NY)

        c_15m = c_15m.loc[c_15m["SPY"].dropna().index]
        o_15m = o_15m.reindex(c_15m.index)
        for d_val, grp_c in c_15m.groupby(c_15m.index.date):
            ts_day = pd.Timestamp(d_val)
            if is_us_open_intraday and (ts_day >= today_ny_ts):
                continue
            last_bar_tm = grp_c.index[-1]
            is_session_closed = (last_bar_tm.hour > 15) or (last_bar_tm.hour == 15 and last_bar_tm.minute >= 45)
            if is_session_closed and (ts_day not in closes.index):
                grp_o = o_15m.loc[grp_c.index]
                for t in CORE_LIVE_ETFS:
                    s_c = grp_c[t].dropna() if t in grp_c.columns else pd.Series(dtype=float)
                    s_o = grp_o[t].dropna() if t in grp_o.columns else pd.Series(dtype=float)
                    if not s_c.empty:
                        closes.loc[ts_day, t] = float(s_c.iloc[-1])
                        opens.loc[ts_day, t] = float(s_o.iloc[0]) if not s_o.empty else float(s_c.iloc[0])
                patch_note = f"盤後 15m 即時合成日K (已補齊 {ts_day.strftime('%Y-%m-%d')} 收盤價)"
        closes = closes.sort_index()
        opens = opens.sort_index()

    # 美東盤中防呆：若歷史 1d 序列本身夾帶尚未收盤之當日盤中 K 線，嚴格剔除
    dropped_unclosed = False
    if is_us_open_intraday and (closes.index[-1] >= today_ny_ts):
        closes = closes.loc[closes.index < today_ny_ts]
        opens = opens.loc[opens.index < today_ny_ts]
        dropped_unclosed = True
        patch_note += " ｜ 已自動排除美股未收盤之盤中報價"

    # 嚴格僅向後 ffill (絕不 bfill)
    for col in closes.columns:
        fv = closes[col].first_valid_index()
        if fv is not None:
            closes.loc[fv:, col] = closes.loc[fv:, col].ffill()
        fvo = opens[col].first_valid_index()
        if fvo is not None:
            opens.loc[fvo:, col] = opens.loc[fvo:, col].ffill()

    rf_ann = rf_ann.reindex(closes.index).ffill().fillna(0.025)
    return closes, opens, rf_ann, patch_note, dropped_unclosed


# ==========================================
# 3. 月度動能排名 (統一四等分標準日曆季度 / Vol252) 與 SMA6 (嚴格美東時區跨月)
# ==========================================
def get_confirmed_month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """僅保留已確認跨月完結之月底交易日，絕不提前鎖定未完結月份。"""
    periods = idx.to_period("M")
    is_eom = np.zeros(len(idx), dtype=bool)
    is_eom[:-1] = periods[:-1] != periods[1:]
    return idx[is_eom]


def safe_row_idxmax_strict(df: pd.DataFrame) -> pd.Series:
    """僅當該列所有欄位皆為有限值時輸出最大值欄名，否則維持 NaN（絕不盲填預設標的）。"""
    out = pd.Series(index=df.index, dtype=object)
    valid_mask = df.notna().all(axis=1)
    if valid_mask.any():
        sub = df.loc[valid_mask]
        cols = np.asarray(sub.columns, dtype=object)
        out.loc[valid_mask] = cols[np.argmax(sub.to_numpy(dtype=float), axis=1)]
    return out


def calculate_monthly_signals_1x(closes: pd.DataFrame, now_ny: datetime, force_eom: bool) -> dict:
    """嚴格以美東時區判定已完結月份，並計算四等分標準日曆季度 (0,3,6,9,12)/Vol252 動能與月度 SMA6。"""
    confirmed_eom = get_confirmed_month_end_dates(closes.index)
    last_date = closes.index[-1]
    last_data_period = last_date.to_period("M")
    current_ny_period = pd.Period(now_ny.strftime("%Y-%m"), freq="M")

    # 僅在「手動勾選月底強制結算」或「美東時間已正式跨入次月」時，才把最新收盤日納入已結算月底
    is_last_month_locked = force_eom or (last_data_period < current_ny_period)
    if is_last_month_locked and (len(confirmed_eom) == 0 or confirmed_eom[-1] != last_date):
        effective_monthly_idx = confirmed_eom.append(pd.DatetimeIndex([last_date]))
    else:
        effective_monthly_idx = confirmed_eom

    # 1. 進攻端 1x 原型 (SPY, VGK, EEM)
    p_1x = closes[PROTO_ETFS_1X]
    m_1x = p_1x.loc[effective_monthly_idx]
    r_1x = (p_1x / p_1x.shift(1)) - 1.0
    v_1x = (r_1x.rolling(252, min_periods=252).std(ddof=1) * np.sqrt(252.0)).reindex(effective_monthly_idx)

    seg_scores, seg_rets = [], {}
    for s_m, e_m in MOM_SEGMENTS:
        r_seg = (m_1x / m_1x.shift(e_m) - 1.0) if s_m == 0 else (m_1x.shift(s_m) / m_1x.shift(e_m) - 1.0)
        seg_rets[(s_m, e_m)] = r_seg
        seg_scores.append(r_seg / (v_1x + 1e-12))
    total_score_df = sum(seg_scores) / float(len(MOM_SEGMENTS))

    sma6_mean_df = m_1x.rolling(SMA_MONTHS, min_periods=SMA_MONTHS).mean()
    sma6_state_df = pd.DataFrame(
        np.where(sma6_mean_df.notna(), (m_1x > sma6_mean_df).astype(float), np.nan),
        index=effective_monthly_idx,
        columns=PROTO_ETFS_1X,
    )

    # 2. 防禦端避險池 (GLD, TLT)
    p_safe = closes[SAFE_POOL]
    m_safe = p_safe.loc[effective_monthly_idx]
    r_safe = (p_safe / p_safe.shift(1)) - 1.0
    v_safe = (r_safe.rolling(252, min_periods=252).std(ddof=1) * np.sqrt(252.0)).reindex(effective_monthly_idx)

    safe_seg_scores, safe_seg_rets = [], {}
    for s_m, e_m in MOM_SEGMENTS:
        r_seg_s = (m_safe / m_safe.shift(e_m) - 1.0) if s_m == 0 else (m_safe.shift(s_m) / m_safe.shift(e_m) - 1.0)
        safe_seg_rets[(s_m, e_m)] = r_seg_s
        safe_seg_scores.append(r_seg_s / (v_safe + 1e-12))
    safe_comp = sum(safe_seg_scores) / float(len(MOM_SEGMENTS))

    ref_date = effective_monthly_idx[-1]

    sel_rows = []
    for proto_t in PROTO_ETFS_1X:
        trade_3x = REVERSE_MAPPING[proto_t]
        sel_rows.append({
            "Trade_ETF_3x": trade_3x,
            "Signal_Proto_1x": proto_t,
            "Close_1x": float(m_1x.loc[ref_date, proto_t]),
            "SMA6_1x": float(sma6_mean_df.loc[ref_date, proto_t]),
            "SMA6_State": int(sma6_state_df.loc[ref_date, proto_t]),
            "Vol_252d_Ann": float(v_1x.loc[ref_date, proto_t]),
            "Ret_0to3M": float(seg_rets[(0, 3)].loc[ref_date, proto_t]),
            "Ret_3to6M": float(seg_rets[(3, 6)].loc[ref_date, proto_t]),
            "Ret_6to9M": float(seg_rets[(6, 9)].loc[ref_date, proto_t]),
            "Ret_9to12M": float(seg_rets[(9, 12)].loc[ref_date, proto_t]),
            "Total_Score": float(total_score_df.loc[ref_date, proto_t]),
        })
    sel_df = pd.DataFrame(sel_rows).set_index("Trade_ETF_3x").sort_values("Total_Score", ascending=False)

    safe_rows = []
    for s_t in SAFE_POOL:
        safe_rows.append({
            "Ticker": s_t,
            "Close_1x": float(m_safe.loc[ref_date, s_t]),
            "Vol_252d_Ann": float(v_safe.loc[ref_date, s_t]),
            "Ret_0to3M": float(safe_seg_rets[(0, 3)].loc[ref_date, s_t]),
            "Ret_3to6M": float(safe_seg_rets[(3, 6)].loc[ref_date, s_t]),
            "Ret_6to9M": float(safe_seg_rets[(6, 9)].loc[ref_date, s_t]),
            "Ret_9to12M": float(safe_seg_rets[(9, 12)].loc[ref_date, s_t]),
            "Total_Score": float(safe_comp.loc[ref_date, s_t]),
        })
    safe_df = pd.DataFrame(safe_rows).set_index("Ticker").sort_values("Total_Score", ascending=False)
    safe_win = str(safe_df.index[0])

    leader_1x_m = safe_row_idxmax_strict(total_score_df)
    leader_3x_m = leader_1x_m.map(REVERSE_MAPPING)
    safe_win_m = safe_row_idxmax_strict(safe_comp)

    daily_leader_3x = leader_3x_m.reindex(closes.index).ffill()
    daily_safe_win = safe_win_m.reindex(closes.index).ffill()
    daily_sma6_by_3x = sma6_state_df.rename(columns=REVERSE_MAPPING).reindex(closes.index).ffill()

    hist_rot_rows = []
    for m_dt in effective_monthly_idx[-24:][::-1]:
        l_1x = leader_1x_m.loc[m_dt]
        s_w = safe_win_m.loc[m_dt]
        if pd.isna(l_1x) or pd.isna(s_w):
            continue
        l_1x_str, s_w_str = str(l_1x), str(s_w)
        sma_st = int(sma6_state_df.loc[m_dt, l_1x_str])
        hist_rot_rows.append({
            "月底訊號鎖定日": m_dt.strftime("%Y-%m-%d"),
            "勝出進攻原型 (1x)": l_1x_str,
            "對應實盤槓桿 ETF (3x)": REVERSE_MAPPING[l_1x_str],
            "進攻 4 季動能總分": round(float(total_score_df.loc[m_dt, l_1x_str]), 3),
            "月度 SMA6 狀態": "多頭 (1)" if sma_st == 1 else "跌破均線 (0)",
            "勝出避險資產 (1x)": s_w_str,
            "避險 4 季動能總分": round(float(safe_comp.loc[m_dt, s_w_str]), 3),
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
# 4. 無近似 5 相位 Student-t GARCH(1,1) 快取載入與即時增量補算引擎 (零假近似)
# ==========================================
def fit_exact_garch_range(
    ret_pct: np.ndarray,
    vol_5ph: np.ndarray,
    start_t: int,
    end_t: int,
    progress_cb=None,
) -> np.ndarray:
    """在 [start_t, end_t) 區間內執行真正的 504 日滾動 Student-t GARCH(1,1) MLE (絕不近似)。"""
    last_vol = [np.nan] * PHASE_COUNT
    if start_t > 0:
        for ph in range(PHASE_COUNT):
            v_prev = vol_5ph[ph, start_t - 1]
            if np.isfinite(v_prev):
                last_vol[ph] = float(v_prev)

    total_steps = max(1, end_t - start_t)
    for step_i, t in enumerate(range(start_t, end_t)):
        due_ph = (t - GARCH_WINDOW) % PHASE_COUNT
        train = ret_pct[t - GARCH_WINDOW + 1 : t + 1]
        if len(train) == GARCH_WINDOW and np.all(np.isfinite(train)):
            try:
                am = arch_model(train, vol="Garch", p=1, q=1, dist="t", rescale=False)
                res = am.fit(disp="off", show_warning=False)
                fc = res.forecast(horizon=1, reindex=False)
                v_ann = float(np.sqrt(fc.variance.iloc[-1].values[0]) * np.sqrt(252.0))
                if np.isfinite(v_ann) and v_ann > 0.0:
                    last_vol[due_ph] = v_ann
            except Exception:
                pass
        for ph in range(PHASE_COUNT):
            vol_5ph[ph, t] = last_vol[ph]
        if progress_cb is not None and (step_i % 100 == 0 or step_i == total_steps - 1):
            progress_cb((step_i + 1) / total_steps)
    return vol_5ph


def serialize_exact_garch_cache(dates: pd.DatetimeIndex, vol_5ph: dict, fit_start_pos: int) -> bytes:
    """匯出含完整日期索引之無近似 GARCH 標準 .npz 快取檔，供提交至 GitHub 根目錄實現秒開。"""
    buf = io.BytesIO()
    save_dict = {
        "dates": np.asarray(dates.strftime("%Y-%m-%d"), dtype="U10"),
        "last_dt": dates[-1].strftime("%Y-%m-%d"),
        "n_total": len(dates),
        "fit_start_pos": fit_start_pos,
    }
    for t3x in TRADE_ETFS_3X:
        save_dict[f"vol_{t3x}"] = vol_5ph[t3x]
    np.savez_compressed(buf, **save_dict)
    return buf.getvalue()


def try_load_exact_garch_from_npz(closes: pd.DataFrame, npz_source) -> tuple[dict | None, int, str]:
    """支援新版(含 dates 索引)與舊版 Colab(last_dt/n_total)兩種 .npz 無近似快取格式，並自動補算最新交易日。"""
    dates = closes.index
    n_total = len(dates)
    fit_start_pos = max(GARCH_WINDOW, int(dates.searchsorted(pd.Timestamp(GARCH_FIT_START_DATE))))

    try:
        npz = np.load(npz_source, allow_pickle=True)
    except Exception as e:
        return None, fit_start_pos, f"快取讀取失敗: {e}"

    if not all(f"vol_{t3x}" in npz.files for t3x in TRADE_ETFS_3X):
        return None, fit_start_pos, "快取檔缺少必要欄位"

    vol_5ph = {t3x: np.full((PHASE_COUNT, n_total), np.nan, dtype=float) for t3x in TRADE_ETFS_3X}
    last_aligned_pos = -1

    if "dates" in npz.files:
        cached_dts = pd.to_datetime(npz["dates"])
        date_to_pos = {d: i for i, d in enumerate(dates)}
        matched_pairs = [(c_i, date_to_pos[d]) for c_i, d in enumerate(cached_dts) if d in date_to_pos]
        if len(matched_pairs) < 500:
            return None, fit_start_pos, "快取日期與當前市場主軸重疊不足"
        c_idx = np.array([p[0] for p in matched_pairs], dtype=int)
        t_idx = np.array([p[1] for p in matched_pairs], dtype=int)
        for t3x in TRADE_ETFS_3X:
            vol_5ph[t3x][:, t_idx] = npz[f"vol_{t3x}"][:, c_idx]
        last_aligned_pos = int(np.max(t_idx))
        cache_mode_desc = f"標準日期索引快取 (對齊 {len(matched_pairs):,} 日)"
    else:
        # 相容舊版 Colab 快取格式 (last_dt, n_total, fit_start_pos)
        c_last_dt = pd.Timestamp(str(npz["last_dt"]))
        c_n = int(npz["n_total"])
        if c_last_dt in dates:
            pos_last = int(dates.get_loc(c_last_dt))
            if pos_last + 1 == c_n:
                for t3x in TRADE_ETFS_3X:
                    vol_5ph[t3x][:, :c_n] = npz[f"vol_{t3x}"][:, :c_n]
                last_aligned_pos = pos_last
                cache_mode_desc = f"Colab 原生快取 (對齊至 {c_last_dt.strftime('%Y-%m-%d')})"
            else:
                return None, fit_start_pos, "舊版快取長度與 SPY 主軸不一致，請使用帶日期索引的新快取"
        else:
            return None, fit_start_pos, "舊版快取截止日不在當前市場索引中"

    # 增量補算：永遠以真實 Student-t GARCH(1,1) 重算最後 5 日及所有新增交易日（確保尾端 100% 即時新鮮）
    incr_start = max(fit_start_pos, last_aligned_pos - 4)
    new_days_cnt = max(0, n_total - 1 - last_aligned_pos)
    for t3x, proto_1x in MAPPING.items():
        ret_pct = (((closes[proto_1x] / closes[proto_1x].shift(1)) - 1.0) * 100.0).to_numpy(dtype=float)
        vol_5ph[t3x] = fit_exact_garch_range(ret_pct, vol_5ph[t3x], incr_start, n_total)

    status_msg = f"100% 無近似 Student-t GARCH(1,1) ｜ {cache_mode_desc} ＋ 即時精確增量補算 {new_days_cnt} 個新交易日"
    return vol_5ph, fit_start_pos, status_msg


def get_exact_garch_5ph_for_session(closes: pd.DataFrame, build_full_now: bool = False) -> tuple[dict, bool, int, str]:
    """
    取得無近似 5 相位 Student-t GARCH(1,1) 序列：
    - 優先載入 GitHub 根目錄或使用者上傳的 .npz 快取並做增量補算（全歷史 25 年完整就緒）。
    - 若無快取且未要求全量重算，僅以真實 arch_model 精確計算近 1,136 天 (含 504 訓練 + 252 中位數 + 380 乾淨回放)，供分頁 1 實盤零偷工減料使用。
    """
    dates = closes.index
    n_total = len(dates)
    last_dt_str = dates[-1].strftime("%Y-%m-%d")
    fit_start_pos = max(GARCH_WINDOW, int(dates.searchsorted(pd.Timestamp(GARCH_FIT_START_DATE))))

    # 1. 檢查同 session 已計算完成之記憶體狀態（避免切換介面元件重複擬合）
    mem = st.session_state.get("exact_garch_runtime_mem")
    if (
        mem is not None
        and mem["last_dt"] == last_dt_str
        and mem["n_total"] == n_total
        and (mem["is_full"] or (not build_full_now))
    ):
        return mem["vol_5ph"], mem["is_full"], mem["fit_start_pos"], mem["status_msg"]

    # 2. 優先嘗試載入使用者上傳或 GitHub 根目錄的無近似 .npz 快取檔
    if not build_full_now:
        npz_src = None
        if "uploaded_garch_npz_bytes" in st.session_state:
            npz_src = io.BytesIO(st.session_state["uploaded_garch_npz_bytes"])
        elif os.path.exists(DEFAULT_GARCH_CACHE_FILE):
            npz_src = DEFAULT_GARCH_CACHE_FILE

        if npz_src is not None:
            vol_5ph, f_pos, msg = try_load_exact_garch_from_npz(closes, npz_src)
            if vol_5ph is not None:
                st.session_state["exact_garch_runtime_mem"] = {
                    "last_dt": last_dt_str,
                    "n_total": n_total,
                    "is_full": True,
                    "fit_start_pos": f_pos,
                    "vol_5ph": vol_5ph,
                    "status_msg": msg,
                }
                return vol_5ph, True, f_pos, msg

    # 3. 若使用者觸發全量重算（產出 2000–至今完整無近似序列）
    if build_full_now:
        vol_5ph = {t3x: np.full((PHASE_COUNT, n_total), np.nan, dtype=float) for t3x in TRADE_ETFS_3X}
        prog = st.progress(0.0, text="正在執行全歷史 2000–至今無近似 5 相位 Student-t GARCH(1,1) MLE 擬合...")
        for idx_a, (t3x, proto_1x) in enumerate(MAPPING.items()):
            ret_pct = (((closes[proto_1x] / closes[proto_1x].shift(1)) - 1.0) * 100.0).to_numpy(dtype=float)

            def _cb(frac, i_a=idx_a, name_a=t3x):
                prog.progress((i_a + frac) / 3.0, text=f"無近似 GARCH 擬合中：{name_a} ({MAPPING[name_a]}) — {(i_a + frac)/3.0*100:.1f}%")

            vol_5ph[t3x] = fit_exact_garch_range(ret_pct, vol_5ph[t3x], fit_start_pos, n_total, progress_cb=_cb)
        prog.empty()
        msg = f"100% 無近似 Student-t GARCH(1,1) ｜ 全歷史即時現算完成 ({dates[fit_start_pos].strftime('%Y-%m-%d')} ～ {last_dt_str})"
        st.session_state["exact_garch_runtime_mem"] = {
            "last_dt": last_dt_str,
            "n_total": n_total,
            "is_full": True,
            "fit_start_pos": fit_start_pos,
            "vol_5ph": vol_5ph,
            "status_msg": msg,
        }
        return vol_5ph, True, fit_start_pos, msg

    # 4. 無快取檔之預設實盤模式：嚴格無 .bfill() 精確計算近 1,136 天 (380 天乾淨狀態機回放)
    live_fit_start = max(GARCH_WINDOW, n_total - (QUANTILE_WINDOW + 380))
    vol_5ph = {t3x: np.full((PHASE_COUNT, n_total), np.nan, dtype=float) for t3x in TRADE_ETFS_3X}
    with st.spinner("未偵測到全歷史 .npz 快取，正在為實盤分頁執行近 1.5 年無近似 Student-t GARCH(1,1) 精確擬合..."):
        for t3x, proto_1x in MAPPING.items():
            ret_pct = (((closes[proto_1x] / closes[proto_1x].shift(1)) - 1.0) * 100.0).to_numpy(dtype=float)
            vol_5ph[t3x] = fit_exact_garch_range(ret_pct, vol_5ph[t3x], live_fit_start, n_total)
    msg = f"實盤精確模式 (近 1.5 年無近似 Student-t GARCH，無 .bfill()) ｜ 尚未載入 25 年全歷史 .npz 快取"
    st.session_state["exact_garch_runtime_mem"] = {
        "last_dt": last_dt_str,
        "n_total": n_total,
        "is_full": False,
        "fit_start_pos": live_fit_start,
        "vol_5ph": vol_5ph,
        "status_msg": msg,
    }
    return vol_5ph, False, live_fit_start, msg


# ==========================================
# 5. 統一單一真相引擎：Q 斷路器狀態機 ＋ 實盤狀態回放 ＋ 獨立子帳戶無近似回測
# ==========================================
def run_q_state_machine_strict(vol_arr: np.ndarray, med_arr: np.ndarray) -> np.ndarray:
    """外層危機峰值回落斷路器 Q (嚴格無 .bfill()，資料未達 252 日暖機維持 NaN)。"""
    n = len(vol_arr)
    q_out = np.full(n, np.nan, dtype=float)
    state = 1
    peak = 0.0
    trough = -1.0
    for t in range(n):
        vt, mt = vol_arr[t], med_arr[t]
        if not (np.isfinite(vt) and np.isfinite(mt) and mt > 0.0):
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
    return q_out


def build_strict_gate_matrices(dates: pd.DatetimeIndex, vol_5ph: dict):
    """嚴格以 min_periods=252 與 shift(1) 建構 Q 斷路器、252 日中位數與舊版 O 分位數閘門。"""
    n_total = len(dates)
    q_5ph = {t3x: np.full((PHASE_COUNT, n_total), np.nan, dtype=float) for t3x in TRADE_ETFS_3X}
    med_5ph = {t3x: np.full((PHASE_COUNT, n_total), np.nan, dtype=float) for t3x in TRADE_ETFS_3X}
    g_o_5ph = {t3x: np.full((PHASE_COUNT, n_total), np.nan, dtype=float) for t3x in TRADE_ETFS_3X}

    for t3x in TRADE_ETFS_3X:
        for ph in range(PHASE_COUNT):
            s_v = pd.Series(vol_5ph[t3x][ph], index=dates)  # 嚴格禁止 .bfill()
            s_med = s_v.rolling(QUANTILE_WINDOW, min_periods=QUANTILE_WINDOW).median().shift(1)
            s_q975 = s_v.rolling(QUANTILE_WINDOW, min_periods=QUANTILE_WINDOW).quantile(O_EXIT_Q).shift(1)
            s_q900 = s_v.rolling(QUANTILE_WINDOW, min_periods=QUANTILE_WINDOW).quantile(O_ENTRY_Q).shift(1)

            v_arr = s_v.to_numpy(dtype=float)
            m_arr = s_med.to_numpy(dtype=float)
            med_5ph[t3x][ph] = m_arr
            q_5ph[t3x][ph] = run_q_state_machine_strict(v_arr, m_arr)

            g_state = 1.0
            q975_arr = s_q975.to_numpy(dtype=float)
            q900_arr = s_q900.to_numpy(dtype=float)
            for k in range(n_total):
                if np.isfinite(v_arr[k]) and np.isfinite(q975_arr[k]) and np.isfinite(q900_arr[k]):
                    if v_arr[k] > q975_arr[k]:
                        g_state = 0.0
                    elif v_arr[k] < q900_arr[k]:
                        g_state = 1.0
                    g_o_5ph[t3x][ph, k] = g_state

    return q_5ph, med_5ph, g_o_5ph


def run_unified_reband_qonly_engine(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    rf_ann: pd.Series,
    m_sig: dict,
    vol_5ph: dict,
    q_5ph: dict,
    med_5ph: dict,
    g_o_5ph: dict,
    start_date_str: str,
    fee_bps: int,
    compute_benchmarks: bool = True,
):
    """
    單一真相引擎（實盤分頁 1 與回測分頁 2 100% 共用）：
    - 嚴格 T+1 開盤成交、納入隔夜跳空漂移與雙邊交易成本。
    - 5 相位獨立子帳戶真實淨值加總（無跨相位免費每日再平衡假象）。
    - 同步輸出完整歷史回測序列、避險三指標診斷矩陣與「明日開盤實盤下單指令」。
    """
    dates = closes.index
    r_on_mat = ((opens[ALL_EXEC_ASSETS] / closes[ALL_EXEC_ASSETS].shift(1)) - 1.0).to_numpy(dtype=float)
    r_id_mat = ((closes[ALL_EXEC_ASSETS] / opens[ALL_EXEC_ASSETS]) - 1.0).to_numpy(dtype=float)

    asset_to_col = {k: i for i, k in enumerate(ALL_EXEC_ASSETS)}
    t3x_to_idx = {k: i for i, k in enumerate(TRADE_ETFS_3X)}
    lead_arr = m_sig["daily_leader_3x"].to_numpy()
    safe_arr = m_sig["daily_safe_win"].to_numpy()
    sma_mat = m_sig["daily_sma6_by_3x"][TRADE_ETFS_3X].to_numpy(dtype=float)

    cand_start = int(dates.searchsorted(pd.Timestamp(start_date_str)))
    start_idx = None
    for idx in range(max(1, cand_start), len(dates)):
        sig_idx = idx - 1
        l_s, sf_s = lead_arr[sig_idx], safe_arr[sig_idx]
        if pd.isna(l_s) or pd.isna(sf_s):
            continue
        if not (np.all(np.isfinite(r_on_mat[idx])) and np.all(np.isfinite(r_id_mat[idx]))):
            continue
        if not np.all(np.isfinite(sma_mat[sig_idx])):
            continue
        if all(
            np.isfinite(q_5ph[t3x][ph, sig_idx]) and np.isfinite(g_o_5ph[t3x][ph, sig_idx])
            for t3x in TRADE_ETFS_3X
            for ph in range(PHASE_COUNT)
        ):
            start_idx = idx
            break

    if start_idx is None:
        raise ValueError(f"找不到有效起算日 (>= {start_date_str})，請確認數據與 GARCH 暖機完成。")

    sim_dates = dates[start_idx:]
    n_sim = len(sim_dates)
    fee_rate = float(fee_bps) / 10000.0

    specs = [("REBAND_QONLY_200_0", "QONLY", True)]
    if compute_benchmarks:
        specs.extend([
            ("SCOPE_QONLY_F0", "QONLY", False),
            ("O_Q4_200_0", "OLD_O", True),
            ("NO_VOL_GATE", "NONE", True),
        ])

    out_rets, out_to, out_skip, out_eq_w = {}, {}, {}, {}
    active_gate_by_strat = {}
    phase_latest_records = []

    active_sma = np.zeros(n_sim, dtype=float)
    active_atk_r = np.zeros(n_sim, dtype=float)
    active_safe_r = np.zeros(n_sim, dtype=float)
    for s_i in range(n_sim):
        t_idx = start_idx + s_i
        sig_idx = t_idx - 1
        l_str, sf_str = str(lead_arr[sig_idx]), str(safe_arr[sig_idx])
        l_col, sf_col = asset_to_col[l_str], asset_to_col[sf_str]
        active_sma[s_i] = float(sma_mat[sig_idx, t3x_to_idx[l_str]])
        active_atk_r[s_i] = (1.0 + r_on_mat[t_idx, l_col]) * (1.0 + r_id_mat[t_idx, l_col]) - 1.0
        active_safe_r[s_i] = (1.0 + r_on_mat[t_idx, sf_col]) * (1.0 + r_id_mat[t_idx, sf_col]) - 1.0

    for name, mode, use_band in specs:
        n_ph = 1 if mode == "NONE" else PHASE_COUNT
        phase_navs = np.ones((n_ph, n_sim), dtype=float)
        phase_to = np.zeros((n_ph, n_sim), dtype=float)
        phase_skip = np.zeros((n_ph, n_sim), dtype=float)
        phase_eq_w = np.zeros((n_ph, n_sim), dtype=float)
        phase_gate = np.ones((n_ph, n_sim), dtype=float)

        for ph in range(n_ph):
            h_prev = np.zeros(5, dtype=float)
            nav_curr = 1.0
            prev_lead, prev_s, prev_gate, prev_safe, prev_b = None, None, None, None, None
            prev_base = np.zeros(5, dtype=float)

            for s_i in range(n_sim):
                t_idx = start_idx + s_i
                sig_idx = t_idx - 1
                l_str, sf_str = str(lead_arr[sig_idx]), str(safe_arr[sig_idx])
                l_col, sf_col = asset_to_col[l_str], asset_to_col[sf_str]
                s_val = float(sma_mat[sig_idx, t3x_to_idx[l_str]])

                if mode == "NONE":
                    gate_flag = 1.0
                    b_val = 0.5 * s_val + 0.5
                elif mode == "OLD_O":
                    gate_flag = float(g_o_5ph[l_str][ph, sig_idx])
                    b_val = 0.5 * s_val + 0.5 * gate_flag
                else:  # QONLY: B = Q * (0.5*S + 0.5)
                    gate_flag = float(q_5ph[l_str][ph, sig_idx])
                    b_val = gate_flag * (0.5 * s_val + 0.5)

                phase_gate[ph, s_i] = gate_flag
                base_w = np.zeros(5, dtype=float)
                base_w[l_col] = b_val
                base_w[sf_col] = 1.0 - b_val

                r_on = r_on_mat[t_idx]
                r_id = r_id_mat[t_idx]

                if not use_band:
                    traded = True
                    plan_w = base_w.copy()
                else:
                    if prev_lead is None:
                        mand = True
                    else:
                        supp_chg = bool(np.any((base_w > 1e-9) != (prev_base > 1e-9)))
                        mand = (
                            (l_str != prev_lead)
                            or (s_val != prev_s)
                            or (gate_flag != prev_gate)
                            or (sf_str != prev_safe)
                            or supp_chg
                            or (b_val < prev_b - 1e-12)
                        )
                    d_max = float(np.max(np.abs(h_prev - base_w)))
                    if mand or (d_max >= BAND_TRIG_TH):
                        traded = True
                        plan_w = base_w.copy()
                    else:
                        traded = False
                        plan_w = h_prev.copy()

                if s_i == 0:
                    nav_on = 1.0
                    turn = float(np.sum(np.abs(plan_w)))
                else:
                    nav_on = float(np.sum(h_prev * (1.0 + r_on)))
                    open_w = (h_prev * (1.0 + r_on)) / nav_on if nav_on > 0 else h_prev.copy()
                    if traded:
                        turn = float(np.sum(np.abs(plan_w - open_w)))
                    else:
                        turn = 0.0
                        plan_w = open_w

                nav_id = float(np.sum(plan_w * (1.0 + r_id)))
                r_day = nav_on * max(0.0, 1.0 - fee_rate * turn) * nav_id - 1.0
                nav_curr *= (1.0 + r_day)

                end_v = plan_w * (1.0 + r_id)
                h_prev = end_v / float(np.sum(end_v)) if np.sum(end_v) > 0 else base_w.copy()

                phase_navs[ph, s_i] = nav_curr
                phase_to[ph, s_i] = turn
                phase_skip[ph, s_i] = 0.0 if traded else 1.0
                phase_eq_w[ph, s_i] = float(np.sum(plan_w[:3]))

                prev_lead, prev_s, prev_gate, prev_safe, prev_base, prev_b = (
                    l_str, s_val, gate_flag, sf_str, base_w.copy(), b_val
                )

            # 針對主系統 REBAND_QONLY_200_0 產出最新收盤後之「下一交易日開盤執行計畫」
            if name == "REBAND_QONLY_200_0":
                idx_now = len(dates) - 1
                winner_now = str(m_sig["sel_df"].index[0])
                safe_now = str(m_sig["safe_win"])
                s_now = float(m_sig["sel_df"].loc[winner_now, "SMA6_State"])
                q_now = float(q_5ph[winner_now][ph, idx_now])
                sig_now = float(vol_5ph[winner_now][ph, idx_now])
                med_now = float(med_5ph[winner_now][ph, idx_now])
                rho_now = sig_now / med_now if (np.isfinite(med_now) and med_now > 0) else np.nan
                g_o_now = float(g_o_5ph[winner_now][ph, idx_now])

                b_now = q_now * (0.5 * s_now + 0.5)
                b_o_now = 0.5 * s_now + 0.5 * g_o_now
                base_w_now = {k: 0.0 for k in ALL_EXEC_ASSETS}
                base_w_now[winner_now] = b_now
                base_w_now[safe_now] = 1.0 - b_now
                holdings_dict = {ALL_EXEC_ASSETS[i]: float(h_prev[i]) for i in range(5)}

                reasons = []
                if winner_now != prev_lead:
                    reasons.append("RANK換股")
                if s_now != prev_s:
                    reasons.append("SMA6翻轉")
                if q_now != prev_gate:
                    reasons.append("Q危機閘門翻轉")
                if safe_now != prev_safe:
                    reasons.append("避險標的換倉")
                if any((base_w_now[ALL_EXEC_ASSETS[i]] > 1e-9) != (prev_base[i] > 1e-9) for i in range(5)):
                    reasons.append("持倉集合改變")
                if b_now < prev_b - 1e-12:
                    reasons.append("股票預算調降(減碼)")

                mand_now = len(reasons) > 0
                d_max_now = max(abs(holdings_dict[k] - base_w_now[k]) for k in ALL_EXEC_ASSETS)
                if mand_now:
                    act_now = 2
                    plan_w_now = dict(base_w_now)
                elif d_max_now >= BAND_TRIG_TH:
                    act_now = 1
                    plan_w_now = dict(base_w_now)
                else:
                    act_now = 0
                    plan_w_now = dict(holdings_dict)

                phase_latest_records.append({
                    "Phase": ph,
                    "Trade_ETF_3x": winner_now,
                    "Safe_ETF_1x": safe_now,
                    "Sigma_Ann_Pct": round(sig_now, 2) if np.isfinite(sig_now) else np.nan,
                    "Med252_Pct": round(med_now, 2) if np.isfinite(med_now) else np.nan,
                    "Rho_Ratio": round(rho_now, 3) if np.isfinite(rho_now) else np.nan,
                    "Q_Gate": int(q_now),
                    "SMA6": int(s_now),
                    "Base_Stock_Weight": round(b_now, 4),
                    "Base_Safe_Weight": round(1.0 - b_now, 4),
                    "O_Stock_Weight": round(b_o_now, 4),
                    "Prior_Close_Weights": holdings_dict,
                    "Base_Target_Weights": base_w_now,
                    "Planned_Open_Weights": plan_w_now,
                    "Decision_Distance_bps": round(d_max_now * 10000.0, 1),
                    "Action_Code": act_now,
                    "Mandatory_Reasons": ", ".join(reasons) if reasons else "無",
                    "Phase_NAV": nav_curr,
                })

        agg_nav = np.mean(phase_navs, axis=0)
        agg_r = np.empty(n_sim, dtype=float)
        agg_r[0] = agg_nav[0] - 1.0
        agg_r[1:] = (agg_nav[1:] / agg_nav[:-1]) - 1.0

        prev_caps = np.ones((n_ph, n_sim), dtype=float)
        prev_caps[:, 1:] = phase_navs[:, :-1]
        cap_w = prev_caps / np.sum(prev_caps, axis=0, keepdims=True)

        out_rets[name] = agg_r
        out_to[name] = np.sum(cap_w * phase_to, axis=0)
        out_skip[name] = np.mean(phase_skip, axis=0)
        out_eq_w[name] = np.sum(cap_w * phase_eq_w, axis=0)
        active_gate_by_strat[name] = phase_gate

    ret_spy = ((closes["SPY"].reindex(sim_dates) / closes["SPY"].reindex(sim_dates).shift(1)) - 1.0).fillna(0.0).to_numpy(dtype=float)
    rf_d = (rf_ann.reindex(sim_dates).ffill().fillna(0.025).to_numpy(dtype=float)) / 252.0

    bt_df = pd.DataFrame(out_rets, index=sim_dates)
    bt_df["SPY_1x"] = ret_spy
    bt_df["RF_Daily"] = rf_d
    for k in out_to:
        bt_df[f"TO_{k}"] = out_to[k]
        bt_df[f"SKIP_{k}"] = out_skip[k]
        bt_df[f"EQW_{k}"] = out_eq_w[k]

    diag = {
        "active_sma": pd.Series(active_sma, index=sim_dates),
        "active_atk_r": pd.Series(active_atk_r, index=sim_dates),
        "active_safe_r": pd.Series(active_safe_r, index=sim_dates),
        "active_gate": active_gate_by_strat,
    }
    return bt_df, diag, phase_latest_records


# ==========================================
# 6. 機構級量化指標與避險三強制指標計算器
# ==========================================
def compute_quant_metrics_table(bt_sub: pd.DataFrame, fee_bps: int) -> pd.DataFrame:
    strat_meta = [
        ("REBAND_QONLY_200_0", "🛡️ REBAND_QONLY_200_0 (本系統：純Q＋200/0護欄)"),
        ("SCOPE_QONLY_F0", "⚡ SCOPE_QONLY_F0 (同訊號無護欄每日調倉)"),
        ("O_Q4_200_0", "📊 公平舊版 O_Q4_200_0 (分位數＋200/0護欄)"),
        ("NO_VOL_GATE", "🧭 NO_VOL_GATE (無波動開關：僅4季動能＋SMA6)"),
        ("SPY_1x", "🇺🇸 SPY (標普 500 買入持有)"),
    ]
    rows = []
    rf_d = bt_sub["RF_Daily"].to_numpy(dtype=float)
    years = max(len(bt_sub) / 252.0, 1e-6)

    for col, label in strat_meta:
        if col not in bt_sub.columns:
            continue
        r = bt_sub[col].to_numpy(dtype=float)
        nav = np.cumprod(1.0 + r)
        tot_mult = float(nav[-1])
        cagr = (tot_mult ** (1.0 / years)) - 1.0 if tot_mult > 0 else -1.0
        ann_vol = float(np.std(r, ddof=1) * np.sqrt(252.0))

        ex_r = r - rf_d
        sharpe = float(np.mean(ex_r) / (np.std(ex_r, ddof=1) + 1e-12) * np.sqrt(252.0))
        down_dev = float(np.sqrt(np.mean(np.minimum(0.0, ex_r) ** 2)) * np.sqrt(252.0))
        sortino = float((np.mean(ex_r) * 252.0) / (down_dev + 1e-12))

        peak = np.maximum.accumulate(nav)
        dd = (nav / peak) - 1.0
        max_dd = float(np.min(dd))
        calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else np.nan
        ulcer_idx = float(np.sqrt(np.mean((dd * 100.0) ** 2)))

        max_uw_days, curr_uw = 0, 0
        for u_flag in (dd < -1e-6):
            curr_uw = curr_uw + 1 if u_flag else 0
            max_uw_days = max(max_uw_days, curr_uw)

        s_nav = pd.Series(nav, index=bt_sub.index)
        r5 = ((s_nav / s_nav.shift(252 * 5)) ** (1.0 / 5.0) - 1.0).dropna() * 100.0
        r10 = ((s_nav / s_nav.shift(252 * 10)) ** (1.0 / 10.0) - 1.0).dropna() * 100.0

        to_col = f"TO_{col}"
        skip_col = f"SKIP_{col}"
        ann_to = float(bt_sub[to_col].mean() * 252.0) if to_col in bt_sub.columns else 0.0
        skip_pct = float(bt_sub[skip_col].mean() * 100.0) if skip_col in bt_sub.columns else 0.0
        fee_drag_bps = ann_to * float(fee_bps)

        rows.append({
            "策略名稱": label,
            "年化報酬 (CAGR)": f"{cagr*100:.2f}%",
            "累積淨值倍數": f"{tot_mult:,.2f}x",
            "年化波動率 (Vol)": f"{ann_vol*100:.2f}%",
            "夏普比率 (Sharpe)": f"{sharpe:.3f}",
            "索提諾比率 (Sortino)": f"{sortino:.3f}",
            "最大回撤 (MaxDD)": f"{max_dd*100:.2f}%",
            "卡瑪比率 (Calmar)": f"{calmar:.2f}",
            "潰瘍指數 (Ulcer)": f"{ulcer_idx:.2f}",
            "最長水下天數": f"{max_uw_days} 天",
            "5Y 最差 CAGR": f"{r5.min():.2f}%" if len(r5) > 0 else "N/A",
            "10Y 最差 CAGR": f"{r10.min():.2f}%" if len(r10) > 0 else "N/A",
            "年化雙邊周轉": f"{ann_to:.2f}x",
            "免交易日比例": f"{skip_pct:.2f}%",
            "年化成本拖累": f"-{fee_drag_bps:.1f} bps",
        })
    return pd.DataFrame(rows).set_index("策略名稱")


def compute_institutional_hedge_reports(bt_sub: pd.DataFrame, diag_full: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """輸出機構級避險模塊三大強制報表：Crisis Alpha、False Alarm Drag、Whipsaw Count。"""
    idx = bt_sub.index
    # 1. Crisis Alpha
    ca_rows = []
    for label, s_dt, e_dt in CRISIS_WINDOWS:
        sub = bt_sub.loc[(idx >= pd.Timestamp(s_dt)) & (idx <= pd.Timestamp(e_dt))]
        if len(sub) < 10:
            continue
        def _rm(col):
            nav = np.cumprod(1.0 + sub[col].to_numpy(dtype=float))
            return (nav[-1] - 1.0) * 100.0, np.min(nav / np.maximum.accumulate(nav) - 1.0) * 100.0

        nv_r, _ = _rm("NO_VOL_GATE")
        q_r, q_dd = _rm("REBAND_QONLY_200_0")
        o_r, o_dd = _rm("O_Q4_200_0")
        ca_rows.append({
            "危機視窗": f"{label} ({sub.index[0].strftime('%Y-%m-%d')}~{sub.index[-1].strftime('%Y-%m-%d')})",
            "日數": len(sub),
            "無波動開關(%)": round(nv_r, 2),
            "REBAND_QONLY(%)": round(q_r, 2),
            "O_Q4_200_0(%)": round(o_r, 2),
            "REBAND CrisisAlpha(pp)": round(q_r - nv_r, 2),
            "O CrisisAlpha(pp)": round(o_r - nv_r, 2),
            "REBAND 區間MDD(%)": round(q_dd, 2),
            "O 區間MDD(%)": round(o_dd, 2),
        })
    ca_df = pd.DataFrame(ca_rows).set_index("危機視窗") if ca_rows else pd.DataFrame()

    # 2. False Alarm Drag (非危機且 SMA6=1 多頭日)
    crisis_mask = np.zeros(len(idx), dtype=bool)
    for label, s_dt, e_dt in CRISIS_WINDOWS:
        if "完整段" in label:
            continue
        crisis_mask |= (idx >= pd.Timestamp(s_dt)) & (idx <= pd.Timestamp(e_dt))
    sma_sub = diag_full["active_sma"].reindex(idx).to_numpy(dtype=float)
    bull_mask = (~crisis_mask) & (sma_sub == 1.0)
    fa_rows = []
    if bull_mask.sum() >= 20:
        nv_log = np.log1p(bt_sub["NO_VOL_GATE"].to_numpy(dtype=float)[bull_mask])
        for s_col, s_lab in [
            ("REBAND_QONLY_200_0", "REBAND_QONLY_200_0 (本系統)"),
            ("O_Q4_200_0", "公平舊版 O_Q4_200_0"),
        ]:
            st_log = np.log1p(bt_sub[s_col].to_numpy(dtype=float)[bull_mask])
            ann_drag = float(np.mean(st_log - nv_log) * 252.0 * 100.0)
            cum_drag = float((np.exp(np.sum(st_log - nv_log)) - 1.0) * 100.0)
            avg_eq = float(np.mean(bt_sub[f"EQW_{s_col}"].to_numpy(dtype=float)[bull_mask]) * 100.0)
            fa_rows.append({
                "策略版本 (非危機且 SMA6=1 多頭日)": s_lab,
                "樣本日數": int(bull_mask.sum()),
                "年化多頭出血 Drag (pp/年)": round(ann_drag, 2),
                "累積多頭相對拖累 (%)": round(cum_drag, 2),
                "多頭區間平均曝險 (%)": round(avg_eq, 1),
            })
    fa_df = pd.DataFrame(fa_rows).set_index("策略版本 (非危機且 SMA6=1 多頭日)") if fa_rows else pd.DataFrame()

    # 3. Whipsaw Count
    full_idx = diag_full["active_sma"].index
    pos_sub = full_idx.get_indexer(idx)
    atk_r = diag_full["active_atk_r"].to_numpy(dtype=float)[pos_sub]
    safe_r = diag_full["active_safe_r"].to_numpy(dtype=float)[pos_sub]
    years = max(len(idx) / 252.0, 1e-6)
    wh_rows = []
    for s_col, s_lab in [
        ("REBAND_QONLY_200_0", "本系統外層斷路器 Q=0 (全清倉)"),
        ("O_Q4_200_0", "舊版 252 日分位數二元開關 G=0"),
    ]:
        g_mat = diag_full["active_gate"][s_col][:, pos_sub]
        tot_ev, w5, w10, w20, fa_cnt = 0, 0, 0, 0, 0
        durs = []
        for ph in range(g_mat.shape[0]):
            off = g_mat[ph] < 0.5
            in_off, s_i = False, 0
            for i in range(len(off)):
                if (not in_off) and off[i]:
                    in_off, s_i = True, i
                elif in_off and (not off[i]):
                    dur = i - s_i
                    tot_ev += 1
                    durs.append(dur)
                    if dur <= 5: w5 += 1
                    if dur <= 10: w10 += 1
                    if dur <= 20: w20 += 1
                    if (np.prod(1.0 + atk_r[s_i:i]) - 1.0) > (np.prod(1.0 + safe_r[s_i:i]) - 1.0):
                        fa_cnt += 1
                    in_off = False
            if in_off:
                dur = len(off) - s_i
                tot_ev += 1
                durs.append(dur)
                if (np.prod(1.0 + atk_r[s_i:]) - 1.0) > (np.prod(1.0 + safe_r[s_i:]) - 1.0):
                    fa_cnt += 1
        avg_ev = tot_ev / g_mat.shape[0]
        wh_rows.append({
            "開關機制": s_lab,
            "單相位平均觸發次數": round(avg_ev, 2),
            "年均頻率(次/年)": round(avg_ev / years, 2),
            "10日內洗盤": round(w10 / g_mat.shape[0], 2),
            "20日內洗盤": round(w20 / g_mat.shape[0], 2),
            "事後誤殺率(%)": round((fa_cnt / max(1, tot_ev)) * 100.0, 1),
            "平均防禦天數": round(float(np.mean(durs)), 1) if durs else 0.0,
        })
    wh_df = pd.DataFrame(wh_rows).set_index("開關機制")
    return ca_df, fa_df, wh_df


# ==========================================
# 7. 主畫面渲染：五大實用分頁
# ==========================================
st.title("🛡️ REBAND_QONLY_200_0 機構實盤部署與無近似量化回測系統")
st.caption(
    "整合：已拔除冗餘內層 a (B = Q × (0.5·SMA6 + 0.5)) ｜ 四等分標準日曆季度 (0,3,6,9,12)/Vol252 ｜ "
    "100% 無近似 5 相位 Student-t GARCH(1,1) ＋ 200/0 bps 護欄 ｜ 黃金代理修正為 GC=F ｜ 美東收盤防呆"
)

closes_df, opens_df, rf_series, patch_source_note, dropped_unclosed_flag = build_clean_live_market_bundle(NOW_NY)

if not closes_df.empty:
    if dropped_unclosed_flag:
        st.warning(
            f"⏳ **美股盤中防呆啟動**：目前為美東交易時段 ({NOW_NY.strftime('%Y-%m-%d %H:%M %Z')})，"
            f"系統已自動排除尚未收盤之當日盤中報價，所有下單與風控訊號嚴格鎖定於上一已收盤交易日 `{closes_df.index[-1].strftime('%Y-%m-%d')}`。"
        )

    last_dt_str = closes_df.index[-1].strftime("%Y-%m-%d")
    m_sig = calculate_monthly_signals_1x(closes_df, NOW_NY, FORCE_EOM)

    # 載入或計算無近似 5 相位 Student-t GARCH(1,1)
    trigger_full_compute = st.session_state.pop("request_full_garch_compute", False)
    vol_5ph, has_full_exact_garch, garch_fit_pos, garch_status_msg = get_exact_garch_5ph_for_session(
        closes_df, build_full_now=trigger_full_compute
    )
    q_5ph, med_5ph, g_o_5ph = build_strict_gate_matrices(closes_df.index, vol_5ph)

    # 執行單一真相引擎（有完整 25 年快取時由 2001-09-04 連續回放；無快取時由近 1.5 年無截斷暖機後連續回放）
    replay_start_str = "2001-09-04" if has_full_exact_garch else closes_df.index[max(0, len(closes_df) - 360)].strftime("%Y-%m-%d")
    bt_full_df, diag_full, phase_records = run_unified_reband_qonly_engine(
        closes_df,
        opens_df,
        rf_series,
        m_sig,
        vol_5ph,
        q_5ph,
        med_5ph,
        g_o_5ph,
        start_date_str=replay_start_str,
        fee_bps=FEE_BPS,
        compute_benchmarks=has_full_exact_garch,
    )

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
        f"📅 **月度訊號鎖定日 (美東完結基準)**: `{ref_date.strftime('%Y-%m-%d')}` ｜ "
        f"📈 **最新已收盤數據**: `{last_dt_str}` ({patch_source_note}) ｜ "
        f"🎯 **下一交易日開盤執行標的**: 進攻 **`{winner_3x}`** (`{MAPPING[winner_3x]}`) ＋ 防禦 **`{safe_win}`**\n\n"
        f"⚙️ **GARCH 引擎狀態**: `{garch_status_msg}`"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏆 本月進攻贏家 (3x)", f"{winner_3x} ({MAPPING[winner_3x]})", f"4季動能總分: {sel_df.loc[winner_3x, 'Total_Score']:.3f}")
    with c2:
        st.metric("🎯 五相位平均基礎股票預算", f"{avg_base_stock*100:.1f}%", f"對照舊版 O: {avg_o_stock*100:.1f}%")
    with c3:
        st.metric("⚡ Q 危機閘門安全相位數", f"{active_q} / 5 相位", f"SMA6: {'多頭 (1)' if s_now_int==1 else '空頭 (0)'}")
    with c4:
        st.metric("🛡️ 本月勝出避險資產", safe_win, f"4季動能總分: {safe_df.loc[safe_win, 'Total_Score']:.3f}")

    st.divider()

    tab_live, tab_bt, tab_doc, tab_rot, tab_raw = st.tabs([
        "💳 1. 單一帳戶實盤下單與五相位監控",
        "📈 2. 無近似超長歷史回測與機構避險報表 (2001–至今)",
        "📖 3. 策略架構與數學規格說明書",
        "📊 4. 當月 1x 四季動能排名與歷史輪動紀錄",
        "🗄️ 5. 原始報價、無近似快取管理與健康檢查",
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

        st.subheader("🔬 五相位 (Phase 0..4) 無近似 Student-t GARCH 與 200/0 bps 護欄明細")
        disp_rows = []
        for r in phase_records:
            act_map = {
                0: "0: 不交易 (偏離 < 200bps，保留原股數)",
                1: "1: 護欄調倉 (偏離 >= 200bps，直接拉回 0bps 基礎目標)",
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
                f"基礎目標 {winner_3x} (%)": round(r["Base_Stock_Weight"] * 100.0, 2),
                f"昨收實際 {winner_3x} (%)": round(r["Prior_Close_Weights"][winner_3x] * 100.0, 2),
                "最大偏離 d_max (bps)": r["Decision_Distance_bps"],
                "今日開盤動作 (Action)": act_map[r["Action_Code"]],
                f"今日計畫 {winner_3x} (%)": round(r["Planned_Open_Weights"][winner_3x] * 100.0, 2),
                f"今日計畫 {safe_win} (%)": round(r["Planned_Open_Weights"][safe_win] * 100.0, 2),
            })
        st.dataframe(pd.DataFrame(disp_rows).set_index("相位 (Phase)"), use_container_width=True)

        with st.expander("🧪 壓力測試模擬器：如果今晚輪值相位觸發危機斷路器 Q=0，我的單一帳戶明天會怎麼動？"):
            sim_rho = st.slider("假設今日輪值相位的波動率警報比 ρ = σ / Med252 (門檻 2.50)", 0.8, 3.5, 2.6, 0.1)
            sim_q = 0.0 if sim_rho >= RHO_EXIT else 1.0
            sim_b = sim_q * (0.5 * float(s_now_int) + 0.5)
            new_acct_stock_w = 0.80 * avg_base_stock + 0.20 * sim_b
            delta_stock_w = new_acct_stock_w - avg_base_stock
            delta_usd = delta_stock_w * ACCOUNT_NAV
            st.markdown(
                f"- **該輪值相位狀態**：`Q = {int(sim_q)}` ({'觸發危機全退！' if sim_q==0 else '安全維持 1'}) ｜ "
                f"該相位股票目標變為 **`{sim_b*100:.1f}%`** (`REBAND_QONLY_200_0` 離散三檔：100% / 50% / 0%)\n"
                f"- **對單一帳戶 (總淨值 `${ACCOUNT_NAV:,.0f}`) 的匯總影響**：帳戶總 `{winner_3x}` 目標由 `{avg_base_stock*100:.2f}%` 變為 **`{new_acct_stock_w*100:.2f}%`** "
                f"👉 明日開盤需調整 `{winner_3x}` `${delta_usd:+,.0f} USD`（約 `{abs(delta_usd)/latest_prices[winner_3x]:.1f}` 股）並反向等額調整 `{safe_win}`。"
            )

    # ==========================================
    # 分頁 2：無近似超長歷史回測與機構避險三指標 (2001–至今)
    # ==========================================
    with tab_bt:
        if not has_full_exact_garch:
            st.warning(
                "🚫 **為杜絕假近似回測誤導，本系統已永久移除原版 MAD 固定係數近似引擎！**\n\n"
                f"目前專案根目錄尚未偵測到無近似 25 年快取檔 `{DEFAULT_GARCH_CACHE_FILE}`。請擇一啟用 100% 無近似歷史回測：\n"
                "1. **推薦（秒開）**：將您在 Colab 跑好的 `reband_exact_garch_5phase_cache.npz` 提交至 GitHub 根目錄，或在下方直接暫時上傳檢視。\n"
                "2. **一次性現算**：點擊下方按鈕執行完整 2000–至今真實 `arch_model` 擬合（約 2～4 分鐘），完成後到分頁 5 下載 `.npz` 放入 GitHub 即可永久秒開。"
            )
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                up_file = st.file_uploader("📤 暫時上傳無近似快取檔 (.npz)", type=["npz"])
                if up_file is not None:
                    st.session_state["uploaded_garch_npz_bytes"] = up_file.getvalue()
                    st.session_state.pop("exact_garch_runtime_mem", None)
                    st.rerun()
            with col_u2:
                st.write("")
                st.write("")
                if st.button("⚙️ 立即執行 2000–至今 100% 無近似 GARCH 全量擬合 (約 2~4 分鐘)"):
                    st.session_state["request_full_garch_compute"] = True
                    st.rerun()
        else:
            st.subheader("⏱️ 選擇歷史回測檢視區間與座標模式 (100% 無近似 Student-t GARCH)")
            ctrl1, ctrl2 = st.columns([2, 1])
            with ctrl1:
                period_choice = st.radio(
                    "回測時間跨度",
                    options=[
                        "🏛️ 純實盤 3x ETF 完整期 (2014-01-23–至今：100% 真實 ETF Open/Close，零代理誤差)",
                        "🌐 無金礦股長週期 (2001-09-04–至今，約 25 年：GLD 以 GC=F 銜接；2014 前 3x 因果合成)",
                        "🔟 近 10 年完整週期 (Last 10 Years)",
                        "5️⃣ 近 5 年完整週期 (Last 5 Years)",
                    ],
                    index=0,
                    horizontal=False,
                )
            with ctrl2:
                scale_type = st.radio(
                    "淨值曲線縱軸座標",
                    options=["log (對數座標：檢視長期複利斜率)", "linear (線性座標：檢視絕對資產倍數)"],
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
                f"（共 `{len(bt_sub):,}` 個交易日，約 `{len(bt_sub)/252.0:.2f}` 年 ｜ 單邊交易成本：`{FEE_BPS} bps` ｜ 5 相位獨立子帳戶核算）"
            )

            # 1. 機構級績效總表
            st.markdown("#### 📊 1. 機構級量化績效指標總表 (無近似 Student-t GARCH)")
            metrics_tbl = compute_quant_metrics_table(bt_sub, FEE_BPS)
            st.dataframe(metrics_tbl, use_container_width=True)

            # 2. 避險模塊三大強制機構報表
            st.markdown("#### 🛡️ 2. 避險模塊三大強制風控報表 (Crisis Alpha / False Alarm Drag / Whipsaw Count)")
            ca_df, fa_df, wh_df = compute_institutional_hedge_reports(bt_sub, diag_full)
            if not ca_df.empty:
                st.caption("A. Crisis Alpha（特定股災防禦力，基準 = 無波動開關 NO_VOL_GATE）")
                st.dataframe(ca_df, use_container_width=True)
            hc1, hc2 = st.columns(2)
            with hc1:
                if not fa_df.empty:
                    st.caption("B. False Alarm Drag（多頭出血率）")
                    st.dataframe(fa_df, use_container_width=True)
            with hc2:
                st.caption("C. Whipsaw Count（假訊號洗盤與事後誤殺率）")
                st.dataframe(wh_df, use_container_width=True)

            # 3. 累積淨值與水下回撤圖
            label_map = {
                "REBAND_QONLY_200_0": "1. REBAND_QONLY_200_0 (本系統)",
                "SCOPE_QONLY_F0": "2. SCOPE_QONLY_F0 (無護欄)",
                "O_Q4_200_0": "3. 公平舊版 O_Q4_200_0",
                "NO_VOL_GATE": "4. NO_VOL_GATE (無波動開關)",
                "SPY_1x": "5. SPY (1x 標普500)",
            }
            color_range = ["#d9534f", "#f0ad4e", "#0275d8", "#5cb85c", "#6c757d"]

            nav_df = pd.DataFrame(index=bt_sub.index)
            dd_df = pd.DataFrame(index=bt_sub.index)
            for col, lab in label_map.items():
                nav_s = np.cumprod(1.0 + bt_sub[col].to_numpy(dtype=float))
                nav_df[lab] = nav_s
                dd_df[lab] = ((nav_s / np.maximum.accumulate(nav_s)) - 1.0) * 100.0

            plot_step = max(1, len(bt_sub) // 1400)
            sample_idx = bt_sub.index[::plot_step].union(bt_sub.index[-1:])

            st.markdown("#### 📈 3. 累積淨值變化圖 (Cumulative Equity Curve, 初始資金 = 1.0)")
            nav_long = nav_df.loc[sample_idx].reset_index()
            nav_long = nav_long.rename(columns={nav_long.columns[0]: "Date"}).melt("Date", var_name="策略", value_name="累積淨值 (倍)")
            chart_nav = (
                alt.Chart(nav_long)
                .mark_line(strokeWidth=2.0)
                .encode(
                    x=alt.X("Date:T", title="年份"),
                    y=alt.Y("累積淨值 (倍):Q", scale=alt.Scale(type="log" if use_log else "linear")),
                    color=alt.Color("策略:N", scale=alt.Scale(domain=list(label_map.values()), range=color_range)),
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

            st.markdown("#### 🌊 4. 歷史水下回撤圖 (Underwater Drawdown %)")
            dd_long = dd_df.loc[sample_idx].reset_index()
            dd_long = dd_long.rename(columns={dd_long.columns[0]: "Date"}).melt("Date", var_name="策略", value_name="水下回撤 (%)")
            chart_dd = (
                alt.Chart(dd_long)
                .mark_line(strokeWidth=1.6)
                .encode(
                    x=alt.X("Date:T", title="年份"),
                    y=alt.Y("水下回撤 (%):Q", title="自歷史高點回撤幅度 (%)"),
                    color=alt.Color("策略:N", scale=alt.Scale(domain=list(label_map.values()), range=color_range)),
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

            # 5. 5Y & 10Y 滾動年化報酬率
            st.markdown("#### 🔄 5. 5 年與 10 年滾動年化複合報酬率 (Rolling CAGR %)")
            full_nav_df = pd.DataFrame(
                {lab: np.cumprod(1.0 + bt_full_df[col].to_numpy(dtype=float)) for col, lab in label_map.items()},
                index=bt_full_df.index,
            )
            roll_5y_df = ((full_nav_df / full_nav_df.shift(252 * 5)) ** (1.0 / 5.0) - 1.0) * 100.0
            roll_10y_df = ((full_nav_df / full_nav_df.shift(252 * 10)) ** (1.0 / 10.0) - 1.0) * 100.0

            rc1, rc2 = st.columns(2)
            for col_ui, r_df, ttl, y_lab in [
                (rc1, roll_5y_df, "📅 5-Year Rolling CAGR (%)", "5年滾動CAGR (%)"),
                (rc2, roll_10y_df, "🏛️ 10-Year Rolling CAGR (%)", "10年滾動CAGR (%)"),
            ]:
                with col_ui:
                    r_sub = r_df.dropna(how="all")
                    if not r_sub.empty:
                        r_idx = r_sub.index[:: max(1, len(r_sub) // 900)].union(r_sub.index[-1:])
                        r_long = r_sub.loc[r_idx].reset_index()
                        r_long = r_long.rename(columns={r_long.columns[0]: "Date"}).melt("Date", var_name="策略", value_name=y_lab)
                        zero_rule = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 4], color="black").encode(y="y:Q")
                        ch_r = (
                            alt.Chart(r_long)
                            .mark_line(strokeWidth=1.8)
                            .encode(
                                x=alt.X("Date:T", title="滾動結束日"),
                                y=alt.Y(f"{y_lab}:Q", title=y_lab),
                                color=alt.Color("策略:N", scale=alt.Scale(domain=list(label_map.values()), range=color_range)),
                                tooltip=[
                                    alt.Tooltip("Date:T", title="日期", format="%Y-%m-%d"),
                                    alt.Tooltip("策略:N", title="策略"),
                                    alt.Tooltip(f"{y_lab}:Q", title=y_lab, format=".2f"),
                                ],
                            )
                            .properties(title=ttl, height=340)
                            .interactive()
                        )
                        st.altair_chart(ch_r + zero_rule, use_container_width=True)

            # 6. 歷年年度報酬對照表
            st.markdown("#### 📅 6. 歷年年度報酬率對照表 (Annual Calendar Returns %)")
            ann_rows = []
            for yr, grp in bt_sub.groupby(bt_sub.index.year):
                r_q = float(np.prod(1.0 + grp["REBAND_QONLY_200_0"].to_numpy()) - 1.0) * 100.0
                r_o = float(np.prod(1.0 + grp["O_Q4_200_0"].to_numpy()) - 1.0) * 100.0
                r_nv = float(np.prod(1.0 + grp["NO_VOL_GATE"].to_numpy()) - 1.0) * 100.0
                r_spy = float(np.prod(1.0 + grp["SPY_1x"].to_numpy()) - 1.0) * 100.0
                ann_rows.append({
                    "年份": str(yr),
                    "REBAND_QONLY (%)": round(r_q, 2),
                    "公平舊版 O (%)": round(r_o, 2),
                    "無波動開關 (%)": round(r_nv, 2),
                    "SPY 1x (%)": round(r_spy, 2),
                    "本系統 vs 舊版 O (pp)": round(r_q - r_o, 2),
                    "本系統免交易日 (%)": round(float(grp["SKIP_REBAND_QONLY_200_0"].mean()) * 100.0, 1),
                })
            st.dataframe(pd.DataFrame(ann_rows[::-1]).set_index("年份"), use_container_width=True)

    # ==========================================
    # 分頁 3：策略架構與數學規格說明書 (同步更新為凍結版 REBAND_QONLY_200_0)
    # ==========================================
    with tab_doc:
        st.subheader("📖 REBAND_QONLY_200_0 最終凍結規格與數學說明書")
        st.markdown(
            r"""
### 一、 經 5 輪無近似消融審計後之最終凍結架構（`REBAND_QONLY_200_0`）

本系統針對全球三倍槓桿指數 ETF（美股 `UPRO`、歐股 `EURL`、新興市場 `EDC`）與防禦型資產（黃金 `GLD`、長天期美債 `TLT`）設計，並已正式完成減法消融，**永久拔除冗餘之內層平方反比縮放器 $a_{p,t}$（移除 41% 與 2.0 兩參數）**，僅保留經無近似 5 相位 Student-$t$ $\text{GARCH}(1,1)$ 驗證之核心四模組：

1. **模組 1 — 進攻與避險統一四等分標準日曆季度動能（零自由參數）**：
   每月美東收盤完結日，將過去 12 個月拆分為四個等長日曆季度 $(0,3]\text{M}$、$(3,6]\text{M}$、$(6,9]\text{M}$、$(9,12]\text{M}$，除以 252 日實現波動率取算術平均，分別選出 1x 進攻贏家（對應 3x 執行標的）與避險贏家（`GLD` 或 `TLT`）。
2. **模組 2 — 月度 `SMA6` 加法半倉趨勢濾網**：
   $S_t = \mathbb{I}(P_{L_t,m} > \text{SMA}_{6,m}) \in \{0, 1\}$。採加法半倉（跌破均線減至 50% 而非清倉），避免盤整期洗盤與 V 轉鎖死。
3. **模組 3 — 五相位無近似 Student-$t$ $\text{GARCH}(1,1)$ 外層危機峰值回落斷路器 $Q_{p,t}$**：
   每交易日僅輪值相位 $p \equiv (t - 504) \pmod 5$ 以 504 日 1x 報酬擬合真正 Student-$t$ $\text{GARCH}(1,1)$ 預測波動率 $\sigma_{p,t}$，並與嚴格因果之過去 252 日中位數比較警報比 $\rho_{p,t} = \sigma_{p,t} / \text{Med}_{252}(\sigma)_{p,t-1}$：
   - **危機全退（$1 \to 0$）**：$\rho_{p,t} \ge 2.50$（復歸後再次觸發須同時滿足 $\sigma_{p,t} \ge 1.25 \cdot \sigma^{\text{trough}}_{p,t}$）。
   - **解除／峰值回落復歸（$0 \to 1$）**：$\rho_{p,t} \le 1.25$ 或 $\sigma_{p,t} \le 0.60 \cdot \sigma^{\text{peak}}_{p,t}$。
"""
        )
        st.latex(r"B_{p,t} = Q_{p,t} \times \left( 0.5 \cdot S_t + 0.5 \right) \in \{100\%,\; 50\%,\; 0\%\}")
        st.markdown(
            r"""
4. **模組 4 — `200 / 0 bps` 零殘差成本走廊與強制覆蓋**：
   - `Action = 2`（強制精確執行）：初始建倉、進攻標的換股、`SMA6` 翻轉、$Q_{p,t}$ 翻轉、避險標的換股、持倉集合改變、股票預算調降。
   - `Action = 1`（護欄一步歸位）：無強制事件但最大權重偏離 $d_{\max} \ge 2.00\%$，一步拉回目標權重。
   - `Action = 0`（免交易抱牢）：$d_{\max} < 2.00\%$，保留原股數不動。

### 二、 無近似機構審計基準表（2026-11-23 Colab 實證鎖定）
"""
        )
        audit_bench_df = pd.DataFrame([
            {
                "無近似審計視窗 (單邊 20 bps)": "純實盤 3x ETF 完整期 (2014-01-23 ～ 2026-11-23, 3,229 日)",
                "REBAND_QONLY_200_0 (本系統)": "CAGR 24.29% ｜ Sharpe 0.683 ｜ MaxDD -49.05% ｜ 5Y最差 +4.83% ｜ 10Y最差 +19.04% ｜ 周轉 6.26x ｜ 免交易 96.17%",
                "公平舊版 O_Q4_200_0": "CAGR 16.88% ｜ Sharpe 0.539 ｜ MaxDD -48.49% ｜ 5Y最差 +0.86% ｜ 10Y最差 +13.23% ｜ 周轉 7.43x ｜ 免交易 95.00%",
                "審計風險提示": "2025 關稅 35 日急跌中兩者皆未觸發 (Crisis Alpha = 0 pp，承擔 -48.67% 跌幅)；實盤應透過總投組倉位上限控管尾部風險。",
            },
            {
                "無近似審計視窗 (單邊 20 bps)": "無金礦股長週期 (2001-09-04 ～ 2026-11-23, 6,343 日, GLD=GC=F)",
                "REBAND_QONLY_200_0 (本系統)": "CAGR 33.75% ｜ Sharpe 0.803 ｜ MaxDD -77.14% ｜ 5Y最差 +4.98% ｜ 10Y最差 +19.31% ｜ 周轉 5.69x ｜ 免交易 94.77%",
                "公平舊版 O_Q4_200_0": "CAGR 28.72% ｜ Sharpe 0.733 ｜ MaxDD -78.43% ｜ 5Y最差 +1.28% ｜ 10Y最差 +13.59% ｜ 周轉 6.98x ｜ 免交易 93.75%",
                "審計風險提示": "歷經多輪測試之歷史回測應強制打折 (Haircut)，並預期未來重演 2008 型百年海嘯時仍可能出現 -70%～-77% 極端回撤。",
            },
        ]).set_index("無近似審計視窗 (單邊 20 bps)")
        st.dataframe(audit_bench_df, use_container_width=True)

    # ==========================================
    # 分頁 4：當月 1x 四季動能排名與歷史輪動紀錄
    # ==========================================
    with tab_rot:
        r_col1, r_col2 = st.columns([3, 2])
        with r_col1:
            st.subheader("📊 當月 1x 原型進攻動能排名表 (SPY / VGK / EEM)")
            st.dataframe(
                sel_df.style.format({
                    "Close_1x": "{:.2f}",
                    "SMA6_1x": "{:.2f}",
                    "Vol_252d_Ann": "{:.2%}",
                    "Ret_0to3M": "{:.2%}",
                    "Ret_3to6M": "{:.2%}",
                    "Ret_6to9M": "{:.2%}",
                    "Ret_9to12M": "{:.2%}",
                    "Total_Score": "{:.3f}",
                }),
                use_container_width=True,
            )
        with r_col2:
            st.subheader("🛡️ 當月避險資產動能比較表 (GLD vs TLT)")
            st.dataframe(
                safe_df.style.format({
                    "Close_1x": "{:.2f}",
                    "Vol_252d_Ann": "{:.2%}",
                    "Ret_0to3M": "{:.2%}",
                    "Ret_3to6M": "{:.2%}",
                    "Ret_6to9M": "{:.2%}",
                    "Ret_9to12M": "{:.2%}",
                    "Total_Score": "{:.3f}",
                }),
                use_container_width=True,
            )

        st.subheader("🗓️ 過去 24 個月進攻與防禦標的歷史輪動紀錄")
        st.dataframe(m_sig["hist_rot_df"].set_index("月底訊號鎖定日"), use_container_width=True)

    # ==========================================
    # 分頁 5：原始報價、無近似快取管理與健康檢查
    # ==========================================
    with tab_raw:
        st.subheader("🗄️ 最新 15 個已收盤交易日 ETF 復權收盤價 (Auto-Adjusted Close)")
        st.dataframe(closes_df[CORE_LIVE_ETFS].tail(15).style.format("{:.2f}"), use_container_width=True)

        st.subheader("💾 無近似 Student-t GARCH(1,1) 標準快取檔匯出")
        if has_full_exact_garch:
            npz_bytes = serialize_exact_garch_cache(closes_df.index, vol_5ph, garch_fit_pos)
            st.download_button(
                label=f"⬇️ 下載最新標準無近似快取檔 ({DEFAULT_GARCH_CACHE_FILE})",
                data=npz_bytes,
                file_name=DEFAULT_GARCH_CACHE_FILE,
                mime="application/octet-stream",
                help="將此檔提交至您的 GitHub 倉庫根目錄（與主程式同層），Streamlit 啟動即可秒讀 25 年無近似 GARCH 並只增量補算最新交易日。",
            )
        else:
            st.info("目前為近 1.5 年實盤精確模式。請至分頁 2 上傳既有 Colab 快取或點擊全量擬合後，即可在此下載帶日期索引的標準 `.npz` 檔。")

        st.caption(
            f"✅ 資料健康檢查：主軸共 `{len(closes_df):,}` 個交易日（`{closes_df.index[0].strftime('%Y-%m-%d')}` ～ `{last_dt_str}`）｜ "
            f"資料來源：`{patch_source_note}` ｜ 早期黃金代理：`GC=F`（無 `FSAGX`、無 `.bfill()`）。"
        )
