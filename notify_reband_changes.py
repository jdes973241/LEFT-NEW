#!/usr/bin/env python3
"""REBAND_HF0_200_0 自動化變動偵測與多管道推播通知腳本 (notify_reband_changes.py).

功能特色：
1. 三階段即時收盤補齊機制 (3-Stage Live Post-Close Bar Patching)：
   - 解決 Yahoo Finance 長天期歷史查詢 (start='2018-01-01') 的 CDN 快取延遲問題
   - 自動結合「近 5 日日K (period='5d', interval='1d')」與「近 5 日正規盤 15 分鐘 K 線 (interval='15m', prepost=False)」，
     只要美東時間 16:00 收盤 (15:45 最後一根 15m K 線完成)，立即合成當日精確 Open 與 Close 納入 GARCH 與護欄計算！
2. 與 Streamlit 旗艦版 (app_reband_hf0_institutional.py) 100% 同步的運算核心：
   - 進攻與避險統一採用四等分標準日曆季度 (0, 3, 6, 9, 12) / Vol252 風險調整動能
   - 月度 SMA6 趨勢濾網 + 5 相位 Student-t GARCH(1,1) + Q 危機斷路器 + a 常態縮放器
   - 200 / 0 bps 零殘差執行護欄與 7 大強制減碼覆蓋規則
"""
from __future__ import annotations

from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json
import os
import smtplib
import urllib.request
import warnings

from arch import arch_model
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.simplefilter(action="ignore")

# ==========================================
# 1. 核心策略凍結常數 (與 Streamlit 主程式 100% 一致)
# ==========================================
MAPPING = {"UPRO": "SPY", "EURL": "VGK", "EDC": "EEM"}
REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}
TRADE_ETFS_3X = ["UPRO", "EURL", "EDC"]
PROTO_ETFS_1X = ["SPY", "VGK", "EEM"]
SAFE_POOL = ["GLD", "TLT"]
ALL_EXEC_ASSETS = ["UPRO", "EURL", "EDC", "GLD", "TLT"]
CORE_LIVE_ETFS = TRADE_ETFS_3X + PROTO_ETFS_1X + SAFE_POOL

SMA_MONTHS = 6
GARCH_WINDOW = 504
QUANTILE_WINDOW = 252
PHASE_COUNT = 5
MOM_SEGMENTS = [(0, 3), (3, 6), (6, 9), (9, 12)]

THETA_VOL_PCT = 41.0
VOL_SCALING_POWER = 2.0
RHO_EXIT = 2.50
RHO_REENTRY = 1.25
PEAK_SIGMA_DROP = 0.40
TROUGH_SIGMA_RISE = 0.25
BAND_TRIG_TH = 0.0200
BAND_DEST_TH = 0.0000

DATA_START_DATE = "2018-01-01"


# ==========================================
# 2. 三階段即時數據下載與收盤價自動補齊引擎
# ==========================================
def safe_row_idxmax(df: pd.DataFrame, default_col: str) -> pd.Series:
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


def extract_close_open_from_yf(raw: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = pd.DataFrame(index=raw.index)
    opens = pd.DataFrame(index=raw.index)
    if raw is None or raw.empty:
        return closes, opens
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))
        for t in tickers:
            if t in lvl0:
                t_df = raw[t]
                if "Close" in t_df.columns:
                    closes[t] = pd.to_numeric(t_df["Close"], errors="coerce")
                elif "Price" in t_df.columns:
                    closes[t] = pd.to_numeric(t_df["Price"], errors="coerce")
                if "Open" in t_df.columns:
                    opens[t] = pd.to_numeric(t_df["Open"], errors="coerce")
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


def download_recent_market_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    tz_ny = pytz.timezone("America/New_York")
    end_tomorrow = (datetime.now(tz_ny) + timedelta(days=2)).strftime("%Y-%m-%d")

    # Stage 1: 下載 2018-01-01 至今之日 K 歷史序列 (明確傳入未來 end 日期避免截斷)
    raw = yf.download(
        CORE_LIVE_ETFS,
        start=DATA_START_DATE,
        end=end_tomorrow,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="ticker",
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance 回傳空資料，請稍後再試。")

    closes, opens = extract_close_open_from_yf(raw, CORE_LIVE_ETFS)
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
        opens.index = opens.index.tz_localize(None)

    valid_spy = closes["SPY"].dropna().index
    closes = closes.loc[valid_spy].copy()
    opens = opens.loc[valid_spy].fillna(closes).copy()
    patch_note = "標準日K (1d)"

    # Stage 2: 單獨抓取 period='5d', interval='1d' 繞過長天期歷史 CDN 快取延遲
    try:
        raw_5d = yf.download(
            CORE_LIVE_ETFS,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )
        c_5d, o_5d = extract_close_open_from_yf(raw_5d, CORE_LIVE_ETFS)
        if not c_5d.empty and "SPY" in c_5d.columns:
            if c_5d.index.tz is not None:
                c_5d.index = c_5d.index.tz_localize(None)
                o_5d.index = o_5d.index.tz_localize(None)
            c_5d = c_5d.loc[c_5d["SPY"].dropna().index]
            for dt in c_5d.index:
                for t in CORE_LIVE_ETFS:
                    if t in c_5d.columns and pd.notna(c_5d.loc[dt, t]):
                        closes.loc[dt, t] = float(c_5d.loc[dt, t])
                    if t in o_5d.columns and pd.notna(o_5d.loc[dt, t]):
                        opens.loc[dt, t] = float(o_5d.loc[dt, t])
            closes = closes.sort_index()
            opens = opens.sort_index()
    except Exception:
        pass

    # Stage 3: 若美股剛收盤數小時內 Yahoo 1d 日表尚未寫入當日數據，自動由正規盤 15m K 線合成當日 Open / Close
    try:
        raw_15m = yf.download(
            CORE_LIVE_ETFS,
            period="5d",
            interval="15m",
            prepost=False,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )
        c_15m, o_15m = extract_close_open_from_yf(raw_15m, CORE_LIVE_ETFS)
        if not c_15m.empty and "SPY" in c_15m.columns:
            if c_15m.index.tz is None:
                c_15m.index = c_15m.index.tz_localize("UTC").tz_convert(tz_ny)
                o_15m.index = o_15m.index.tz_localize("UTC").tz_convert(tz_ny)
            else:
                c_15m.index = c_15m.index.tz_convert(tz_ny)
                o_15m.index = o_15m.index.tz_convert(tz_ny)

            c_15m = c_15m.loc[c_15m["SPY"].dropna().index]
            o_15m = o_15m.reindex(c_15m.index)

            for d_val, grp_c in c_15m.groupby(c_15m.index.date):
                ts_day = pd.Timestamp(d_val)
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
    except Exception:
        pass

    return closes.ffill().bfill(), opens.ffill().bfill(), patch_note


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


def evaluate_latest_reband_state(
    closes: pd.DataFrame, opens: pd.DataFrame, patch_note: str = "標準日K (1d)", account_nav: float = 100000.0
) -> dict:
    prices_1x = closes[PROTO_ETFS_1X].dropna(how="all")
    monthly_1x = get_monthly_data(prices_1x)
    ret_1x_daily = (prices_1x / prices_1x.shift(1)) - 1.0
    daily_vol_252 = ret_1x_daily.rolling(252).std(ddof=1) * np.sqrt(252.0)
    monthly_vol_252 = daily_vol_252.reindex(monthly_1x.index)

    seg_scores = []
    for start_m, end_m in MOM_SEGMENTS:
        r_seg = (monthly_1x / monthly_1x.shift(end_m)) - 1.0 if start_m == 0 else (monthly_1x.shift(start_m) / monthly_1x.shift(end_m)) - 1.0
        seg_scores.append(r_seg / (monthly_vol_252 + 1e-6))
    total_score_df = sum(seg_scores) / float(len(MOM_SEGMENTS))

    sma6_mean_df = monthly_1x.rolling(SMA_MONTHS).mean()
    sma6_state_df = (monthly_1x > sma6_mean_df).astype(float)

    prices_safe = closes[SAFE_POOL].dropna(how="all")
    safe_monthly = get_monthly_data(prices_safe)
    ret_safe_daily = (prices_safe / prices_safe.shift(1)) - 1.0
    safe_daily_vol_252 = ret_safe_daily.rolling(252).std(ddof=1) * np.sqrt(252.0)
    safe_monthly_vol_252 = safe_daily_vol_252.reindex(safe_monthly.index)

    safe_seg_scores = []
    for start_m, end_m in MOM_SEGMENTS:
        r_seg_s = (safe_monthly / safe_monthly.shift(end_m)) - 1.0 if start_m == 0 else (safe_monthly.shift(start_m) / safe_monthly.shift(end_m)) - 1.0
        safe_seg_scores.append(r_seg_s / (safe_monthly_vol_252 + 1e-6))
    safe_comp = sum(safe_seg_scores) / float(len(MOM_SEGMENTS))

    last_date = closes.index[-1]
    next_bday = last_date + pd.tseries.offsets.BDay(1)
    is_month_end_session = (next_bday.month != last_date.month)

    effective_monthly_idx = monthly_1x.index if is_month_end_session else monthly_1x.index[:-1]
    ref_date = effective_monthly_idx[-1]

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

    tail_n = GARCH_WINDOW + QUANTILE_WINDOW + 140
    sub_closes = closes.tail(tail_n).copy()
    sub_opens = opens.reindex(sub_closes.index).copy()
    abs_indices = closes.index.get_indexer(sub_closes.index)
    dates_sub = sub_closes.index

    vol_5ph = {t3x: np.full((PHASE_COUNT, len(dates_sub)), np.nan) for t3x in TRADE_ETFS_3X}
    q_5ph = {t3x: np.ones((PHASE_COUNT, len(dates_sub)), dtype=float) for t3x in TRADE_ETFS_3X}
    a_5ph = {t3x: np.ones((PHASE_COUNT, len(dates_sub)), dtype=float) for t3x in TRADE_ETFS_3X}
    med_5ph = {t3x: np.full((PHASE_COUNT, len(dates_sub)), np.nan) for t3x in TRADE_ETFS_3X}

    winner_now = str(daily_leader_3x.iloc[-1])
    safe_now = str(daily_safe_win.iloc[-1])
    active_3x_set = set(daily_leader_3x.tail(130).unique()) | {winner_now}

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
            v_arr = s_v.to_numpy(dtype=float)
            m_arr = s_med.to_numpy(dtype=float)
            q_arr, a_arr = run_q_state_machine_1d(v_arr, m_arr)
            vol_5ph[t3x][ph] = v_arr
            med_5ph[t3x][ph] = m_arr
            q_5ph[t3x][ph] = q_arr
            a_5ph[t3x][ph] = a_arr

    replay_start = max(GARCH_WINDOW + QUANTILE_WINDOW, len(dates_sub) - 120)
    r_on_df = ((sub_opens[ALL_EXEC_ASSETS] / sub_closes[ALL_EXEC_ASSETS].shift(1)) - 1.0).fillna(0.0)
    r_id_df = ((sub_closes[ALL_EXEC_ASSETS] / sub_opens[ALL_EXEC_ASSETS]) - 1.0).fillna(0.0)

    phase_records = []
    change_events = []

    prev_day_winner = str(daily_leader_3x.iloc[-2])
    prev_day_safe = str(daily_safe_win.iloc[-2])
    if winner_now != prev_day_winner:
        change_events.append(f"進攻標的月度轉換: {prev_day_winner} -> {winner_now}")
    if safe_now != prev_day_safe:
        change_events.append(f"避險標的月度轉換: {prev_day_safe} -> {safe_now}")

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
            lead_3x = str(daily_leader_3x.loc[sig_dt])
            safe_1x = str(daily_safe_win.loc[sig_dt])
            s_val = float(daily_sma6_by_3x.loc[sig_dt, lead_3x])
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
        s_now = float(daily_sma6_by_3x.iloc[-1][winner_now])
        q_now = float(q_5ph[winner_now][ph, idx_now])
        a_now = float(a_5ph[winner_now][ph, idx_now])
        sig_now = float(vol_5ph[winner_now][ph, idx_now])
        med_now = float(med_5ph[winner_now][ph, idx_now])
        rho_now = sig_now / med_now if (np.isfinite(med_now) and med_now > 0) else np.nan
        b_now = q_now * (0.5 * s_now + 0.5 * a_now)

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
            change_events.append(f"Phase {ph} 觸發 Action=2 強制調倉 ({', '.join(reasons)})")
        elif d_max_now >= BAND_TRIG_TH and d_max_now > 0:
            act_now = 1
            ratio = BAND_DEST_TH / d_max_now
            plan_w_now = {k: base_w_now[k] + ratio * (holdings[k] - base_w_now[k]) for k in ALL_EXEC_ASSETS}
            change_events.append(f"Phase {ph} 觸發 Action=1 護欄歸位 (偏離 {d_max_now*10000:.1f} bps >= 200 bps)")
        else:
            act_now = 0
            plan_w_now = dict(holdings)

        phase_records.append({
            "Phase": ph,
            "Sigma_Pct": round(sig_now, 2),
            "Med252_Pct": round(med_now, 2),
            "Rho": round(rho_now, 3),
            "Q_Gate": int(q_now),
            "SMA6": int(s_now),
            "a_Scalar": round(a_now, 4),
            "Base_Stock_Pct": round(b_now * 100.0, 2),
            "Prior_Close_Stock_Pct": round(holdings[winner_now] * 100.0, 2),
            "Drift_bps": round(d_max_now * 10000.0, 1),
            "Action_Code": act_now,
            "Reasons": ", ".join(reasons) if reasons else "無",
            "Prior_Weights": holdings,
            "Base_Weights": base_w_now,
            "Planned_Weights": plan_w_now,
        })

    agg_prior_w = {k: float(np.mean([r["Prior_Weights"][k] for r in phase_records])) for k in ALL_EXEC_ASSETS}
    agg_base_w = {k: float(np.mean([r["Base_Weights"][k] for r in phase_records])) for k in ALL_EXEC_ASSETS}
    agg_plan_w = {k: float(np.mean([r["Planned_Weights"][k] for r in phase_records])) for k in ALL_EXEC_ASSETS}

    latest_prices = {t: float(closes[t].iloc[-1]) for t in ALL_EXEC_ASSETS}
    orders = []
    for t in ALL_EXEC_ASSETS:
        w_curr = agg_prior_w[t]
        w_targ = agg_plan_w[t]
        w_diff = w_targ - w_curr
        if len(change_events) == 0:
            w_diff = 0.0
            w_targ = w_curr
        usd_diff = w_diff * account_nav
        px = latest_prices[t]
        shares_diff = usd_diff / px if px > 0 else 0.0
        if abs(w_diff) < 1e-4:
            act_str = "HOLD (不交易)"
        elif w_diff > 0:
            act_str = "BUY (開盤買進)"
        else:
            act_str = "SELL (開盤賣出)"
        orders.append({
            "Ticker": t,
            "Prior_Weight_Pct": round(w_curr * 100.0, 2),
            "Base_Target_Pct": round(agg_base_w[t] * 100.0, 2),
            "Planned_Weight_Pct": round(w_targ * 100.0, 2),
            "Delta_Weight_Pct": round(w_diff * 100.0, 2),
            "Delta_USD": round(usd_diff, 1),
            "Latest_Price": round(px, 2),
            "Delta_Shares": round(shares_diff, 2),
            "Whole_Shares": int(round(shares_diff)),
            "Command": act_str,
        })

    return {
        "signal_close_date": last_date.strftime("%Y-%m-%d"),
        "patch_source_note": patch_note,
        "monthly_lock_date": ref_date.strftime("%Y-%m-%d"),
        "is_month_end_session": bool(is_month_end_session),
        "winner_3x": winner_now,
        "winner_1x": MAPPING[winner_now],
        "safe_win": safe_now,
        "has_change": len(change_events) > 0,
        "change_events": change_events,
        "agg_prior_weights_pct": {k: round(v * 100.0, 2) for k, v in agg_prior_w.items()},
        "agg_base_weights_pct": {k: round(v * 100.0, 2) for k, v in agg_base_w.items()},
        "agg_planned_weights_pct": {k: round(v * 100.0, 2) for k, v in agg_plan_w.items()},
        "phase_records": phase_records,
        "orders": orders,
    }


# ==========================================
# 3. 訊息排版與多管道推播函式 (Email / Bark iOS / Telegram)
# ==========================================
def format_notification_messages(report: dict, account_nav: float) -> tuple[str, str, str]:
    sig_dt = report["signal_close_date"]
    w3x = report["winner_3x"]
    sf = report["safe_win"]
    has_chg = report["has_change"]
    src_note = report.get("patch_source_note", "標準日K (1d)")

    if has_chg:
        subject = f"🚨 [REBAND 調倉通知] {sig_dt} 收盤觸發變動！目標: {report['agg_planned_weights_pct'][w3x]:.1f}% {w3x} / {report['agg_planned_weights_pct'][sf]:.1f}% {sf}"
    else:
        subject = f"✅ [REBAND 每日巡檢] {sig_dt} 無變動 (Action=0 抱牢 {report['agg_planned_weights_pct'][w3x]:.1f}% {w3x})"

    active_orders = [o for o in report["orders"] if not o["Command"].startswith("HOLD")]
    if active_orders:
        order_lines = "\n".join(
            f"• {o['Command']} {o['Ticker']}: 權重 {o['Prior_Weight_Pct']:.2f}% -> {o['Planned_Weight_Pct']:.2f}% "
            f"({o['Delta_Weight_Pct']:+.2f}% / ${abs(o['Delta_USD']):,.0f} USD / 約 {abs(o['Delta_Shares']):.2f} 股)"
            for o in active_orders
        )
    else:
        order_lines = "• 今日全部 5 個相位均為 Action=0，單一帳戶 0 筆委託單，請抱牢現有股數不動。"

    events_str = "\n".join(f"• {ev}" for ev in report["change_events"]) if report["change_events"] else "• 無任何標的轉換或護欄觸發事件"

    text_body = (
        f"{subject}\n"
        f"==================================================\n"
        f"📅 最新美股收盤日: {sig_dt} ({src_note})\n"
        f"🗓️ 月度訊號鎖定日: {report['monthly_lock_date']}\n"
        f"🎯 當前進攻贏家: {w3x} ({report['winner_1x']}) ｜ 當前避險贏家: {sf}\n"
        f"💰 預設帳戶總淨值: ${account_nav:,.0f} USD\n\n"
        f"【變動觸發摘要】\n{events_str}\n\n"
        f"【次一交易日開盤下單指令】\n{order_lines}\n\n"
        f"【五相位 (Phase 0..4) 狀態明細】\n"
    )
    for r in report["phase_records"]:
        text_body += (
            f"- Phase {r['Phase']}: σ={r['Sigma_Pct']:.1f}%, ρ={r['Rho']:.2f}, Q={r['Q_Gate']}, SMA6={r['SMA6']}, "
            f"a={r['a_Scalar']:.3f} | 基礎目標={r['Base_Stock_Pct']:.1f}%, 昨收={r['Prior_Close_Stock_Pct']:.1f}%, "
            f"偏離={r['Drift_bps']:.1f}bps -> Action={r['Action_Code']} ({r['Reasons']})\n"
        )

    order_rows_html = ""
    for o in report["orders"]:
        bg = "#fff3cd" if not o["Command"].startswith("HOLD") else "#ffffff"
        order_rows_html += (
            f"<tr style='background-color:{bg};'>"
            f"<td style='padding:6px;border:1px solid #ddd;'><b>{o['Ticker']}</b></td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'>{o['Prior_Weight_Pct']:.2f}%</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'>{o['Base_Target_Pct']:.2f}%</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'><b>{o['Planned_Weight_Pct']:.2f}%</b></td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'>{o['Delta_Weight_Pct']:+.2f}%</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'>${o['Delta_USD']:+,.0f}</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'>{o['Delta_Shares']:+.2f} ({o['Whole_Shares']:+d})</td>"
            f"<td style='padding:6px;border:1px solid #ddd;'><b>{o['Command']}</b></td>"
            f"</tr>"
        )

    phase_rows_html = ""
    for r in report["phase_records"]:
        phase_rows_html += (
            f"<tr>"
            f"<td style='padding:6px;border:1px solid #ddd;'>Phase {r['Phase']}</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'>{r['Sigma_Pct']:.2f}%</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'>{r['Rho']:.3f}</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:center;'>{r['Q_Gate']}</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:center;'>{r['SMA6']}</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'>{r['a_Scalar']:.4f}</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'>{r['Base_Stock_Pct']:.2f}%</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'>{r['Prior_Close_Stock_Pct']:.2f}%</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:right;'>{r['Drift_bps']:.1f}</td>"
            f"<td style='padding:6px;border:1px solid #ddd;'><b>Action={r['Action_Code']}</b> ({r['Reasons']})</td>"
            f"</tr>"
        )

    html_body = f"""
    <html><body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #212529;">
      <h2 style="color: {'#d9534f' if has_chg else '#28a745'};">{subject}</h2>
      <p>
        📅 <b>最新美股收盤日</b>: <code>{sig_dt}</code> ({src_note}) ｜
        <b>月度訊號鎖定日</b>: <code>{report['monthly_lock_date']}</code> ｜
        <b>當前標的</b>: 進攻 <b>{w3x}</b> ({report['winner_1x']}) ＋ 避險 <b>{sf}</b>
      </p>
      <h4>🔔 變動觸發摘要</h4>
      <pre style="background:#f8f9fa;padding:12px;border-radius:6px;border:1px solid #dee2e6;">{events_str}\n\n{order_lines}</pre>
      <h4>💳 單一帳戶匯總下單指令表 (帳戶淨值預設 ${account_nav:,.0f} USD)</h4>
      <table style="border-collapse:collapse;width:100%;font-size:13px;">
        <thead>
          <tr style="background-color:#f1f3f5;">
            <th style="padding:6px;border:1px solid #ddd;">ETF</th>
            <th style="padding:6px;border:1px solid #ddd;">昨日收盤權重</th>
            <th style="padding:6px;border:1px solid #ddd;">基礎目標權重</th>
            <th style="padding:6px;border:1px solid #ddd;">今日計畫權重</th>
            <th style="padding:6px;border:1px solid #ddd;">調倉幅度</th>
            <th style="padding:6px;border:1px solid #ddd;">買賣金額 (USD)</th>
            <th style="padding:6px;border:1px solid #ddd;">買賣股數 (整數)</th>
            <th style="padding:6px;border:1px solid #ddd;">下單動作</th>
          </tr>
        </thead>
        <tbody>{order_rows_html}</tbody>
      </table>
      <h4>🔬 五相位 (Phase 0..4) GARCH 波動率與 200/0 bps 護欄明細</h4>
      <table style="border-collapse:collapse;width:100%;font-size:13px;">
        <thead>
          <tr style="background-color:#f1f3f5;">
            <th style="padding:6px;border:1px solid #ddd;">相位</th>
            <th style="padding:6px;border:1px solid #ddd;">預測波動率 σ</th>
            <th style="padding:6px;border:1px solid #ddd;">警報比 ρ</th>
            <th style="padding:6px;border:1px solid #ddd;">Q 閘門</th>
            <th style="padding:6px;border:1px solid #ddd;">SMA6</th>
            <th style="padding:6px;border:1px solid #ddd;">a 縮放</th>
            <th style="padding:6px;border:1px solid #ddd;">基礎股票目標</th>
            <th style="padding:6px;border:1px solid #ddd;">昨收實際股票</th>
            <th style="padding:6px;border:1px solid #ddd;">偏離 (bps)</th>
            <th style="padding:6px;border:1px solid #ddd;">開盤動作與原因</th>
          </tr>
        </thead>
        <tbody>{phase_rows_html}</tbody>
      </table>
    </body></html>
    """
    return subject, text_body, html_body


def send_email_notification(subject: str, text_body: str, html_body: str) -> bool:
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASSWORD", "").strip()
    notify_to = os.environ.get("NOTIFY_EMAIL_TO", smtp_user).strip()
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))

    if not smtp_user or not smtp_pass or not notify_to:
        print("ℹ️ 未設定 SMTP_USER / SMTP_PASSWORD，跳過 Email 寄送。")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = notify_to
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(smtp_user, smtp_pass)
        recipients = [addr.strip() for addr in notify_to.split(",") if addr.strip()]
        server.sendmail(smtp_user, recipients, msg.as_string())
    print(f"✅ 已成功寄送 Email 通知至: {notify_to}")
    return True


def send_ios_bark_or_webhook(subject: str, text_body: str) -> bool:
    sent_any = False
    bark_url = os.environ.get("BARK_URL", "").strip().rstrip("/")
    if bark_url:
        payload = json.dumps({
            "title": subject,
            "body": text_body,
            "group": "REBAND_HF0",
            "sound": "calypso",
        }).encode("utf-8")
        req = urllib.request.Request(bark_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"✅ 已發送 iPhone Bark 原生推播 (HTTP {resp.status})")
            sent_any = True

    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if tg_token and tg_chat_id:
        tg_api = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        payload = json.dumps({"chat_id": tg_chat_id, "text": text_body}).encode("utf-8")
        req = urllib.request.Request(tg_api, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"✅ 已發送 Telegram 推播通知 (HTTP {resp.status})")
            sent_any = True

    return sent_any


def main() -> int:
    account_nav = float(os.environ.get("ACCOUNT_NAV", "100000"))
    notify_only_on_change = os.environ.get("NOTIFY_ONLY_ON_CHANGE", "false").lower() == "true"
    force_send = os.environ.get("FORCE_SEND", "false").lower() == "true"

    print("🚀 正在下載最新 ETF 報價並執行三階段收盤價補齊與 REBAND_HF0_200_0 狀態回放...")
    closes, opens, patch_note = download_recent_market_data()
    report = evaluate_latest_reband_state(closes, opens, patch_note=patch_note, account_nav=account_nav)

    subject, text_body, html_body = format_notification_messages(report, account_nav=account_nav)
    print("\n" + text_body)

    if report["has_change"] or (not notify_only_on_change) or force_send:
        print("🔔 觸發推播條件，開始發送通知...")
        send_email_notification(subject, text_body, html_body)
        send_ios_bark_or_webhook(subject, text_body)
    else:
        print("✅ 今日無任何標的轉換或持股比例變動 (全部相位均為 Action=0)，依設定靜默不打擾。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
