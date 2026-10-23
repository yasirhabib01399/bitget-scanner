# ═══════════════════════════════════════════════════════════════════
# MEGA BOT v8 — P1 + P8 + P21-v5 + P21-v6 + P22 + P23
# ═══════════════════════════════════════════════════════════════════
import csv
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ENABLE_TELEGRAM = True

API_URL = "https://api.bitget.com/api/v2/spot/market/candles"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SIGNAL_LOG_FILE = "signals_log.csv"

# ─── Coin lists ───
P1_COINS = ["BTCUSDT","SOLUSDT","ADAUSDT","NEARUSDT","LTCUSDT",
            "BNBUSDT","UNIUSDT","APTUSDT","ARBUSDT","DOTUSDT",
            "DOGEUSDT","TIAUSDT"]
P8_COINS = ["OPUSDT","ADAUSDT","SOLUSDT","LTCUSDT","ETHUSDT","SEIUSDT"]
P21_V5_COINS = ["BTCUSDT","SOLUSDT","ADAUSDT","NEARUSDT","LTCUSDT",
                "BNBUSDT","UNIUSDT","ARBUSDT","DOTUSDT",
                "DOGEUSDT","TIAUSDT","ETHUSDT","SUIUSDT"]
P21_V6_COINS = P21_V5_COINS
P22_COINS = ["TIAUSDT","ARBUSDT","APTUSDT","UNIUSDT","OPUSDT",
             "DOTUSDT","SUIUSDT","SOLUSDT","ETHUSDT","NEARUSDT","ADAUSDT"]

# 🆕 P23 (Only best coins)
P23_COINS = ["DOTUSDT", "SOLUSDT", "NEARUSDT"]

ALL_SYMBOLS = list(set(P1_COINS + P8_COINS + P21_V5_COINS + P21_V6_COINS + P22_COINS + P23_COINS))

# ─── Bot config ───
CANDLES_TO_CHECK = 3
TIME_STOP = 20
COOLDOWN_BARS = 8
FEE_PCT = 0.2
TRAIL_PCT = 0.3
MAX_BARS_AFTER_TP = 15

# Pattern specific
P21_TIME_STOP = 12
P21_TP = 1.5
P21_SL = 2.0

P22_TIME_STOP = 12
P22_TP = 1.2
P22_SL = 1.5

# 🆕 P23 Config B
P23_TIME_STOP = 12
P23_TP = 1.2
P23_SL = 1.2

# Priority: P21-v5 > P22 > P21-v6 > P23 > P1 > P8
PATTERN_PRIORITY = {"P21-v5": 6, "P22": 5, "P21-v6": 4, "P23": 3, "P1": 2, "P8": 1}

CSV_FIELDS = [
    "signal_ts_ms", "signal_time_utc", "signal_time_pkt",
    "symbol", "pattern",
    "entry_close", "entry_open",
    "tp_pct", "sl_pct",
    "tp_price", "sl_price",
    "tp_triggered", "tp_trigger_ts",
    "highest_seen", "current_sl",
    "bars_since_tp",
    "exit_price", "exit_ts_ms", "exit_time_pkt",
    "result", "bars_held", "hours_to_exit",
    "mfe_pct", "mae_pct", "bars_to_mfe", "bars_to_mae",
    "pnl_pct", "status"
]

def safe_int(val):
    if val is None or val == "": return 0
    try: return int(float(val))
    except (ValueError, TypeError): return 0

def safe_float(val, default=0.0):
    if val is None or val == "": return default
    try: return float(val)
    except (ValueError, TypeError): return default

def send_telegram_message(message):
    if not ENABLE_TELEGRAM or not TELEGRAM_BOT_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

def fetch_candles(symbol):
    params = {"symbol": symbol, "granularity": "15min", "limit": "300"}
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        data = r.json()
        if data.get("code") != "00000": return None
        rows = data.get("data", [])
        if len(rows) < 250: return None
        df = pd.DataFrame([{
            "ts": int(x[0]), "open": float(x[1]), "high": float(x[2]),
            "low": float(x[3]), "close": float(x[4]), "volume": float(x[5])
        } for x in rows])
        df = df.sort_values("ts").reset_index(drop=True)
        now_ms = int(time.time() * 1000)
        if df.iloc[-1]["ts"] + (15 * 60 * 1000) > now_ms:
            df = df.iloc[:-1]
        return df
    except Exception as e:
        print(f"❌ Fetch Error {symbol}: {e}")
        return None

def calc_rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_adx(df, n=14):
    high, low, close = df['high'], df['low'], df['close']
    up = high.diff(); down = -low.diff()
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    mdm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=high.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100 * pd.Series(mdm, index=high.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def calc_atr(df, n=20):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    close = df['close']
    df['rsi14'] = calc_rsi(close, 14)
    df['rsi2'] = calc_rsi(close, 2)
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    df['ema50'] = close.ewm(span=50, adjust=False).mean()
    df['ema20'] = close.ewm(span=20, adjust=False).mean()
    df['adx'] = calc_adx(df, 14)
    df['bb_mid'] = close.rolling(20).mean()
    df['bb_std'] = close.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2.0 * df['bb_std'])
    df['bb_lower'] = df['bb_mid'] - (2.0 * df['bb_std'])
    
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_pos'] = (close - df['bb_lower']) / bb_range.replace(0, np.nan)
    
    df['atr20'] = calc_atr(df, 20)
    df['kc_lower'] = df['ema20'] - (2.0 * df['atr20'])
    
    # P21/P22 indicators
    df['swing_low_20'] = df['low'].rolling(20).min().shift(1)
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20'].replace(0, np.nan)
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    df['prev_3_high'] = df['high'].rolling(3).max().shift(1)
    df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200'] * 100
    return df

# ─── Pattern signals ───
def check_p1(row):
    if pd.isna(row['rsi14']) or pd.isna(row['ema200']) or pd.isna(row['adx']): return False
    if pd.isna(row['bb_lower']): return False
    return (row['close'] < row['bb_lower'] and row['rsi14'] < 32.0 and
            row['close'] > row['ema200'] and row['adx'] < 50.0)

def check_p8(row):
    if pd.isna(row['rsi14']) or pd.isna(row['ema200']) or pd.isna(row['adx']): return False
    if pd.isna(row['kc_lower']): return False
    return (row['close'] < row['kc_lower'] and row['rsi14'] < 35.0 and
            row['close'] > row['ema200'] and row['adx'] < 50.0)

def check_p21_v5(df, idx):
    if idx < 1: return False
    r = df.iloc[idx]; prev = df.iloc[idx-1]
    if any(pd.isna(r[k]) for k in ["ema200","rsi14","swing_low_20","vol_ratio"]): return False
    if not ((r["low"] < r["swing_low_20"]) and (r["close"] > r["swing_low_20"])): return False
    if r["close"] <= r["open"]: return False
    if r["close"] <= prev["high"]: return False
    if r["close"] >= r["ema200"]: return False
    if (r["close"] - r["ema200"]) / r["ema200"] * 100 < -15: return False
    if r["rsi14"] >= 40: return False
    if r["vol_ratio"] < 1.5: return False
    return True

def check_p21_v6(df, idx):
    if idx < 1: return False
    r = df.iloc[idx]; prev = df.iloc[idx-1]
    if any(pd.isna(r[k]) for k in ["rsi14","swing_low_20","vol_ratio"]): return False
    if not ((r["low"] < r["swing_low_20"]) and (r["close"] > r["swing_low_20"])): return False
    if r["close"] <= r["open"]: return False
    if r["close"] <= prev["high"]: return False
    if r["rsi14"] >= 40: return False
    if r["vol_ratio"] < 1.2: return False
    return True

def check_p22(df, idx):
    if idx < 25: return False
    r = df.iloc[idx]
    if any(pd.isna(r[k]) for k in ["ema200","rsi14","swing_low_20","vol_ratio","range_pct","prev_3_high"]): return False
    
    recent_sweep = False
    for k in range(idx-10, idx):
        if df.iloc[k]["low"] < df.iloc[k]["swing_low_20"]:
            recent_sweep = True; break
    if not recent_sweep: return False
    if df["range_pct"].iloc[idx-3:idx].max() >= 1.5: return False
    if r["close"] <= r["prev_3_high"]: return False
    if r["vol_ratio"] < 1.5: return False
    if r["rsi14"] >= 50 or r["rsi14"] <= 30: return False
    if r["dist_ema200"] < -15: return False
    return True

# 🆕 P23 Logic (Hook Up)
def check_p23(df, idx):
    if idx < 1: return False
    r = df.iloc[idx]; p = df.iloc[idx-1]
    
    if any(pd.isna(r[k]) for k in ["rsi14", "adx", "bb_pos", "vol_ratio"]): return False
    if any(pd.isna(p[k]) for k in ["rsi2"]): return False
    if any(pd.isna(r[k]) for k in ["ema50", "ema200"]): return False
    
    if r['close'] <= r['ema200']: return False
    if r['close'] <= r['ema50']: return False
    if r['adx'] >= 25.0: return False
    if p['rsi2'] >= 25.0: return False
    if r['close'] <= r['open']: return False
    if r['close'] <= p['high']: return False
    if r['bb_pos'] <= 0.2: return False
    if r['rsi14'] <= 40.0: return False
    if r['vol_ratio'] <= 0.5: return False
    
    return True

def check_signals_at(symbol, df, idx):
    if idx < 200 or idx >= len(df): return None
    row = df.iloc[idx]
    candidates = []

    if symbol in P1_COINS and check_p1(row):
        candidates.append({"pattern":"P1", "tp":1.2, "sl":2.5,
                           "time_stop":TIME_STOP, "ts":row['ts'], "entry_close":row['close']})

    if symbol in P8_COINS and check_p8(row):
        candidates.append({"pattern":"P8", "tp":1.0, "sl":2.5,
                           "time_stop":TIME_STOP, "ts":row['ts'], "entry_close":row['close']})

    if symbol in P21_V5_COINS and check_p21_v5(df, idx):
        candidates.append({"pattern":"P21-v5", "tp":P21_TP, "sl":P21_SL,
                           "time_stop":P21_TIME_STOP, "ts":row['ts'], "entry_close":row['close']})

    if symbol in P21_V6_COINS and check_p21_v6(df, idx):
        candidates.append({"pattern":"P21-v6", "tp":P21_TP, "sl":P21_SL,
                           "time_stop":P21_TIME_STOP, "ts":row['ts'], "entry_close":row['close']})

    if symbol in P22_COINS and check_p22(df, idx):
        candidates.append({"pattern":"P22", "tp":P22_TP, "sl":P22_SL,
                           "time_stop":P22_TIME_STOP, "ts":row['ts'], "entry_close":row['close']})

    # 🆕 P23 Check
    if symbol in P23_COINS and check_p23(df, idx):
        candidates.append({"pattern":"P23", "tp":P23_TP, "sl":P23_SL,
                           "time_stop":P23_TIME_STOP, "ts":row['ts'], "entry_close":row['close']})

    if not candidates: return None
    return max(candidates, key=lambda x: PATTERN_PRIORITY[x["pattern"]])

# ─── CSV & Telegram helpers (Same as v7) ───
def load_log():
    if not os.path.exists(SIGNAL_LOG_FILE): return []
    try:
        with open(SIGNAL_LOG_FILE, mode='r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        return [r for r in rows if r.get('pattern')]
    except Exception as e:
        print(f"⚠️ CSV read error: {e}")
        return []

def save_log(rows):
    try:
        with open(SIGNAL_LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore', restval='')
            w.writeheader()
            for r in rows: w.writerow(r)
    except Exception as e:
        print(f"⚠️ CSV save error: {e}")

def can_open_trade(sym, sig_ts, existing_rows):
    cooldown_ms = COOLDOWN_BARS * 15 * 60 * 1000
    last_exit_ts = None
    has_open = False
    for r in existing_rows:
        if r.get("symbol") != sym: continue
        status = r.get("status", "")
        if status in ("PENDING", "OPEN"):
            has_open = True; break
        if status == "DONE" and r.get("exit_ts_ms"):
            ets = safe_int(r["exit_ts_ms"])
            if ets > 0 and (last_exit_ts is None or ets > last_exit_ts):
                last_exit_ts = ets
    if has_open: return False
    if last_exit_ts is not None and (sig_ts - last_exit_ts) < cooldown_ms:
        return False
    return True

def notify_signal(sig, t_pkt):
    msg = (
        f"🚨 <b>{sig['pattern']} SIGNAL</b>\n\n"
        f"🪙 <b>Symbol:</b> {sig['symbol']}\n"
        f"⏰ <b>Time:</b> {t_pkt.strftime('%Y-%m-%d %H:%M')} PKT\n"
        f"🎯 <b>Entry:</b> {sig['entry_close']:.8g}\n"
        f"📊 <b>TP:</b> +{sig['tp']}% | <b>SL:</b> -{sig['sl']}%\n"
        f"\n🔥 <b>TP-Trail:</b> After TP, trail at 0.3%"
    )
    send_telegram_message(msg)

def notify_tp_trigger(sym, trade, highest, current_sl):
    msg = (
        f"🎯 <b>TP HIT → TRAILING</b> {sym} ({trade['pattern']})\n"
        f"🕐 Signal: {trade['signal_time_pkt']} PKT\n"
        f"🎯 Entry: {safe_float(trade['entry_open']):.8g}\n"
        f"📈 TP Level: {safe_float(trade['tp_price']):.8g}\n"
        f"📊 Current High: {highest:.8g}\n"
        f"🛡️ Trail SL: {current_sl:.8g} (locked +{((current_sl/safe_float(trade['entry_open']))-1)*100:.2f}%)\n"
        f"\n💡 <b>Watching for more upside...</b>"
    )
    send_telegram_message(msg)

def notify_result(sym, trade, result, exit_price, pnl_pct, mfe, mae):
    emoji = "✅" if "TP" in result else ("🛑" if result == "SL" else "⏱")
    pattern = trade.get('pattern', 'N/A')
    if result == "TP_TRAIL":
        note = "\n🔥 Trailing profit captured!"
    elif result == "TP_TRAIL_TIMEOUT":
        note = "\n⏱ Trailing timeout"
    else:
        note = ""
    msg = (
        f"{emoji} <b>{result}</b> #{sym} ({pattern})\n"
        f"🕐 Signal: {trade['signal_time_pkt']} PKT\n"
        f"🎯 Entry: {safe_float(trade['entry_open']):.8g}\n"
        f"🚪 Exit: {exit_price:.8g}\n"
        f"💰 P&L: <b>{pnl_pct:+.3f}%</b>\n"
        f"📊 MFE: +{mfe:.2f}% | MAE: {mae:.2f}%"
        f"{note}"
    )
    send_telegram_message(msg)

def update_outcomes(coin_data, existing_rows):
    updated = False
    for r in existing_rows:
        sym = r.get("symbol", "")
        status = r.get("status", "")
        if status not in ("PENDING", "OPEN"): continue
        if sym not in coin_data: continue

        df = coin_data[sym]
        sig_ts = safe_int(r.get("signal_ts_ms", 0))
        if sig_ts == 0: continue

        match = df[df['ts'] == sig_ts]
        if match.empty:
            r["result"] = "UNKNOWN"
            r["status"] = "DONE"
            updated = True
            continue
        idx = match.index[0]

        if status == "PENDING":
            if idx + 1 >= len(df): continue
            next_open = float(df.iloc[idx + 1]['open'])
            tp_pct = safe_float(r.get("tp_pct", 1.2))
            sl_pct = safe_float(r.get("sl_pct", 2.5))

            r["entry_open"] = f"{next_open:.10g}"
            r["tp_price"] = f"{next_open * (1 + tp_pct/100):.10g}"
            r["sl_price"] = f"{next_open * (1 - sl_pct/100):.10g}"
            r["highest_seen"] = f"{next_open:.10g}"
            r["current_sl"] = r["sl_price"]
            r["status"] = "OPEN"
            updated = True

        entry = safe_float(r.get("entry_open", 0))
        tp = safe_float(r.get("tp_price", 0))
        orig_sl = safe_float(r.get("sl_price", 0))
        if entry == 0 or tp == 0 or orig_sl == 0: continue

        pattern = r.get("pattern", "P1")
        if pattern.startswith("P21"):
            time_stop = P21_TIME_STOP
        elif pattern == "P22":
            time_stop = P22_TIME_STOP
        elif pattern == "P23":
            time_stop = P23_TIME_STOP
        else:
            time_stop = TIME_STOP

        future_bars = df.iloc[idx+1 : idx+time_stop+2]
        if len(future_bars) == 0: continue

        current_sl = orig_sl
        highest = entry
        tp_hit = False
        tp_hit_bar = None
        bars_held = 0
        result = None
        exit_ts = None
        exit_price = None
        mae_pct = 0.0
        was_tp_triggered = safe_int(r.get("tp_triggered", 0)) == 1

        for i, (_, bar) in enumerate(future_bars.iterrows()):
            bars_held = i + 1
            bar_ts = bar['ts']
            bar_high = bar['high']
            bar_low = bar['low']
            bar_close = bar['close']

            bar_mae = ((bar_low / entry) - 1) * 100
            if bar_mae < mae_pct: mae_pct = bar_mae

            if not tp_hit:
                if bar_low <= current_sl:
                    result, exit_ts, exit_price = "SL", bar_ts, current_sl
                    break
                if bar_high >= tp:
                    tp_hit = True
                    tp_hit_bar = i
                    highest = bar_high
                    new_sl = highest * (1 - TRAIL_PCT/100)
                    current_sl = max(tp, new_sl)
                    if not was_tp_triggered:
                        notify_tp_trigger(sym, r, highest, current_sl)
                        was_tp_triggered = True
                        r["tp_triggered"] = "1"
                        r["tp_trigger_ts"] = str(int(bar_ts))
                    continue
                if bars_held >= time_stop:
                    result, exit_ts, exit_price = "TIMEOUT", bar_ts, bar_close
                    break
            else:
                bars_since = i - tp_hit_bar
                highest = max(highest, bar_high)
                new_sl = highest * (1 - TRAIL_PCT/100)
                current_sl = max(current_sl, new_sl)

                if bar_low <= current_sl:
                    result, exit_ts, exit_price = "TP_TRAIL", bar_ts, current_sl
                    break
                if bars_since >= MAX_BARS_AFTER_TP:
                    result, exit_ts, exit_price = "TP_TRAIL_TIMEOUT", bar_ts, bar_close
                    break

        if result:
            pnl_pct = ((exit_price / entry) - 1) * 100 - FEE_PCT
            r["result"] = result
            r["exit_price"] = f"{exit_price:.10g}"
            r["exit_ts_ms"] = str(int(exit_ts))
            r["bars_held"] = str(bars_held)
            r["hours_to_exit"] = f"{bars_held * 0.25:.2f}"
            t_exit = datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc) + timedelta(hours=5)
            r["exit_time_pkt"] = t_exit.strftime("%Y-%m-%d %H:%M")
            r["pnl_pct"] = f"{pnl_pct:+.3f}"
            r["status"] = "DONE"
            mfe_pct = ((highest / entry) - 1) * 100
            r["mfe_pct"] = f"{mfe_pct:+.3f}"
            r["mae_pct"] = f"{mae_pct:+.3f}"
            r["highest_seen"] = f"{highest:.10g}"
            r["current_sl"] = f"{current_sl:.10g}"
            updated = True
            notify_result(sym, r, result, exit_price, pnl_pct, mfe_pct, mae_pct)
        else:
            r["highest_seen"] = f"{highest:.10g}"
            r["current_sl"] = f"{current_sl:.10g}"
            r["tp_triggered"] = "1" if tp_hit else "0"
            r["bars_since_tp"] = str(bars_held - (tp_hit_bar + 1) if tp_hit else 0)
            r["mae_pct"] = f"{mae_pct:+.3f}"
            updated = True

    return updated

def scan_market():
    print("="*50)
    print("🚀 MEGA BOT v8 — P1 + P8 + P21-v5 + P21-v6 + P22 + P23")
    print("="*50)
    print(f"   Coins: {len(ALL_SYMBOLS)}")
    print(f"   P1: {len(P1_COINS)} | P8: {len(P8_COINS)} | P21-v5: {len(P21_V5_COINS)} | P21-v6: {len(P21_V6_COINS)} | P22: {len(P22_COINS)} | P23: {len(P23_COINS)}")

    print("⏳ Waiting 45s...")
    time.sleep(45)

    coin_data = {}
    for sym in ALL_SYMBOLS:
        df = fetch_candles(sym)
        if df is not None:
            coin_data[sym] = add_indicators(df)

    existing_rows = load_log()

    if update_outcomes(coin_data, existing_rows):
        save_log(existing_rows)
        print("📒 Outcomes updated")

    new_signals = 0
    seen_keys = {(r.get('symbol'), safe_int(r.get('signal_ts_ms', 0)))
                 for r in existing_rows if r.get('signal_ts_ms')}

    for sym, df in coin_data.items():
        n = len(df)
        for offset in range(1, CANDLES_TO_CHECK + 1):
            idx = n - offset
            if idx < 200: continue
            sig = check_signals_at(sym, df, idx)
            if not sig: continue

            key = (sym, sig['ts'])
            if key in seen_keys: continue

            if not can_open_trade(sym, sig['ts'], existing_rows):
                print(f"⏸ {sym} skipped")
                continue

            t_utc = datetime.fromtimestamp(sig['ts'] / 1000, tz=timezone.utc)
            t_pkt = t_utc + timedelta(hours=5)

            new_row = {
                "signal_ts_ms": str(sig['ts']),
                "signal_time_utc": t_utc.strftime("%Y-%m-%d %H:%M"),
                "signal_time_pkt": t_pkt.strftime("%Y-%m-%d %H:%M"),
                "symbol": sym, "pattern": sig['pattern'],
                "entry_close": f"{sig['entry_close']:.10g}",
                "entry_open": "",
                "tp_pct": str(sig['tp']), "sl_pct": str(sig['sl']),
                "tp_price": "", "sl_price": "",
                "tp_triggered": "0", "tp_trigger_ts": "",
                "highest_seen": "", "current_sl": "",
                "bars_since_tp": "0",
                "exit_price": "", "exit_ts_ms": "",
                "exit_time_pkt": "", "bars_held": "", "hours_to_exit": "",
                "mfe_pct": "", "mae_pct": "", "bars_to_mfe": "", "bars_to_mae": "",
                "pnl_pct": "", "result": "", "status": "PENDING"
            }
            existing_rows.append(new_row)
            seen_keys.add(key)

            sig["symbol"] = sym
            notify_signal(sig, t_pkt)
            print(f"🚨 SIGNAL: {sig['pattern']} {sym} @ {sig['entry_close']:.6g}")
            new_signals += 1

    if new_signals > 0:
        save_log(existing_rows)

    print(f"✅ Scan complete. New signals: {new_signals}")

if __name__ == "__main__":
    scan_market()
