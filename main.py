# ═══════════════════════════════════════════════════════════════════
# MEGA BOT v9 — P1 + P8 + P21-v5 + P21-v6 + P22 + P23 + P25
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
P23_COINS = ["DOTUSDT", "SOLUSDT", "NEARUSDT"]
# 🆕 P25 Coins (Momentum Continuation)
P25_COINS = ["BTCUSDT","SOLUSDT","ADAUSDT","NEARUSDT","LTCUSDT",
             "BNBUSDT","UNIUSDT","APTUSDT","ARBUSDT","DOTUSDT",
             "DOGEUSDT","TIAUSDT","OPUSDT","ETHUSDT","SEIUSDT"]

ALL_SYMBOLS = list(set(P1_COINS + P8_COINS + P21_V5_COINS + P21_V6_COINS + P22_COINS + P23_COINS + P25_COINS))

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
P23_TIME_STOP = 12
P23_TP = 1.2
P23_SL = 1.2
# 🆕 P25 Config
P25_TIME_STOP = 12
P25_TP = 1.2
P25_SL = 1.5

# Priority
PATTERN_PRIORITY = {"P21-v5": 7, "P22": 6, "P25": 5, "P21-v6": 4, "P23": 3, "P1": 2, "P8": 1}

CSV_FIELDS = [
    "signal_ts_ms", "signal_time_utc", "signal_time_pkt",
    "symbol", "pattern", "entry_close", "entry_open", "tp_pct", "sl_pct",
    "tp_price", "sl_price", "tp_triggered", "tp_trigger_ts",
    "highest_seen", "current_sl", "bars_since_tp",
    "exit_price", "exit_ts_ms", "exit_time_pkt", "result", "bars_held",
    "hours_to_exit", "mfe_pct", "mae_pct", "bars_to_mfe", "bars_to_mae",
    "pnl_pct", "status"
]

def safe_int(val):
    if val is None or val == "": return 0
    try: return int(float(val))
    except: return 0

def safe_float(val, default=0.0):
    if val is None or val == "": return default
    try: return float(val)
    except: return default

def send_telegram_message(message):
    if not ENABLE_TELEGRAM or not TELEGRAM_BOT_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e: print(f"❌ Telegram Error: {e}")

def fetch_candles(symbol):
    params = {"symbol": symbol, "granularity": "15min", "limit": "300"}
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        data = r.json()
        if data.get("code") != "00000": return None
        rows = data.get("data", [])
        if len(rows) < 250: return None
        df = pd.DataFrame([{"ts": int(x[0]), "open": float(x[1]), "high": float(x[2]),
            "low": float(x[3]), "close": float(x[4]), "volume": float(x[5])} for x in rows])
        df = df.sort_values("ts").reset_index(drop=True)
        if df.iloc[-1]["ts"] + (15 * 60 * 1000) > int(time.time() * 1000): df = df.iloc[:-1]
        return df
    except Exception as e:
        print(f"❌ Fetch Error {symbol}: {e}")
        return None

def calc_rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, adjust=False).mean()
    return 100 - (100 / (1 + ag/al.replace(0, np.nan)))

def calc_adx(df, n=14):
    h, lo, c = df['high'], df['low'], df['close']
    up = h.diff(); dn = -lo.diff()
    pdm = np.where((up>dn)&(up>0), up, 0.0)
    mdm = np.where((dn>up)&(dn>0), dn, 0.0)
    tr = pd.concat([h-lo, (h-c.shift()).abs(), (lo-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*pd.Series(pdm,index=h.index).ewm(alpha=1/n,adjust=False).mean()/atr.replace(0,np.nan)
    mdi = 100*pd.Series(mdm,index=h.index).ewm(alpha=1/n,adjust=False).mean()/atr.replace(0,np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean()

def calc_atr(df, n=20):
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    c = df['close']
    df['rsi14'] = calc_rsi(c, 14)
    df['rsi2'] = calc_rsi(c, 2)
    df['ema200'] = c.ewm(span=200, adjust=False).mean()
    df['ema50'] = c.ewm(span=50, adjust=False).mean()
    df['ema20'] = c.ewm(span=20, adjust=False).mean()
    df['adx'] = calc_adx(df, 14)
    df['bb_mid'] = c.rolling(20).mean()
    df['bb_std'] = c.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2.0 * df['bb_std'])
    df['bb_lower'] = df['bb_mid'] - (2.0 * df['bb_std'])
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_pos'] = (c - df['bb_lower']) / bb_range.replace(0, np.nan)
    df['atr20'] = calc_atr(df, 20)
    df['kc_lower'] = df['ema20'] - (2.0 * df['atr20'])
    df['swing_low_20'] = df['low'].rolling(20).min().shift(1)
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20'].replace(0, np.nan)
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    df['prev_3_high'] = df['high'].rolling(3).max().shift(1)
    df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200'] * 100
    return df

def check_p1(row):
    if any(pd.isna(row[k]) for k in ["rsi14","ema200","adx","bb_lower"]): return False
    return (row['close'] < row['bb_lower'] and row['rsi14'] < 32.0 and row['close'] > row['ema200'] and row['adx'] < 50.0)

def check_p8(row):
    if any(pd.isna(row[k]) for k in ["rsi14","ema200","adx","kc_lower"]): return False
    return (row['close'] < row['kc_lower'] and row['rsi14'] < 35.0 and row['close'] > row['ema200'] and row['adx'] < 50.0)

def check_p21_v5(df, idx):
    if idx < 1: return False
    r = df.iloc[idx]; prev = df.iloc[idx-1]
    if any(pd.isna(r[k]) for k in ["ema200","rsi14","swing_low_20","vol_ratio"]): return False
    return ((r["low"] < r["swing_low_20"]) and (r["close"] > r["swing_low_20"]) and (r["close"] > r["open"]) and (r["close"] > prev["high"]) and (r["close"] < r["ema200"]) and ((r["close"]-r["ema200"])/r["ema200"]*100 > -15) and (r["rsi14"] < 40) and (r["vol_ratio"] >= 1.5))

def check_p21_v6(df, idx):
    if idx < 1: return False
    r = df.iloc[idx]; prev = df.iloc[idx-1]
    if any(pd.isna(r[k]) for k in ["rsi14","swing_low_20","vol_ratio"]): return False
    return ((r["low"] < r["swing_low_20"]) and (r["close"] > r["swing_low_20"]) and (r["close"] > r["open"]) and (r["close"] > prev["high"]) and (r["rsi14"] < 40) and (r["vol_ratio"] >= 1.2))

def check_p22(df, idx):
    if idx < 25: return False
    r = df.iloc[idx]
    if any(pd.isna(r[k]) for k in ["ema200","rsi14","swing_low_20","vol_ratio","range_pct","prev_3_high"]): return False
    recent_sweep = False
    for k in range(idx-10, idx):
        if df.iloc[k]["low"] < df.iloc[k]["swing_low_20"]: recent_sweep = True; break
    return (recent_sweep and df["range_pct"].iloc[idx-3:idx].max() < 1.5 and r["close"] > r["prev_3_high"] and r["vol_ratio"] >= 1.5 and 30 < r["rsi14"] < 50 and r["dist_ema200"] > -15)

def check_p23(df, idx):
    if idx < 1: return False
    r = df.iloc[idx]; p = df.iloc[idx-1]
    if any(pd.isna(r[k]) for k in ["rsi14","adx","bb_pos","vol_ratio","ema50","ema200"]): return False
    if any(pd.isna(p[k]) for k in ["rsi2"]): return False
    return (r['close'] > r['ema200'] and r['close'] > r['ema50'] and r['adx'] < 25.0 and p['rsi2'] < 25.0 and r['close'] > r['open'] and r['close'] > p['high'] and r['bb_pos'] > 0.2 and r['rsi14'] > 40.0 and r['vol_ratio'] > 0.5)

# 🆕 P25 Logic (Momentum Continuation)
def check_p25(df, idx):
    r = df.iloc[idx]
    if any(pd.isna(r[k]) for k in ["rsi14", "rsi2", "adx", "bb_pos", "vol_ratio", "dist_ema200"]): return False
    
    # Signature: ADX > 30, RSI14 > 55, dist_ema200 > 2, RSI2 < 40, BB_pos > 0.5
    if r['adx'] < 30.0: return False
    if r['rsi14'] < 55.0: return False
    if r['dist_ema200'] < 2.0 or r['dist_ema200'] > 10.0: return False
    if r['rsi2'] > 40.0: return False
    if r['bb_pos'] < 0.5: return False
    if r['vol_ratio'] < 0.8: return False
    if r['close'] <= r['open']: return False # Green candle
    
    return True

def check_signals_at(symbol, df, idx):
    if idx < 200 or idx >= len(df): return None
    row = df.iloc[idx]
    candidates = []

    if symbol in P1_COINS and check_p1(row): candidates.append({"pattern":"P1", "tp":1.2, "sl":2.5, "time_stop":TIME_STOP, "ts":row['ts'], "entry_close":row['close']})
    if symbol in P8_COINS and check_p8(row): candidates.append({"pattern":"P8", "tp":1.0, "sl":2.5, "time_stop":TIME_STOP, "ts":row['ts'], "entry_close":row['close']})
    if symbol in P21_V5_COINS and check_p21_v5(df, idx): candidates.append({"pattern":"P21-v5", "tp":P21_TP, "sl":P21_SL, "time_stop":P21_TIME_STOP, "ts":row['ts'], "entry_close":row['close']})
    if symbol in P21_V6_COINS and check_p21_v6(df, idx): candidates.append({"pattern":"P21-v6", "tp":P21_TP, "sl":P21_SL, "time_stop":P21_TIME_STOP, "ts":row['ts'], "entry_close":row['close']})
    if symbol in P22_COINS and check_p22(df, idx): candidates.append({"pattern":"P22", "tp":P22_TP, "sl":P22_SL, "time_stop":P22_TIME_STOP, "ts":row['ts'], "entry_close":row['close']})
    if symbol in P23_COINS and check_p23(df, idx): candidates.append({"pattern":"P23", "tp":P23_TP, "sl":P23_SL, "time_stop":P23_TIME_STOP, "ts":row['ts'], "entry_close":row['close']})
    if symbol in P25_COINS and check_p25(df, idx): candidates.append({"pattern":"P25", "tp":P25_TP, "sl":P25_SL, "time_stop":P25_TIME_STOP, "ts":row['ts'], "entry_close":row['close']})

    if not candidates: return None
    return max(candidates, key=lambda x: PATTERN_PRIORITY[x["pattern"]])

def load_log():
    if not os.path.exists(SIGNAL_LOG_FILE): return []
    try:
        with open(SIGNAL_LOG_FILE, mode='r', encoding='utf-8') as f: return [r for r in csv.DictReader(f) if r.get('pattern')]
    except: return []

def save_log(rows):
    try:
        with open(SIGNAL_LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore', restval='')
            w.writeheader()
            for r in rows: w.writerow(r)
    except Exception as e: print(f"⚠️ CSV save error: {e}")

def can_open_trade(sym, sig_ts, existing_rows):
    cooldown_ms = COOLDOWN_BARS * 15 * 60 * 1000
    last_exit_ts = None; has_open = False
    for r in existing_rows:
        if r.get("symbol") != sym: continue
        status = r.get("status", "")
        if status in ("PENDING", "OPEN"): has_open = True; break
        if status == "DONE" and r.get("exit_ts_ms"):
            ets = safe_int(r["exit_ts_ms"])
            if ets > 0 and (last_exit_ts is None or ets > last_exit_ts): last_exit_ts = ets
    if has_open: return False
    if last_exit_ts is not None and (sig_ts - last_exit_ts) < cooldown_ms: return False
    return True

def notify_signal(sig, t_pkt):
    msg = f"🚨 <b>{sig['pattern']} SIGNAL</b>\n\n🪙 <b>Symbol:</b> {sig['symbol']}\n⏰ <b>Time:</b> {t_pkt.strftime('%Y-%m-%d %H:%M')} PKT\n🎯 <b>Entry:</b> {sig['entry_close']:.8g}\n📊 <b>TP:</b> +{sig['tp']}% | <b>SL:</b> -{sig['sl']}%\n\n🔥 <b>TP-Trail:</b> 0.3%"
    send_telegram_message(msg)

def notify_tp_trigger(sym, trade, highest, current_sl):
    msg = f"🎯 <b>TP HIT → TRAILING</b> {sym} ({trade['pattern']})\n🎯 Entry: {safe_float(trade['entry_open']):.8g}\n📈 High: {highest:.8g}\n🛡️ Trail SL: {current_sl:.8g} (+{((current_sl/safe_float(trade['entry_open']))-1)*100:.2f}%)"
    send_telegram_message(msg)

def notify_result(sym, trade, result, exit_price, pnl_pct, mfe, mae):
    emoji = "✅" if "TP" in result else ("🛑" if result == "SL" else "⏱")
    msg = f"{emoji} <b>{result}</b> #{sym} ({trade['pattern']})\n🚪 Exit: {exit_price:.8g}\n💰 P&L: <b>{pnl_pct:+.3f}%</b>\n📊 MFE: +{mfe:.2f}% | MAE: {mae:.2f}%"
    send_telegram_message(msg)

def update_outcomes(coin_data, existing_rows):
    updated = False
    for r in existing_rows:
        sym = r.get("symbol", ""); status = r.get("status", "")
        if status not in ("PENDING", "OPEN") or sym not in coin_data: continue
        df = coin_data[sym]; sig_ts = safe_int(r.get("signal_ts_ms", 0))
        if sig_ts == 0: continue
        match = df[df['ts'] == sig_ts]
        if match.empty: r["result"] = "UNKNOWN"; r["status"] = "DONE"; updated = True; continue
        idx = match.index[0]

        if status == "PENDING":
            if idx + 1 >= len(df): continue
            next_open = float(df.iloc[idx + 1]['open'])
            tp_pct = safe_float(r.get("tp_pct", 1.2)); sl_pct = safe_float(r.get("sl_pct", 2.5))
            r["entry_open"] = f"{next_open:.10g}"; r["tp_price"] = f"{next_open * (1 + tp_pct/100):.10g}"; r["sl_price"] = f"{next_open * (1 - sl_pct/100):.10g}"
            r["highest_seen"] = f"{next_open:.10g}"; r["current_sl"] = r["sl_price"]; r["status"] = "OPEN"; updated = True

        entry = safe_float(r.get("entry_open", 0)); tp = safe_float(r.get("tp_price", 0)); orig_sl = safe_float(r.get("sl_price", 0))
        if entry == 0 or tp == 0 or orig_sl == 0: continue

        pattern = r.get("pattern", "P1")
        time_stop = P21_TIME_STOP if pattern.startswith("P21") else (P22_TIME_STOP if pattern == "P22" else (P23_TIME_STOP if pattern == "P23" else (P25_TIME_STOP if pattern == "P25" else TIME_STOP)))
        future_bars = df.iloc[idx+1 : idx+time_stop+2]
        if len(future_bars) == 0: continue

        current_sl = orig_sl; highest = entry; tp_hit = False; tp_hit_bar = None; bars_held = 0; result = None; exit_ts = None; exit_price = None; mae_pct = 0.0
        was_tp_triggered = safe_int(r.get("tp_triggered", 0)) == 1

        for i, (_, bar) in enumerate(future_bars.iterrows()):
            bars_held = i + 1; bar_ts = bar['ts']; bar_high = bar['high']; bar_low = bar['low']; bar_close = bar['close']
            bar_mae = ((bar_low / entry) - 1) * 100
            if bar_mae < mae_pct: mae_pct = bar_mae

            if not tp_hit:
                if bar_low <= current_sl: result, exit_ts, exit_price = "SL", bar_ts, current_sl; break
                if bar_high >= tp:
                    tp_hit = True; tp_hit_bar = i; highest = bar_high; current_sl = max(tp, highest * (1 - TRAIL_PCT/100))
                    if not was_tp_triggered: notify_tp_trigger(sym, r, highest, current_sl); was_tp_triggered = True; r["tp_triggered"] = "1"; r["tp_trigger_ts"] = str(int(bar_ts))
                    continue
                if bars_held >= time_stop: result, exit_ts, exit_price = "TIMEOUT", bar_ts, bar_close; break
            else:
                bars_since = i - tp_hit_bar; highest = max(highest, bar_high); new_sl = highest * (1 - TRAIL_PCT/100); current_sl = max(current_sl, new_sl)
                if bar_low <= current_sl: result, exit_ts, exit_price = "TP_TRAIL", bar_ts, current_sl; break
                if bars_since >= MAX_BARS_AFTER_TP: result, exit_ts, exit_price = "TP_TRAIL_TIMEOUT", bar_ts, bar_close; break

        if result:
            pnl_pct = ((exit_price / entry) - 1) * 100 - FEE_PCT
            r["result"] = result; r["exit_price"] = f"{exit_price:.10g}"; r["exit_ts_ms"] = str(int(exit_ts)); r["bars_held"] = str(bars_held); r["hours_to_exit"] = f"{bars_held * 0.25:.2f}"
            r["exit_time_pkt"] = (datetime.fromtimestamp(exit_ts/1000, tz=timezone.utc) + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
            r["pnl_pct"] = f"{pnl_pct:+.3f}"; r["status"] = "DONE"
            mfe_pct = ((highest / entry) - 1) * 100
            r["mfe_pct"] = f"{mfe_pct:+.3f}"; r["mae_pct"] = f"{mae_pct:+.3f}"; r["highest_seen"] = f"{highest:.10g}"; r["current_sl"] = f"{current_sl:.10g}"
            updated = True; notify_result(sym, r, result, exit_price, pnl_pct, mfe_pct, mae_pct)
        else:
            r["highest_seen"] = f"{highest:.10g}"; r["current_sl"] = f"{current_sl:.10g}"; r["tp_triggered"] = "1" if tp_hit else "0"; r["bars_since_tp"] = str(bars_held - (tp_hit_bar + 1) if tp_hit else 0); r["mae_pct"] = f"{mae_pct:+.3f}"; updated = True

    return updated

def scan_market():
    print("="*50)
    print("🚀 MEGA BOT v9 — P1 + P8 + P21 + P22 + P23 + P25")
    print("="*50)
    print("⏳ Waiting 45s...")
    time.sleep(45)

    coin_data = {}
    for sym in ALL_SYMBOLS:
        df = fetch_candles(sym)
        if df is not None: coin_data[sym] = add_indicators(df)

    existing_rows = load_log()
    if update_outcomes(coin_data, existing_rows): save_log(existing_rows); print("📒 Outcomes updated")

    new_signals = 0
    seen_keys = {(r.get('symbol'), safe_int(r.get('signal_ts_ms', 0))) for r in existing_rows if r.get('signal_ts_ms')}

    for sym, df in coin_data.items():
        n = len(df)
        for offset in range(1, CANDLES_TO_CHECK + 1):
            idx = n - offset
            if idx < 200: continue
            sig = check_signals_at(sym, df, idx)
            if not sig or (sym, sig['ts']) in seen_keys: continue

            if not can_open_trade(sym, sig['ts'], existing_rows): continue

            t_utc = datetime.fromtimestamp(sig['ts'] / 1000, tz=timezone.utc)
            t_pkt = t_utc + timedelta(hours=5)

            new_row = {
                "signal_ts_ms": str(sig['ts']), "signal_time_utc": t_utc.strftime("%Y-%m-%d %H:%M"), "signal_time_pkt": t_pkt.strftime("%Y-%m-%d %H:%M"),
                "symbol": sym, "pattern": sig['pattern'], "entry_close": f"{sig['entry_close']:.10g}", "entry_open": "",
                "tp_pct": str(sig['tp']), "sl_pct": str(sig['sl']), "tp_price": "", "sl_price": "",
                "tp_triggered": "0", "tp_trigger_ts": "", "highest_seen": "", "current_sl": "", "bars_since_tp": "0",
                "exit_price": "", "exit_ts_ms": "", "exit_time_pkt": "", "bars_held": "", "hours_to_exit": "",
                "mfe_pct": "", "mae_pct": "", "bars_to_mfe": "", "bars_to_mae": "", "pnl_pct": "", "result": "", "status": "PENDING"
            }
            existing_rows.append(new_row); seen_keys.add((sym, sig['ts']))

            sig["symbol"] = sym
            notify_signal(sig, t_pkt)
            print(f"🚨 SIGNAL: {sig['pattern']} {sym} @ {sig['entry_close']:.6g}")
            new_signals += 1

    if new_signals > 0: save_log(existing_rows)
    print(f"✅ Scan complete. New signals: {new_signals}")

if __name__ == "__main__":
    scan_market()
