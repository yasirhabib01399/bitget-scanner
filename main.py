import csv
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ─── TELEGRAM ───
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ENABLE_TELEGRAM = True

# ─── CONFIG ───
API_URL = "https://api.bitget.com/api/v2/spot/market/candles"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SIGNAL_LOG_FILE = "signals_log.csv"

SYMBOLS_P1 = ["BTCUSDT", "SOLUSDT", "ADAUSDT", "NEARUSDT", "LTCUSDT", "BNBUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT", "DOTUSDT", "DOGEUSDT", "TIAUSDT"]
SYMBOLS_P8 = ["OPUSDT", "ADAUSDT", "SOLUSDT", "LTCUSDT", "ETHUSDT", "SEIUSDT"]
ALL_SYMBOLS = list(set(SYMBOLS_P1 + SYMBOLS_P8))

CANDLES_TO_CHECK = 3
TIME_STOP = 20            # Backtest matched
COOLDOWN_BARS = 8         # Backtest matched (2 hours)
FEE_PCT = 0.2             # Round-trip fee for P&L calc

# ─── CSV FIELDS (FIX: Added exit_price, pnl_pct, entry_open, tp_pct, sl_pct, exit_ts_ms) ───
CSV_FIELDS = [
    "signal_ts_ms", "signal_time_utc", "signal_time_pkt", 
    "symbol", "pattern",
    "entry_close",       # Signal candle close (reference)
    "entry_open",        # Actual entry = next candle open (updated on next scan)
    "tp_pct", "sl_pct",
    "tp_price", "sl_price",
    "exit_price",        # NEW: Actual exit price
    "exit_ts_ms",        # NEW: For cooldown tracking
    "exit_time_pkt",
    "result", "bars_held", "hours_to_exit",
    "pnl_pct",           # NEW: Net P&L after fee
    "status"             # PENDING / OPEN / DONE
]

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def send_telegram_message(message):
    if not ENABLE_TELEGRAM or not TELEGRAM_BOT_TOKEN:
        return
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
    df['rsi'] = calc_rsi(close, 14)
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    df['adx'] = calc_adx(df, 14)
    df['bb_mid'] = close.rolling(20).mean()
    df['bb_std'] = close.rolling(20).std()
    df['bb_lower'] = df['bb_mid'] - (2.0 * df['bb_std'])
    df['ema20'] = close.ewm(span=20, adjust=False).mean()
    df['atr20'] = calc_atr(df, 20)
    df['kc_lower'] = df['ema20'] - (2.0 * df['atr20'])
    return df

# ═══════════════════════════════════════════════════════════════════
# SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════════
def check_signals_at(symbol, df, idx):
    if idx < 200 or idx >= len(df): return None
    row = df.iloc[idx]
    if pd.isna(row['rsi']) or pd.isna(row['ema200']) or pd.isna(row['adx']): return None
    if pd.isna(row['bb_lower']) or pd.isna(row['kc_lower']): return None
    
    signals = []
    if symbol in SYMBOLS_P1:
        if (row['close'] < row['bb_lower'] and row['rsi'] < 32.0 and 
            row['close'] > row['ema200'] and row['adx'] < 50.0):
            signals.append({"pattern": "P1", "tp": 1.2, "sl": 1.5, "ts": row['ts'], "entry_close": row['close']})
    if symbol in SYMBOLS_P8:
        if (row['close'] < row['kc_lower'] and row['rsi'] < 35.0 and 
            row['close'] > row['ema200'] and row['adx'] < 50.0):
            signals.append({"pattern": "P8", "tp": 1.0, "sl": 1.5, "ts": row['ts'], "entry_close": row['close']})
    
    if not signals: return None
    return next((s for s in signals if s["pattern"] == "P1"), signals[0])

# ═══════════════════════════════════════════════════════════════════
# CSV LOG
# ═══════════════════════════════════════════════════════════════════
def load_log():
    if not os.path.exists(SIGNAL_LOG_FILE): return []
    with open(SIGNAL_LOG_FILE, mode='r') as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get('pattern')]

def save_log(rows):
    with open(SIGNAL_LOG_FILE, mode='w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore', restval='')
        w.writeheader()
        for r in rows: w.writerow(r)

# ═══════════════════════════════════════════════════════════════════
# COOLDOWN CHECK (FIX #4)
# ═══════════════════════════════════════════════════════════════════
def can_open_trade(sym, sig_ts, existing_rows):
    """New trade only if: (1) no PENDING/OPEN trade, (2) last DONE exit >= 8 bars (2hrs) ago"""
    cooldown_ms = COOLDOWN_BARS * 15 * 60 * 1000
    last_exit_ts = None
    has_open = False
    
    for r in existing_rows:
        if r.get("symbol") != sym: continue
        status = r.get("status", "")
        if status in ("PENDING", "OPEN"):
            has_open = True
            break
        if status == "DONE" and r.get("exit_ts_ms"):
            ets = int(r["exit_ts_ms"])
            if last_exit_ts is None or ets > last_exit_ts:
                last_exit_ts = ets
    
    if has_open: return False
    if last_exit_ts is not None and (sig_ts - last_exit_ts) < cooldown_ms:
        return False
    return True

# ═══════════════════════════════════════════════════════════════════
# OUTCOME TRACKING (FIX #1, #2, #3)
# ═══════════════════════════════════════════════════════════════════
def notify_result(sym, trade, result, exit_price, pnl_pct):
    emoji = "✅" if result == "TP" else ("🛑" if result == "SL" else "⏱")
    msg = (
        f"{emoji} <b>{result} HIT</b> #{sym} ({trade['pattern']})\n"
        f"🕐 Signal: {trade['signal_time_pkt']} PKT\n"
        f"🎯 Entry: {float(trade['entry_open']):.8g}\n"
        f"🚪 Exit: {exit_price:.8g}\n"
        f"💰 P&L: <b>{pnl_pct:+.3f}%</b> (after 0.2% fee)\n"
        f"⏱ Bars: {trade['bars_held']} ({trade['hours_to_exit']}h)"
    )
    send_telegram_message(msg)

def update_outcomes(coin_data, existing_rows):
    """
    Handle PENDING → OPEN (next candle open available)
    Handle OPEN → DONE (TP/SL/timeout, max 20 bars)
    """
    updated = False
    
    for r in existing_rows:
        sym = r["symbol"]
        status = r.get("status", "")
        if status not in ("PENDING", "OPEN"): continue
        if sym not in coin_data: continue
        
        df = coin_data[sym]
        sig_ts = int(r["signal_ts_ms"])
        
        match = df[df['ts'] == sig_ts]
        if match.empty:
            # Signal candle fell out of data window — cannot resolve
            r["result"] = "UNKNOWN"
            r["status"] = "DONE"
            updated = True
            continue
        idx = match.index[0]
        
        # ⭐ FIX #3: PENDING → OPEN (entry = next candle's open, matches backtest)
        if status == "PENDING":
            if idx + 1 >= len(df):
                continue  # Next candle not complete yet
            next_open = float(df.iloc[idx + 1]['open'])
            tp_pct = float(r["tp_pct"])
            sl_pct = float(r["sl_pct"])
            
            r["entry_open"] = f"{next_open:.10g}"
            r["tp_price"] = f"{next_open * (1 + tp_pct/100):.10g}"
            r["sl_price"] = f"{next_open * (1 - sl_pct/100):.10g}"
            r["status"] = "OPEN"
            updated = True
        
        if r["status"] != "OPEN": continue
        
        entry = float(r["entry_open"])
        tp = float(r["tp_price"])
        sl = float(r["sl_price"])
        
        # ⭐ FIX #1: Only check 20 bars from entry (candle N+1 to N+20)
        future_bars = df.iloc[idx+1 : idx+21]
        if len(future_bars) == 0:
            continue
        
        result, exit_ts, bars_held, exit_price = None, None, 0, None
        
        for i, (_, bar) in enumerate(future_bars.iterrows()):
            bars_held = i + 1
            if bar['low'] <= sl:
                result, exit_ts, exit_price = "SL", bar['ts'], sl
                break
            if bar['high'] >= tp:
                result, exit_ts, exit_price = "TP", bar['ts'], tp
                break
        
        # ⭐ FIX #1 & #2: Timeout with exit_price (20th bar close)
        if result is None:
            if len(future_bars) < TIME_STOP:
                continue  # Wait for all 20 bars
            result = "TIMEOUT"
            exit_ts = future_bars.iloc[TIME_STOP - 1]['ts']
            exit_price = future_bars.iloc[TIME_STOP - 1]['close']
            bars_held = TIME_STOP
        
        # ⭐ FIX #2: Calculate P&L
        pnl_pct = ((exit_price / entry) - 1) * 100 - FEE_PCT
        
        r["result"] = result
        r["exit_price"] = f"{exit_price:.10g}"
        r["exit_ts_ms"] = str(exit_ts)
        r["bars_held"] = str(bars_held)
        r["hours_to_exit"] = f"{bars_held * 0.25:.2f}"
        t_exit = datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc) + timedelta(hours=5)
        r["exit_time_pkt"] = t_exit.strftime("%Y-%m-%d %H:%M")
        r["pnl_pct"] = f"{pnl_pct:+.3f}"
        r["status"] = "DONE"
        updated = True
        notify_result(sym, r, result, exit_price, pnl_pct)
    
    return updated

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def scan_market():
    print("="*50)
    print("🚀 FINAL MEGA BOT SCANNER (P1 + P8) v2")
    print("="*50)
    
    print("⏳ Waiting 45s for candle to finalize...")
    time.sleep(45)
    
    coin_data = {}
    for sym in ALL_SYMBOLS:
        df = fetch_candles(sym)
        if df is not None:
            coin_data[sym] = add_indicators(df)
    
    existing_rows = load_log()
    
    # Phase 1: Update pending/open trades
    if update_outcomes(coin_data, existing_rows):
        save_log(existing_rows)
        print("📒 Outcomes updated")
    
    # Phase 2: Check new signals
    new_signals = 0
    for sym, df in coin_data.items():
        n = len(df)
        for offset in range(1, CANDLES_TO_CHECK + 1):
            idx = n - offset
            if idx < 200: continue
            
            sig = check_signals_at(sym, df, idx)
            if not sig: continue
            
            # Dedup
            key = (sym, sig['ts'])
            if any(r.get('symbol') == sym and int(r.get('signal_ts_ms', 0)) == sig['ts'] for r in existing_rows):
                continue
            
            # ⭐ FIX #4: Cooldown + no overlapping trades
            if not can_open_trade(sym, sig['ts'], existing_rows):
                print(f"⏸ {sym} skipped (cooldown or open trade)")
                continue
            
            t_utc = datetime.fromtimestamp(sig['ts'] / 1000, tz=timezone.utc)
            t_pkt = t_utc + timedelta(hours=5)
            
            new_row = {
                "signal_ts_ms": str(sig['ts']),
                "signal_time_utc": t_utc.strftime("%Y-%m-%d %H:%M"),
                "signal_time_pkt": t_pkt.strftime("%Y-%m-%d %H:%M"),
                "symbol": sym, "pattern": sig['pattern'],
                "entry_close": f"{sig['entry_close']:.10g}",
                "entry_open": "",  # Will be set on next scan
                "tp_pct": str(sig['tp']), "sl_pct": str(sig['sl']),
                "tp_price": "", "sl_price": "",  # Set on next scan
                "exit_price": "", "exit_ts_ms": "",
                "exit_time_pkt": "", "bars_held": "", "hours_to_exit": "",
                "pnl_pct": "", "result": "", "status": "PENDING"
            }
            existing_rows.append(new_row)
            
            msg = (
                f"🚨 <b>{sig['pattern']} SIGNAL DETECTED</b>\n\n"
                f"🪙 <b>Symbol:</b> #{sym}\n"
                f"⏰ <b>Signal Time:</b> {t_pkt.strftime('%Y-%m-%d %H:%M')} PKT\n"
                f"🎯 <b>Signal Close:</b> {sig['entry_close']:.8g}\n"
                f"⏳ <b>Status:</b> PENDING (entry at next candle open)\n"
                f"📊 <b>Est. TP:</b> +{sig['tp']}% | <b>SL:</b> -{sig['sl']}%\n"
                f"\n✅ TP/SL will be confirmed on next scan."
            )
            send_telegram_message(msg)
            print(f"🚨 SIGNAL: {sig['pattern']} {sym} @ {sig['entry_close']:.6g} (PENDING)")
            new_signals += 1
    
    if new_signals > 0:
        save_log(existing_rows)
    
    print(f"✅ Scan complete. New signals: {new_signals}")

if __name__ == "__main__":
    scan_market()
