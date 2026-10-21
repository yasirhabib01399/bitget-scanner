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

# Final Mega Bot Coins
SYMBOLS_P1 = ["BTCUSDT", "SOLUSDT", "ADAUSDT", "NEARUSDT", "LTCUSDT", "BNBUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT", "DOTUSDT", "DOGEUSDT", "TIAUSDT"]
SYMBOLS_P8 = ["OPUSDT", "ADAUSDT", "SOLUSDT", "LTCUSDT", "ETHUSDT", "SEIUSDT"]
ALL_SYMBOLS = list(set(SYMBOLS_P1 + SYMBOLS_P8))

# ─── LOG CSV FIELDS ───
CSV_FIELDS = [
    "signal_ts_ms", "signal_time_utc", "signal_time_pkt", 
    "symbol", "pattern", "entry_close", "tp_price", "sl_price", 
    "result", "hours_to_exit", "exit_time_pkt", "status"
]

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
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
    """Fetch 15m candles from Bitget, return DataFrame of last 300 candles (drop incomplete)"""
    params = {"symbol": symbol, "granularity": "15min", "limit": "300"}
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        data = r.json()
        if data.get("code") != "00000":
            return None
        rows = data.get("data", [])
        if len(rows) < 250:
            return None
            
        df = pd.DataFrame([{
            "ts": int(x[0]), "open": float(x[1]), "high": float(x[2]),
            "low": float(x[3]), "close": float(x[4]), "volume": float(x[5])
        } for x in rows])
        
        df = df.sort_values("ts").reset_index(drop=True)
        
        # Drop last candle if incomplete
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

# ═══════════════════════════════════════════════════════════════════
# SIGNAL LOGIC
# ═══════════════════════════════════════════════════════════════════
def check_signals(symbol, df):
    if len(df) < 250: return None
    
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
    
    row = df.iloc[-1]
    if pd.isna(row['rsi']) or pd.isna(row['ema200']) or pd.isna(row['adx']) or pd.isna(row['bb_lower']) or pd.isna(row['kc_lower']):
        return None
    
    signals = []
    
    # P1 Logic
    if symbol in SYMBOLS_P1:
        if (row['close'] < row['bb_lower'] and row['rsi'] < 32.0 and 
            row['close'] > row['ema200'] and row['adx'] < 50.0):
            signals.append({"pattern": "P1", "tp": 1.2, "sl": 1.5, "ts": row['ts'], "entry": row['close']})
            
    # P8 Logic
    if symbol in SYMBOLS_P8:
        if (row['close'] < row['kc_lower'] and row['rsi'] < 35.0 and 
            row['close'] > row['ema200'] and row['adx'] < 50.0):
            signals.append({"pattern": "P8", "tp": 1.0, "sl": 1.5, "ts": row['ts'], "entry": row['close']})
            
    if not signals: return None
    
    # Deduplication: P1 preferred over P8
    return next((s for s in signals if s["pattern"] == "P1"), signals[0])

# ═══════════════════════════════════════════════════════════════════
# CSV TRACKING
# ═══════════════════════════════════════════════════════════════════
def load_log():
    if not os.path.exists(SIGNAL_LOG_FILE): return []
    with open(SIGNAL_LOG_FILE, mode='r') as f:
        return list(csv.DictReader(f))

def save_log(rows):
    with open(SIGNAL_LOG_FILE, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def notify_result(sym, trade, result):
    emoji = "✅" if result == "TP" else ("🛑" if result == "SL" else "⏱")
    tp = float(trade['tp_price'])
    sl = float(trade['sl_price'])
    entry = float(trade['entry_close'])
    
    msg = (
        f"{emoji} <b>{result} HIT</b> #{sym} ({trade['pattern']})\n"
        f"🕐 Signal: {trade['signal_time_pkt']} PKT\n"
        f"🎯 Entry: {entry:.8g}\n"
    )
    if result == "TP":
        msg += f"✅ TP {tp:.8g} touch hua"
    elif result == "SL":
        msg += f"🛑 SL {sl:.8g} touch hua"
    else:
        msg += f"⏱ Timeout (20 bars) - Exit at market close"
        
    msg += f"\n⏱ Duration: {trade['hours_to_exit']} hours"
    send_telegram_message(msg)

def update_outcomes(coin_data):
    rows = load_log()
    open_trades = [r for r in rows if r.get("status") == "OPEN"]
    if not open_trades: return False
    
    updated = False
    for r in open_trades:
        sym = r["symbol"]
        if sym not in coin_data: continue
        
        df = coin_data[sym]
        sig_ts = int(r["signal_ts_ms"])
        entry = float(r["entry_close"])
        tp = float(r["tp_price"])
        sl = float(r["sl_price"])
        
        match = df[df['ts'] == sig_ts]
        if match.empty: continue
        idx = match.index[0]
        
        future_bars = df.iloc[idx+1:]
        if len(future_bars) == 0: continue
        
        result = None
        exit_ts = None
        bars_held = 0
        
        for i, (_, bar) in enumerate(future_bars.iterrows()):
            bars_held = i + 1
            if bar['low'] <= sl:
                result, exit_ts = "SL", bar['ts']; break
            if bar['high'] >= tp:
                result, exit_ts = "TP", bar['ts']; break
                
        if result is None and bars_held >= 20:
            result = "TIMEOUT"
            exit_ts = future_bars.iloc[19]['ts']
            
        if result:
            r["result"] = result
            r["hours_to_exit"] = f"{bars_held * 0.25:.2f}"
            t_exit = datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc) + timedelta(hours=5)
            r["exit_time_pkt"] = t_exit.strftime("%Y-%m-%d %H:%M")
            r["status"] = "DONE"
            updated = True
            notify_result(sym, r, result)
            
    if updated:
        save_log(rows)
    return updated

# ═══════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ═══════════════════════════════════════════════════════════════════
def scan_market():
    print("="*50)
    print("🚀 FINAL MEGA BOT SCANNER (P1 + P8)")
    print("="*50)
    
    time.sleep(10) # Wait for API data to settle after candle close
    
    coin_data = {}
    for sym in ALL_SYMBOLS:
        df = fetch_candles(sym)
        if df is not None:
            coin_data[sym] = df
            
    # 1. Update open trades
    update_outcomes(coin_data)
    
    # 2. Check new signals
    existing_rows = load_log()
    new_signals = 0
    
    for sym, df in coin_data.items():
        sig = check_signals(sym, df)
        if sig:
            # Check if already logged
            if any(r['symbol'] == sym and int(r['signal_ts_ms']) == sig['ts'] for r in existing_rows):
                continue
                
            tp_price = sig['entry'] * (1 + sig['tp']/100)
            sl_price = sig['entry'] * (1 - sig['sl']/100)
            
            t_utc = datetime.fromtimestamp(sig['ts'] / 1000, tz=timezone.utc)
            t_pkt = t_utc + timedelta(hours=5)
            
            # Save to CSV
            new_row = {
                "signal_ts_ms": str(sig['ts']),
                "signal_time_utc": t_utc.strftime("%Y-%m-%d %H:%M"),
                "signal_time_pkt": t_pkt.strftime("%Y-%m-%d %H:%M"),
                "symbol": sym, "pattern": sig['pattern'],
                "entry_close": f"{sig['entry']:.10g}",
                "tp_price": f"{tp_price:.10g}",
                "sl_price": f"{sl_price:.10g}",
                "result": "", "hours_to_exit": "", "exit_time_pkt": "", "status": "OPEN"
            }
            existing_rows.append(new_row)
            save_log(existing_rows)
            
            # Telegram Alert
            msg = (
                f"🚨 <b>{sig['pattern']} SIGNAL</b>\n\n"
                f"🪙 <b>Symbol:</b> #{sym}\n"
                f"⏰ <b>Time:</b> {t_pkt.strftime('%Y-%m-%d %H:%M')} PKT\n"
                f"🎯 <b>Entry:</b> {sig['entry']:.8g}\n"
                f"✅ <b>TP (+{sig['tp']}%):</b> {tp_price:.8g}\n"
                f"🛑 <b>SL (-{sig['sl']}%):</b> {sl_price:.8g}"
            )
            send_telegram_message(msg)
            print(f"🚨 NEW SIGNAL: {sig['pattern']} {sym} @ {sig['entry']:.6g}")
            new_signals += 1
            
    print(f"✅ Scan complete. New signals: {new_signals}")

if __name__ == "__main__":
    scan_market()
