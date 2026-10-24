# ═══════════════════════════════════════════════════════════════════
# MEGA BOT v12 — 9 Patterns (P1 + P8 + P21-v5 + P21-v6 + P23 + P25 + P27 + P28 + P29-ML)
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
P1_COINS = ["BTCUSDT","SOLUSDT","ADAUSDT","NEARUSDT","LTCUSDT","BNBUSDT","UNIUSDT","APTUSDT","ARBUSDT","DOTUSDT","DOGEUSDT","TIAUSDT"]
P8_COINS = ["OPUSDT","ADAUSDT","SOLUSDT","LTCUSDT","ETHUSDT","SEIUSDT"]
P21_V5_COINS = ["BTCUSDT","SOLUSDT","ADAUSDT","NEARUSDT","LTCUSDT","BNBUSDT","UNIUSDT","ARBUSDT","DOTUSDT","DOGEUSDT","TIAUSDT","ETHUSDT","SUIUSDT"]
P21_V6_COINS = P21_V5_COINS
P23_COINS = ["DOTUSDT", "SOLUSDT", "NEARUSDT"]
P25_COINS = ["BTCUSDT","SOLUSDT","ADAUSDT","NEARUSDT","LTCUSDT","BNBUSDT","UNIUSDT","APTUSDT","ARBUSDT","DOTUSDT","DOGEUSDT","TIAUSDT","OPUSDT","ETHUSDT","SEIUSDT"]
P27_COINS = ["BTCUSDT","SOLUSDT","ADAUSDT","NEARUSDT","LTCUSDT","BNBUSDT","UNIUSDT","APTUSDT","ARBUSDT","DOTUSDT","DOGEUSDT","TIAUSDT","OPUSDT","ETHUSDT","SEIUSDT"]
P28_COINS = ["UNIUSDT", "ETHUSDT", "SEIUSDT", "SOLUSDT"]
P29_COINS = ["ETHUSDT", "LTCUSDT", "APTUSDT", "DOGEUSDT", "SOLUSDT", "OPUSDT", "ARBUSDT", "ADAUSDT", "UNIUSDT"]

ALL_SYMBOLS = list(set(P1_COINS + P8_COINS + P21_V5_COINS + P21_V6_COINS + P23_COINS + P25_COINS + P27_COINS + P28_COINS + P29_COINS))

CANDLES_TO_CHECK = 3
TIME_STOP = 20
COOLDOWN_BARS = 8
FEE_PCT = 0.2
TRAIL_PCT = 0.3
MAX_BARS_AFTER_TP = 15

P21_TIME_STOP = 12; P21_TP = 1.5; P21_SL = 2.0
P23_TIME_STOP = 12; P23_TP = 1.2; P23_SL = 1.2
P25_TIME_STOP = 12; P25_TP = 1.2; P25_SL = 1.5
P27_TIME_STOP = 15; P27_TP = 1.5; P27_SL = 1.5
P28_TIME_STOP = 15; P28_TP = 1.5; P28_SL = 1.5
# 🆕 P29 ML Config (Wide TP/SL + 40 bars)
P29_TIME_STOP = 40; P29_TP = 2.5; P29_SL = 2.0

# Priority: P29 > P21-v5 > P25 > P27 > P28 > P21-v6 > P23 > P1 > P8
PATTERN_PRIORITY = {"P29-ML": 10, "P21-v5": 8, "P25": 7, "P27": 6, "P28": 5, "P21-v6": 4, "P23": 3, "P1": 2, "P8": 1}

CSV_FIELDS = [
    "signal_ts_ms", "signal_time_utc", "signal_time_pkt", "symbol", "pattern", "entry_close", "entry_open", 
    "tp_pct", "sl_pct", "tp_price", "sl_price", "tp_triggered", "tp_trigger_ts", "highest_seen", "current_sl", 
    "bars_since_tp", "exit_price", "exit_ts_ms", "exit_time_pkt", "result", "bars_held", "hours_to_exit", 
    "mfe_pct", "mae_pct", "bars_to_mfe", "bars_to_mae", "tp_hit_price", "tp_hit_bar", "pnl_pct", "status"
]

def safe_int(v):
    try: return int(float(v)) if v not in (None, "") else 0
    except: return 0
def safe_float(v, d=0.0):
    try: return float(v) if v not in (None, "") else d
    except: return d

def send_telegram_message(msg):
    if not ENABLE_TELEGRAM or not TELEGRAM_BOT_TOKEN: return
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e: print(f"❌ Telegram Error: {e}")

def fetch_candles(sym):
    try:
        r = requests.get(API_URL, params={"symbol": sym, "granularity": "15min", "limit": "300"}, headers=HEADERS, timeout=15).json()
        if r.get("code") != "00000": return None
        rows = r.get("data", [])
        if len(rows) < 250: return None
        df = pd.DataFrame([{"ts": int(x[0]), "open": float(x[1]), "high": float(x[2]), "low": float(x[3]), "close": float(x[4]), "volume": float(x[5])} for x in rows])
        df = df.sort_values("ts").reset_index(drop=True)
        if df.iloc[-1]["ts"] + 900000 > int(time.time() * 1000): df = df.iloc[:-1]
        return df
    except: return None

def calc_rsi(close, n):
    d = close.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/n, adjust=False).mean(); al = l.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + ag/al.replace(0, np.nan))

def calc_adx(df, n=14):
    h, lo, c = df['high'], df['low'], df['close']
    up = h.diff(); dn = -lo.diff()
    pdm = np.where((up>dn)&(up>0), up, 0.0); mdm = np.where((dn>up)&(dn>0), dn, 0.0)
    tr = pd.concat([h-lo, (h-c.shift()).abs(), (lo-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*pd.Series(pdm, index=h.index).ewm(alpha=1/n, adjust=False).mean()/atr.replace(0, np.nan)
    mdi = 100*pd.Series(mdm, index=h.index).ewm(alpha=1/n, adjust=False).mean()/atr.replace(0, np.nan)
    return (100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)).ewm(alpha=1/n, adjust=False).mean()

def calc_atr(df, n=20):
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df, btc_df):
    c = df['close']; h = df['high']; l = df['low']; o = df['open']
    df['rsi14'] = calc_rsi(c, 14); df['rsi2'] = calc_rsi(c, 2); df['rsi7'] = calc_rsi(c, 7)
    df['ema20'] = c.ewm(span=20, adjust=False).mean()
    df['ema50'] = c.ewm(span=50, adjust=False).mean()
    df['ema200'] = c.ewm(span=200, adjust=False).mean()
    df['adx'] = calc_adx(df)
    df['atr14'] = calc_atr(df, 14)
    df['atr_pct'] = (df['atr14'] / c) * 100
    df['bb_mid'] = c.rolling(20).mean(); df['bb_std'] = c.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2.0*df['bb_std']; df['bb_lower'] = df['bb_mid'] - 2.0*df['bb_std']
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_pos'] = (c - df['bb_lower']) / bb_range.replace(0, np.nan)
    df['kc_lower'] = df['ema20'] - 2.0*calc_atr(df, 20)
    df['swing_low_20'] = l.rolling(20).min().shift(1)
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20'].replace(0, np.nan)
    df['dist_ema200'] = (c - df['ema200']) / df['ema200'] * 100
    df['dist_ema20'] = (c - df['ema20']) / df['ema20'] * 100
    df['lower_wick_pct'] = (df[['open','close']].min(axis=1) - l) / o * 100
    df['range_pct'] = (h - l) / o * 100
    
    # 🆕 P29 Specific Indicators (VWAP & BTC Context)
    typical_price = (h + l + c) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    df['dist_vwap'] = (c - df['vwap']) / df['vwap'] * 100
    
    if btc_df is not None:
        df = pd.merge_asof(df.sort_values('ts'), btc_df[['ts', 'btc_close']].sort_values('ts'), on='ts', direction='backward')
        df['btc_ret_4h'] = df['btc_close'].pct_change(16) * 100
        df['btc_corr'] = df['close'].pct_change().rolling(20).corr(df['btc_close'].pct_change())
    else:
        df['btc_corr'] = 0.5
        df['btc_ret_4h'] = 0.0
        
    df['hour_pkt'] = df['ts'].apply(lambda x: (datetime.fromtimestamp(x/1000, tz=timezone.utc) + timedelta(hours=5)).hour)
    return df

def check_p1(r): return (r['close'] < r['bb_lower'] and r['rsi14'] < 32 and r['close'] > r['ema200'] and r['adx'] < 50)
def check_p8(r): return (r['close'] < r['kc_lower'] and r['rsi14'] < 35 and r['close'] > r['ema200'] and r['adx'] < 50)
def check_p21_v5(df, i):
    if i < 1: return False
    r, p = df.iloc[i], df.iloc[i-1]
    return (r["low"] < r["swing_low_20"] and r["close"] > r["swing_low_20"] and r["close"] > r["open"] and r["close"] > p["high"] and r["close"] < r["ema200"] and (r["close"]-r["ema200"])/r["ema200"]*100 > -15 and r["rsi14"] < 40 and r["vol_ratio"] >= 1.5)
def check_p21_v6(df, i):
    if i < 1: return False
    r, p = df.iloc[i], df.iloc[i-1]
    return (r["low"] < r["swing_low_20"] and r["close"] > r["swing_low_20"] and r["close"] > r["open"] and r["close"] > p["high"] and r["rsi14"] < 40 and r["vol_ratio"] >= 1.2)
def check_p23(df, i):
    if i < 1: return False
    r, p = df.iloc[i], df.iloc[i-1]
    return (r['close'] > r['ema200'] and r['close'] > r['ema50'] and r['adx'] < 25 and p['rsi2'] < 25 and r['close'] > r['open'] and r['close'] > p['high'] and r['bb_pos'] > 0.2 and r['rsi14'] > 40 and r['vol_ratio'] > 0.5)
def check_p25(df, i):
    r = df.iloc[i]
    return (r['adx'] > 30 and r['rsi14'] > 55 and 2 < r['dist_ema200'] < 10 and r['rsi2'] < 40 and r['bb_pos'] > 0.5 and r['vol_ratio'] >= 0.8 and r['close'] > r['open'])
def check_p27(r):
    return (r['atr_pct'] > 1.07 and 1.10 < r['lower_wick_pct'] <= 2.56 and r['dist_ema20'] > -0.60 and r['close'] > r['open'])
def check_p28(r):
    return (r['atr_pct'] <= 1.07 and r['range_pct'] > 2.50 and r['rsi7'] > 65.0 and r['close'] > r['open'])

# 🆕 P29 ML Logic (VWAP + ATR + BTC Correlation)
def check_p29(r):
    if pd.isna(r['dist_vwap']) or pd.isna(r['atr_pct']) or pd.isna(r['adx']) or pd.isna(r['btc_corr']): return False
    if r['close'] <= r['open']: return False
    # ML Discovered Rule: Price below VWAP, Decoupled from BTC, Trending
    return (r['dist_vwap'] < -0.5 and      # Below VWAP (Oversold)
            r['atr_pct'] > 0.5 and         # Has volatility
            r['adx'] > 25.0 and            # Strong trend
            r['btc_corr'] < 0.7 and        # Not just following BTC
            r['rsi14'] > 40.0)             # Not dead

def check_signals_at(sym, df, i):
    if i < 200 or i >= len(df): return None
    r = df.iloc[i]; cands = []
    if sym in P1_COINS and check_p1(r): cands.append({"pattern":"P1", "tp":1.2, "sl":2.5, "ts":TIME_STOP, "ts_ms":r['ts'], "entry":r['close']})
    if sym in P8_COINS and check_p8(r): cands.append({"pattern":"P8", "tp":1.0, "sl":2.5, "ts":TIME_STOP, "ts_ms":r['ts'], "entry":r['close']})
    if sym in P21_V5_COINS and check_p21_v5(df, i): cands.append({"pattern":"P21-v5", "tp":P21_TP, "sl":P21_SL, "ts":P21_TIME_STOP, "ts_ms":r['ts'], "entry":r['close']})
    if sym in P21_V6_COINS and check_p21_v6(df, i): cands.append({"pattern":"P21-v6", "tp":P21_TP, "sl":P21_SL, "ts":P21_TIME_STOP, "ts_ms":r['ts'], "entry":r['close']})
    if sym in P23_COINS and check_p23(df, i): cands.append({"pattern":"P23", "tp":P23_TP, "sl":P23_SL, "ts":P23_TIME_STOP, "ts_ms":r['ts'], "entry":r['close']})
    if sym in P25_COINS and check_p25(df, i): cands.append({"pattern":"P25", "tp":P25_TP, "sl":P25_SL, "ts":P25_TIME_STOP, "ts_ms":r['ts'], "entry":r['close']})
    if sym in P27_COINS and check_p27(r): cands.append({"pattern":"P27", "tp":P27_TP, "sl":P27_SL, "ts":P27_TIME_STOP, "ts_ms":r['ts'], "entry":r['close']})
    if sym in P28_COINS and check_p28(r): cands.append({"pattern":"P28", "tp":P28_TP, "sl":P28_SL, "ts":P28_TIME_STOP, "ts_ms":r['ts'], "entry":r['close']})
    if sym in P29_COINS and check_p29(r): cands.append({"pattern":"P29-ML", "tp":P29_TP, "sl":P29_SL, "ts":P29_TIME_STOP, "ts_ms":r['ts'], "entry":r['close']})
    
    if not cands: return None
    return max(cands, key=lambda x: PATTERN_PRIORITY[x["pattern"]])

def load_log():
    if not os.path.exists(SIGNAL_LOG_FILE): return []
    try:
        with open(SIGNAL_LOG_FILE, 'r', encoding='utf-8') as f: return [x for x in csv.DictReader(f) if x.get('pattern')]
    except: return []

def save_log(rows):
    try:
        with open(SIGNAL_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore', restval=''); w.writeheader()
            for r in rows: w.writerow(r)
    except Exception as e: print(f"⚠️ CSV error: {e}")

def can_open(sym, ts, rows):
    cd = COOLDOWN_BARS * 15 * 60 * 1000; last_exit = None
    for r in rows:
        if r.get("symbol") != sym: continue
        if r.get("status") in ("PENDING", "OPEN"): return False
        if r.get("status") == "DONE" and r.get("exit_ts_ms"):
            e = safe_int(r["exit_ts_ms"])
            if e > 0 and (last_exit is None or e > last_exit): last_exit = e
    if last_exit and (ts - last_exit) < cd: return False
    return True

def notify_sig(sig, tpkt):
    send_telegram_message(f"🚨 <b>{sig['pattern']} SIGNAL</b>\n🪙 {sig['symbol']}\n⏰ {tpkt.strftime('%Y-%m-%d %H:%M')} PKT\n🎯 Entry: {sig['entry']:.8g}\n📊 TP: +{sig['tp']}% | SL: -{sig['sl']}%\n🔥 TP-Trail: 0.3%")

def notify_tp_trigger(sym, tr, hi, csl):
    send_telegram_message(f"🎯 <b>TP HIT → TRAILING</b> {sym} ({tr['pattern']})\n🎯 Entry: {safe_float(tr['entry_open']):.8g}\n📈 High: {hi:.8g}\n🛡️ Trail SL: {csl:.8g} (+{((csl/safe_float(tr['entry_open']))-1)*100:.2f}%)")

def notify_res(sym, tr, res, xp, pnl, mfe, mae):
    em = "✅" if "TP" in res else ("🛑" if res == "SL" else "⏱")
    send_telegram_message(f"{em} <b>{res}</b> #{sym} ({tr['pattern']})\n🚪 Exit: {xp:.8g}\n💰 P&L: <b>{pnl:+.3f}%</b>\n📊 MFE: +{mfe:.2f}% | MAE: {mae:.2f}%")

def update_outcomes(coin_data, rows):
    updated = False
    for r in rows:
        sym, status = r.get("symbol", ""), r.get("status", "")
        if status not in ("PENDING", "OPEN") or sym not in coin_data: continue
        df = coin_data[sym]; sig_ts = safe_int(r.get("signal_ts_ms", 0))
        if sig_ts == 0: continue
        m = df[df['ts'] == sig_ts]
        if m.empty: r["result"] = "UNKNOWN"; r["status"] = "DONE"; updated = True; continue
        idx = m.index[0]
        
        if status == "PENDING":
            if idx + 1 >= len(df): continue
            no = float(df.iloc[idx + 1]['open'])
            tp_p, sl_p = safe_float(r.get("tp_pct", 1.2)), safe_float(r.get("sl_pct", 2.5))
            r["entry_open"] = f"{no:.10g}"; r["tp_price"] = f"{no*(1+tp_p/100):.10g}"; r["sl_price"] = f"{no*(1-sl_p/100):.10g}"
            r["highest_seen"] = f"{no:.10g}"; r["current_sl"] = r["sl_price"]; r["status"] = "OPEN"; updated = True
        
        entry, tp, osl = safe_float(r.get("entry_open", 0)), safe_float(r.get("tp_price", 0)), safe_float(r.get("sl_price", 0))
        if entry == 0 or tp == 0 or osl == 0: continue
        
        pat = r.get("pattern", "P1")
        ts = P29_TIME_STOP if pat == "P29-ML" else (P21_TIME_STOP if pat.startswith("P21") else (P23_TIME_STOP if pat == "P23" else (P25_TIME_STOP if pat == "P25" else (P27_TIME_STOP if pat == "P27" else (P28_TIME_STOP if pat == "P28" else TIME_STOP)))))
        fb = df.iloc[idx+1 : idx+ts+2]
        if len(fb) == 0: continue
        
        csl, hi, tp_hit, tp_bar, held, res, xt, xp, mae = osl, entry, False, None, 0, None, None, None, 0.0
        wt = safe_int(r.get("tp_triggered", 0)) == 1
        lowest_price = entry; bars_to_mfe = 0; bars_to_mae = 0
        
        for i, (_, b) in enumerate(fb.iterrows()):
            held = i + 1; bts, bh, bl, bc = b['ts'], b['high'], b['low'], b['close']
            if bh > hi: hi = bh; bars_to_mfe = held
            if bl < lowest_price: lowest_price = bl; bars_to_mae = held
            if ((bl / entry) - 1) * 100 < mae: mae = ((bl / entry) - 1) * 100
            
            if not tp_hit:
                if bl <= csl: res, xt, xp = "SL", bts, csl; break
                if bh >= tp:
                    tp_hit = True; tp_bar = i; hi = bh; csl = max(tp, hi * (1 - TRAIL_PCT/100))
                    r["tp_hit_price"] = f"{tp:.10g}"; r["tp_hit_bar"] = str(held)
                    if not wt: notify_tp_trigger(sym, r, hi, csl); wt = True; r["tp_triggered"] = "1"; r["tp_trigger_ts"] = str(int(bts))
                    continue
                if held >= ts: res, xt, xp = "TIMEOUT", bts, bc; break
            else:
                bs = i - tp_bar; hi = max(hi, bh); csl = max(csl, hi * (1 - TRAIL_PCT/100))
                if bl <= csl: res, xt, xp = "TP_TRAIL", bts, csl; break
                if bs >= MAX_BARS_AFTER_TP: res, xt, xp = "TP_TRAIL_TIMEOUT", bts, bc; break
        
        if res:
            pnl = ((xp / entry) - 1) * 100 - FEE_PCT
            r["result"] = res; r["exit_price"] = f"{xp:.10g}"; r["exit_ts_ms"] = str(int(xt)); r["bars_held"] = str(held)
            r["hours_to_exit"] = f"{held * 0.25:.2f}"
            r["exit_time_pkt"] = (datetime.fromtimestamp(xt/1000, tz=timezone.utc) + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
            r["pnl_pct"] = f"{pnl:+.3f}"; r["status"] = "DONE"
            mfe = ((hi / entry) - 1) * 100; mae_final = ((lowest_price / entry) - 1) * 100
            r["mfe_pct"] = f"{mfe:+.3f}"; r["mae_pct"] = f"{mae_final:+.3f}" 
            r["bars_to_mfe"] = str(bars_to_mfe); r["bars_to_mae"] = str(bars_to_mae)
            r["highest_seen"] = f"{hi:.10g}"; r["current_sl"] = f"{csl:.10g}"
            updated = True; notify_res(sym, r, res, xp, pnl, mfe, mae_final)
        else:
            r["highest_seen"] = f"{hi:.10g}"; r["current_sl"] = f"{csl:.10g}"; r["tp_triggered"] = "1" if tp_hit else "0"
            r["bars_since_tp"] = str(held - (tp_bar + 1) if tp_hit else 0); r["mae_pct"] = f"{mae:+.3f}"; updated = True
            mfe = ((hi / entry) - 1) * 100; mae_final = ((lowest_price / entry) - 1) * 100
            r["mfe_pct"] = f"{mfe:+.3f}"; r["mae_pct"] = f"{mae_final:+.3f}" 
            r["bars_to_mfe"] = str(bars_to_mfe); r["bars_to_mae"] = str(bars_to_mae)
    return updated

def scan_market():
    print("="*50)
    print("🚀 MEGA BOT v12 — 9 Patterns (ML P29 Added)")
    print("="*50)
    print("⏳ Waiting 45s..."); time.sleep(45)

    print("📥 Fetching BTC context...")
    btc_df = fetch_candles("BTCUSDT")
    btc_close = btc_df[['ts', 'close']].rename(columns={'close': 'btc_close'}) if btc_df is not None else None

    coin_data = {}
    for sym in ALL_SYMBOLS:
        df = fetch_candles(sym)
        if df is not None: coin_data[sym] = add_indicators(df, btc_close)

    rows = load_log()
    if update_outcomes(coin_data, rows): save_log(rows); print("📒 Outcomes updated")

    ns = 0; seen = {(r.get('symbol'), safe_int(r.get('signal_ts_ms', 0))) for r in rows if r.get('signal_ts_ms')}

    for sym, df in coin_data.items():
        n = len(df)
        for off in range(1, CANDLES_TO_CHECK + 1):
            idx = n - off
            if idx < 200: continue
            sig = check_signals_at(sym, df, idx)
            if not sig or (sym, sig['ts_ms']) in seen: continue
            if not can_open(sym, sig['ts_ms'], rows): continue

            tu = datetime.fromtimestamp(sig['ts_ms'] / 1000, tz=timezone.utc)
            tp = tu + timedelta(hours=5)
            
            nr = {
                "signal_ts_ms": str(sig['ts_ms']), "signal_time_utc": tu.strftime("%Y-%m-%d %H:%M"), "signal_time_pkt": tp.strftime("%Y-%m-%d %H:%M"),
                "symbol": sym, "pattern": sig['pattern'], "entry_close": f"{sig['entry']:.10g}", "entry_open": "",
                "tp_pct": str(sig['tp']), "sl_pct": str(sig['sl']), "tp_price": "", "sl_price": "",
                "tp_triggered": "0", "tp_trigger_ts": "", "highest_seen": "", "current_sl": "", "bars_since_tp": "0",
                "exit_price": "", "exit_ts_ms": "", "exit_time_pkt": "", "bars_held": "", "hours_to_exit": "",
                "mfe_pct": "", "mae_pct": "", "bars_to_mfe": "", "bars_to_mae": "", 
                "tp_hit_price": "", "tp_hit_bar": "", 
                "pnl_pct": "", "result": "", "status": "PENDING"
            }
            rows.append(nr); seen.add((sym, sig['ts_ms']))
            sig["symbol"] = sym
            notify_sig(sig, tp)
            print(f"🚨 SIGNAL: {sig['pattern']} {sym} @ {sig['entry']:.6g}")
            ns += 1

    if ns > 0: save_log(rows)
    print(f"✅ Scan complete. New signals: {ns}")

if __name__ == "__main__":
    scan_market()
