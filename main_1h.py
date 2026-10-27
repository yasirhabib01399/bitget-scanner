# ═══════════════════════════════════════════════════════════════════
# MEGA BOT 1H — Enhanced CNN+BiLSTM+Attention (Filtered Coins)
# ═══════════════════════════════════════════════════════════════════
import csv
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.get_logger().setLevel('ERROR')
    HAS_DL = True
except ImportError:
    HAS_DL = False
    print("⚠️ TensorFlow not found")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ENABLE_TELEGRAM = True

API_URL = "https://api.bitget.com/api/v2/spot/market/candles"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SIGNAL_LOG_FILE = "signals_log_1h.csv"

# 🆕 Filtered Coins (Dropped ARB, SUI, ATOM, APT)
COINS = [
    "ETHUSDT", "OPUSDT", "TIAUSDT", "BTCUSDT", "LINKUSDT", 
    "XRPUSDT", "DOGEUSDT", "SEIUSDT", "LTCUSDT", "BNBUSDT",
    "INJUSDT", "SOLUSDT", "ADAUSDT", "HYPEUSDT"
]

SEQ_LEN = 24; TIME_STOP = 24; TP_PCT = 3.0; SL_PCT = 2.0
TRAIL_PCT = 0.5; MAX_BARS_AFTER_TP = 24
COOLDOWN_BARS = 12
FEE_PCT = 0.2
DL_THRESHOLD = 0.4113722  # Kaggle se aya hua 97% quantile threshold

# ─── Load DL Model ───
DL_MODEL = None
if HAS_DL:
    try:
        if os.path.exists("1h_v2_weights.weights.h5"):
            inp = layers.Input(shape=(SEQ_LEN, 17))
            x = layers.Conv1D(64, 3, activation='relu', padding='same')(inp)
            x = layers.BatchNormalization()(x)
            x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Bidirectional(layers.LSTM(48, return_sequences=True))(x)
            attn = layers.Attention()([x, x])
            x = layers.Add()([x, attn])
            x = layers.Bidirectional(layers.LSTM(32))(x)
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(0.35)(x)
            out = layers.Dense(1, activation='sigmoid')(x)
            
            DL_MODEL = models.Model(inputs=inp, outputs=out)
            DL_MODEL.load_weights("1h_v2_weights.weights.h5")
            print("✅ 1H Enhanced DL Model loaded")
    except Exception as e: print(f"⚠️ DL load error: {e}")

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
    except: pass

def fetch_candles(sym):
    try:
        r = requests.get(API_URL, params={"symbol": sym, "granularity": "1h", "limit": "300"}, headers=HEADERS, timeout=15).json()
        if r.get("code") != "00000": return None
        rows = r.get("data", [])
        if len(rows) < 250: return None
        df = pd.DataFrame([{"ts": int(x[0]), "open": float(x[1]), "high": float(x[2]), "low": float(x[3]), "close": float(x[4]), "volume": float(x[5])} for x in rows])
        df = df.sort_values("ts").reset_index(drop=True)
        now_ms = int(time.time() * 1000)
        if df.iloc[-1]["ts"] + 3600000 > now_ms: df = df.iloc[:-1]
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

def add_indicators(df):
    c = df['close']; h = df['high']; l = df['low']; o = df['open']
    df['rsi14'] = calc_rsi(c, 14); df['rsi2'] = calc_rsi(c, 2)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr_pct'] = (tr.ewm(alpha=1/14, adjust=False).mean() / c) * 100
    df['bb_mid'] = c.rolling(20).mean(); df['bb_std'] = c.rolling(20).std()
    df['bb_pos'] = (c - (df['bb_mid'] - 2*df['bb_std'])) / ((df['bb_mid'] + 2*df['bb_std']) - (df['bb_mid'] - 2*df['bb_std'])).replace(0, np.nan)
    df['vol_sma20'] = df['volume'].rolling(20).mean(); df['vol_ratio'] = df['volume'] / df['vol_sma20'].replace(0, np.nan)
    
    df['ema50'] = c.ewm(span=50, adjust=False).mean()
    df['ema200'] = c.ewm(span=200, adjust=False).mean()
    df['dist_ema200'] = (c - df['ema200']) / df['ema200'] * 100
    df['trend_macro'] = (c > df['ema200']).astype(int)
    df['lower_wick_pct'] = (df[['open','close']].min(axis=1) - l) / o * 100
    df['upper_wick_pct'] = (h - df[['open','close']].max(axis=1)) / o * 100
    df['momentum_5'] = c.pct_change(5) * 100
    tp_vwap = (h + l + c) / 3
    df['vwap'] = (tp_vwap * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['dist_vwap'] = (c - df['vwap']) / df['vwap'] * 100
    df['hour'] = df['ts'].apply(lambda x: datetime.fromtimestamp(x/1000, tz=timezone.utc).hour)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['rsi_div'] = df['rsi14'] - df['rsi2']
    return df

def check_dl_signal(df, i):
    if DL_MODEL is None or i < 24: return False, 0
    
    # EMA200 Filter
    if df.iloc[i]['close'] < df.iloc[i]['ema200']:
        return False, 0
        
    try:
        seq = df.iloc[i-SEQ_LEN+1 : i+1]
        if seq.isna().any().any(): return False, 0
        lc = seq['close'].iloc[-1]
        seq_feat = np.column_stack((
            (seq['open'] - lc) / lc * 100, (seq['high'] - lc) / lc * 100,
            (seq['low'] - lc) / lc * 100, (seq['close'] - lc) / lc * 100,
            seq['rsi14'].values, seq['atr_pct'].values, seq['vol_ratio'].values,
            seq['bb_pos'].values, seq['dist_ema200'].values, seq['trend_macro'].values,
            seq['lower_wick_pct'].values, seq['upper_wick_pct'].values,
            seq['momentum_5'].values, seq['dist_vwap'].values,
            seq['hour_sin'].values, seq['hour_cos'].values, seq['rsi_div'].values
        )).reshape(1, SEQ_LEN, 17)
        prob = DL_MODEL.predict(seq_feat, verbose=0)[0][0]
        return prob >= DL_THRESHOLD, prob
    except: return False, 0

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
    cd = COOLDOWN_BARS * 3600000; last_exit = None
    for r in rows:
        if r.get("symbol") != sym: continue
        if r.get("status") in ("PENDING", "OPEN"): return False
        if r.get("status") == "DONE" and r.get("exit_ts_ms"):
            e = safe_int(r["exit_ts_ms"])
            if e > 0 and (last_exit is None or e > last_exit): last_exit = e
    if last_exit and (ts - last_exit) < cd: return False
    return True

def notify_sig(sig, tpkt):
    send_telegram_message(f"🚨 <b>1H DL SIGNAL</b>\n🪙 {sig['symbol']}\n⏰ {tpkt.strftime('%Y-%m-%d %H:%M')} PKT\n🎯 Entry: {sig['entry']:.8g}\n📊 TP: +{TP_PCT}% | SL: -{SL_PCT}%\n🧠 DL Prob: {sig.get('dl_prob', 0)*100:.1f}%\n🔥 TP-Trail: 0.5%")

def notify_tp_trigger(sym, tr, hi, csl):
    send_telegram_message(f"🎯 <b>1H TP HIT → TRAILING</b> {sym}\n🎯 Entry: {safe_float(tr['entry_open']):.8g}\n📈 High: {hi:.8g}\n🛡️ Trail SL: {csl:.8g} (+{((csl/safe_float(tr['entry_open']))-1)*100:.2f}%)")

def notify_res(sym, tr, res, xp, pnl, mfe, mae):
    em = "✅" if "TP" in res else ("🛑" if res == "SL" else "⏱")
    send_telegram_message(f"{em} <b>{res}</b> #{sym} (1H)\n🚪 Exit: {xp:.8g}\n💰 P&L: <b>{pnl:+.3f}%</b>\n📊 MFE: +{mfe:.2f}% | MAE: {mae:.2f}%")

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
            tp_p, sl_p = safe_float(r.get("tp_pct", TP_PCT)), safe_float(r.get("sl_pct", SL_PCT))
            r["entry_open"] = f"{no:.10g}"; r["tp_price"] = f"{no*(1+tp_p/100):.10g}"; r["sl_price"] = f"{no*(1-sl_p/100):.10g}"
            r["highest_seen"] = f"{no:.10g}"; r["current_sl"] = r["sl_price"]; r["status"] = "OPEN"; updated = True

        entry, tp, osl = safe_float(r.get("entry_open", 0)), safe_float(r.get("tp_price", 0)), safe_float(r.get("sl_price", 0))
        if entry == 0 or tp == 0 or osl == 0: continue

        fb = df.iloc[idx+1 : idx+TIME_STOP+2]
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
                if held >= TIME_STOP: res, xt, xp = "TIMEOUT", bts, bc; break
            else:
                bs = i - tp_bar; hi = max(hi, bh); csl = max(csl, hi * (1 - TRAIL_PCT/100))
                if bl <= csl: res, xt, xp = "TP_TRAIL", bts, csl; break
                if bs >= MAX_BARS_AFTER_TP: res, xt, xp = "TP_TRAIL_TIMEOUT", bts, bc; break

        if res:
            pnl = ((xp / entry) - 1) * 100 - FEE_PCT
            r["result"] = res; r["exit_price"] = f"{xp:.10g}"; r["exit_ts_ms"] = str(int(xt)); r["bars_held"] = str(held)
            r["hours_to_exit"] = f"{held * 1.0:.2f}"
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
    print("🚀 MEGA BOT 1H — Filtered Coins")
    print("="*50)
    if DL_MODEL is not None: print("   ✅ 1H DL Model Active")
    
    print("⏳ Waiting 45s..."); time.sleep(45)

    coin_data = {}
    for sym in COINS:
        df = fetch_candles(sym)
        if df is not None: coin_data[sym] = add_indicators(df)

    rows = load_log()
    if update_outcomes(coin_data, rows): save_log(rows); print("📒 Outcomes updated")

    ns = 0; seen = {(r.get('symbol'), safe_int(r.get('signal_ts_ms', 0))) for r in rows if r.get('signal_ts_ms')}

    for sym, df in coin_data.items():
        n = len(df)
        for off in range(1, 3):
            idx = n - off
            if idx < 50: continue
            
            dl_ok, dl_prob = check_dl_signal(df, idx)
            if not dl_ok: continue
            
            r = df.iloc[idx]
            if (sym, r['ts']) in seen: continue
            if not can_open(sym, r['ts'], rows): continue

            tu = datetime.fromtimestamp(r['ts'] / 1000, tz=timezone.utc)
            tp = tu + timedelta(hours=5)

            nr = {
                "signal_ts_ms": str(r['ts']), "signal_time_utc": tu.strftime("%Y-%m-%d %H:%M"), "signal_time_pkt": tp.strftime("%Y-%m-%d %H:%M"),
                "symbol": sym, "pattern": "DL-1H", "entry_close": f"{r['close']:.10g}", "entry_open": "",
                "tp_pct": str(TP_PCT), "sl_pct": str(SL_PCT), "tp_price": "", "sl_price": "",
                "tp_triggered": "0", "tp_trigger_ts": "", "highest_seen": "", "current_sl": "", "bars_since_tp": "0",
                "exit_price": "", "exit_ts_ms": "", "exit_time_pkt": "", "bars_held": "", "hours_to_exit": "",
                "mfe_pct": "", "mae_pct": "", "bars_to_mfe": "", "bars_to_mae": "", 
                "tp_hit_price": "", "tp_hit_bar": "", 
                "pnl_pct": "", "result": "", "status": "PENDING"
            }
            rows.append(nr); seen.add((sym, r['ts']))
            
            sig_data = {"symbol": sym, "entry": r['close'], "dl_prob": dl_prob}
            notify_sig(sig_data, tp)
            print(f"🚨 1H SIGNAL: {sym} @ {r['close']:.6g} (DL {dl_prob*100:.0f}%)")
            ns += 1

    if ns > 0: save_log(rows)
    print(f"✅ Scan complete. New 1H signals: {ns}")

if __name__ == "__main__":
    scan_market()
