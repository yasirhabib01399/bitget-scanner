import csv
import os
import time
import requests
from datetime import datetime, timedelta, timezone

# Telegram credentials from GitHub Secrets
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ENABLE_TELEGRAM = True

# True rakhein to har coin ki indicator values log mein print hongi (chart se milane ke liye)
DEBUG = True

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT", "LINKUSDT",
    "DOTUSDT", "AVAXUSDT", "SUIUSDT",
    "ZECUSDT", "HYPEUSDT", "BGBUSDT",
    "BNBUSDT", "NEARUSDT", "QNTUSDT"
]

# ======================= STRATEGY (sirf 15m chart) =======================
# SIGNAL tab aata hai jab 15m ki band candle:
#   1) CLOSE teeno (EMA9, EMA21, VWAP) ke upar ho, AUR
#   2) un teeno mein se kam az kam aik ko cross kare (yani OPEN us level ke neeche tha), AUR
#   3) volume pichli 9 candles ke average (SMA9) x VOLUME_MULT se zyada ho, AUR
#   4) close se Resistance kam az kam MIN_RESISTANCE_GAP_PCT % door ho.
# Is mein teeno cases shamil hain:
#   - teeno ko cross (open teeno ke neeche)
#   - aik level pehle se neeche, baaki do ko cross
#   - do levels pehle se neeche, baaki aik ko cross
# Agar teeno levels pehle se open ke neeche hon (koi cross nahi) to signal nahi.
TIMEFRAME = "15min"
CANDLE_LIMIT = 100
CANDLE_MS = 15 * 60 * 1000
VOLUME_PERIOD = 9
VOLUME_MULT = 1.0          # 1.0 = SMA9 se zyada. "Bara volume" chahiye to 1.5 / 2.0
VWAP_PERIOD = 15

# Resistance: pichli SR_PERIOD candles ka sab se ooncha High (Bitget S/R indicator, parameter 15, jaisa).
SR_PERIOD = 15
MIN_RESISTANCE_GAP_PCT = 0.8
# False = agar close pehle hi resistance ke upar nikal chuka ho to signal nahi aaye ga
ALLOW_CLOSE_ABOVE_RESISTANCE = False

# Alert mein dikhane ke liye TP/SL (close se %)
TP_PCT = 0.8
SL_PCT = 0.4

# Signal log (paper trade): har signal signals_log.csv mein save hota hai aur baad ke runs mein
# khud result (TP/SL) bhar diya jata hai. Entry = signal candle ka close.
ENABLE_SIGNAL_LOG = True
SIGNAL_LOG_FILE = "signals_log.csv"
LOG_TRACK_CANDLES = 88     # itni 15m candles (~22 ghantay) tak har signal ko dekhte hain (result + price kitni upar gayi)

# 1H chart sirf RECORD ke liye (signal ko rokta nahi): har signal par 1H ki position CSV mein save hoti hai
H1_TIMEFRAME = "1h"
H1_CANDLE_MS = 60 * 60 * 1000

API_URL = "https://api.bitget.com/api/v2/spot/market/candles"


def send_telegram_message(message):
    if not ENABLE_TELEGRAM or not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Telegram Notification Failed: {e}")


def fetch_candles(symbol, retries=3, timeframe=TIMEFRAME, candle_ms=CANDLE_MS, only_completed=True):
    """Sirf band ho chuki candles (only_completed=True)."""
    params = {"symbol": symbol, "granularity": timeframe, "limit": str(CANDLE_LIMIT)}
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            result = response.json()
            if result.get("code") != "00000":
                raise RuntimeError(result.get("msg", "Bitget API error"))

            candles = []
            for row in result.get("data", []):
                candles.append({
                    "timestamp": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5])
                })
            candles.sort(key=lambda x: x["timestamp"])
            if only_completed:
                now_ms = int(time.time() * 1000)
                candles = [c for c in candles if c["timestamp"] + candle_ms <= now_ms]
            if len(candles) < 30:
                raise RuntimeError("Insufficient completed candles")
            return candles
        except Exception as e:
            if attempt == retries - 1:
                print(f"❌ {symbol}: API error: {e}")
                return None
            time.sleep(2)
    return None


def add_indicators(candles):
    closes = [c["close"] for c in candles]
    ema9, ema21 = [], []
    alpha9 = 2.0 / (9 + 1)
    alpha21 = 2.0 / (21 + 1)
    e9, e21 = closes[0], closes[0]

    for close in closes:
        e9 = alpha9 * close + (1 - alpha9) * e9
        e21 = alpha21 * close + (1 - alpha21) * e21
        ema9.append(e9)
        ema21.append(e21)

    # Rolling VWAP (chart ke VWAP(15) jaisa)
    for i, candle in enumerate(candles):
        window = candles[max(0, i - VWAP_PERIOD + 1): i + 1]
        pv = sum(((c["high"] + c["low"] + c["close"]) / 3.0) * c["volume"] for c in window)
        vol = sum(c["volume"] for c in window)
        candle["VWAP"] = pv / vol if vol > 0 else float("nan")

    for i, candle in enumerate(candles):
        candle["EMA9"] = ema9[i]
        candle["EMA21"] = ema21[i]
        if i >= VOLUME_PERIOD:
            previous_volumes = [candles[j]["volume"] for j in range(i - VOLUME_PERIOD, i)]
            candle["Volume_SMA9"] = sum(previous_volumes) / VOLUME_PERIOD
        else:
            candle["Volume_SMA9"] = None
    return candles


def add_support_resistance(candles, index):
    """
    candles[index] ke liye Resistance (aur Support, sirf jankari ke liye) nikaal kar usi candle mein save karta hai.
    Resistance = pichli SR_PERIOD candles ka max High (candles[index] khud shamil nahi).
    Gap % = (Resistance - Close) / Close * 100
    """
    current = candles[index]
    if index < SR_PERIOD:
        current["Resistance"] = None
        current["Support"] = None
        current["Res_gap_pct"] = None
        return
    window = candles[index - SR_PERIOD:index]
    resistance = max(c["high"] for c in window)
    support = min(c["low"] for c in window)
    current["Resistance"] = resistance
    current["Support"] = support
    current["Res_gap_pct"] = (resistance - current["close"]) / current["close"] * 100.0


def check_single_candle(candles, index):
    """Signal ho to candle (dict) wapas, warna None. current["Crossed"] mein cross kiye hue levels ki list."""
    if index < VOLUME_PERIOD:
        return None
    current = candles[index]
    add_support_resistance(candles, index)
    if current["Volume_SMA9"] is None or current["Resistance"] is None:
        return None

    levels = {"EMA9": current["EMA9"], "EMA21": current["EMA21"], "VWAP": current["VWAP"]}
    close_above_all = all(current["close"] > v for v in levels.values())
    crossed = [name for name, v in levels.items() if current["open"] < v]     # open neeche, close upar
    volume_confirmed = current["volume"] > current["Volume_SMA9"] * VOLUME_MULT

    gap = current["Res_gap_pct"]
    if gap >= MIN_RESISTANCE_GAP_PCT:
        resistance_ok = True
    else:
        resistance_ok = ALLOW_CLOSE_ABOVE_RESISTANCE and gap < 0

    if close_above_all and len(crossed) >= 1 and volume_confirmed and resistance_ok:
        current["Crossed"] = crossed
        return current
    return None


# ======================= SIGNAL LOG (signals_log.csv) =======================
# Har signal ek row. Do hisse: (1) signal ke waqt ki tafseel (features), (2) baad mein khud bharne wale results.
# Naam ki pehchan:  p1..p5 = signal se pehle ki 1..5 candles (p1 = sab se qareeb)
#                   h1_*   = 1H chart ki wo candle jis mein signal candle aati hai
#                   m15_*  = 15m chart,  btc_* = BTC ka rukh
#   *_pos  = 3 harf, order EMA9, EMA21, VWAP:  A = close us level ke upar (Above), B = neeche (Below)
#            jaise "BBA" = EMA9 ke neeche, EMA21 ke neeche, VWAP ke upar
#   *_vol_x = us candle ka volume / us waqt ka volume SMA9 (2.0 = SMA9 se doguna)
#   Liquidity sweep = candle ka low pichli N candles ke sab se neeche low se NEECHE gaya (stop-hunt) lekin
#            close us low se UPAR aaya. 15m: N=15, 1H: N=10.
# Waqt ke hisaab se peak: signal ke baad pehle 1/2/4/8 ghantay (= 4/8/16/32 candles) ke andar sab se oonchi/neechi price.
# Column tabhi bharta hai jab wo poori muddat guzar jaye (warna adhoora number ghalat tasur deta).
PEAK_HORIZONS = [(1, 4), (2, 8), (4, 16), (8, 32)]       # (ghantay, candles)
RESULT_GRID_EXTRA = [(0.5, 0.4), (1.0, 0.4), (1.5, 0.4), (2.0, 0.4), (0.8, 0.3), (0.8, 0.6), (0.8, 0.8)]
RESULT_GRID = [(g[0], g[1]) for g in RESULT_GRID_EXTRA if g != (TP_PCT, SL_PCT)]


def grid_col(tp, sl):
    return f"r_tp{tp}_sl{sl}"


def _log_fields():
    f = ["signal_time_pkt", "signal_time_utc", "symbol", "result", "hours_to_exit", "exit_time_pkt",
         "entry_close", "tp_price", "sl_price",
         "peak_up_pct", "hours_to_peak", "peak_after_tp_extra_pct", "peak_down_pct"]
    f += [f"peak_up_{h}h_pct" for h, _ in PEAK_HORIZONS] + [f"peak_down_{h}h_pct" for h, _ in PEAK_HORIZONS]
    f += [
         "mfe_before_exit_pct", "mae_before_exit_pct", "ret_1h_after_pct", "ret_4h_after_pct"]
    f += [grid_col(tp, sl) for tp, sl in RESULT_GRID]
    f += ["status", "candles_tracked",
          # 15m candle
          "open", "high", "low", "volume", "vol_ratio", "vol_ratio_96", "body_pct", "range_pct", "upper_wick_pct",
          "lower_wick_pct", "close_pos", "atr_pct", "n_crossed", "crossed_levels",
          "d_ema9", "d_ema21", "d_vwap", "levels_spread_pct", "res_gap_pct", "sup_gap_pct",
          "ret_1h", "ret_4h", "ret_24h", "hour_pkt", "weekday",
          "m15_sweep_recent", "m15_sweep_candles_ago", "m15_sweep_depth_pct"]
    for k in range(1, 6):
        f += [f"p{k}_vol_x", f"p{k}_pos", f"p{k}_color"]
    f += ["btc_ret_1h", "btc_ret_4h", "btc_candle",
          # 1H chart
          "h1_color", "h1_body_pct", "h1_range_pct", "h1_pos", "h1_d_ema9", "h1_d_ema21", "h1_d_vwap",
          "h1_res_gap_pct", "h1_sup_gap_pct", "h1_nearest_up_pct", "h1_nearest_up_name", "h1_broke_res",
          "h1_sweep_now", "h1_sweep_last3", "h1_sweep_depth_pct", "h1_last3", "h1_prev_vol_x",
          "signal_ts_ms"]
    return f


def _num(x, nd=2):
    """Number ko text banata hai; None/NaN ho to khali."""
    if x is None:
        return ""
    try:
        if x != x or x in (float("inf"), float("-inf")):
            return ""
    except TypeError:
        return ""
    return f"{x:.{nd}f}"


def _pct(a, b):
    """(a / b - 1) * 100, ghalat input par None"""
    try:
        return (a / b - 1.0) * 100.0
    except (ZeroDivisionError, TypeError):
        return None


def pos_code(price, c):
    """EMA9, EMA21, VWAP ke muqable price: A = upar, B = neeche"""
    return "".join("A" if price > c[k] else "B" for k in ("EMA9", "EMA21", "VWAP"))


def sweep_info(candles, i, lookback):
    """candles[i] ne pichli `lookback` candles ke sab se neeche low ko toor kar close us se upar kiya?"""
    if i < lookback or i >= len(candles):
        return False, 0.0
    prior_low = min(c["low"] for c in candles[i - lookback:i])
    c = candles[i]
    if c["low"] < prior_low and c["close"] > prior_low:
        return True, (prior_low - c["low"]) / prior_low * 100.0
    return False, 0.0


def atr_pct(candles, index, n=14):
    if index < n:
        return None
    trs = []
    for j in range(index - n + 1, index + 1):
        c, pc = candles[j], candles[j - 1]["close"]
        trs.append(max(c["high"] - c["low"], abs(c["high"] - pc), abs(c["low"] - pc)))
    return sum(trs) / n / candles[index]["close"] * 100.0


def btc_context(btc_candles, sig_ts):
    out = {"btc_ret_1h": "", "btc_ret_4h": "", "btc_candle": ""}
    if not btc_candles:
        return out
    idx = next((i for i, c in enumerate(btc_candles) if c["timestamp"] == sig_ts), None)
    if idx is None:
        return out
    c = btc_candles[idx]
    if idx >= 4:
        out["btc_ret_1h"] = _num(_pct(c["close"], btc_candles[idx - 4]["close"]))
    if idx >= 16:
        out["btc_ret_4h"] = _num(_pct(c["close"], btc_candles[idx - 16]["close"]))
    out["btc_candle"] = "G" if c["close"] > c["open"] else "R"
    return out


def h1_features(symbol, sig_ts, price):
    """
    1H chart ki wo candle jis mein signal candle aati hai (chalti ho to us waqt tak ki). Indicators 1H candles par.
    1H data na mile to khali dict (log mein khali columns).
    """
    cs = fetch_candles(symbol, timeframe=H1_TIMEFRAME, candle_ms=H1_CANDLE_MS, only_completed=False)
    if cs is None:
        return {}
    hs = (sig_ts // H1_CANDLE_MS) * H1_CANDLE_MS
    idx = next((i for i, c in enumerate(cs) if c["timestamp"] == hs), None)
    if idx is None or idx < 20:
        return {}
    h = add_indicators(cs[:idx + 1])
    add_support_resistance(h, idx)
    cur = h[idx]
    res, sup = cur["Resistance"], cur["Support"]

    levels = {"EMA9": cur["EMA9"], "EMA21": cur["EMA21"], "VWAP": cur["VWAP"], "RES": res}
    near_name, near_gap = "", None
    for name, v in levels.items():
        if v is None or v != v:
            continue
        g = _pct(v, price)
        if g is not None and g > 0 and (near_gap is None or g < near_gap):
            near_name, near_gap = name, g

    sweeps = [sweep_info(h, idx - k, 10) for k in range(3)]
    depth = max([d for ok, d in sweeps if ok], default=0.0)
    colors = ""
    for j in range(idx - 2, idx + 1):
        close_j = price if j == idx else h[j]["close"]
        colors += "G" if close_j > h[j]["open"] else "R"
    prev_sma = h[idx - 1].get("Volume_SMA9")

    return {
        "h1_color": "G" if price > cur["open"] else ("R" if price < cur["open"] else "F"),
        "h1_body_pct": _num(_pct(price, cur["open"])),
        "h1_range_pct": _num(_pct(cur["high"], cur["low"])),
        "h1_pos": pos_code(price, cur),
        "h1_d_ema9": _num(_pct(price, cur["EMA9"])),
        "h1_d_ema21": _num(_pct(price, cur["EMA21"])),
        "h1_d_vwap": _num(_pct(price, cur["VWAP"])),
        "h1_res_gap_pct": _num(_pct(res, price)),
        "h1_sup_gap_pct": _num(_pct(price, sup)),
        "h1_nearest_up_pct": _num(near_gap),
        "h1_nearest_up_name": near_name,
        "h1_broke_res": "1" if cur["high"] > res else "0",
        "h1_sweep_now": "1" if sweeps[0][0] else "0",
        "h1_sweep_last3": "1" if any(ok for ok, _ in sweeps) else "0",
        "h1_sweep_depth_pct": _num(depth),
        "h1_last3": colors,
        "h1_prev_vol_x": _num(h[idx - 1]["volume"] / prev_sma) if prev_sma else "",
    }


def build_features(symbol, candles, index, signal, btc_candles):
    """Signal ke waqt ki poori tafseel (15m + pichli candles + BTC + 1H). Sab text mein."""
    c = signal
    price = c["close"]
    rng = c["high"] - c["low"]
    levels = [c["EMA9"], c["EMA21"], c["VWAP"]]
    t_pkt = datetime.fromtimestamp(c["timestamp"] / 1000, tz=timezone.utc) + timedelta(hours=5)

    f = {
        "open": f"{c['open']:.10g}", "high": f"{c['high']:.10g}", "low": f"{c['low']:.10g}",
        "volume": _num(c["volume"]),
        "vol_ratio": _num(c["volume"] / c["Volume_SMA9"]) if c.get("Volume_SMA9") else "",
        "vol_ratio_96": (_num(c["volume"] / (sum(x["volume"] for x in candles[index - 96:index]) / 96.0))
                         if index >= 96 else ""),
        "body_pct": _num(_pct(price, c["open"])),
        "range_pct": _num(_pct(c["high"], c["low"])),
        "upper_wick_pct": _num((c["high"] - max(c["open"], price)) / price * 100.0),
        "lower_wick_pct": _num((min(c["open"], price) - c["low"]) / price * 100.0),
        "close_pos": _num((price - c["low"]) / rng) if rng > 0 else "",
        "atr_pct": _num(atr_pct(candles, index)),
        "n_crossed": str(len(c.get("Crossed", []))),
        "crossed_levels": "+".join(c.get("Crossed", [])),
        "d_ema9": _num(_pct(price, c["EMA9"])),
        "d_ema21": _num(_pct(price, c["EMA21"])),
        "d_vwap": _num(_pct(price, c["VWAP"])),
        "levels_spread_pct": _num(_pct(max(levels), min(levels))),
        "res_gap_pct": _num(c.get("Res_gap_pct")),
        "sup_gap_pct": _num(_pct(price, c["Support"])) if c.get("Support") else "",
        "ret_1h": _num(_pct(price, candles[index - 4]["close"])) if index >= 4 else "",
        "ret_4h": _num(_pct(price, candles[index - 16]["close"])) if index >= 16 else "",
        "ret_24h": _num(_pct(price, candles[index - 96]["close"])) if index >= 96 else "",
        "hour_pkt": str(t_pkt.hour),
        "weekday": t_pkt.strftime("%a"),
    }
    # 15m liquidity sweep: signal candle ya pichli 5 candles mein
    f["m15_sweep_recent"], f["m15_sweep_candles_ago"], f["m15_sweep_depth_pct"] = "0", "", ""
    for k in range(0, 6):
        ok, depth = sweep_info(candles, index - k, SR_PERIOD)
        if ok:
            f["m15_sweep_recent"], f["m15_sweep_candles_ago"], f["m15_sweep_depth_pct"] = "1", str(k), _num(depth)
            break
    # Pichli 5 candles: volume aur location
    for k in range(1, 6):
        j = index - k
        pc = candles[j] if j >= 0 else None
        if pc is None or pc.get("Volume_SMA9") is None:
            f[f"p{k}_vol_x"], f[f"p{k}_pos"], f[f"p{k}_color"] = "", "", ""
            continue
        f[f"p{k}_vol_x"] = _num(pc["volume"] / pc["Volume_SMA9"])
        f[f"p{k}_pos"] = pos_code(pc["close"], pc)
        f[f"p{k}_color"] = "G" if pc["close"] > pc["open"] else "R"
    f.update(btc_context(btc_candles, c["timestamp"]))
    f.update(h1_features(symbol, c["timestamp"], price))
    return f


def load_signal_log():
    if not os.path.exists(SIGNAL_LOG_FILE):
        return []
    try:
        with open(SIGNAL_LOG_FILE, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception as e:
        print(f"⚠️ Signal log padh nahi saka: {e}")
        return []


def save_signal_log(rows):
    try:
        with open(SIGNAL_LOG_FILE, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_log_fields(), extrasaction="ignore", restval="")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception as e:
        print(f"⚠️ Signal log save nahi hua: {e}")


def log_signal(symbol, signal, tp_price, sl_price, features):
    """Signal ko CSV mein jorta hai (same symbol + candle dobara nahi likhta)."""
    if not ENABLE_SIGNAL_LOG:
        return
    rows = load_signal_log()
    key = str(signal["timestamp"])
    if any(r.get("symbol") == symbol and r.get("signal_ts_ms") == key for r in rows):
        return
    t_utc = datetime.fromtimestamp(signal["timestamp"] / 1000, tz=timezone.utc)
    t_pkt = t_utc + timedelta(hours=5)
    row = {
        "signal_ts_ms": key,
        "signal_time_utc": t_utc.strftime("%Y-%m-%d %H:%M"),
        "signal_time_pkt": t_pkt.strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "entry_close": f"{signal['close']:.10g}",
        "tp_price": f"{tp_price:.10g}",
        "sl_price": f"{sl_price:.10g}",
        "status": "OPEN",
    }
    row.update(features)
    rows.append(row)
    save_signal_log(rows)
    print(f"📝 Signal log mein save: {symbol} {t_pkt.strftime('%Y-%m-%d %H:%M')} PKT")


def _first_hit(after, entry, tp_pct, sl_pct):
    """Entry ke baad pehli candle jis mein TP ya SL touch hua. Dono ek hi candle mein ho to SL maana jata hai."""
    tp = entry * (1 + tp_pct / 100.0)
    sl = entry * (1 - sl_pct / 100.0)
    for k, c in enumerate(after, 1):
        if c["low"] <= sl:
            return "SL", k
        if c["high"] >= tp:
            return "TP", k
    return None, None


def notify_result(r, res, k, after, entry):
    """TP ya SL pehli baar hit hone par Telegram message (candle band hone ke baad, agle run mein)."""
    tp_price = entry * (1 + TP_PCT / 100.0)
    sl_price = entry * (1 - SL_PCT / 100.0)
    hit_candle = after[k - 1]
    t_hit = datetime.fromtimestamp(hit_candle["timestamp"] / 1000, tz=timezone.utc) + timedelta(hours=5)
    peak_so_far = _pct(max(c["high"] for c in after), entry)
    low_so_far = (1 - min(c["low"] for c in after) / entry) * 100.0
    if res == "TP":
        head = f"✅ <b>TP HIT</b> #{r['symbol']}"
        line = f"🎯 TP {tp_price:.8g} (+{TP_PCT}%) touch hua"
    else:
        head = f"🛑 <b>SL HIT</b> #{r['symbol']}"
        line = f"🛑 SL {sl_price:.8g} (-{SL_PCT}%) touch hua"
    both = hit_candle["low"] <= sl_price and hit_candle["high"] >= tp_price
    note = ("\n⚠️ Isi candle mein TP aur SL dono touch hue, SL pehle maana gaya (asal mein pehle kaun hua, "
            "chart se dekh lein)") if both else ""
    msg = (
        f"{head}\n\n"
        f"🕐 <b>Signal:</b> {r['signal_time_pkt']} PKT\n"
        f"🎯 <b>Entry (close):</b> {entry:.8g}\n"
        f"{line}\n"
        f"⏱ <b>Hit candle:</b> {t_hit.strftime('%Y-%m-%d %H:%M')} PKT ({k * 0.25:.2f} ghantay baad)\n"
        f"📈 <b>Ab tak ki sab se oonchi price:</b> +{peak_so_far:.2f}%  |  📉 sab se neechi: -{low_so_far:.2f}%"
        f"{note}"
    )
    send_telegram_message(msg)


def update_signal_outcomes():
    """
    Khule (OPEN) signals ka result bharta hai. Har signal ko LOG_TRACK_CANDLES (~22 ghantay) tak dekhte hain:
    TP/SL kis ne pehle hit kiya (alag alag TP/SL jooron par), TP ke baad price kitni upar gayi, wagaira.
    """
    if not ENABLE_SIGNAL_LOG:
        return
    rows = load_signal_log()
    open_rows = [r for r in rows if r.get("status") != "DONE"]
    if not open_rows:
        return
    cache = {}
    changed = False
    for r in open_rows:
        sym = r["symbol"]
        if sym not in cache:
            cache[sym] = fetch_candles(sym)
        candles = cache[sym]
        if candles is None:
            continue
        sig_ts = int(r["signal_ts_ms"])
        entry = float(r["entry_close"])
        all_cols = ["result"] + [grid_col(tp, sl) for tp, sl in RESULT_GRID]

        if sig_ts < candles[0]["timestamp"]:
            # scanner kai ghantay band raha: signal candle ab data window se bahar, result pakka nahi nikal sakte
            for col in all_cols:
                if r.get(col) not in ("TP", "SL", "NONE"):
                    r[col] = "UNKNOWN"
            r["status"] = "DONE"
            changed = True
            continue

        after = [c for c in candles if c["timestamp"] > sig_ts][:LOG_TRACK_CANDLES]
        if not after:
            continue
        complete = len(after) >= LOG_TRACK_CANDLES

        # Asli TP/SL
        prev_result = r.get("result", "")
        res, k = _first_hit(after, entry, TP_PCT, SL_PCT)
        if res and prev_result not in ("TP", "SL"):
            notify_result(r, res, k, after, entry)      # pehli baar hit hua: Telegram alert
        r["result"] = res if res else ("NONE" if complete else "")
        r["hours_to_exit"] = f"{k * 0.25:.2f}" if k else ""
        if k:
            t0 = int(r["signal_ts_ms"]) + k * CANDLE_MS
            r["exit_time_pkt"] = (datetime.fromtimestamp(t0 / 1000, tz=timezone.utc) + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
        # Baqi TP/SL jore
        for tp, sl in RESULT_GRID:
            g, _ = _first_hit(after, entry, tp, sl)
            r[grid_col(tp, sl)] = g if g else ("NONE" if complete else "")

        # Price kitni upar/neeche gayi (poori tracking muddat mein)
        highs = [c["high"] for c in after]
        lows = [c["low"] for c in after]
        peak = max(highs)
        r["peak_up_pct"] = _num(_pct(peak, entry))
        r["hours_to_peak"] = f"{(highs.index(peak) + 1) * 0.25:.2f}"
        r["peak_down_pct"] = _num((1 - min(lows) / entry) * 100.0)
        for h_, n_ in PEAK_HORIZONS:                       # sirf tab jab wo muddat poori guzar chuki ho
            if len(after) >= n_:
                r[f"peak_up_{h_}h_pct"] = _num(_pct(max(c["high"] for c in after[:n_]), entry))
                r[f"peak_down_{h_}h_pct"] = _num((1 - min(c["low"] for c in after[:n_]) / entry) * 100.0)
        r["peak_after_tp_extra_pct"] = _num(_pct(peak, entry) - TP_PCT) if res == "TP" else ""
        upto = after[:k] if k else after                   # exit tak (ya ab tak)
        r["mfe_before_exit_pct"] = _num(_pct(max(c["high"] for c in upto), entry))
        r["mae_before_exit_pct"] = _num((1 - min(c["low"] for c in upto) / entry) * 100.0)
        r["ret_1h_after_pct"] = _num(_pct(after[3]["close"], entry)) if len(after) >= 4 else ""
        r["ret_4h_after_pct"] = _num(_pct(after[15]["close"], entry)) if len(after) >= 16 else ""
        r["candles_tracked"] = str(len(after))
        if complete:
            r["status"] = "DONE"
        changed = True
    if changed:
        save_signal_log(rows)
        n_done = sum(1 for r in rows if r.get("status") == "DONE")
        print(f"📒 Signal log update: {len(rows)} signals, {n_done} mukammal")


# ============================== SCAN ==============================
def scan_market():
    # Bitget candle close hone ke baad data fetch karne ke liye thora wait
    time.sleep(20)

    print("=" * 50)
    print("BITGET SPOT - 15m BULLISH CROSS SCAN")
    print("=" * 50)

    # Pehle purane signals ke result (TP/SL) update karein
    update_signal_outcomes()

    # BTC ka rukh (har signal ke saath record hota hai)
    btc_candles = fetch_candles("BTCUSDT") if ENABLE_SIGNAL_LOG else None

    for symbol in SYMBOLS:
        candles = fetch_candles(symbol)
        if candles is None:
            continue
        candles = add_indicators(candles)

        # Sirf sab se aakhri (latest completed) candle check karein
        index = len(candles) - 1
        current = candles[index]
        signal = check_single_candle(candles, index)

        if DEBUG:
            t = datetime.fromtimestamp(current["timestamp"] / 1000, tz=timezone.utc).strftime('%H:%M')
            sma = current["Volume_SMA9"]
            sma_str = f"{sma:.0f}" if sma is not None else "NA"
            res = current.get("Resistance")
            gap = current.get("Res_gap_pct")
            res_str = f"{res:.5g}" if res is not None else "NA"
            gap_str = f"{gap:.2f}%" if gap is not None else "NA"
            print(
                f"{symbol} {t}UTC O:{current['open']:.5g} C:{current['close']:.5g} "
                f"EMA9:{current['EMA9']:.5g} EMA21:{current['EMA21']:.5g} "
                f"VWAP:{current['VWAP']:.5g} Vol:{current['volume']:.0f} SMA9:{sma_str} "
                f"Res:{res_str} Gap:{gap_str}"
            )

        if signal is None:
            print(f"❌ {symbol}: Not Matched")
            continue

        t_utc = datetime.fromtimestamp(signal["timestamp"] / 1000, tz=timezone.utc)
        time_str = t_utc.strftime('%Y-%m-%d %H:%M UTC')
        pkt_str = (t_utc + timedelta(hours=5)).strftime('%H:%M PKT')
        print(f"🎯 15m MATCHED: {symbol} at {time_str} (crossed: {'+'.join(signal['Crossed'])})")

        entry = signal["close"]
        tp_price = entry * (1 + TP_PCT / 100.0)
        sl_price = entry * (1 - SL_PCT / 100.0)
        vol_ratio = signal["volume"] / signal["Volume_SMA9"] if signal.get("Volume_SMA9") else 0.0
        crossed = signal["Crossed"]
        res_gap = signal["Res_gap_pct"]

        try:
            features = build_features(symbol, candles, index, signal, btc_candles)
        except Exception as e:                      # record mein masla ho tab bhi alert zaroor jaye
            print(f"⚠️ {symbol}: features nahi ban sake ({e})")
            features = {}
        color_txt = {"G": "🟢 hari", "R": "🔴 laal"}.get(features.get("h1_color", ""), "")
        if color_txt:
            near = features.get("h1_nearest_up_pct", "")
            near_txt = f"{features.get('h1_nearest_up_name', '')} {near}% upar" if near else "koi level upar nahi"
            sweep_txt = "haan" if features.get("h1_sweep_last3") == "1" else "nahi"
            h1_block = (
                f"\n\n⏱ <b>1H:</b> candle {color_txt} | EMA9/EMA21/VWAP: {features.get('h1_pos', '')} "
                f"(A=upar, B=neeche)\n"
                f"   Qareeb tareen rukawat: {near_txt} | Sweep (pichli 3 candles): {sweep_txt}"
            )
        else:
            h1_block = "\n\n⏱ <b>1H:</b> data nahi mila"

        telegram_msg = (
            f"🚨 <b>BULLISH CROSS DETECTED (15m)</b>\n\n"
            f"🪙 <b>Symbol:</b> #{symbol}\n"
            f"🕐 <b>Candle Time:</b> {time_str} ({pkt_str})\n"
            f"📊 <b>Open:</b> {signal['open']:.8g}\n"
            f"🎯 <b>Close (Entry):</b> {entry:.8g}\n"
            f"🔀 <b>Cross kiye:</b> {' + '.join(crossed)} ({len(crossed)}/3)\n"
            f"🔥 <b>Volume:</b> {signal['volume']:.2f} ({vol_ratio:.2f}x SMA9)\n"
            f"🧱 <b>Resistance:</b> {signal['Resistance']:.8g} ({res_gap:.2f}% door)\n\n"
            f"✅ <b>TP (+{TP_PCT}%):</b> {tp_price:.8g}\n"
            f"🛑 <b>SL (-{SL_PCT}%):</b> {sl_price:.8g}"
            f"{h1_block}"
        )
        send_telegram_message(telegram_msg)
        log_signal(symbol, signal, tp_price, sl_price, features)


if __name__ == "__main__":
    scan_market()
