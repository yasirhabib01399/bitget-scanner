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
LOG_TRACK_CANDLES = 88     # itni 15m candles (~22 ghantay) tak dekhte hain, phir bhi kuch hit na ho to NONE

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
def _log_fields():
    return ["signal_ts_ms", "signal_time_utc", "signal_time_pkt", "symbol", "entry_close", "tp_price", "sl_price",
            "open", "volume", "vol_ratio", "body_pct", "res_gap_pct", "n_crossed", "crossed_levels", "status",
            f"result_tp{TP_PCT}_sl{SL_PCT}", "hours_to_exit", "max_up_pct", "max_down_pct", "candles_tracked"]


RESULT_COL = f"result_tp{TP_PCT}_sl{SL_PCT}"


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
            w = csv.DictWriter(fh, fieldnames=_log_fields(), extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception as e:
        print(f"⚠️ Signal log save nahi hua: {e}")


def log_signal(symbol, signal, tp_price, sl_price):
    """Signal ko CSV mein jorta hai (same symbol + candle dobara nahi likhta)."""
    if not ENABLE_SIGNAL_LOG:
        return
    rows = load_signal_log()
    key = str(signal["timestamp"])
    if any(r.get("symbol") == symbol and r.get("signal_ts_ms") == key for r in rows):
        return
    t_utc = datetime.fromtimestamp(signal["timestamp"] / 1000, tz=timezone.utc)
    t_pkt = t_utc + timedelta(hours=5)
    crossed = signal.get("Crossed", [])
    rows.append({
        "signal_ts_ms": key,
        "signal_time_utc": t_utc.strftime("%Y-%m-%d %H:%M"),
        "signal_time_pkt": t_pkt.strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "entry_close": f"{signal['close']:.10g}",
        "tp_price": f"{tp_price:.10g}",
        "sl_price": f"{sl_price:.10g}",
        "open": f"{signal['open']:.10g}",
        "volume": f"{signal['volume']:.2f}",
        "vol_ratio": f"{signal['volume'] / signal['Volume_SMA9']:.2f}" if signal.get("Volume_SMA9") else "",
        "body_pct": f"{(signal['close'] / signal['open'] - 1) * 100:.2f}",
        "res_gap_pct": f"{signal['Res_gap_pct']:.2f}" if signal.get("Res_gap_pct") is not None else "",
        "n_crossed": str(len(crossed)),
        "crossed_levels": "+".join(crossed),
        "status": "OPEN",
    })
    save_signal_log(rows)
    print(f"📝 Signal log mein save: {symbol} {t_pkt.strftime('%Y-%m-%d %H:%M')} PKT")


def _first_hit(after, entry):
    """Entry ke baad pehli candle jis mein TP ya SL touch hua. Dono ek hi candle mein ho to SL maana jata hai."""
    tp = entry * (1 + TP_PCT / 100.0)
    sl = entry * (1 - SL_PCT / 100.0)
    for k, c in enumerate(after, 1):
        if c["low"] <= sl:
            return "SL", k
        if c["high"] >= tp:
            return "TP", k
    return None, None


def update_signal_outcomes():
    """Khule (OPEN) signals ka result bharta hai: TP pehle hit hua ya SL, aur price kitni upar/neeche gayi."""
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
        lost = sig_ts < candles[0]["timestamp"]            # signal candle ab data window se bahar
        after = [c for c in candles if c["timestamp"] > sig_ts][:LOG_TRACK_CANDLES]
        res, k = _first_hit(after, entry)
        done = True
        if res is not None:
            r[RESULT_COL], r["hours_to_exit"] = res, f"{k * 0.25:.2f}"
        elif lost:
            r[RESULT_COL] = "UNKNOWN"
        elif len(after) >= LOG_TRACK_CANDLES:
            r[RESULT_COL] = "NONE"
        else:
            done = False
        if after:
            r["max_up_pct"] = f"{(max(c['high'] for c in after) / entry - 1) * 100:.2f}"
            r["max_down_pct"] = f"{(1 - min(c['low'] for c in after) / entry) * 100:.2f}"
        r["candles_tracked"] = str(len(after))
        if done:
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
        )
        send_telegram_message(telegram_msg)
        log_signal(symbol, signal, tp_price, sl_price)


if __name__ == "__main__":
    scan_market()
