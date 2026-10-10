import os
import time
import requests
from datetime import datetime, timezone

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

TIMEFRAME = "15min"
CANDLE_LIMIT = 100
CANDLE_MS = 15 * 60 * 1000
VOLUME_PERIOD = 9
VWAP_PERIOD = 15

# Support / Resistance: pichli SR_PERIOD candles ka sab se ooncha High / neecha Low
# (Bitget ke S/R indicator, parameter 15, se match kiya hua). Signal candle shamil nahi.
SR_PERIOD = 15
# Close price resistance se kam az kam itne % neeche ho (maximum ki koi hadd nahi)
MIN_RESISTANCE_GAP_PCT = 0.9
# False = agar close pehle hi resistance ke upar nikal chuka ho to signal nahi aaye ga
ALLOW_CLOSE_ABOVE_RESISTANCE = False
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


def fetch_candles(symbol, retries=3):
    params = {"symbol": symbol, "granularity": TIMEFRAME, "limit": str(CANDLE_LIMIT)}
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
            now_ms = int(time.time() * 1000)
            candles = [c for c in candles if c["timestamp"] + CANDLE_MS <= now_ms]
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
    candles[index] ke liye Support/Resistance nikaal kar usi candle mein save karta hai.
    Resistance = pichli SR_PERIOD candles ka max High, Support = min Low
    (candles[index] khud shamil nahi). Gap % = (Resistance - Close) / Close * 100
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
    if index < VOLUME_PERIOD:
        return None
    current = candles[index]
    add_support_resistance(candles, index)
    if current["Volume_SMA9"] is None or current["Resistance"] is None:
        return None

    open_below_all = current["open"] < current["EMA9"] and current["open"] < current["EMA21"] and current["open"] < current["VWAP"]
    close_above_all = current["close"] > current["EMA9"] and current["close"] > current["EMA21"] and current["close"] > current["VWAP"]
    volume_confirmed = current["volume"] > current["Volume_SMA9"]

    gap = current["Res_gap_pct"]
    if gap >= MIN_RESISTANCE_GAP_PCT:
        resistance_ok = True
    else:
        resistance_ok = ALLOW_CLOSE_ABOVE_RESISTANCE and gap < 0

    if open_below_all and close_above_all and volume_confirmed and resistance_ok:
        return current
    return None


def scan_market():
    # Bitget candle close hone ke baad data fetch karne ke liye thora wait
    time.sleep(20)

    print("=" * 50)
    print("BITGET SPOT - BULLISH BREAKOUT SCAN")
    print("=" * 50)

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
            sup = current.get("Support")
            gap = current.get("Res_gap_pct")
            res_str = f"{res:.5g}" if res is not None else "NA"
            sup_str = f"{sup:.5g}" if sup is not None else "NA"
            gap_str = f"{gap:.2f}%" if gap is not None else "NA"
            print(
                f"{symbol} {t}UTC O:{current['open']:.5g} C:{current['close']:.5g} "
                f"EMA9:{current['EMA9']:.5g} EMA21:{current['EMA21']:.5g} "
                f"VWAP:{current['VWAP']:.5g} Vol:{current['volume']:.0f} SMA9:{sma_str} "
                f"Res:{res_str} Sup:{sup_str} Gap:{gap_str}"
            )

        if signal is not None:
            time_str = datetime.fromtimestamp(signal["timestamp"] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
            print(f"🎯 MATCHED: {symbol} at {time_str}")

            telegram_msg = (
                f"🚨 <b>BULLISH BREAKOUT DETECTED</b>\n\n"
                f"🪙 <b>Symbol:</b> #{symbol}\n"
                f"🕐 <b>Candle Time:</b> {time_str}\n"
                f"📊 <b>Open:</b> {signal['open']:.8g}\n"
                f"🎯 <b>Close:</b> {signal['close']:.8g}\n"
                f"🔥 <b>Volume:</b> {signal['volume']:.2f}\n"
                f"🧱 <b>Resistance:</b> {signal['Resistance']:.8g} ({signal['Res_gap_pct']:.2f}% door)\n"
                f"🛟 <b>Support:</b> {signal['Support']:.8g}"
            )
            send_telegram_message(telegram_msg)
        else:
            print(f"❌ {symbol}: Not Matched")


if __name__ == "__main__":
    scan_market()
