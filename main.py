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
MIN_RESISTANCE_GAP_PCT = 1.3
# False = agar close pehle hi resistance ke upar nikal chuka ho to signal nahi aaye ga
ALLOW_CLOSE_ABOVE_RESISTANCE = False

# 1H confirmation: 15m signal ke baad 1H ki CURRENT (chalti hui) candle ki price par:
#  - UPAR: EMA9, EMA21, VWAP(15), Resistance(15) mein se jo level price ke UPAR ho, us se kam az kam
#    H1_MIN_GAP_PCT % faasla ho. Jo level price ke neeche ho use ignore kiya jata hai.
#  - NEECHE: sirf Support dekhte hain. Support price se kam az kam H1_MIN_SUPPORT_GAP_PCT % neeche
#    (door) ho to shart poori. Support is se qareeb ho ya price support se neeche aa chuki ho
#    (support toot gayi) to match complete nahi hota.
ENABLE_1H_CONFIRMATION = True
H1_TIMEFRAME = "1h"
H1_CANDLE_MS = 60 * 60 * 1000
H1_MIN_GAP_PCT = 1.3
H1_MIN_SUPPORT_GAP_PCT = 1.3

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
    """
    only_completed=True  -> sirf band ho chuki candles (15m scan ke liye)
    only_completed=False -> chalti hui (current) candle bhi shamil (1H confirmation ke liye)
    """
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


def check_1h_confirmation(symbol):
    """
    1H ki CURRENT (chalti hui) candle ki price (close) par:
      - EMA9, EMA21, VWAP(15), Resistance(15): jo level price ke UPAR ho us se faasla
        (level - price) / price * 100 kam az kam H1_MIN_GAP_PCT hona chahiye. Jo level price ke
        neeche (ya barabar) ho wo ignore hota hai.
      - Support(15): faasla (price - support) / price * 100 kam az kam H1_MIN_SUPPORT_GAP_PCT
        hona chahiye (support door neeche). Support qareeb ho, ya price support se neeche aa chuki
        ho (faasla negative), to FAIL.

    Wapas: None (data nahi mila) ya (sab_pass, [(naam, value, gap_pct, ok, kind), ...], 1H_price)
    kind: "above" (upar ki rukawat), "below" (neeche, ignore), "support", "na" (data nahi)
    """
    candles = fetch_candles(symbol, timeframe=H1_TIMEFRAME, candle_ms=H1_CANDLE_MS, only_completed=False)
    if candles is None:
        return None
    candles = add_indicators(candles)
    index = len(candles) - 1
    add_support_resistance(candles, index)
    current = candles[index]
    price = current["close"]

    rows = []
    all_pass = True

    # Upar ki rukawatein: sirf wohi levels ginay jaate hain jo price ke upar hon
    for name in ("EMA9", "EMA21", "VWAP", "Resistance"):
        value = current.get(name)
        if value is None or price <= 0:
            rows.append((name, None, None, False, "na"))
            all_pass = False
            continue
        gap = (value - price) / price * 100.0
        if gap <= 0:
            rows.append((name, value, gap, True, "below"))   # price ke neeche: ignore
            continue
        ok = gap >= H1_MIN_GAP_PCT
        rows.append((name, value, gap, ok, "above"))
        if not ok:
            all_pass = False

    # Neeche: sirf Support. Support kam az kam H1_MIN_SUPPORT_GAP_PCT % door ho to PASS,
    # qareeb ho ya toot chuki ho (price support se neeche) to FAIL
    support = current.get("Support")
    if support is None or price <= 0:
        rows.append(("Support", None, None, False, "na"))
        all_pass = False
    else:
        sup_gap = (price - support) / price * 100.0
        ok = sup_gap >= H1_MIN_SUPPORT_GAP_PCT
        rows.append(("Support", support, sup_gap, ok, "support"))
        if not ok:
            all_pass = False

    return all_pass, rows, price


def format_1h_rows(rows):
    parts = []
    for name, value, gap, ok, kind in rows:
        if kind == "na":
            parts.append(f"{name} NA ❌")
        elif kind == "below":
            parts.append(f"{name} neeche (ignore)")
        elif kind == "above":
            parts.append(f"{name} {gap:.2f}% upar {'✅' if ok else '❌'}")
        elif gap >= 0:
            parts.append(f"{name} {gap:.2f}% neeche {'✅' if ok else '❌'}")
        else:
            parts.append(f"{name} toot gayi (price {abs(gap):.2f}% neeche) ❌")
    return " | ".join(parts)


def format_1h_debug(rows):
    """Debug log ke liye: har level ki value aur price se faasla (chart se milane ke liye)"""
    parts = []
    for name, value, gap, ok, kind in rows:
        if kind == "na":
            parts.append(f"{name}:NA")
        elif kind == "below":
            parts.append(f"{name}:{value:.5g}(neeche {abs(gap):.2f}% ignore)")
        elif kind == "above":
            parts.append(f"{name}:{value:.5g}(upar {gap:.2f}% {'OK' if ok else 'FAIL'})")
        elif gap >= 0:
            parts.append(f"{name}:{value:.5g}(neeche {gap:.2f}% {'OK' if ok else 'FAIL'})")
        else:
            parts.append(f"{name}:{value:.5g}(toot gayi, price {abs(gap):.2f}% neeche FAIL)")
    return " | ".join(parts)


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

        # 1H data: DEBUG mein har coin ke liye (log dekhne ke liye), warna sirf 15m signal par
        h1 = None
        if ENABLE_1H_CONFIRMATION and (DEBUG or signal is not None):
            h1 = check_1h_confirmation(symbol)

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
            if ENABLE_1H_CONFIRMATION:
                if h1 is None:
                    print(f"   1H: data nahi mila")
                else:
                    h1_ok_dbg, h1_rows_dbg, h1_price_dbg = h1
                    print(
                        f"   1H [{'PASS' if h1_ok_dbg else 'FAIL'}] Price:{h1_price_dbg:.5g} "
                        f"{format_1h_debug(h1_rows_dbg)}"
                    )

        if signal is not None:
            time_str = datetime.fromtimestamp(signal["timestamp"] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
            print(f"🎯 15m MATCHED: {symbol} at {time_str}")

            # 15m signal ban gaya, ab 1H confirmation check karein
            h1_line = ""
            if ENABLE_1H_CONFIRMATION:
                if h1 is None:
                    # Sirf 1H ka data na milne par alert phir bhi jaye ga (warning ke saath)
                    print(f"⚠️ {symbol}: 1H data nahi mila, 15m signal warning ke saath bheja ja raha hai")
                    h1_line = (
                        f"\n⚠️ <b>1H Confirmation:</b> 1H data nahi mila, check nahi ho saka. "
                        f"Khud 1H chart dekh lein."
                    )
                else:
                    h1_ok, h1_rows, _h1_price = h1
                    h1_summary = format_1h_rows(h1_rows)
                    if not h1_ok:
                        # 1H check hua aur shart poori nahi hui -> alert nahi
                        print(f"❌ {symbol}: 15m matched lekin 1H confirmation FAIL ({h1_summary})")
                        continue
                    print(f"✅ {symbol}: 1H confirmation PASS ({h1_summary})")
                    h1_line = (
                        f"\n⏱ <b>1H Confirmation:</b> ✅ (upar ki rukawatein {H1_MIN_GAP_PCT}%+ door, "
                        f"support {H1_MIN_SUPPORT_GAP_PCT}%+ neeche)\n"
                        f"   {h1_summary}"
                    )

            telegram_msg = (
                f"🚨 <b>BULLISH BREAKOUT DETECTED</b>\n\n"
                f"🪙 <b>Symbol:</b> #{symbol}\n"
                f"🕐 <b>Candle Time:</b> {time_str}\n"
                f"📊 <b>Open:</b> {signal['open']:.8g}\n"
                f"🎯 <b>Close:</b> {signal['close']:.8g}\n"
                f"🔥 <b>Volume:</b> {signal['volume']:.2f}\n"
                f"🧱 <b>Resistance:</b> {signal['Resistance']:.8g} ({signal['Res_gap_pct']:.2f}% door)\n"
                f"🛟 <b>Support:</b> {signal['Support']:.8g}"
                f"{h1_line}"
            )
            send_telegram_message(telegram_msg)
        else:
            print(f"❌ {symbol}: Not Matched")


if __name__ == "__main__":
    scan_market()
