"""
Multi-Symbol Automated Trading Bot for WEEX (Futures/Swap) -> EMA+RSI trend-filtered strategy.
Runs on a schedule via GitHub Actions. Places REAL orders with REAL money, on
multiple symbols independently (each symbol has its own position/state).

⚠ Educational template. Not financial advice. Test extensively with tiny size
before trusting this with real capital. You are fully responsible for
whatever this bot does to your account. Running more symbols means more
simultaneous real-money positions -> higher total exposure. Adjust
FIXED_TRADE_USD accordingly if you want to keep total risk the same.
"""

import os
import sys
import json
import time
import requests
import ccxt

# ---- Config from environment variables (GitHub Secrets) -------------------
WEEX_API_KEY    = os.environ["WEEX_API_KEY"]
WEEX_SECRET_KEY = os.environ["WEEX_SECRET_KEY"]
WEEX_PASSPHRASE = os.environ["WEEX_PASSPHRASE"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY")

# ccxt unified symbols for WEEX USDT-margined perpetuals. Remove/add as needed.
SYMBOLS = [
    "BTC/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "LTC/USDT:USDT",
    "DOGE/USDT:USDT",
]

TIMEFRAME    = "5m"
EMA_FAST     = 9
EMA_SLOW     = 21
EMA_TREND    = 200
RSI_PERIOD   = 14
ATR_PERIOD   = 14
ATR_SL_MULT  = 1.5
ATR_TP_MULT  = 3.0

LEVERAGE        = 3          # deliberately low leverage despite exchange allowing much more
MARGIN_MODE     = "isolated" # isolated, not cross - limits risk to this position's margin only
FIXED_TRADE_USD = 2.0        # fixed position size in USD (notional) PER SYMBOL, regardless of account equity
STATE_FILE      = "state.json"

# Small delay between symbols to be gentle on rate limits / avoid bursts.
SYMBOL_DELAY_SEC = 1.0

# ---------------------------------------------------------------------------


def get_exchange():
    exchange = ccxt.weex({
        "apiKey": WEEX_API_KEY,
        "secret": WEEX_SECRET_KEY,
        "password": WEEX_PASSPHRASE,
        "options": {"defaultType": "swap"},
    })
    return exchange


def ema(values, period):
    k = 2 / (period + 1)
    out = [None] * len(values)
    if len(values) < period:
        return out
    sma = sum(values[:period]) / period
    out[period - 1] = sma
    prev = sma
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(closes, period=RSI_PERIOD):
    out = [None] * len(closes)
    if len(closes) <= period:
        return out
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
    out[period] = 100 - (100 / (1 + rs))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        out[i + 1] = 100 - (100 / (1 + rs))
    return out


def atr(highs, lows, closes, period=ATR_PERIOD):
    out = [None] * len(closes)
    if len(closes) <= period:
        return out
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    avg_tr = sum(trs[:period]) / period
    out[period] = avg_tr
    for i in range(period, len(trs)):
        avg_tr = (avg_tr * (period - 1) + trs[i]) / period
        out[i + 1] = avg_tr
    return out


def load_state():
    """State is now keyed per symbol: {symbol: {"last_candle":.., "position_open":..}}"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            data = json.load(f)
        # Migrate old single-symbol state file format if found.
        if "last_candle" in data or "position_open" in data:
            print("Old single-symbol state file detected - migrating to BTC entry.")
            data = {"BTC/USDT:USDT": {
                "last_candle": data.get("last_candle"),
                "position_open": data.get("position_open", False),
            }}
        return data
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def get_ai_analysis(symbol, signal, price, sl, tp, rsi_val, atr_val, closes):
    if not GROQ_API_KEY:
        return None
    recent = ", ".join(f"{c:.4f}" for c in closes[-10:])
    prompt = (
        f"یک سیگنال معاملاتی خودکار روی {symbol} بر اساس EMA+RSI صادر شده:\n"
        f"جهت: {signal}\nقیمت: {price}\nRSI: {rsi_val:.1f}\nATR: {atr_val}\n"
        f"حد ضرر: {sl} | حد سود: {tp}\nقیمت‌های اخیر: {recent}\n\n"
        f"در حداکثر ۳ جمله‌ی فارسی خلاصه بگو این سیگنال با روند اخیر هم‌راستاست یا نه. "
        f"لحن محتاطانه، نه توصیه قطعی."
    )
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "openai/gpt-oss-120b",
                  "messages": [{"role": "user", "content": prompt}], "max_tokens": 300},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip() or None
    except Exception as e:
        print(f"AI analysis skipped (error: {e})")
        return None


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
    r.raise_for_status()


def process_symbol(exchange, symbol, state):
    """Run the strategy for a single symbol. Mutates state[symbol] in place."""
    sym_state = state.setdefault(symbol, {"last_candle": None, "position_open": False})

    try:
        exchange.set_margin_mode(MARGIN_MODE, symbol)
    except Exception as e:
        print(f"[{symbol}] Could not set margin mode (may already be set): {e}")

    try:
        exchange.set_leverage(
            LEVERAGE, symbol,
            params={
                "isolatedLongLeverage": LEVERAGE,
                "isolatedShortLeverage": LEVERAGE,
            },
        )
    except Exception as e:
        print(f"[{symbol}] Could not set leverage (may already be set): {e}")

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=250)
    closes = [c[4] for c in ohlcv]
    highs  = [c[2] for c in ohlcv]
    lows   = [c[3] for c in ohlcv]
    times  = [c[0] for c in ohlcv]

    ema_fast = ema(closes, EMA_FAST)
    ema_slow = ema(closes, EMA_SLOW)
    ema_trend = ema(closes, EMA_TREND)
    rsi_vals = rsi(closes)
    atr_vals = atr(highs, lows, closes)

    if any(v[-1] is None or v[-2] is None for v in (ema_fast, ema_slow, ema_trend, rsi_vals, atr_vals)):
        print(f"[{symbol}] Not enough data yet.")
        return

    last_candle_time = times[-1]
    if sym_state.get("last_candle") == last_candle_time:
        print(f"[{symbol}] Already checked this candle, skipping.")
        return

    if sym_state.get("position_open"):
        print(f"[{symbol}] Position already open, not opening a new one.")
        sym_state["last_candle"] = last_candle_time
        return

    price = closes[-1]
    trend_up = price > ema_trend[-1]
    trend_down = price < ema_trend[-1]
    cross_up = ema_fast[-2] <= ema_slow[-2] and ema_fast[-1] > ema_slow[-1]
    cross_down = ema_fast[-2] >= ema_slow[-2] and ema_fast[-1] < ema_slow[-1]

    signal = None
    if cross_up and rsi_vals[-1] > 50 and trend_up:
        signal = "BUY"
    elif cross_down and rsi_vals[-1] < 50 and trend_down:
        signal = "SELL"

    if not signal:
        print(f"[{symbol}] No signal. RSI={rsi_vals[-1]:.1f} price={price}")
        sym_state["last_candle"] = last_candle_time
        return

    atr_val = atr_vals[-1]
    sl_dist = atr_val * ATR_SL_MULT
    tp_dist = atr_val * ATR_TP_MULT

    if signal == "BUY":
        sl_price = price - sl_dist
        tp_price = price + tp_dist
        side = "buy"
    else:
        sl_price = price + sl_dist
        tp_price = price - tp_dist
        side = "sell"

    amount = FIXED_TRADE_USD / price
    risk_amount = amount * sl_dist

    if amount <= 0:
        print(f"[{symbol}] Invalid amount={amount}")
        return

    order = exchange.create_order(
        symbol, "market", side, amount,
        params={
            "marginMode": MARGIN_MODE,
            "stopLoss": {"triggerPrice": sl_price},
            "takeProfit": {"triggerPrice": tp_price},
        },
    )

    ai_note = get_ai_analysis(symbol, signal, price, sl_price, tp_price, rsi_vals[-1], atr_val, closes)

    emoji = "🟢" if signal == "BUY" else "🔴"
    msg = (
        f"{emoji} سیگنال {signal} روی {symbol} (M5)\n"
        f"قیمت ورود: {price}\n"
        f"اندازه پوزیشن: {amount:.6f} (~{FIXED_TRADE_USD:.2f} USDT, اهرم {LEVERAGE}x, {MARGIN_MODE})\n"
        f"حد ضرر: {sl_price} | حد سود: {tp_price}\n"
        f"RSI: {rsi_vals[-1]:.1f} | ATR: {atr_val}\n"
        f"ریسک این معامله در صورت خوردن حد ضرر: ~{risk_amount:.2f} USDT\n"
    )
    if ai_note:
        msg += f"\n🤖 تحلیل هوش مصنوعی:\n{ai_note}\n"
    msg += f"\n⚠ معامله به‌صورت خودکار ثبت شد. Order ID: {order.get('id', 'N/A')}"

    send_telegram(msg)
    print(f"[{symbol}] Opened {signal} position. size={amount:.6f} SL={sl_price} TP={tp_price}")

    sym_state["last_candle"] = last_candle_time
    sym_state["position_open"] = True


def main():
    exchange = get_exchange()
    exchange.load_markets()
    state = load_state()

    for symbol in SYMBOLS:
        try:
            process_symbol(exchange, symbol, state)
        except Exception as e:
            # One symbol failing (e.g. bad market data, API hiccup) should not
            # stop the others from being checked.
            print(f"[{symbol}] Error: {e}", file=sys.stderr)
        finally:
            save_state(state)  # persist progress after each symbol, not just at the end
        time.sleep(SYMBOL_DELAY_SEC)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
