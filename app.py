import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="NSE FII Momentum Scanner", page_icon="🚀", layout="wide")

st.title("🚀 NSE FII Momentum Scanner")
st.caption("10 EMA Cross 50 EMA | Price Above 200 EMA | RSI > 50 | High Momentum | Institutional Holding > 7%")

@st.cache_data(ttl=86400)
def get_symbols():
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        return df["SYMBOL"].dropna().astype(str).str.upper().str.strip().unique().tolist()
    except Exception:
        return []

def scan_stock(symbol):
    try:
        df = yf.download(
            symbol + ".NS",
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df.empty or len(df) < 210:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        high = df["High"]
        volume = df["Volume"]

        df["EMA10"] = close.ewm(span=10, adjust=False).mean()
        df["EMA50"] = close.ewm(span=50, adjust=False).mean()
        df["EMA200"] = close.ewm(span=200, adjust=False).mean()

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        df["MACD"] = ema12 - ema26
        df["SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        df["VOL20"] = volume.rolling(20).mean()
        df["HIGH20"] = high.shift(1).rolling(20).max()

        today = df.iloc[-1]
        recent = df.tail(6)

        price = float(today["Close"])
        ema10 = float(today["EMA10"])
        ema50 = float(today["EMA50"])
        ema200 = float(today["EMA200"])
        rsi = float(today["RSI"])
        macd = float(today["MACD"])
        signal = float(today["SIGNAL"])
        vol = float(today["Volume"])
        avg_vol = float(today["VOL20"])
        high20 = float(today["HIGH20"])

        if avg_vol <= 0:
            return None

        volume_x = vol / avg_vol

        cross_recent = False

        for i in range(1, len(recent)):
            p = recent.iloc[i - 1]
            c = recent.iloc[i]

            if p["EMA10"] <= p["EMA50"] and c["EMA10"] > c["EMA50"]:
                cross_recent = True

        above_200 = price > ema200
        breakout_200 = above_200 and price > ema200 * 1.01
        breakout_20 = price > high20
        macd_bullish = macd > signal

        momentum = (
            price > ema10
            and rsi > 50
            and volume_x >= 1.3
            and macd_bullish
        )

        score = 0

        if cross_recent:
            score += 25
        if above_200:
            score += 20
        if breakout_200:
            score += 15
        if breakout_20:
            score += 15
        if rsi > 50:
            score += 10
        if rsi >= 60:
            score += 5
        if volume_x >= 1.3:
            score += 10
        if volume_x >= 2:
            score += 5
        if macd_bullish:
            score += 10

        return {
            "Symbol": symbol,
            "Price": round(price, 2),
            "EMA10": round(ema10, 2),
            "EMA50": round(ema50, 2),
            "EMA200": round(ema200, 2),
            "RSI": round(rsi, 2),
            "MACD": round(macd, 2),
            "Volume X": round(volume_x, 2),
            "EMA Cross": cross_recent,
            "Above 200 EMA": above_200,
            "200 Breakout": breakout_200,
            "20D Breakout": breakout_20,
            "MACD Bullish": macd_bullish,
            "Momentum": momentum,
            "Score": score
        }

    except Exception:
        return None

def get_institutional_holding(symbol):
    try:
        info = yf.Ticker(symbol + ".NS").info
        value = info.get("heldPercentInstitutions")

        if value is None:
            return np.nan

        return float(value) * 100

    except Exception:
        return np.nan

if st.button("🚀 SCAN NSE", use_container_width=True):

    symbols = get_symbols()

    if not symbols:
        st.error("NSE stock list could not be loaded.")
        st.stop()

    st.info("Scanning " + str(len(symbols)) + " NSE stocks...")

    technical_results = []
    progress = st.progress(0)

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = {
            executor.submit(scan_stock, symbol): symbol
            for symbol in symbols
        }

        for i, future in enumerate(as_completed(futures), 1):

            try:
                result = future.result()

                if result is not None:
                    technical_results.append(result)

            except Exception:
                pass

            progress.progress(i / len(futures))

    progress.empty()

    df = pd.DataFrame(technical_results)

    if df.empty:
        st.error("No technical data received.")
        st.stop()

    candidates = df[
        (df["Above 200 EMA"] == True)
        &
        (df["RSI"] > 50)
        &
        (df["Momentum"] == True)
    ].copy()

    candidates = candidates.sort_values(
        "Score",
        ascending=False
    ).head(200)

    if candidates.empty:
        st.warning("No momentum candidates found.")
        st.stop()

    st.info("Checking institutional holdings...")

    holdings = []
    progress = st.progress(0)

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = {
            executor.submit(
                get_institutional_holding,
                symbol
            ): symbol
            for symbol in candidates["Symbol"]
        }

        for i, future in enumerate(as_completed(futures), 1):

            symbol = futures[future]

            try:
                value = future.result()
            except Exception:
                value = np.nan

            holdings.append({
                "Symbol": symbol,
                "Institutional Holding": value
            })

            progress.progress(i / len(futures))

    progress.empty()

    hdf = pd.DataFrame(holdings)

    result = candidates.merge(
        hdf,
        on="Symbol",
        how="left"
    )

    result = result.sort_values(
        "Score",
        ascending=False
    )

    strict = result[
        result["Institutional Holding"] > 7
    ].copy()

    if not strict.empty:

        st.success(
            str(len(strict)) +
            " stocks match Institutional Holding > 7%."
        )

        display = strict.head(50)

    else:

        st.warning(
            "No stock has Institutional Holding > 7% among the current momentum candidates."
        )

        st.subheader("Top Momentum Near-Matches")

        display = result.head(30)

    display["Signal"] = np.select(
        [
            display["Score"] >= 80,
            display["Score"] >= 60
        ],
        [
            "STRONG",
            "BUY"
        ],
        default="WATCH"
    )

    columns = [
        "Symbol",
        "Price",
        "EMA10",
        "EMA50",
        "EMA200",
        "RSI",
        "MACD",
        "Volume X",
        "Institutional Holding",
        "Score",
        "Signal"
    ]

    st.dataframe(
        display[columns],
        use_container_width=True,
        hide_index=True
    )

    csv = display[columns].to_csv(index=False).encode("utf-8")

    st.download_button(
        "DOWNLOAD RESULTS",
        data=csv,
        file_name="nse_momentum_scanner.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("Press SCAN NSE to start.")
