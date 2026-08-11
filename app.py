import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(
    page_title="NSE FII Momentum Scanner",
    page_icon="🚀",
    layout="wide"
)

st.title("NSE FII Momentum Scanner")
st.write("10 EMA > 50 EMA | Fresh 200 EMA Breakout | RSI > 50 | High Momentum | FII > 7%")

FII_MIN = 7.0


@st.cache_data(ttl=86400)
def get_nse_symbols():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    try:
        df = pd.read_csv(url)

        return (
            df["SYMBOL"]
            .dropna()
            .astype(str)
            .str.upper()
            .str.strip()
            .unique()
            .tolist()
        )

    except Exception:
        return []


def calculate_indicators(df):

    close = df["Close"]
    high = df["High"]
    volume = df["Volume"]

    df["EMA10"] = close.ewm(
        span=10,
        adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    df["VOL20"] = volume.rolling(20).mean()

    df["HIGH20"] = high.shift(1).rolling(20).max()

    return df


def scan_stock(symbol):

    try:

        data = yf.download(
            symbol + ".NS",
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data.empty or len(data) < 210:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = calculate_indicators(data)

        today = data.iloc[-1]
        yesterday = data.iloc[-2]

        price = float(today["Close"])
        ema10 = float(today["EMA10"])
        ema50 = float(today["EMA50"])
        ema200 = float(today["EMA200"])
        rsi = float(today["RSI"])

        macd = float(today["MACD"])
        macd_signal = float(today["MACD_SIGNAL"])

        volume = float(today["Volume"])
        avg_volume = float(today["VOL20"])

        previous_high = float(today["HIGH20"])

        if avg_volume <= 0:
            return None

        volume_ratio = volume / avg_volume

        ema_cross = (
            yesterday["EMA10"] <= yesterday["EMA50"]
            and ema10 > ema50
        )

        fresh_200_breakout = (
            yesterday["Close"] <= yesterday["EMA200"]
            and price > ema200
        )

        twenty_day_breakout = (
            price > previous_high
        )

        macd_bullish = (
            macd > macd_signal
            and macd > 0
        )

        high_momentum = (
            price > ema10
            and rsi >= 55
            and volume_ratio >= 1.5
            and macd_bullish
        )

        score = 0

        if ema_cross:
            score += 25

        if fresh_200_breakout:
            score += 25

        if rsi > 50:
            score += 10

        if rsi >= 60:
            score += 5

        if volume_ratio >= 1.5:
            score += 15

        if volume_ratio >= 2:
            score += 5

        if twenty_day_breakout:
            score += 10

        if macd_bullish:
            score += 5

        return {
            "Symbol": symbol,
            "Price": round(price, 2),
            "EMA10": round(ema10, 2),
            "EMA50": round(ema50, 2),
            "EMA200": round(ema200, 2),
            "RSI": round(rsi, 2),
            "MACD": round(macd, 2),
            "Volume X": round(volume_ratio, 2),
            "EMA Cross": ema_cross,
            "200 EMA Breakout": fresh_200_breakout,
            "20D Breakout": twenty_day_breakout,
            "MACD Bullish": macd_bullish,
            "High Momentum": high_momentum,
            "Momentum Score": score
        }

    except Exception:
        return None


def get_fii_holding(symbol):

    try:

        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info

        value = info.get("heldPercentInstitutions")

        if value is None:
            return None

        return float(value) * 100

    except Exception:
        return None


if st.button(
    "SCAN ALL NSE STOCKS",
    use_container_width=True
):

    symbols = get_nse_symbols()

    if not symbols:

        st.error(
            "Unable to load NSE stock list."
        )

        st.stop()

    st.info(
        "Scanning " + str(len(symbols)) + " NSE stocks..."
    )

    results = []

    progress = st.progress(0)

    completed = 0

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        futures = {
            executor.submit(
                scan_stock,
                symbol
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):

            result = future.result()

            if result is not None:
                results.append(result)

            completed += 1

            progress.progress(
                completed / len(futures)
            )

    progress.empty()

    technical = pd.DataFrame(results)

    if technical.empty:

        st.warning(
            "No market data received."
        )

        st.stop()

    st.info(
        "Checking FII holdings..."
    )

    fii_results = []

    progress = st.progress(0)

    completed = 0

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        futures = {
            executor.submit(
                get_fii_holding,
                symbol
            ): symbol
            for symbol in technical["Symbol"]
        }

        for future in as_completed(futures):

            symbol = futures[future]

            try:
                fii = future.result()
            except Exception:
                fii = None

            fii_results.append(
                {
                    "Symbol": symbol,
                    "FII Holding": fii
                }
            )

            completed += 1

            progress.progress(
                completed / len(futures)
            )

    progress.empty()

    fii_df = pd.DataFrame(fii_results)

    final = technical.merge(
        fii_df,
        on="Symbol",
        how="left"
    )

    final["FII Holding"] = pd.to_numeric(
        final["FII Holding"],
        errors="coerce"
    )

    final = final[
        (final["EMA Cross"] == True)
        &
        (final["200 EMA Breakout"] == True)
        &
        (final["RSI"] > 50)
        &
        (final["High Momentum"] == True)
        &
        (final["FII Holding"] > FII_MIN)
    ].copy()

    final = final.sort_values(
        "Momentum Score",
        ascending=False
    )

    if final.empty:

        st.warning(
            "No stocks match all conditions."
        )

    else:

        final["Signal"] = np.select(
            [
                final["Momentum Score"] >= 90,
                final["Momentum Score"] >= 75,
                final["Momentum Score"] >= 60
            ],
            [
                "STRONG MOMENTUM",
                "BUY",
                "WATCH"
            ],
            default="WEAK"
        )

        st.success(
            str(len(final)) +
            " stocks found."
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
            "FII Holding",
            "Momentum Score",
            "Signal"
        ]

        st.dataframe(
            final[columns],
            use_container_width=True,
            hide_index=True
        )

        csv = final[columns].to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "DOWNLOAD RESULTS",
            data=csv,
            file_name="nse_fii_momentum_scanner.csv",
            mime="text/csv",
            use_container_width=True
        )

else:

    st.info(
        "Press SCAN ALL NSE STOCKS to start."
    )
