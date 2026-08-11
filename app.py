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

st.title("🚀 NSE FII Momentum Scanner")
st.caption(
    "10 EMA > 50 EMA | 200 EMA Breakout | RSI > 50 | "
    "Volume Momentum | MACD | Institutional Holding > 7%"
)


@st.cache_data(ttl=86400)
def get_nse_symbols():

    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    try:
        df = pd.read_csv(url)

        symbols = (
            df["SYMBOL"]
            .dropna()
            .astype(str)
            .str.upper()
            .str.strip()
            .unique()
            .tolist()
        )

        return symbols

    except Exception:
        return []


def indicators(data):

    close = data["Close"]
    high = data["High"]
    volume = data["Volume"]

    data["EMA10"] = close.ewm(
        span=10,
        adjust=False
    ).mean()

    data["EMA50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    data["EMA200"] = close.ewm(
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

    data["MACD"] = ema12 - ema26

    data["MACD_SIGNAL"] = data["MACD"].ewm(
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

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    data["VOL20"] = volume.rolling(20).mean()

    data["HIGH20"] = high.shift(1).rolling(20).max()

    data["HIGH52"] = high.rolling(252).max()

    return data


def scan_stock(symbol):

    try:

        data = yf.download(
            symbol + ".NS",
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data.empty or len(data) < 210:
            return None

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):
            data.columns = (
                data.columns
                .get_level_values(0)
            )

        data = indicators(data)

        today = data.iloc[-1]

        recent = data.tail(6)

        price = float(today["Close"])
        ema10 = float(today["EMA10"])
        ema50 = float(today["EMA50"])
        ema200 = float(today["EMA200"])
        rsi = float(today["RSI"])

        macd = float(today["MACD"])
        macd_signal = float(
            today["MACD_SIGNAL"]
        )

        volume = float(today["Volume"])
        avg_volume = float(
            today["VOL20"]
        )

        high20 = float(
            today["HIGH20"]
        )

        high52 = float(
            today["HIGH52"]
        )

        if avg_volume <= 0:
            return None

        volume_x = volume / avg_volume

        cross_recent = False

        for i in range(1, len(recent)):

            previous = recent.iloc[i - 1]
            current = recent.iloc[i]

            if (
                previous["EMA10"]
                <= previous["EMA50"]
                and
                current["EMA10"]
                >
                current["EMA50"]
            ):
                cross_recent = True

        price_above_200 = (
            price > ema200
        )

        breakout_200 = (
            price > ema200
            and
            price > ema50
        )

        fresh_20_breakout = (
            price > high20
        )

        rsi_ok = (
            rsi > 50
        )

        volume_momentum = (
            volume_x >= 1.5
        )

        macd_bullish = (
            macd > macd_signal
            and
            macd > 0
        )

        strong_momentum = (
            price > ema10
            and
            rsi >= 55
            and
            volume_x >= 1.5
            and
            macd_bullish
        )

        score = 0

        if cross_recent:
            score += 25

        if breakout_200:
            score += 20

        if fresh_20_breakout:
            score += 15

        if price_above_200:
            score += 10

        if rsi_ok:
            score += 10

        if rsi >= 60:
            score += 5

        if volume_x >= 1.5:
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
            "Volume X": round(
                volume_x,
                2
            ),
            "52W High": round(
                high52,
                2
            ),
            "EMA Cross Recent": cross_recent,
            "Above 200 EMA": price_above_200,
            "20D Breakout": fresh_20_breakout,
            "MACD Bullish": macd_bullish,
            "High Momentum": strong_momentum,
            "Score": score
        }

    except Exception:
        return None


def get_institutional_holding(symbol):

    try:

        ticker = yf.Ticker(
            symbol + ".NS"
        )

        info = ticker.info

        value = info.get(
            "heldPercentInstitutions"
        )

        if value is None:
            return np.nan

        return float(value) * 100

    except Exception:
        return np.nan


if st.button(
    "🚀 SCAN NSE",
    use_container_width=True
):

    symbols = get_nse_symbols()

    if not symbols:

        st.error(
            "NSE stock list could not be loaded."
        )

        st.stop()

    st.info(
        "Scanning "
        + str(len(symbols))
        + " NSE stocks..."
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

        for future in as_completed(
            futures
        ):

            result = future.result()

            if result is not None:
                results.append(result)

            completed += 1

            progress.progress(
                completed
                /
                len(futures)
            )

    progress.empty()

    technical = pd.DataFrame(
        results
    )

    if technical.empty:

        st.error(
            "No technical data received."
        )

        st.stop()

    st.info(
        "Checking institutional holdings..."
    )

    holdings = []

    progress = st.progress(0)

    completed = 0

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        futures = {
            executor.submit(
                get_institutional_holding,
                symbol
            ): symbol
            for symbol in technical[
                "Symbol"
            ]
        }

        for future in as_completed(
            futures
        ):

            symbol = futures[
                future
            ]

            try:
                holding = future.result()
            except Exception:
                holding = np.nan

            holdings.append(
                {
                    "Symbol": symbol,
                    "Institutional Holding": holding
                }
            )

            completed += 1

            progress.progress(
                completed
                /
                len(futures)
            )

    progress.empty()

    holding_df = pd.DataFrame(
        holdings
    )

    df = technical.merge(
        holding_df,
        on="Symbol",
        how="left"
    )

    df[
        "Institutional Holding"
    ] = pd.to_numeric(
        df[
            "Institutional Holding"
        ],
        errors="coerce"
    )

    strict = df[
        (df["EMA Cross Recent"] == True)
        &
        (df["Above 200 EMA"] == True)
        &
        (df["20D Breakout"] == True)
        &
        (df["RSI"] > 50)
        &
        (df["High Momentum"] == True)
        &
        (
            df[
                "Institutional Holding"
            ] > 7
        )
    ].copy()

    strict = strict.sort_values(
        "Score",
        ascending=False
    )

    if not strict.empty:

        st.success(
            str(len(strict))
            +
            " stocks match all conditions."
        )

        strict["Signal"] = np.select(
            [
                strict["Score"] >= 90,
                strict["Score"] >= 75,
                strict["Score"] >= 60
            ],
            [
                "🔥 STRONG",
                "🟢 BUY",
                "🟡 WATCH"
            ],
            default="WEAK"
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
            strict[columns],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No stock matches every condition today."
        )

        st.subheader(
            "Top Momentum Near-Matches"
        )

        near = df[
            (df["Above 200 EMA"] == True)
            &
            (df["RSI"] > 50)
            &
            (df["High Momentum"] == True)
        ].copy()

        near = near.sort_values(
            "Score",
            ascending=False
        ).head(30)

        if not near.empty:

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
                "Score"
            ]

            st.dataframe(
                near[columns],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No strong momentum stocks found."
            )

    download_df = (
        strict
        if not strict.empty
        else near
    )

    if not download_df.empty:

        csv = download_df.to_csv(
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
        "Press SCAN NSE to start."
    )
requirements.txt
streamlit
pandas
numpy
yfinance
