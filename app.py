Bilkul. Neeche sirf code blocks hain. Har block ke andar jo hai wahi copy karna hai.
app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(
    page_title="NSE AI Stock Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("NSE AI Stock Scanner")
st.caption("EMA + MACD + RSI + ROE + FII + DII + 4 Month High")


@st.cache_data(ttl=86400)
def get_nse_stocks():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    try:
        df = pd.read_csv(url)
        return (
            df["SYMBOL"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
    except Exception:
        return [
            "RELIANCE", "TCS", "INFY", "HDFCBANK",
            "ICICIBANK", "SBIN", "LT", "AXISBANK",
            "KOTAKBANK", "ITC", "BHARTIARTL",
            "MARUTI", "SUNPHARMA", "M&M",
            "TATAMOTORS", "TATASTEEL", "HINDALCO",
            "NTPC", "POWERGRID"
        ]


def calculate_indicators(df):
    close = df["Close"]

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

    df["HIGH_4M"] = (
        df["High"]
        .rolling(84)
        .max()
    )

    return df


def scan_stock(symbol):
    try:
        ticker = symbol + ".NS"

        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 210:
            return None

        df = calculate_indicators(df)

        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        price = float(today["Close"])
        ema10 = float(today["EMA10"])
        ema50 = float(today["EMA50"])
        ema200 = float(today["EMA200"])
        rsi = float(today["RSI"])
        macd = float(today["MACD"])
        macd_signal = float(today["MACD_SIGNAL"])
        high4m = float(today["HIGH_4M"])

        ema_cross = (
            yesterday["EMA10"] <= yesterday["EMA50"]
            and today["EMA10"] > today["EMA50"]
        )

        return {
            "Symbol": symbol,
            "Price": round(price, 2),
            "EMA10": round(ema10, 2),
            "EMA50": round(ema50, 2),
            "EMA200": round(ema200, 2),
            "RSI": round(rsi, 2),
            "MACD": round(macd, 2),
            "MACD Signal": round(macd_signal, 2),
            "4M High": round(high4m, 2),
            "EMA Cross": ema_cross,
            "Above 200 EMA": price > ema200,
            "MACD Bullish": macd > macd_signal,
            "RSI > 50": rsi > 50,
            "4M Breakout": price >= high4m * 0.995
        }

    except Exception:
        return None


def calculate_score(row):
    score = 0

    if row["Above 200 EMA"]:
        score += 20

    if row["EMA Cross"]:
        score += 20

    if row["MACD Bullish"]:
        score += 15

    if row["RSI > 50"]:
        score += 10

    if row["4M Breakout"]:
        score += 15

    if row["ROE"] > 10:
        score += 5

    if row["FII Holding"] > 7:
        score += 10

    if row["DII Holding"] > 45:
        score += 5

    return score


st.sidebar.header("Scanner Settings")

roe_min = st.sidebar.number_input(
    "Minimum ROE",
    min_value=0.0,
    value=10.0
)

fii_min = st.sidebar.number_input(
    "Minimum FII Holding",
    min_value=0.0,
    value=7.0
)

dii_min = st.sidebar.number_input(
    "Minimum DII Holding",
    min_value=0.0,
    value=45.0
)

rsi_min = st.sidebar.number_input(
    "Minimum RSI",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

uploaded_file = st.sidebar.file_uploader(
    "Upload fundamentals CSV",
    type=["csv"]
)

run_scanner = st.sidebar.button(
    "RUN NSE SCANNER",
    use_container_width=True
)


fundamentals = None

if uploaded_file is not None:
    try:
        fundamentals = pd.read_csv(uploaded_file)

        fundamentals.columns = [
            str(c).strip().lower()
            for c in fundamentals.columns
        ]

        required = [
            "symbol",
            "roe",
            "fii_holding",
            "dii_holding"
        ]

        if not all(
            column in fundamentals.columns
            for column in required
        ):
            st.error(
                "CSV columns must be: "
                "symbol, roe, fii_holding, dii_holding"
            )
            fundamentals = None
        else:
            fundamentals["symbol"] = (
                fundamentals["symbol"]
                .astype(str)
                .str.upper()
                .str.strip()
            )

    except Exception as error:
        st.error(str(error))
        fundamentals = None


if run_scanner:

    symbols = get_nse_stocks()

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
        st.error("No market data received.")
        st.stop()

    if fundamentals is None:

        st.warning(
            "Upload fundamentals CSV for ROE, FII and DII filtering."
        )

        technical["ROE"] = 0.0
        technical["FII Holding"] = 0.0
        technical["DII Holding"] = 0.0

        final = technical

    else:

        final = technical.merge(
            fundamentals,
            left_on="Symbol",
            right_on="symbol",
            how="inner"
        )

        final.rename(
            columns={
                "roe": "ROE",
                "fii_holding": "FII Holding",
                "dii_holding": "DII Holding"
            },
            inplace=True
        )

        final.drop(
            columns=["symbol"],
            inplace=True,
            errors="ignore"
        )

    final = final[
        (final["Price"] > final["EMA200"]) &
        (final["EMA Cross"] == True) &
        (final["MACD Bullish"] == True) &
        (final["RSI"] > rsi_min) &
        (final["4M Breakout"] == True) &
        (final["ROE"] > roe_min) &
        (final["FII Holding"] > fii_min) &
        (final["DII Holding"] > dii_min)
    ].copy()

    if final.empty:

        st.warning(
            "No stocks match all scanner conditions."
        )

    else:

        final["AI Score"] = final.apply(
            calculate_score,
            axis=1
        )

        final["Signal"] = np.select(
            [
                final["AI Score"] >= 85,
                final["AI Score"] >= 70,
                final["AI Score"] >= 55
            ],
            [
                "STRONG BUY",
                "BUY",
                "WATCH"
            ],
            default="WEAK"
        )

        final = final.sort_values(
            "AI Score",
            ascending=False
        )

        st.success(
            str(len(final)) + " stocks found"
        )

        columns = [
            "Symbol",
            "Price",
            "EMA10",
            "EMA50",
            "EMA200",
            "RSI",
            "MACD",
            "ROE",
            "FII Holding",
            "DII Holding",
            "AI Score",
            "Signal"
        ]

        st.dataframe(
            final[columns],
            use_container_width=True,
            hide_index=True
        )

        csv_data = final[columns].to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "DOWNLOAD RESULTS",
            data=csv_data,
            file_name="nse_ai_scanner_results.csv",
            mime="text/csv",
            use_container_width=True
        )

else:

    st.info(
        "Upload fundamentals.csv and click RUN NSE SCANNER."
    )
requirements.txt
streamlit
pandas
numpy
yfinance
fundamentals.csv
symbol,roe,fii_holding,dii_holding
RELIANCE,12.5,18.2,38.5
TCS,45.2,12.8,35.1
INFY,29.4,15.6,32.7
HDFCBANK,17.1,24.3,28.6
ICICIBANK,18.9,22.1,31.4
Ab GitHub me purana app.py poora delete karke pehla code poora paste karo. requirements.txt me doosra code, aur fundamentals.csv me teesra code.
Phir Commit changes → Streamlit → Reboot app.
