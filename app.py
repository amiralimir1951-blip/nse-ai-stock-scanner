Yes. Below is a complete GitHub + Streamlit starter scanner for your personal NSE AI scanner.
It scans the NSE universe using:
Price > 200 EMA
10 EMA crossing above 50 EMA
MACD bullish
RSI > 50
4-month high
ROE > 10%
FII holding > 7%
DII holding > 45%
AI-style score
CSV download
Important: market-price/technical data can be fetched from Yahoo Finance, but ROE/FII/DII data is not reliably available for every NSE stock from the same free source. So this version includes a fundamentals CSV input, which makes the scanner actually usable rather than pretending those fields are available everywhere.
1. app.py
Writing
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

st.title("📈 NSE AI Stock Scanner")
st.caption("10 EMA × 50 EMA • 200 EMA • MACD • RSI • 4-Month High • ROE • FII • DII")

# =========================================================
# NSE STOCK UNIVERSE
# =========================================================

@st.cache_data(ttl=86400)
def get_nse_symbols():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    try:
        df = pd.read_csv(url)

        symbols = (
            df["SYMBOL"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        return symbols

    except Exception:
        # Backup list if NSE blocks the request
        return [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "SBIN", "LT", "AXISBANK", "KOTAKBANK", "ITC",
            "BHARTIARTL", "MARUTI", "SUNPHARMA", "TATAMOTORS",
            "M&M", "TATASTEEL", "HINDALCO", "NTPC", "POWERGRID"
        ]


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(df):

    close = df["Close"]

    df["EMA10"] = close.ewm(span=10, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # 4-month high ≈ 84 trading sessions
    df["HIGH_4M"] = df["High"].rolling(84).max()

    return df


# =========================================================
# DOWNLOAD PRICE DATA
# =========================================================

def scan_stock(symbol):

    ticker = symbol + ".NS"

    try:

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

        required = ["Close", "High"]

        if not all(x in df.columns for x in required):
            return None

        if len(df) < 210:
            return None

        df = calculate_indicators(df)

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        price = float(latest["Close"])

        ema10 = float(latest["EMA10"])
        ema50 = float(latest["EMA50"])
        ema200 = float(latest["EMA200"])

        macd = float(latest["MACD"])
        macd_signal = float(latest["MACD_SIGNAL"])

        rsi = float(latest["RSI"])

        high4m = float(latest["HIGH_4M"])

        # 10 EMA CROSS ABOVE 50 EMA
        ema_cross = (
            previous["EMA10"] <= previous["EMA50"]
            and latest["EMA10"] > latest["EMA50"]
        )

        price_above_200 = price > ema200

        macd_bullish = macd > macd_signal

        rsi_ok = rsi > 50

        four_month_high = price >= high4m * 0.995

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
            "Above 200 EMA": price_above_200,
            "MACD Bullish": macd_bullish,
            "RSI > 50": rsi_ok,
            "4M High": four_month_high
        }

    except Exception:
        return None


# =========================================================
# FUNDAMENTAL DATA
# =========================================================

def load_fundamentals(uploaded_file):

    if uploaded_file is None:
        return None

    try:

        df = pd.read_csv(uploaded_file)

        df.columns = [
            str(c).strip().lower().replace(" ", "_")
            for c in df.columns
        ]

        required = [
            "symbol",
            "roe",
            "fii_holding",
            "dii_holding"
        ]

        if not all(c in df.columns for c in required):
            st.error(
                "CSV must contain: symbol, roe, fii_holding, dii_holding"
            )
            return None

        df["symbol"] = (
            df["symbol"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        return df

    except Exception as e:

        st.error(f"Fundamental CSV error: {e}")

        return None


# =========================================================
# AI SCORE
# =========================================================

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

    if row["4M High"]:
        score += 15

    if row["ROE"] > 10:
        score += 5

    if row["FII Holding"] > 7:
        score += 10

    if row["DII Holding"] > 45:
        score += 5

    return score


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Scanner Settings")

roe_min = st.sidebar.number_input(
    "Minimum ROE %",
    value=10.0
)

fii_min = st.sidebar.number_input(
    "Minimum FII Holding %",
    value=7.0
)

dii_min = st.sidebar.number_input(
    "Minimum DII Holding %",
    value=45.0
)

rsi_min = st.sidebar.number_input(
    "Minimum RSI",
    value=50.0
)

scan_button = st.sidebar.button(
    "🚀 RUN NSE SCANNER",
    use_container_width=True
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload fundamentals.csv containing:\n\n"
    "symbol, roe, fii_holding, dii_holding"
)


# =========================================================
# FUNDAMENTAL CSV
# =========================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload fundamentals CSV",
    type=["csv"]
)

fundamentals = load_fundamentals(uploaded_file)


# =========================================================
# MAIN SCANNER
# =========================================================

if scan_button:

    symbols = get_nse_symbols()

    st.info(f"Scanning {len(symbols)} NSE stocks...")

    results = []

    progress = st.progress(0)

    completed = 0

    # Parallel downloads
    with ThreadPoolExecutor(max_workers=8) as executor:

        futures = {
            executor.submit(scan_stock, symbol): symbol
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

    if not results:

        st.warning("No technical data returned.")
        st.stop()

    technical = pd.DataFrame(results)

    # =====================================================
    # MERGE FUNDAMENTALS
    # =====================================================

    if fundamentals is not None:

        technical["Symbol"] = (
            technical["Symbol"]
            .astype(str)
            .str.upper()
        )

        fundamentals["symbol"] = (
            fundamentals["symbol"]
            .astype(str)
            .str.upper()
        )

        final = technical.merge(
            fundamentals,
            left_on="Symbol",
            right_on="symbol",
            how="inner"
        )

        final = final.drop(columns=["symbol"])

    else:

        st.warning(
            "No fundamentals CSV uploaded. "
            "ROE/FII/DII filters cannot be verified."
        )

        technical["ROE"] = np.nan
        technical["FII Holding"] = np.nan
        technical["DII Holding"] = np.nan

        final = technical


    # =====================================================
    # FILTER
    # =====================================================

    if fundamentals is not None:

        final = final[
            (final["ROE"] > roe_min) &
            (final["FII Holding"] > fii_min) &
            (final["DII Holding"] > dii_min) &
            (final["Above 200 EMA"] == True) &
            (final["EMA Cross"] == True) &
            (final["MACD Bullish"] == True) &
            (final["RSI"] > rsi_min) &
            (final["4M High"] == True)
        ].copy()

    else:

        final = final[
            (final["Above 200 EMA"] == True) &
            (final["MACD Bullish"] == True) &
            (final["RSI"] > rsi_min)
        ].copy()


    # =====================================================
    # AI SCORE
    # =====================================================

    if len(final) > 0:

        final["AI Score"] = final.apply(
            calculate_score,
            axis=1
        )

        final["Signal"] = np.where(
            final["AI Score"] >= 85,
            "🔥 STRONG BUY",
            np.where(
                final["AI Score"] >= 70,
                "🟢 BUY",
                "🟡 WATCH"
            )
        )

        final = final.sort_values(
            "AI Score",
            ascending=False
        )

        st.success(
            f"Found {len(final)} stocks matching the scanner."
        )

        # =================================================
        # TOP RESULTS
        # =================================================

        display_columns = [
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
            "4M High",
            "AI Score",
            "Signal"
        ]

        display_columns = [
            c for c in display_columns
            if c in final.columns
        ]

        st.dataframe(
            final[display_columns],
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # DOWNLOAD
        # =================================================

        csv = final[display_columns].to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Scanner Results",
            data=csv,
            file_name="nse_ai_scanner_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.warning(
            "No stocks currently satisfy all selected conditions."
        )

else:

    st.info(
        "Upload fundamentals.csv and click "
        "'🚀 RUN NSE SCANNER'."
    )

    st.markdown(
        """
### Scanner Conditions

✅ Price > 200 EMA  
✅ 10 EMA crosses above 50 EMA  
✅ MACD bullish  
✅ RSI > 50  
✅ Near/at 4-month high  
✅ ROE > 10%  
✅ FII holding > 7%  
✅ DII holding > 45%  
"""
    )
2. requirements.txt
Writing
streamlit
pandas
numpy
yfinance
3. fundamentals.csv
Is file ko bhi GitHub me rakho. Abhi sample format:
Writing
symbol,roe,fii_holding,dii_holding
RELIANCE,12.5,18.2,38.5
TCS,45.2,12.8,35.1
INFY,29.4,15.6,32.7
HDFCBANK,17.1,24.3,28.6
ICICIBANK,18.9,22.1,31.4
Lekin sample numbers ko real investment data mat samajhna—apne current ROE/FII/DII data se CSV update karna hoga.
GitHub structure
nse-ai-scanner/
│
├── app.py
├── requirements.txt
└── fundamentals.csv
Streamlit Community Cloud repository se app deploy kar sakta hai, aur requirements.txt root me rakhna recommended hai. �
Streamlit Docs +1
Ek limitation: upar wala version technically complete hai, lekin FII/DII/ROE ko manually CSV se supply karna padta hai. Agar tumhara goal “button dabao → automatically all NSE stocks + automatic ROE + FII + DII + technicals + AI ranking” hai, to next version me main automatic Indian fundamental-data source/API layer add karunga.
