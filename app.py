import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
import os
import time

# --- Setup ---
st.set_page_config(
    page_title="SGX Stock Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up Gemini API key from Streamlit secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    st.error("Please add your Google API key to Streamlit secrets. Instructions are in the code comments.")
    st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()


# --- Data and Ticker List ---
# List of Straits Times Index (STI) components
# This list can be updated based on the latest STI constituents.
STI_COMPANIES = {
    'D05.SI': 'DBS Group Holdings',
    'O39.SI': 'Oversea-Chinese Banking Corp',
    'U11.SI': 'United Overseas Bank',
    'C31.SI': 'CapitaLand Integrated Commercial Trust',
    'J36.SI': 'Jardine Matheson Holdings',
    'C6L.SI': 'Singapore Airlines',
    'Z74.SI': 'Singtel',
    'G13.SI': 'Genting Singapore',
    'S63.SI': 'Singapore Technologies Engineering',
    'BN4.SI': 'Keppel Corporation',
    'C09.SI': 'City Developments',
    'Y92.SI': 'Thai Beverage',
    'S58.SI': 'SATS',
    'U96.SI': 'Sembcorp Industries',
    'H78.SI': 'Hongkong Land Holdings',
    'V03.SI': 'Venture Corporation',
    'F34.SI': 'Wilmar International',
    'A17U.SI': 'CapitaLand Ascendas REIT',
    'J69U.SI': 'Frasers Centrepoint Trust',
    'M44U.SI': 'Mapletree Logistics Trust',
    'N2IU.SI': 'Mapletree Pan Asia Commercial Trust',
    'ME8U.SI': 'Mapletree Industrial Trust',
    'BS6.SI': 'Yangzijiang Shipbuilding',
    'C38U.SI': 'CapitaLand Ascott Trust',
    '5E2.SI': 'Seatrium Limited',
    'S68.SI': 'Singapore Exchange',
    'U14.SI': 'UOL Group',
    'Q01.SI': 'ComfortDelGro Corporation',
    'N52.SI': 'NetLink NBN Trust',
    'T39.SI': 'Thai Beverage Public Co Ltd'
}


# --- Functions ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_stock_data(ticker_list):
    """
    Fetches historical stock data from Yahoo Finance.
    Returns a dictionary of DataFrames.
    """
    data = {}
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="3mo")
            if not df.empty:
                data[ticker] = df
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {e}")
            continue
    return data

def get_llm_insight(company_name, ticker, drop_percentage):
    """
    Generates a brief LLM-based insight for a stock.
    The prompt is designed to produce a short, concise analysis.
    """
    prompt = f"""
    You are a professional financial analyst. Provide a very brief, high-level insight
    (2-3 sentences max) for the stock of {company_name} ({ticker}).
    It has dropped by {abs(drop_percentage):.2f}% over the past two weeks.
    Focus on potential reasons for the drop and its status as a potential investment
    highlighting that it is a blue-chip company. Do not use complex financial jargon.
    Do not mention specific prices or dates.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        response = model.generate_content(prompt)
        insight = response.text
        return insight
    except Exception as e:
        return f"Could not generate LLM insight: {e}"

# --- Main App Logic ---
st.title("🇸🇬 SGX Blue Chip Stock Insights")
st.markdown("""
<style>
    .reportview-container {
        background: #141414;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stCard {
        background-color: #2a2a2a;
        color: white;
        border-radius: 12px;
        border: 2px solid #2a2a2a;
        padding: 20px;
        transition: transform 0.2s, border-color 0.2s;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stCard:hover {
        transform: scale(1.02);
        border-color: #e50914;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
    }
    .card-ticker {
        font-size: 0.9rem;
        color: #b3b3b3;
        margin-top: 0;
    }
    .price {
        font-size: 1.5rem;
        font-weight: bold;
        color: #fff;
    }
    .change-positive {
        color: #4CAF50;
    }
    .change-negative {
        color: #F44336;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: white;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e50914;
        padding-bottom: 0.5rem;
    }
    .highlight-card {
        border-color: #e50914;
        border-width: 4px;
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.4);
    }
    .insight-text {
        font-size: 0.95rem;
        color: #d1d1d1;
        margin-top: 1rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Filter & Settings")
significant_drop_pct = st.sidebar.slider("Significant Drop Threshold (%)", 0.0, 15.0, 5.0)

with st.spinner("Fetching stock data and generating insights..."):
    # Fetch all data at once to reduce API calls
    all_stocks_data = fetch_stock_data(list(STI_COMPANIES.keys()))

    highlighted_stocks = []
    other_stocks = []

    for ticker, name in STI_COMPANIES.items():
        if ticker in all_stocks_data:
            df = all_stocks_data[ticker]
            if df.empty:
                continue

            # Check for data over the last 14 days
            last_two_weeks_data = df.tail(14)
            if len(last_two_weeks_data) < 2:
                continue

            current_price = last_two_weeks_data['Close'].iloc[-1]
            price_two_weeks_ago = last_two_weeks_data['Close'].iloc[0]

            if price_two_weeks_ago > 0:
                drop_percentage = ((current_price - price_two_weeks_ago) / price_two_weeks_ago) * 100
            else:
                drop_percentage = 0

            # Determine if it's a significant drop
            if drop_percentage < -significant_drop_pct:
                insight = get_llm_insight(name, ticker, drop_percentage)
                highlighted_stocks.append({
                    'ticker': ticker,
                    'name': name,
                    'current_price': current_price,
                    'drop_percentage': drop_percentage,
                    'insight': insight
                })
            else:
                other_stocks.append({
                    'ticker': ticker,
                    'name': name,
                    'current_price': current_price,
                    'drop_percentage': drop_percentage
                })

# --- Display Content ---
if highlighted_stocks:
    st.markdown('<div class="section-title">📉 Potential Opportunities: Significant Drops</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#b3b3b3;">These blue-chip stocks have dropped by more than the selected threshold over the last two weeks, potentially offering a buying opportunity.</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, stock in enumerate(highlighted_stocks):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="stCard highlight-card">
                <div class="card-title">{stock['name']}</div>
                <div class="card-ticker">{stock['ticker']}</div>
                <div class="price">S${stock['current_price']:.2f}</div>
                <div class="change-negative">
                    <span style='font-size:1.2rem;'>▼</span>
                    <span style='font-weight:bold;'>{stock['drop_percentage']:.2f}%</span>
                </div>
                <div class="insight-text">
                    <strong>Insight:</strong> {stock['insight']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

st.markdown('<div class="section-title">📊 All SGX Blue Chips</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#b3b3b3;">Performance of the Straits Times Index (STI) constituents.</p>', unsafe_allow_html=True)

cols = st.columns(4)
for i, stock in enumerate(other_stocks):
    with cols[i % 4]:
        change_class = "change-positive" if stock['drop_percentage'] >= 0 else "change-negative"
        change_arrow = "▲" if stock['drop_percentage'] >= 0 else "▼"
        st.markdown(f"""
        <div class="stCard">
            <div class="card-title">{stock['name']}</div>
            <div class="card-ticker">{stock['ticker']}</div>
            <div class="price">S${stock['current_price']:.2f}</div>
            <div class="{change_class}">
                <span style='font-size:1.2rem;'>{change_arrow}</span>
                <span style='font-weight:bold;'>{stock['drop_percentage']:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
