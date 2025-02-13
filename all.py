import os
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import datetime
from concurrent.futures import ThreadPoolExecutor
from pandas.api.types import is_datetime64tz_dtype

# ---------------------------
# إعداد صفحة التطبيق
# ---------------------------
st.set_page_config(
    page_title="Global Sector Performance Tracker",  
    layout="wide",
    page_icon="🛢️"
)

# ---------------------------
# قواميس القطاعات (أمثلة)
# ---------------------------
# شركات النفط (30 شركة)
oil_companies = {
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "SHEL": "Shell plc",
    "TTE": "TotalEnergies SE",
    "BP": "BP p.l.c.",
    "COP": "ConocoPhillips",
    "EOG": "EOG Resources, Inc.",
    "PBR": "Petróleo Brasileiro S.A. - Petrobras",
    "PSX": "Phillips 66",
    "VLO": "Valero Energy Corporation",
    "HES": "Hess Corporation",
    "OXY": "Occidental Petroleum Corporation",
    "DVN": "Devon Energy Corporation",
    "APA": "Apache Corporation",
    "SLB": "Schlumberger Limited",
    "HAL": "Halliburton Company",
    "BKR": "Baker Hughes Company",
    "NOV": "National Oilwell Varco, Inc.",
    "RIG": "Transocean Ltd.",
    "NE": "Noble Energy, Inc.",
    "FANG": "Diamondback Energy, Inc.",
    "SU": "Suncor Energy Inc.",
    "CNX": "CNX Resources Corporation",
    "EQNR": "Equinor ASA",
    "PXD": "Pioneer Natural Resources Company",
    "MPC": "Marathon Petroleum Corporation",
    "KMI": "Kinder Morgan, Inc.",
    "TRP": "TC Energy Corporation",
    "ENB": "Enbridge Inc.",
    "ET": "Energy Transfer LP"
}

# البنوك (30 شركة)
banks = {
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America",
    "WFC": "Wells Fargo",
    "C": "Citigroup Inc.",
    "GS": "Goldman Sachs Group",
    "MS": "Morgan Stanley",
    "USB": "U.S. Bancorp",
    "PNC": "PNC Financial Services",
    "TD": "Toronto-Dominion Bank",
    "RBC": "Royal Bank of Canada",
    "BMO": "Bank of Montreal",
    "BNS": "Bank of Nova Scotia",
    "STT": "State Street Corporation",
    "SCHW": "Charles Schwab Corporation",
    "CFG": "Citizens Financial Group",
    "FITB": "Fifth Third Bancorp",
    "HBAN": "Huntington Bancshares",
    "KEY": "KeyCorp",
    "MTB": "M&T Bank Corporation",
    "RF": "Regions Financial Corporation",
    "ZION": "Zions Bancorporation",
    "AMP": "Ameriprise Financial",
    "BK": "The Bank of New York Mellon",
    "FRC": "First Republic Bank",
    "CMA": "Comerica Incorporated",
    "NWG": "National Western Group",
    "AMTD": "Shanghai AMTD International",
    "SCB": "Standard Chartered Bank",
    "UBS": "UBS Group",
    "HSBC": "HSBC Holdings"
}

# شركات العقارات (30 شركة)
real_estate = {
    "SPG": "Simon Property Group",
    "O": "Realty Income Corporation",
    "AVB": "AvalonBay Communities",
    "PLD": "Prologis, Inc.",
    "EXR": "Extra Space Storage",
    "VTR": "Ventas, Inc.",
    "EQR": "Equity Residential",
    "PSA": "Public Storage",
    "UDR": "UDR, Inc.",
    "DLR": "Digital Realty Trust",
    "IRM": "Iron Mountain",
    "REG": "Regency Centers",
    "BXP": "Boston Properties",
    "SLG": "SL Green Realty Corp",
    "FRT": "Federal Realty Investment Trust",
    "DRE": "Douglas Emmett",
    "KIM": "Kimco Realty Corporation",
    "CCI": "Crown Castle International",
    "MAA": "Mid-America Apartment Communities",
    "ARE": "Alexandria Real Estate Equities",
    "PEAK": "Healthpeak Properties",
    "OHI": "Omega Healthcare Investors",
    "NLY": "Annaly Capital Management",
    "LXP": "Lexington Realty Trust",
    "VNO": "Vornado Realty Trust",
    "ESS": "Essex Property Trust",
    "EPR": "EPR Properties",
    "EXI": "Extra Income Inc",  # مثال تجريبي
    "RXP": "Rexford Industrial Realty",
    "CPT": "Capital Properties Trust"
}

# شركات الاتصالات (30 شركة)
telecom = {
    "T": "AT&T Inc.",
    "VZ": "Verizon Communications",
    "TMUS": "T-Mobile US, Inc.",
    "VOD": "Vodafone Group Plc",
    "TEF": "Telefonica SA",
    "ORAN": "Orange S.A.",
    "NTT": "Nippon Telegraph and Telephone",
    "SKT": "SK Telecom",
    "KT": "KT Corporation",
    "LUMN": "Lumen Technologies",
    "BCE": "BCE Inc.",
    "Rogers": "Rogers Communications",
    "Singtel": "Singtel",
    "Telus": "Telus Corporation",
    "Bell": "Bell Canada",
    "CT": "China Telecom",
    "CMCC": "China Mobile",
    "CUG": "China Unicom",
    "MTS": "MTS",  # مثال
    "MTN": "MTN Group",
    "Zain": "Zain Group",
    "Ooredoo": "Ooredoo Qatari",
    "Etisalat": "Etisalat Group",
    "STC": "Saudi Telecom Company",
    "Telia": "Telia Company",
    "Telenor": "Telenor Group",
    "Tele2": "Tele2 AB",
    "TelecomItalia": "Telecom Italia",
    "Vodacom": "Vodacom Group"
}

# ربط أسماء القطاعات بالقواميس
sector_mapping = {
    "Oil": oil_companies,
    "Banks": banks,
    "Real Estate": real_estate,
    "Telecommunications": telecom
}

# ---------------------------
# شريط الجانبي: اختيار القطاع
# ---------------------------
sector = st.sidebar.selectbox("Select Sector", list(sector_mapping.keys()))
companies = sector_mapping[sector]

# ---------------------------
# دوال جلب البيانات
# ---------------------------
@st.cache_data(show_spinner="Fetching data from Yahoo Finance...")
def fetch_data_yahoo(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="3y")
        data.reset_index(inplace=True)
        if not is_datetime64tz_dtype(data["Date"]):
            data["Date"] = data["Date"].dt.tz_localize("UTC")
        data["Date"] = data["Date"].dt.tz_convert("America/New_York")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_all_data(tickers):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_data_yahoo, tickers))
    return {ticker: df for ticker, df in zip(tickers, results) if not df.empty}

def preprocess_data(data_dict, start_date, end_date):
    df_all = pd.concat(
        [df.assign(Company=ticker) for ticker, df in data_dict.items()],
        ignore_index=True
    )
    df_filtered = df_all[(df_all["Date"] >= start_date) & (df_all["Date"] <= end_date)].copy()
    scaler = MinMaxScaler()
    df_filtered.loc[:, "Close_Scaled"] = scaler.fit_transform(df_filtered[["Close"]])
    return df_filtered

def ml_forecast(data_dict, start_date, end_date):
    forecast_results = {}
    for ticker, df in data_dict.items():
        df_ml = df.copy()
        df_ml = df_ml[(df_ml["Date"] >= start_date) & (df_ml["Date"] <= end_date)]
        df_ml.sort_values("Date", inplace=True)
        if df_ml.empty or df_ml.shape[0] < 10:
            continue
        df_ml["Date_ordinal"] = df_ml["Date"].apply(lambda x: x.timestamp())
        df_ml["Close_lag1"] = df_ml["Close"].shift(1)
        df_ml = df_ml.dropna(subset=["Date_ordinal", "Close", "Close_lag1"])
        if df_ml.shape[0] < 10:
            continue
        X = df_ml[["Date_ordinal", "Close_lag1"]].values
        y = df_ml["Close"].values
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X, y)
        last_date_ordinal = df_ml["Date_ordinal"].iloc[-1]
        last_close = df_ml["Close"].iloc[-1]
        X_next = np.array([[last_date_ordinal + 86400, last_close]])
        predicted_price = model.predict(X_next)[0]
        predicted_growth = (predicted_price - last_close) / last_close
        forecast_results[ticker] = predicted_growth
    return forecast_results

# ---------------------------
# التطبيق الرئيسي
# ---------------------------
def main():
    st.title("Global Sector Performance Tracker")
    st.markdown("""
    This application provides a comprehensive overview of the performance of companies in various sectors (Oil, Banks, Real Estate, and Telecommunications) over the past three years.
    Data is fetched from Yahoo Finance and visualized using interactive charts.
    """)
    
    # استخدام الشركات للقطاع المحدد
    tickers = list(companies.keys())
    with st.spinner("Fetching data..."):
        data_dict = fetch_all_data(tickers)
    if not data_dict:
        st.error("No data available for the selected companies.")
        st.stop()
    
    st.sidebar.header("Filter Options")
    available_tickers = sorted(data_dict.keys())
    display_names = [f"{ticker} - {companies.get(ticker, '')}" for ticker in available_tickers]
    selected_display = st.sidebar.selectbox("Select a company:", display_names)
    selected_ticker = selected_display.split(" - ")[0]
    
    min_date = pd.to_datetime(min(df["Date"].min() for df in data_dict.values())).date()
    max_date = pd.to_datetime(max(df["Date"].max() for df in data_dict.values())).date()
    date_range = st.sidebar.date_input("Select date range:", [min_date, max_date])
    start_date, end_date = [pd.to_datetime(date).tz_localize("America/New_York") for date in date_range]
    
    df_filtered = preprocess_data(data_dict, start_date, end_date)
    if start_date == end_date:
        st.warning("The selected date range is a single day; some charts may not display meaningful trends.")
    
    ticker_data_filtered = df_filtered[df_filtered["Company"] == selected_ticker]
    st.subheader(f"{selected_ticker} - {companies.get(selected_ticker, '')} (Last 30 Days)")
    if not ticker_data_filtered.empty:
        st.dataframe(ticker_data_filtered.tail(30))
    else:
        st.info("No data available for the selected date range.")
    
    st.subheader(f"{selected_ticker} - {companies.get(selected_ticker, '')} Stock Price")
    fig_line = px.line(
        data_dict[selected_ticker],
        x="Date",
        y="Close",
        title=f"{selected_ticker} Stock Price",
        labels={"Close": "Stock Price", "Date": "Date"}
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.subheader("Normalized Stock Price Comparison")
    fig_compare = px.line(
        df_filtered,
        x="Date",
        y="Close_Scaled",
        color="Company",
        title="Normalized Stock Price Comparison",
        labels={"Close_Scaled": "Normalized Price", "Date": "Date"}
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    
    if st.button("Load Efficiency Comparison"):
        st.subheader("Efficiency Comparison")
        efficiency_df = df_filtered.groupby("Company")["Close_Scaled"].mean().reset_index()
        efficiency_df = efficiency_df.sort_values(by="Close_Scaled", ascending=False)
        fig_bar = px.bar(
            efficiency_df,
            x="Company",
            y="Close_Scaled",
            title="Average Normalized Stock Price per Company",
            labels={"Close_Scaled": "Average Normalized Price", "Company": "Company"},
            color="Close_Scaled",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Click the button above to load the efficiency comparison chart.")
    
    st.subheader("Machine Learning Forecast & Recommendation")
    st.markdown("#### Forecasting Next Day's Closing Price using Gradient Boosting Regression")
    forecast_results = ml_forecast(data_dict, start_date, end_date)
    if forecast_results:
        forecast_df = pd.DataFrame.from_dict(forecast_results, orient="index", columns=["Predicted Growth Rate"])
        forecast_df.sort_values(by="Predicted Growth Rate", ascending=False, inplace=True)
        forecast_df["Predicted Growth Rate (%)"] = forecast_df["Predicted Growth Rate"] * 100
        st.dataframe(forecast_df[["Predicted Growth Rate (%)"]].style.format("{:.2f}"))
        best_ticker = forecast_df.index[0]
        best_growth = forecast_df.iloc[0]["Predicted Growth Rate (%)"]
        worst_ticker = forecast_df.index[-1]
        worst_growth = forecast_df.iloc[-1]["Predicted Growth Rate (%)"]
        st.success(f"**Buy Recommendation: {best_ticker} - {companies.get(best_ticker, '')}** with a predicted growth of {best_growth:.2f}%.")
        st.error(f"**Sell Recommendation: {worst_ticker} - {companies.get(worst_ticker, '')}** with a predicted growth of {worst_growth:.2f}%.")
    else:
        st.warning("Insufficient data for machine learning forecast.")
    
    st.markdown("<center><small>Designed by Chief Engineer Tareq Majeed alkarimi - Iraqi Ministry of Oil</small></center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
