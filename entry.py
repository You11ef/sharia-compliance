import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import financedatabase as fd

# --------------------------------------
# Import yahooquery for Sharia Filtering
# --------------------------------------
from yahooquery import Ticker

# ======================================
# 1. PAGE CONFIG
# ======================================
st.set_page_config(page_title="Sharia-Compliant Portfolio Optimization", layout="wide")

# ======================================
# 2. HELPER FUNCTIONS & CACHING
# ======================================

@st.cache_data
def load_ticker_list():
    """Load & prepare a comprehensive ticker list from financedatabase."""
    df_etfs = fd.ETFs().select().reset_index()[['symbol', 'name']]
    df_equities = fd.Equities().select().reset_index()[['symbol', 'name']]
    df_all = pd.concat([df_etfs, df_equities], ignore_index=True)
    df_all.dropna(subset=['symbol'], inplace=True)
    df_all = df_all[~df_all['symbol'].str.startswith('^')]
    df_all['symbol_name'] = df_all['symbol'] + " - " + df_all['name'].fillna('')
    df_all.drop_duplicates(subset='symbol', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    return df_all

@st.cache_data
def fetch_prices(tickers, start, end):
    """Download price data once, to avoid re-downloading for every parameter change."""
    return yf.download(tickers, start=start, end=end, auto_adjust=True)

def resample_prices(df_prices, freq):
    """
    Resample prices to user-selected frequency.
    freq = "Daily", "Weekly", or "Monthly".
    """
    if freq == "Daily":
        return df_prices  # no resampling
    elif freq == "Weekly":
        # Use 'W-FRI' so data lines up with last available price on Friday
        return df_prices.resample('W-FRI').last().dropna(how='all')
    elif freq == "Monthly":
        return df_prices.resample('M').last().dropna(how='all')

def portfolio_return(weights, returns, ann_factor):
    """Annualized portfolio return."""
    return np.sum(returns.mean() * weights) * ann_factor

def portfolio_std(weights, cov, ann_factor):
    """Annualized portfolio volatility."""
    return np.sqrt(weights.T @ cov @ weights) * (ann_factor ** 0.5)

def sharpe_ratio(weights, returns, cov, rf, ann_factor):
    """Annualized Sharpe ratio."""
    ret = portfolio_return(weights, returns, ann_factor)
    vol = portfolio_std(weights, cov, ann_factor)
    return (ret - rf) / vol if vol != 0 else 0

def sortino_ratio(weights, returns, rf, ann_factor):
    """
    Annualized Sortino ratio:
    (Return - RF) / Downside Deviation
    Using 0 as the 'target' for downside risk, or you can use rf if you prefer.
    """
    portfolio_daily = returns @ weights  # daily returns of the portfolio
    avg_return = portfolio_daily.mean() * ann_factor
    # Downside deviation relative to 0:
    negative_excess = portfolio_daily[portfolio_daily < 0]
    if len(negative_excess) == 0:
        # no negative returns => effectively infinite sortino
        return float('inf')
    downside_std = np.sqrt((negative_excess ** 2).mean()) * (ann_factor ** 0.5)
    if downside_std == 0:
        return float('inf')
    return (avg_return - rf) / downside_std

def objective_function(weights, returns, cov, rf, ann_factor, objective):
    """
    Unified objective function for:
      - Max Sharpe => minimize negative Sharpe
      - Min Vol    => minimize volatility
      - Max Return => minimize negative return
      - Max Sortino => minimize negative sortino
    """
    if objective == "Max Sharpe Ratio":
        return -sharpe_ratio(weights, returns, cov, rf, ann_factor)
    elif objective == "Min Volatility":
        return portfolio_std(weights, cov, ann_factor)
    elif objective == "Max Return":
        return -portfolio_return(weights, returns, ann_factor)
    elif objective == "Max Sortino Ratio":
        return -sortino_ratio(weights, returns, rf, ann_factor)

@st.cache_data
def generate_random_portfolios(num_portfolios, n_assets, min_weight, max_weight, 
                               returns, cov, rf, ann_factor):
    """Generate random feasible portfolios, compute metrics for plotting an approximate efficient frontier."""
    results = []
    for _ in range(num_portfolios):
        w = np.random.random(n_assets)
        w /= w.sum()  # ensure sum=1
        # enforce bounds
        if np.any(w < min_weight) or np.any(w > max_weight):
            continue
        ret = portfolio_return(w, returns, ann_factor)
        vol = portfolio_std(w, cov, ann_factor)
        sr = sharpe_ratio(w, returns, cov, rf, ann_factor)
        results.append((ret, vol, sr, w))
    columns = ['Return', 'Volatility', 'Sharpe', 'Weights']
    df_rand = pd.DataFrame(results, columns=columns)
    return df_rand

# ===============================
# SHARIA FILTERING LOGIC
# ===============================

def safe_str(x):
    """Convert non-string or NaN to empty string."""
    return x if isinstance(x, str) else ""

def is_haram_industry_or_sector(industry, sector, haram_list):
    """Check if sector/industry contains any haram keyword."""
    industry_str = safe_str(industry)
    sector_str   = safe_str(sector)
    combined_lower = (industry_str + " " + sector_str).lower()
    for kw in haram_list:
        if kw in combined_lower:
            return True
    return False

def sharia_filter(tickers, haram_keywords, ratio_threshold=0.33):
    """
    1) Exclude if 'industry' or 'sector' is haram (via yahooquery).
    2) Exclude if ratioDebt>33% or data missing (debt or marketCap).
    Returns: 
      - final_tickers: the ones that pass
      - logs: list of messages about why certain tickers got excluded
    """
    logs = []
    # Pull data from yahooquery
    t = Ticker(tickers, asynchronous=True)
    data_modules = t.get_modules(["summaryProfile", "financialData"])
    df_modules = pd.DataFrame.from_dict(data_modules).T
    
    # Normalization
    def normalize_module(df, module_name):
        if module_name not in df.columns:
            return pd.DataFrame()
        items = df[module_name].dropna()
        dict_list = []
        index_list = []
        for idx, val in items.items():
            if isinstance(val, dict):
                dict_list.append(val)
                index_list.append(idx)
        if not dict_list:
            return pd.DataFrame()
        df_norm = pd.json_normalize(dict_list)
        df_norm.index = index_list
        return df_norm

    df_sum = normalize_module(df_modules, "summaryProfile")
    df_fin = normalize_module(df_modules, "financialData")
    df_merged = pd.concat([df_sum, df_fin], axis=1)
    df_merged = df_merged.reindex(tickers)

    final_list = []

    for sym in tickers:
        row = df_merged.loc[sym]
        indus  = row.get('industry', np.nan)
        sect   = row.get('sector', np.nan)
        debt   = row.get('totalDebt', np.nan)

        # 1) Check haram sector
        if is_haram_industry_or_sector(indus, sect, haram_keywords):
            logs.append(f"{sym} => Exclu car secteur/industrie haram.")
            continue
        
        # 2) MarketCap from yfinance
        yf_info = yf.Ticker(sym).info
        mcap = yf_info.get("marketCap", np.nan)
        if pd.isna(mcap):
            logs.append(f"{sym} => Exclu car marketCap manquant via yfinance.")
            continue
        
        # 3) Ratio debt / marketcap
        ratio = None
        if not pd.isna(debt) and not pd.isna(mcap) and mcap > 0:
            ratio = debt / mcap
        if ratio is None or ratio > ratio_threshold:
            logs.append(f"{sym} => Exclu car ratioDette = {ratio} > {ratio_threshold} ou data manquante.")
            continue

        # If we get here, it's valid
        final_list.append(sym)
    
    return final_list, logs

# ===============================
# 3. MAIN STREAMLIT APP
# ===============================

# -- Liste de mots-clés haram --
haram_keywords = [
    "tobacco", "casino", "gambling", "brew", "beer",
    "distill", "winery", "adult entertainment"
]

ticker_list = load_ticker_list()

st.title("Sharia-Compliant Portfolio Optimization")
st.sidebar.header("Settings")

# 3.1 Ticker selection
st.sidebar.subheader("Choose Tickers")
sel_tickers = st.sidebar.multiselect(
    "Search and Select Tickers",
    options=ticker_list["symbol_name"],
    default=[]
)
sel_symbol_list = ticker_list.loc[ticker_list.symbol_name.isin(sel_tickers), 'symbol'].tolist()

# 3.2 Date Range
start_date_input = st.sidebar.date_input("Start Date", value=datetime(2015, 1, 1))
end_date_input = st.sidebar.date_input("End Date", value=datetime.today())

# 3.3 Frequency selection
freq_choice = st.sidebar.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
freq_map = {"Daily": 252, "Weekly": 52, "Monthly": 12}
ann_factor = freq_map[freq_choice]

# 3.4 Risk-free Rate
st.sidebar.subheader("Risk-Free Rate (%)")
risk_free_rate_input = st.sidebar.number_input("Enter Risk-Free Rate (%)", value=1.5, step=0.1)
risk_free_rate = risk_free_rate_input / 100

# 3.5 Weight Bounds
st.sidebar.subheader("Weight Bounds per Asset")
min_weight = st.sidebar.slider("Minimum Weight", 0.0, 0.5, 0.0, 0.05)
max_weight = st.sidebar.slider("Maximum Weight", 0.0, 1.0, 0.4, 0.05)

# 3.6 Optimization Objective
opt_objective = st.sidebar.selectbox(
    "Optimization Objective",
    ["Max Sharpe Ratio", "Min Volatility", "Max Return", "Max Sortino Ratio"]
)

# 3.7 Sharia Ratio Threshold
ratio_threshold = st.sidebar.slider("Max Debt/MarketCap Ratio", 0.0, 1.0, 0.33, 0.01)

st.write("## Selected Tickers")
if sel_symbol_list:
    st.write(", ".join(sel_symbol_list))
else:
    st.info("Select at least one ticker in the sidebar.")

# ===============================
# 4. Button to run Sharia filter
# ===============================
if st.button("Apply Sharia Filter"):
    if not sel_symbol_list:
        st.warning("No tickers selected. Please select tickers first.")
    else:
        with st.spinner("Applying Sharia Filter..."):
            final_list, logs = sharia_filter(sel_symbol_list, haram_keywords, ratio_threshold)
        st.subheader("Sharia Filter Logs")
        if logs:
            for line in logs:
                st.write("-", line)
        if final_list:
            st.success(f"Sharia-Compliant Tickers: {', '.join(final_list)}")
        else:
            st.error("No tickers passed the Sharia filter.")
        # Store final list in session for subsequent optimization
        st.session_state['sharia_list'] = final_list

# ===============================
# 5. Price Download & Optimization
# ===============================
if 'sharia_list' not in st.session_state:
    st.info("Please click 'Apply Sharia Filter' to store the final Sharia list in session.")
else:
    final_sharia_tickers = st.session_state['sharia_list']
    if st.button("Optimize Sharia-Compliant Portfolio"):
        if not final_sharia_tickers:
            st.warning("No Sharia-compliant tickers to optimize.")
        else:
            st.write("### Sharia-Compliant Tickers to be Optimized:")
            st.write(", ".join(final_sharia_tickers))

            data_raw = fetch_prices(final_sharia_tickers, start=start_date_input, end=end_date_input)
            if data_raw.empty:
                st.warning("No data available for the chosen Sharia tickers & date range.")
            else:
                # Use Close prices if available
                if 'Close' in data_raw:
                    price_data_raw = data_raw['Close']
                else:
                    price_data_raw = data_raw

                price_data_raw = price_data_raw.ffill().dropna(how='all')

                # Check which tickers truly have data
                valid_tickers = []
                no_data_tickers = []
                for t in final_sharia_tickers:
                    if t in price_data_raw.columns and not price_data_raw[t].isnull().all():
                        valid_tickers.append(t)
                    else:
                        no_data_tickers.append(t)

                if no_data_tickers:
                    st.warning(f"These Sharia tickers have no data: {', '.join(no_data_tickers)}")

                if not valid_tickers:
                    st.error("No valid price data for the final Sharia list.")
                else:
                    st.write("Valid Tickers with Price Data:", ", ".join(valid_tickers))
                    full_price_data = price_data_raw[valid_tickers]

                    # Show line chart
                    st.subheader("Price History (Sharia-Compliant)")
                    st.line_chart(full_price_data)

                    # Resample
                    freq_price_data = resample_prices(full_price_data, freq_choice)

                    # Determine intersection date range
                    first_valid = full_price_data.apply(lambda col: col.first_valid_index())
                    last_valid  = full_price_data.apply(lambda col: col.last_valid_index())
                    non_null_starts = [d for d in first_valid if d is not None]
                    non_null_ends = [d for d in last_valid if d is not None]

                    max_start = None
                    min_end = None
                    if non_null_starts and non_null_ends:
                        max_start = max(non_null_starts)
                        min_end = min(non_null_ends)

                    if not max_start or not min_end or max_start >= min_end:
                        st.warning("No overlapping date among chosen tickers for optimization.")
                    else:
                        # Slice to intersection
                        freq_price_data = freq_price_data.loc[max_start:min_end].dropna(how='all')
                        if freq_price_data.empty:
                            st.warning("No price data in the overlapping date range.")
                        else:
                            # Calculate log returns
                            log_returns = np.log(freq_price_data / freq_price_data.shift(1)).dropna()
                            cov_matrix = log_returns.cov()

                            # SLSQP constraints
                            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
                            bounds = [(min_weight, max_weight) for _ in valid_tickers]
                            init_weights = np.array([1 / len(valid_tickers)] * len(valid_tickers))

                            # Optimize
                            optimized = minimize(
                                objective_function,
                                init_weights,
                                args=(log_returns, cov_matrix, risk_free_rate, ann_factor, opt_objective),
                                method="SLSQP",
                                constraints=constraints,
                                bounds=bounds
                            )

                            if not optimized.success:
                                st.error("Optimization failed. Please adjust inputs.")
                            else:
                                opt_weights = optimized.x
                                w_df = pd.DataFrame({
                                    "Ticker": valid_tickers,
                                    "Weight": opt_weights
                                })

                                st.markdown("### Optimal Portfolio Weights")
                                st.dataframe(w_df.set_index("Ticker"), use_container_width=True)

                                ret_val = portfolio_return(opt_weights, log_returns, ann_factor)
                                vol_val = portfolio_std(opt_weights, cov_matrix, ann_factor)
                                sr_val = sharpe_ratio(opt_weights, log_returns, cov_matrix, risk_free_rate, ann_factor)
                                so_val = sortino_ratio(opt_weights, log_returns, risk_free_rate, ann_factor)

                                st.markdown("### Portfolio Metrics")
                                metrics_df = pd.DataFrame({
                                    "Metric": [
                                        "Expected Annual Return",
                                        "Expected Annual Volatility",
                                        "Sharpe Ratio",
                                        "Sortino Ratio"
                                    ],
                                    "Value": [
                                        f"{ret_val:.4f}",
                                        f"{vol_val:.4f}",
                                        f"{sr_val:.4f}",
                                        f"{so_val:.4f}"
                                    ]
                                })
                                st.table(metrics_df)

                                # Pie chart
                                fig, ax = plt.subplots()
                                ax.pie(
                                    opt_weights,
                                    labels=w_df["Ticker"],
                                    autopct='%1.1f%%',
                                    startangle=90
                                )
                                ax.axis("equal")
                                st.markdown("### Optimal Allocation")
                                st.pyplot(fig)

                                # Download CSV
                                csv_data = w_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Weights CSV",
                                    data=csv_data.encode('utf-8'),
                                    file_name="sharia_optimized_weights.csv",
                                    mime="text/csv",
                                )

                                # Show random portfolios + frontier
                                st.markdown("### Efficient Frontier (Random Portfolios)")
                                df_rand = generate_random_portfolios(
                                    num_portfolios=2000,
                                    n_assets=len(valid_tickers),
                                    min_weight=min_weight,
                                    max_weight=max_weight,
                                    returns=log_returns,
                                    cov=cov_matrix,
                                    rf=risk_free_rate,
                                    ann_factor=ann_factor
                                )
                                if not df_rand.empty:
                                    fig_ef, ax_ef = plt.subplots()
                                    scatter = ax_ef.scatter(
                                        df_rand["Volatility"],
                                        df_rand["Return"],
                                        c=df_rand["Sharpe"],  
                                        cmap="viridis",
                                        alpha=0.6
                                    )
                                    cbar = fig_ef.colorbar(scatter, ax=ax_ef)
                                    cbar.set_label("Sharpe Ratio")

                                    # Plot the optimized portfolio
                                    ax_ef.scatter(
                                        vol_val, ret_val,
                                        c="red", s=80, edgecolors="black",
                                        label="Optimized"
                                    )
                                    ax_ef.set_xlabel("Annual Volatility")
                                    ax_ef.set_ylabel("Annual Return")
                                    ax_ef.set_title("Random Portfolios & Optimized Portfolio (Sharia)")
                                    ax_ef.legend()
                                    st.pyplot(fig_ef)
                                else:
                                    st.warning("No random portfolios generated under given constraints.")
