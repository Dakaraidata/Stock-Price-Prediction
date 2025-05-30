import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from stocknews import StockNews # Note: StockNews might make its own requests, outside of yfinance
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
# import matplotlib.pyplot as plt # Not used for displaying plots in Streamlit in your current code
# import seaborn as sns # Not used for displaying plots in Streamlit in your current code
import warnings
import time
from requests.exceptions import HTTPError # Import for retry logic
import random # Import random for jitter in retry

warnings.filterwarnings('ignore')

# --- Data Fetching Functions (with caching and retry) ---

# Add retry logic and caching to the primary data loader
@st.cache_data
def load_data_with_retry(symbol, start, end, max_retries=5):
    """Retrieve Data from Yahoo Finance with retry logic and caching."""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(start=start, end=end)
            if not data.empty:
                return data
            else:
                 # Handle cases where no data is returned (e.g., invalid symbol or dates)
                 st.error(f"No data found for symbol {symbol} between {start.strftime('%Y-%m-%d')} and {end.strftime('%Y-%m-%d')}.")
                 return pd.DataFrame() # Return empty DataFrame to indicate failure

        except Exception as e:
            if "Rate limit" in str(e) or isinstance(e, HTTPError):
                wait_time = (2 ** attempt) + random.uniform(0, 1) # Exponential backoff with jitter
                st.warning(f"Rate limited for {symbol}. Waiting {wait_time:.2f} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                # Re-raise other exceptions or return empty data on error
                st.error(f"An error occurred while fetching data for {symbol}: {e}")
                return pd.DataFrame() # Return empty DataFrame on other errors

    st.error(f"Failed to fetch data for {symbol} after {max_retries} retries.")
    return pd.DataFrame() # Return empty DataFrame if retries are exhausted

# Cache other data fetching functions
@st.cache_data
def get_stock_name(symbol):
    """Get the long name of the stock symbol with caching."""
    stock = yf.Ticker(symbol)
    try:
        # Use .info.get to safely retrieve longName
        return stock.info.get('longName', symbol)
    except Exception as e:
        st.warning(f"Could not fetch stock name for {symbol}: {e}")
        return symbol # Return symbol if name fetching fails

@st.cache_data
def getDividends(symbol, data): # Pass data as it's used for avg_stock_price
    """Retrieve the dividends for a Stock and calculate rating with caching."""
    stock = yf.Ticker(symbol)
    try:
        currentYear = datetime.today().year
        dividends = stock.dividends

        # Filter dividends for the current year
        if not dividends.empty:
             currentDiv = dividends[dividends.index.year == currentYear]
        else:
             currentDiv = pd.Series(dtype=float) # Handle case with no dividends

        # Calculate the total dividends paid over the past year
        total_dividends = currentDiv.sum()

        # Calculate the average stock price over the past year - Use provided data DataFrame
        if not data.empty and 'Close' in data.columns:
             avg_stock_price = data['Close'].mean()
        else:
             avg_stock_price = 0 # Handle case with no data or no Close column

        # Calculate the average dividend yield over the past year
        average_dividend_yield = (total_dividends / avg_stock_price * 100) if avg_stock_price else 0 # Avoid division by zero

        # Various Ratings based on Dividend Yield
        if average_dividend_yield < 0:
            rating = 1
        elif average_dividend_yield <= 0.01:
            rating = 2
        elif average_dividend_yield <= 0.10:
            rating = 3
        elif average_dividend_yield <= 0.25:
            rating = 4
        elif average_dividend_yield <= 0.5:
            rating = 5
        elif average_dividend_yield <= 1.0:
            rating = 6
        elif average_dividend_yield <= 2.0:
            rating = 7
        elif average_dividend_yield <= 3.0:
            rating = 8
        elif average_dividend_yield <= 5.0:
            rating = 9
        else:
            rating = 10

        return rating
    except Exception as e:
        st.warning(f"Could not fetch or calculate dividends for {symbol}: {e}")
        return 0 # Return a default rating on error

@st.cache_data
def getROE(symbol):
    """Calculate ROE and return rating with caching."""
    stock = yf.Ticker(symbol)
    try:
        balanceSheet = stock.balance_sheet
        incomeStatement = stock.financials

        # Check if required data exists and is not empty
        if incomeStatement.empty or balanceSheet.empty or 'Net Income' not in incomeStatement.index or 'Total Assets' not in balanceSheet.index or 'Total Liabilities Net Minority Interest' not in balanceSheet.index:
             st.warning(f"Missing required financial data for ROE calculation for {symbol}.")
             return 0 # Return default rating if data is missing

        netIncome = float(incomeStatement.loc['Net Income'].iloc[0])
        totalAssets = float(balanceSheet.loc['Total Assets'].iloc[0])
        totalLiabilities = float(balanceSheet.loc['Total Liabilities Net Minority Interest'].iloc[0])
        shareHolderEquity = totalAssets - totalLiabilities

        if shareHolderEquity <= 0: # Avoid division by zero or negative equity
            return 1 # Very low rating if equity is zero or negative

        roe = (netIncome / shareHolderEquity) * 100

        if roe < 0:
            rating = 1
        elif roe <= 3:
            rating = 2
        elif roe <= 8:
            rating = 3
        elif roe <= 14:
            rating = 4
        elif roe <= 20:
            rating = 5
        elif roe <= 25:
            rating = 6
        elif roe <= 30:
            rating = 7
        elif roe <= 35:
            rating = 8
        elif roe <= 40:
            rating = 9
        else:
            rating = 10
        return rating
    except Exception as e:
        st.warning(f"Could not fetch or calculate ROE for {symbol}: {e}")
        return 0 # Return default rating on error

@st.cache_data
def getEPSGrowthRating(symbol):
    """Calculate EPS Growth and return rating with caching."""
    stock = yf.Ticker(symbol)
    try:
        eps = stock.earnings.reset_index()
        if len(eps) < 2: # Need at least 2 years for growth calculation
             st.warning(f"Insufficient EPS data for {symbol} to calculate growth.")
             return 1 # Insufficient data

        eps['Year'] = eps['Date'].dt.year
        # Ensure years are sorted to get the min/max year correctly
        eps = eps.sort_values('Year')
        start_eps = eps.loc[eps['Year'].min(), 'Earnings']
        end_eps = eps.loc[eps['Year'].max(), 'Earnings']
        num_years = eps['Year'].max() - eps['Year'].min()

        if num_years <= 0 or start_eps == 0: # Avoid division by zero or zero start EPS for growth calc
            return 1 # Cannot calculate growth or growth is undefined/infinite

        # Calculate Compound Annual Growth Rate (CAGR) if num_years > 0
        if num_years > 0 and start_eps != 0:
             eps_growth = ((end_eps / start_eps) ** (1/num_years) - 1) * 100
        else:
             eps_growth = 0 # Default to 0 growth if cannot calculate

        if eps_growth < 0:
            rating = 1
        elif eps_growth <= 5:
            rating = 2
        elif eps_growth <= 10:
            rating = 3
        elif eps_growth <= 15:
            rating = 4
        elif eps_growth <= 20:
            rating = 5
        elif eps_growth <= 25:
            rating = 6
        elif eps_growth <= 30:
            rating = 7
        elif eps_growth <= 35:
            rating = 8
        elif eps_growth <= 40:
            rating = 9
        else:
            rating = 10

        return rating
    except Exception as e:
        st.warning(f"Could not fetch or calculate EPS growth for {symbol}: {e}")
        return 0 # Return default rating on error

@st.cache_data
def get1YearReturnRating(symbol):
    """Calculate 1-Year Return and return rating with caching."""
    stock = yf.Ticker(symbol)
    try:
        hist = stock.history(period="1y")
        if len(hist) < 2: # Need at least two data points
            st.warning(f"Insufficient 1-year historical data for {symbol} to calculate return.")
            return 1 # Insufficient data

        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]

        if start_price == 0: # Avoid division by zero
             return 1 # Cannot calculate return if start price is zero

        one_year_return = ((end_price - start_price) / start_price) * 100

        if one_year_return < 0:
            rating = 1
        elif one_year_return <= 3:
            rating = 2
        elif one_year_return <= 6: # Assuming this was the intended break point
            rating = 3
        # Removed the duplicate <= 6 condition
        elif one_year_return <= 9:
            rating = 5
        elif one_year_return <= 12:
            rating = 6
        elif one_year_return <= 15:
            rating = 7
        elif one_year_return <= 18:
            rating = 8
        elif one_year_return <= 21:
            rating = 9
        else:
            rating = 10

        return rating
    except Exception as e:
        st.warning(f"Could not fetch or calculate 1-year return for {symbol}: {e}")
        return 0 # Return default rating on error

@st.cache_data
def get5_year_dividend_growth(symbol):
    """Calculate 5-year dividend growth and return rating with caching."""
    stock = yf.Ticker(symbol)
    try:
        dividends = stock.dividends
        if dividends.empty:
            st.warning(f"No dividend data available for {symbol}.")
            return 1 # No data

        dividends = dividends.resample('Y').sum()

        # Need at least 2 annual data points to calculate growth
        if len(dividends) < 2:
            st.warning(f"Insufficient annual dividend data for {symbol} to calculate growth.")
            return 1 # Insufficient data

        # Get the last two available annual dividend values
        start_dividend = dividends.iloc[-2]
        end_dividend = dividends.iloc[-1]

        if start_dividend == 0: # Avoid division by zero
             return 1 # Cannot calculate growth if start dividend is zero

        dividend_growth_rate = ((end_dividend - start_dividend) / start_dividend) * 100
        # Note: This calculates growth over the last available period, not a 5-year CAGR.

        if dividend_growth_rate > 10:
            rating = 10
        elif dividend_growth_rate > 6:
            rating = 9
        elif dividend_growth_rate > 4:
            rating = 8
        elif dividend_growth_rate > 3:
            rating = 7
        elif dividend_growth_rate > 2:
            rating = 6
        elif dividend_growth_rate > 1:
            rating = 5
        elif dividend_growth_rate > -1:
            rating = 4
        elif dividend_growth_rate > -3:
            rating = 3
        elif dividend_growth_rate > -5:
            rating = 2
        else:
            rating = 1

        return rating
    except Exception as e:
        st.warning(f"Could not fetch or calculate 5-year dividend growth for {symbol}: {e}")
        return 0 # Return default rating on error


# Note: Original code had two getnet_profit_margin functions. Using the second one.
@st.cache_data
def getNet_profit_margin(symbol): # Using the second function definition
    """Calculate Net Profit Margin and return rating with caching."""
    stock = yf.Ticker(symbol)
    try:
        # Get financial data
        income_statement = stock.financials

        # Check if income statement and required keys exist and are not empty
        if income_statement.empty or 'Net Income' not in income_statement.index or 'Total Revenue' not in income_statement.index:
             st.warning(f"Missing Net Income or Total Revenue in financial data for {symbol} for Net Profit Margin.")
             return 0

        # Ensure data is not empty before accessing
        net_income = income_statement.loc['Net Income'].iloc[0]
        revenue = income_statement.loc['Total Revenue'].iloc[0] # Revenue is in income statement

        if revenue == 0: # Avoid division by zero
             return 1 # Cannot calculate margin if revenue is zero

        # Calculate net profit margin
        net_profit_margin = (net_income / revenue) * 100

        if net_profit_margin > 80:
            rating = 10
        elif net_profit_margin > 60:
            rating = 9
        elif net_profit_margin > 40:
            rating = 8
        elif net_profit_margin > 20:
            rating = 7
        elif net_profit_margin > 10:
            rating = 6
        elif net_profit_margin > 5:
            rating = 5
        elif net_profit_margin > 0:
            rating = 4
        elif net_profit_margin > -5:
            rating = 3
        elif net_profit_margin > -10:
            rating = 2
        else:
            rating = 1

        return rating
    except Exception as e:
        st.warning(f"Could not fetch or calculate Net Profit Margin for {symbol}: {e}")
        return 0 # Return default rating on error

@st.cache_data
def getroa(symbol):
    """Calculate ROA and return rating with caching."""
    stock = yf.Ticker(symbol)
    try:
        balanceSheet = stock.balance_sheet
        incomeStatement = stock.financials

        # Check if required data exists and is not empty
        if incomeStatement.empty or balanceSheet.empty or 'Net Income' not in incomeStatement.index or 'Total Assets' not in balanceSheet.index:
             st.warning(f"Missing Net Income or Total Assets in financial data for {symbol} for ROA.")
             return 0

        netIncome = float(incomeStatement.loc['Net Income'].iloc[0])
        totalAssets = float(balanceSheet.loc['Total Assets'].iloc[0])

        if totalAssets == 0: # Avoid division by zero
             return 1 # Cannot calculate ROA if assets are zero

        # Extract return on assets (ROA) value
        roa = (netIncome / totalAssets) * 100

        if roa > 35:
            rating = 10
        elif roa > 25:
            rating = 9
        elif roa > 20:
            rating = 8
        elif roa > 15:
            rating = 7
        elif roa > 10:
            rating = 6
        elif roa > 5:
            rating = 5
        elif roa > 0:
            rating = 4
        elif roa > -5:
            rating = 3
        elif roa > -10:
            rating = 2
        else:
            rating = 1

        return rating
    except Exception as e:
        st.warning(f"Could not fetch or calculate ROA for {symbol}: {e}")
        return 0 # Return default rating on error

# Note: get3MonthVolatility was defined but not called anywhere. Keeping it for completeness.
@st.cache_data
def get3MonthVolatility(symbol):
    """Calculate 3-Month Volatility with caching."""
    stock = yf.Ticker(symbol)
    try:
        hist = stock.history(period="3mo")
        if len(hist) < 2:
            st.warning(f"Insufficient 3-month historical data for {symbol} to calculate volatility.")
            return None # Insufficient data

        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility for daily returns
        return volatility
    except Exception as e:
        st.warning(f"Could not fetch or calculate 3-month volatility for {symbol}: {e}")
        return None # Return None on error


# --- Streamlit Interface ---
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Reducing whitespace on the top of the page
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

st.title("🔮 Ticker Predicter ")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

# Suppress the specific Matplotlib deprecation warning globally for Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

# Title
st.sidebar.header(' Ticker Predicter')
st.sidebar.info("This app uses a Random Forest model to predict future stock prices. Select a stock symbol as well as the start and end date to get started.")
# Define the list of famous ticker symbols and an option to enter a custom ticker symbol
famous_ticker_symbols = sorted(['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA','META', 'NFLX', 'DIS', 'V', 'JPM', 'BA', 'INTC', 'GM', 'KO', 'IBM', 'AMD', 'PFE', 'UPS', 'WMT', 'SBUX'])
options = famous_ticker_symbols + ["Other"]

# Sidebar input for selecting a stock ticker symbol
symbol = st.sidebar.selectbox("Select a Stock Ticker Symbol", options=options)

# If user selects to enter a custom ticker symbol
if symbol == "Other":
    symbol = st.sidebar.text_input("Please input your other Ticker Symbol")

# Check if the symbol input is empty or just whitespace
if not symbol or symbol.strip() == "":
    st.warning("Please enter a Ticker Symbol to proceed.")
    st.stop() # Stop execution if no symbol is provided

# Convert the input to uppercase
symbol = symbol.upper()

# Get the long name of the stock symbol using the cached function
stock_name = get_stock_name(symbol)

# Choose the date range
current_date = datetime.today()
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", max_value=current_date)

# Check date validity
if start_date > end_date:
    st.sidebar.error('Error: End date must fall after Start date.')
    st.stop()

# User interface pages
page = st.sidebar.radio("Navigation", ["Ticker Predicter Dashboard", "Model Performance Metrics", "Stock Charts", "Additional Stock Company Info"])

# Load data with retry and caching
data = load_data_with_retry(symbol, start_date, end_date)

# Check if data is successfully loaded and is not empty before proceeding
if data is not None and not data.empty:

    # Display a loading spinner while processing
    with st.spinner(f'Analyzing data for {stock_name}...'):

        # Model Training and Prediction
        # Feature preparation
        # Ensure 'Close' column exists before shifting
        if 'Close' in data.columns:
            data['Next Close'] = data['Close'].shift(-1)
            data.dropna(inplace=True)

            if data.empty:
                st.error("Not enough historical data to train the model after adding 'Next Close'. Please select a longer date range.")
                # Continue execution but indicate inability to train model
                can_train_model = False
            else:
                 can_train_model = True
        else:
             st.error(f"Could not find 'Close' price data for {symbol}. Cannot train the model.")
             can_train_model = False


        if can_train_model:
            # Prepare feature set and target variable
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Volume', 'Next Close']
            if not all(col in data.columns for col in required_cols):
                 st.error(f"Missing required data columns for model training. Ensure data includes: {', '.join(required_cols)}")
                 can_train_model = False
            else:
                X = data[['Open', 'High', 'Low', 'Volume']]
                y = data['Next Close']

                # Split the data - Ensure enough data points for splitting
                if len(data) < 2: # Need at least 2 data points for training/testing
                     st.error("Insufficient data to train the model. Please select a longer date range.")
                     can_train_model = False
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Generate polynomial features for both train and test sets
                    poly = PolynomialFeatures(degree=2)
                    X_poly_train = poly.fit_transform(X_train)
                    X_poly_test = poly.transform(X_test)

                    # Train the model
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_poly_train, y_train)

                    # Predict for the next day
                    # Ensure there is at least one row in the data
                    if not data.empty:
                        last_known_data = data[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
                        last_known_data_poly = poly.transform(last_known_data)
                        predicted_price = model.predict(last_known_data_poly)
                    else:
                        predicted_price = [np.nan] # Set to NaN if no data

                    # Make predictions on the test set
                    predicted_prices = model.predict(X_poly_test)

                    # Performance Metrics
                    # Ensure y_test is not empty before calculating metrics
                    if not y_test.empty:
                        y_pred = model.predict(X_poly_test)
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                    else:
                         mae, mse, r2 = np.nan, np.nan, np.nan # Set metrics to NaN if no test data

                    # Sort the data points by date for plotting
                    if not y_test.empty:
                        sorted_dates = y_test.index
                        sorted_indices = np.argsort(sorted_dates)
                        sorted_dates = sorted_dates[sorted_indices]
                        sorted_actual_prices = y_test[sorted_indices]
                        sorted_predicted_prices = predicted_prices[sorted_indices]

                        # Create a line plot for actual vs predicted prices
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=sorted_dates, y=sorted_actual_prices, mode='lines+markers', name='Actual Prices', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=sorted_dates, y=sorted_predicted_prices, mode='lines+markers', name='Predicted Prices', line=dict(color='green')))
                        fig.update_layout(title=f'Random Forest Actual vs Predicted Closing Stock Prices for {stock_name}', xaxis_title='Date', yaxis_title='Stock Prices (USD)', autosize=True, width=800, height=500)
                    else:
                         fig = go.Figure() # Create an empty figure if no test data for plotting
                         fig.update_layout(title=f'Not enough data to plot Actual vs Predicted Prices for {stock_name}')

        # --- Financial Parameter Calculations and Radar Chart ---

        # This function now calls the cached data fetching functions
        def calculate_parameters(symbol, data): # Pass data for dividend calculation
            parameters = {
                'ROE': getROE(symbol),
                '1-Year Return': get1YearReturnRating(symbol),
                'Dividend Yield': getDividends(symbol, data), # Pass data here
                '5-Year Dividend Growth': get5_year_dividend_growth(symbol),
                'Net Profit Margin': getNet_profit_margin(symbol), # Using the second function name
                'ROA': getroa(symbol),
                 # EPS Growth is calculated separately, add if needed in the radar chart
                 # 'EPS Growth': getEPSGrowthRating(symbol)
            }
            return parameters

        def create_radar_chart(symbol, data): # Pass data
            # Get ratings
            ratings = calculate_parameters(symbol, data) # Pass data here

            # Create a DataFrame from the ratings data
            df = pd.DataFrame(list(ratings.items()), columns=['Parameter', 'Rating'])

            # Create a radar chart to visualize ratings for each parameter
            fig2 = px.line_polar(df, r='Rating', theta='Parameter', line_close=True, title=f"Financial Ratings for {stock_name}")
            fig2.update_traces(fill='toself')
            return fig2


        # --- Streamlit Page Layouts ---

        # Page: Ticker Predicter Dashboard
        if page == "Ticker Predicter Dashboard":

            col11, col12 = st.columns([2, 3])
            co11 = col11.container(border=False)
            co12 = col12.container(border=False)
            # Display content in the first column (two rows)
            with co11:
                con11 = st.container(border=True)
                con111 = st.container(border=True)

                with con11:
                    st.markdown(f"## 📈 Selected Stock: <br>{stock_name}", unsafe_allow_html=True)

                # Display the radar chart in Streamlit
                with con111:
                    # Only create and display radar chart if data was loaded successfully
                    if data is not None and not data.empty:
                        radar_chart = create_radar_chart(symbol, data) # Pass data
                        st.plotly_chart(radar_chart, use_container_width=True)
                    else:
                         st.warning("Cannot display financial ratings due to data fetching issues.")


            # Display content in the second column (one row)
            with co12:
                con12 = st.container(border=True)
                con121 = st.container(border=True)
                # Display the interactive plot using st.plotly_chart
                with con121:
                     # Only display plot if model was trained and test data was available
                    if can_train_model and not y_test.empty:
                         st.plotly_chart(fig, use_container_width=True)
                    elif can_train_model and y_test.empty:
                         st.warning("Not enough test data to plot Actual vs Predicted Prices.")
                    else:
                         st.warning("Cannot display Actual vs Predicted Prices plot because the model could not be trained.")


                with con12:
                    if can_train_model and predicted_price is not None and not np.isnan(predicted_price[0]):
                        st.markdown(f"## 🔭 Next Forecasted Closing Price: <br>${predicted_price[0]:.2f}", unsafe_allow_html=True)
                    elif can_train_model and (predicted_price is None or np.isnan(predicted_price[0])):
                         st.markdown(f"## 🔭 Next Forecasted Closing Price: <br>Calculation Failed", unsafe_allow_html=True)
                    else:
                        st.markdown(f"## 🔭 Next Forecasted Closing Price: <br>Model Not Trained", unsafe_allow_html=True)


            # Calculate the start date as 3 months before the current date
            start_date_returns = current_date - timedelta(days=90)

            # End date is the current date
            end_date_returns = current_date

            # Convert start_date and end_date to string format for the function
            start_date_str_returns = start_date_returns.strftime('%Y-%m-%d')
            end_date_str_returns = end_date_returns.strftime('%Y-%m-%d')

            # Calculate and plot daily returns
            top_5_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'META'] # Top 5
            display_symbols_returns = list(top_5_symbols)
            if symbol not in display_symbols_returns:
                display_symbols_returns.append(symbol) # Add selected symbol if not in top 5

            # Function to get daily returns for a specific stock
            @st.cache_data
            def get_daily_returns(sym, start, end, max_retries=3): # Added retry for this fetch too
                 for attempt in range(max_retries):
                    try:
                        data_returns = yf.download(sym, start=start, end=end, progress=False)
                        if not data_returns.empty and 'Close' in data_returns.columns:
                            daily_returns = data_returns['Close'].pct_change().dropna()
                            return daily_returns
                        else:
                            st.warning(f"No daily returns data found for {sym} between {start} and {end}.")
                            return pd.Series(dtype=float) # Return empty Series

                    except Exception as e:
                        if "Rate limit" in str(e) or isinstance(e, HTTPError):
                            wait_time = (1.5 ** attempt) + random.uniform(0, 0.5) # Shorter backoff for this
                            st.warning(f"Rate limited for {sym}. Waiting {wait_time:.2f} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                            time.sleep(wait_time)
                        else:
                            st.warning(f"An error occurred while fetching daily returns for {sym}: {e}")
                            return pd.Series(dtype=float) # Return empty Series on other errors
                 st.warning(f"Failed to fetch daily returns for {sym} after {max_retries} retries.")
                 return pd.Series(dtype=float) # Return empty Series if retries exhausted


           # Create a Plotly figure for daily returns of top companies + selected
            figreturn = go.Figure()
            has_return_data = False

            for sym in display_symbols_returns:
                daily_returns = get_daily_returns(sym, start_date_str_returns, end_date_str_returns)
                if not daily_returns.empty:
                    daily_returns.index = pd.to_datetime(daily_returns.index)
                    figreturn.add_trace(go.Scatter(x=daily_returns.index, y=daily_returns, mode='lines', name=sym))
                    has_return_data = True

            if has_return_data:
                figreturn.update_layout(title='Daily Returns for Selected Stocks (Past 3 Months)', xaxis_title='Date', yaxis_title='Daily Return')
            else:
                 figreturn.update_layout(title='No Daily Returns Data Available for Plotting')


            # Create a heatmap of the stock data correlation
            # Ensure 'data' is not empty before calculating correlation
            if not data.empty and len(data.columns) > 1: # Need at least 2 columns for correlation
                 # Select only numeric columns for correlation matrix
                 numeric_cols = data.select_dtypes(include=np.number)
                 if not numeric_cols.empty and len(numeric_cols.columns) > 1:
                    corr_matrix = numeric_cols.corr()
                    # Create a Plotly heatmap
                    figheat = px.imshow(corr_matrix,
                                    text_auto=True,
                                    color_continuous_scale='Blues',
                                    title='Correlation Heatmap of Stock Data',
                                    labels=dict(x="Features", y="Features", color="Correlation"))
                 else:
                    figheat = go.Figure()
                    figheat.update_layout(title='Not Enough Numeric Data for Correlation Heatmap')

            else:
                 figheat = go.Figure() # Create empty figure if no data for heatmap
                 figheat.update_layout(title='Not Enough Data to Generate Correlation Heatmap')


            #Two columns for Returns chart and Heatmap chart
            col13, col14 = st.columns([2, 3])
            co13 = col13.container(border=True)
            co14 = col14.container(border=True)
            # Display content in the first column (returns chart)
            with co13:
                st.plotly_chart(figreturn, use_container_width=True)

            # Display content in the second column (heatmap chart)
            with co14:
                st.plotly_chart(figheat, use_container_width=True)


        # Page: Model Performance Metrics
        elif page == "Model Performance Metrics":
            st.markdown('### 📊 Random Forest Model Performance Metrics:')
            # Only display metrics if the model was trained and they were successfully calculated
            if can_train_model and not y_test.empty:
                col1, col2, col3 = st.columns(3)

                con1 = col1.container(border=True)
                con2 = col2.container(border=True)
                con3 = col3.container(border=True)
                # Add borders to each column
                with con1:
                    st.subheader('Mean Absolute Error (MAE)')
                    st.write(f"{mae:.2f}", unsafe_allow_html=True)

                with con2:
                    st.subheader('R-squared (R²) Score')
                    st.write(f"{r2:.2f}", unsafe_allow_html=True)

                with con3:
                    st.subheader('Mean Squared Error (MSE)')
                    st.write(f"{mse:.2f}", unsafe_allow_html=True)
            elif can_train_model and y_test.empty:
                 st.warning("Model performance metrics cannot be displayed due to insufficient test data.")
            else:
                 st.warning("Model performance metrics cannot be displayed because the model could not be trained.")


        # Page: Stock Charts
        elif page == "Stock Charts":

            st.header("Stock Charts")
            st.markdown("<br>", unsafe_allow_html=True)

            # Create charts only if data is available and has 'Close' column
            if not data.empty and 'Close' in data.columns:

                # Moving Average Chart
                fig_moving_avg = px.line(data, x=data.index, y='Close', title=f'Stock Closing Price with Moving Averages for {stock_name}')
                # Ensure enough data points for rolling calculations
                if len(data) >= 10: fig_moving_avg.add_scatter(x=data.index, y=data['Close'].rolling(window=10).mean(), mode='lines', name='10-Day Moving Average')
                if len(data) >= 30: fig_moving_avg.add_scatter(x=data.index, y=data['Close'].rolling(window=30).mean(), mode='lines', name='30-Day Moving Average')
                if len(data) >= 50: fig_moving_avg.add_scatter(x=data.index, y=data['Close'].rolling(window=50).mean(), mode='lines', name='50-Day Moving Average')
                fig_moving_avg.update_layout(autosize=True, width=550, height=450)


                # Candlestick Chart
                # Ensure enough data points for rolling calculations (at least window size for Bollinger Bands)
                if len(data) >= 20: # Need at least 20 for the 20-day BB
                    fig1 = go.Figure()
                    fig1.add_traces([
                        go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candle Stick Chart'),
                        go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean(), line=dict(color='orange', width=1), name='Middle Band'),
                        go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean() + 1.96*data['Close'].rolling(window=20).std(), line=dict(color='gray', width=1), name='Upper Band'),
                        go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean() - 1.96*data['Close'].rolling(window=20).std(), line=dict(color='gray', width=1), name='Lower Band')
                    ])
                    fig1.update_layout(title_text=f'🕯️ {stock_name} Candlestick Chart with Bollinger Bands', autosize=True, width=550, height=450)

                else:
                     fig1 = go.Figure()
                     fig1.update_layout(title=f'Not enough data for Candlestick Chart with Bollinger Bands for {stock_name} (requires >= 20 days)')


                # Organize the charts into a dashboard layout using st.columns and st.plotly_chart
                c1, c2 = st.columns([1, 1])
                co1 = c1.container(border=True)
                co2 = c2.container(border=True)

                with co1:
                    st.plotly_chart(fig_moving_avg, use_container_width=True)  # Stock Closing Price with Moving Average

                with co2:
                    st.plotly_chart(fig1, use_container_width=True)  # Candlestick Chart

            else:
                 st.warning("Cannot display stock charts due to data fetching issues.")


        # Page: Additional Stock Company Info
        # Tabs for Historical data, fundamental data and News on relative Stock
        elif page == "Additional Stock Company Info":
            historical_pricing_data, fundamental_data, news = st.tabs(["Historical Pricing Data", "Fundamental Data", "Top 10 News"])

            # Define the content for each tab
            with historical_pricing_data:  # Historical Pricing Data tab
                st.header(f'💸 {stock_name} Price Movements')
                # Only display data if it's not empty
                if not data.empty:
                    st.write(data)
                else:
                    st.warning("No historical pricing data available.")

            with fundamental_data:  # Fundamental Data tab
                stock = yf.Ticker(symbol) # Need a Ticker object here for accessing .balance_sheet, .financials, .cashflow directly
                st.header("🧾 Balance Sheet")
                try:
                    balance_sheet = stock.balance_sheet
                    if not balance_sheet.empty:
                        st.write(balance_sheet)
                    else:
                        st.warning("No balance sheet data available.")
                except Exception as e:
                    st.warning(f"Could not fetch balance sheet: {e}")


                st.header("🤑 Income Statement")
                try:
                    income_statement = stock.financials
                    if not income_statement.empty:
                        st.write(income_statement)
                    else:
                        st.warning("No income statement data available.")
                except Exception as e:
                    st.warning(f"Could not fetch income statement: {e}")

                st.header("💵 Cash Flow Statement")
                try:
                    cash_flow = stock.cashflow
                    if not cash_flow.empty:
                         st.write(cash_flow)
                    else:
                         st.warning("No cash flow data available.")
                except Exception as e:
                    st.warning(f"Could not fetch cash flow statement: {e}")


            with news:  # Top 10 News tab
                st.header(f'📢 News for {stock_name}')
                try:
                    # StockNews also makes network requests, may need its own error handling or delay
                    # Added caching here as well
                    @st.cache_data
                    def get_stock_news(sym):
                         try:
                            sn = StockNews(sym, save_news=False)
                            df_news = sn.read_rss() # This might be where another rate limit occurs
                            return df_news
                         except Exception as e:
                             st.warning(f"Could not fetch news using StockNews for {sym}: {e}")
                             return pd.DataFrame() # Return empty DataFrame on error

                    df_news = get_stock_news(symbol)

                    if not df_news.empty:
                        for i in range(min(10, len(df_news))): # Ensure we don't go out of bounds
                            st.subheader(f'News {i+1}:')
                            # Check if keys exist before accessing
                            st.write(df_news['published'][i] if 'published' in df_news.columns else 'N/A')
                            st.write(df_news['title'][i] if 'title' in df_news.columns else 'N/A')
                            st.write(df_news['summary'][i] if 'summary' in df_news.columns else 'N/A')
                            title_sentiment = df_news['sentiment_title'][i] if 'sentiment_title' in df_news.columns else 'N/A'
                            st.write(f'Title Sentiment: {title_sentiment}')
                            news_sentiment = df_news['sentiment_summary'][i] if 'sentiment_summary' in df_news.columns else 'N/A'
                            st.write(f'News Sentiment: {news_sentiment}')
                            st.markdown("---") # Separator between news articles
                    else:
                         st.warning(f"Could not fetch news for {stock_name}.")

                except Exception as e:
                    st.warning(f"An error occurred while fetching news: {e}")

# End of main data check
else:
    # This block will be executed if the initial data load failed or returned empty data
    # The error message is already displayed by load_data_with_retry
    pass # No need for a duplicate error message here


# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.info("Created by Dakarai Lewis and Asukele Harewood.")