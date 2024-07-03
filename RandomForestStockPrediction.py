import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from stocknews import StockNews
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')

# Define helper functions
def load_data(symbol, start, end):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start, end=end)
    return data

def get_earnings(symbol):
    stock = yf.Ticker(symbol)
    earnings = stock.earnings
    return earnings


def getDividends(symbol):
    stock = yf.Ticker(symbol)
    currentYear = datetime.today().year
    dividends = stock.dividends
    currentDiv = dividends[dividends.index.year == currentYear]
    stock = pd.DataFrame(currentDiv).reset_index()
    stock['Date'] = pd.to_datetime(stock['Date'])
    stock['year'] = stock['Date'].dt.year
    stock['month'] = stock['Date'].dt.month
    stock['day'] = stock['Date'].dt.day
    stock['Date'] = pd.to_datetime(stock[['year', 'month', 'day']])
    stock.drop(columns=['year', 'month', 'day'], inplace=True)
    stock['Dividends'] = stock['Dividends'].astype(float)
    
    # Calculate the total dividends paid over the past year
    total_dividends = currentDiv.sum()
    
    # Calculate the average stock price over the past year
    avg_stock_price = data['Close'].mean()
    
    # Calculate the average dividend yield over the past year
    average_dividend_yield = total_dividends / avg_stock_price * 100
    

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

def getROE(symbol):
    stock = yf.Ticker(symbol)
    balanceSheet = stock.balance_sheet
    incomeStatement = stock.financials
    netIncome = float(incomeStatement.loc['Net Income'].iloc[0])
    totalAssets = float(balanceSheet.loc['Total Assets'].iloc[0])
    totalLiabilities = float(balanceSheet.loc['Total Liabilities Net Minority Interest'].iloc[0])
    shareHolderEquity = totalAssets - totalLiabilities
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

def getEPSGrowthRating(symbol):
    stock = yf.Ticker(symbol)
    eps = stock.earnings.reset_index()
    eps['Year'] = eps['Date'].dt.year
    eps_growth = ((eps.loc[eps['Year'].max(), 'Earnings'] / eps.loc[eps['Year'].min(), 'Earnings']) ** (1/5) - 1) * 100
    
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

def get1YearReturnRating(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")
    start_price = hist['Close'].iloc[0]
    end_price = hist['Close'].iloc[-1]
    one_year_return = ((end_price - start_price) / start_price) * 100
    
    if one_year_return < 0:
        rating = 1
    elif one_year_return <= 3:
        rating = 2
    elif one_year_return <= 6:
        rating = 3
    elif one_year_return <= 6:
        rating = 4
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

def get3MonthVolatility(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="3mo")
    daily_returns = hist['Close'].pct_change().dropna()
    volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility for daily returns
    
    return volatility


def get5_year_dividend_growth(symbol):
    stock = yf.Ticker(symbol)
    
    # Get historical dividend data for the past 5 years
    dividends = stock.dividends
    dividends = dividends.resample('Y').sum()  # Resample dividends annually and aggregate
    
    # Filter out dividends data for the last 5 years
    last_5_years_dividends = dividends[-5:]
    
    if len(last_5_years_dividends) < 2:
        return None  # Insufficient data to calculate growth rate
    
    # Calculate the 5-year dividend growth rate
    start_dividend = last_5_years_dividends.iloc[0]
    end_dividend = last_5_years_dividends.iloc[-1]
    dividend_growth_rate = ((end_dividend - start_dividend) / start_dividend) * 100
 
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

#def getpe_ratio(symbol):
    stock = yf.Ticker(symbol)
    
    # Get historical price and earnings data
    
    stock_price = data['Close'][0]
    eps = stock.earnings.reset_index()
    pe_ratio = stock_price / eps
    
    if pe_ratio < 10.0:
        rating = 10
    elif pe_ratio < 15.0:
        rating = 9
    elif pe_ratio < 20.0:
        rating = 8
    elif pe_ratio < 25.0:
        rating = 7
    elif pe_ratio < 30.0:
        rating = 6
    elif pe_ratio < 35.0:
        rating = 5
    elif pe_ratio < 40.0:
        rating = 4
    elif pe_ratio < 45.0:
        rating = 3
    elif pe_ratio < 50.0:
        rating = 2
    else:
        rating = 1
    
    return rating

#def getpfcf_ratio(symbol):
    stock = yf.Ticker(symbol)
    
    # Get historical price and free cash flow data
    historical_data = stock.history(period="1y")
    cash_flow = stock.cashflow.reset_index()
    latest_fcf = cash_flow['Free Cash Flow'].iloc[-1]  # Latest free cash flow value
    
    # Calculate the P/FCF ratio
    latest_price = historical_data['Close'].iloc[-1]  # Latest closing price
    pfcf_ratio = latest_price / latest_fcf
    
    if pfcf_ratio < 8.0:
        rating = 10
    elif pfcf_ratio < 12.0:
        rating = 9
    elif pfcf_ratio < 16.0:
        rating = 8
    elif pfcf_ratio < 20.0:
        rating = 7
    elif pfcf_ratio < 24.0:
        rating = 6
    elif pfcf_ratio < 28.0:
        rating = 5
    elif pfcf_ratio < 32.0:
        rating = 4
    elif pfcf_ratio < 36.0:
        rating = 3
    elif pfcf_ratio < 40.0:
        rating = 2
    else:
        rating = 1
    
    return rating

#def getdebt_equity_ratio(symbol):
    stock = yf.Ticker(symbol)
    
    # Get key financial ratios data
    key_ratios = stock.info['keyRatio']
    
    # Extract total debt and total equity values
    total_debt = key_ratios.get('totalDebt')
    total_equity = key_ratios.get('totalStockholderEquity')
    
    # Calculate the Debt/Equity ratio
    debt_equity_ratio = total_debt / total_equity
    
    if debt_equity_ratio < 0.5:
        rating = 10
    elif debt_equity_ratio < 1.0:
        rating = 9
    elif debt_equity_ratio < 1.5:
        rating = 8
    elif debt_equity_ratio < 2.0:
        rating = 7
    elif debt_equity_ratio < 2.5:
        rating = 6
    elif debt_equity_ratio < 3.0:
        rating = 5
    elif debt_equity_ratio < 3.5:
        rating = 4
    elif debt_equity_ratio < 4.0:
        rating = 3
    elif debt_equity_ratio < 4.5:
        rating = 2
    else:
        rating = 1
    
    return rating

def getnet_profit_margin(symbol):
    
    stock = yf.Ticker(symbol)
    
    stock = yf.Ticker(symbol)
    balanceSheet = stock.balance_sheet
    incomeStatement = stock.financials
    netIncome = float(incomeStatement.loc['Net Income'].iloc[0])
    
    # Extract revenue (total sales) from the balance sheet
    revenue = float(incomeStatement.loc['Total Revenue'].iloc[0])

    # Calculate net profit margin
    net_profit_margin = (netIncome / revenue) * 100

    if net_profit_margin is not None:
        if net_profit_margin > 40:
            rating = 10
        elif net_profit_margin > 30:
            rating = 9
        elif net_profit_margin > 20:
            rating = 8
        elif net_profit_margin > 15:
            rating = 7
        elif net_profit_margin > 10:
            rating = 6
        elif net_profit_margin > 5:
            rating = 5
        elif net_profit_margin > -0:
            rating = 4
        elif net_profit_margin > -5:
            rating = 3
        elif net_profit_margin > -10:
            rating = 2
        else:
            rating = 1
    else:
        rating = 0  # Handle the case where net_profit_margin is None
    
    return rating

def getroa(symbol):
    stock = yf.Ticker(symbol)
    
    balanceSheet = stock.balance_sheet
    incomeStatement = stock.financials    
    netIncome = float(incomeStatement.loc['Net Income'].iloc[0])
    totalAssets = float(balanceSheet.loc['Total Assets'].iloc[0])
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

#def getactual_eps_growth(symbol):
    stock = yf.Ticker(symbol)
    
    # Get historical EPS data
    quarterly_eps = stock.earnings['yearlyEarnings']
    
    # Select the EPS values for the past 12 months
    eps_values = quarterly_eps['revenue'][-5:]  # Selecting the last 5 quarters
    
    # Calculate EPS growth rate for each consecutive pair of quarters
    eps_growth = [(eps_values[i] / eps_values[i-1]) - 1 for i in range(1, len(eps_values))]
    
    # Average the EPS growth rates to obtain the 12-month EPS growth rate
    actual_eps_growth_rate = sum(eps_growth) / len(eps_growth)
  
    if eps_growth > 0.25:
        rating = 10
    elif eps_growth > 0.20:
        rating = 9
    elif eps_growth > 0.15:
        rating = 8
    elif eps_growth > 0.10:
        rating = 7
    elif eps_growth > 0.05:
        rating = 6
    elif eps_growth > 0.0:
        rating = 5
    elif eps_growth > -0.05:
        rating = 4
    elif eps_growth > -0.10:
        rating = 3
    elif eps_growth > -0.15:
        rating = 2
    else:
        rating = 1
    
    return rating

#def get5year_eps_growth(symbol):
    stock = yf.Ticker(symbol)
    
    # Get key statistics data
    key_stats = stock.info['defaultKeyStatistics']
    
    # Extract the 5-year EPS growth rate value
    eps_growth_5y = key_stats.get('5YearAverageEPS')
    
    if eps_growth_5y > 0.20:
        rating = 10
    elif eps_growth_5y > 0.15:
        rating = 9
    elif eps_growth_5y > 0.10:
        rating = 8
    elif eps_growth_5y > 0.05:
        rating = 7
    elif eps_growth_5y > 0.0:
        rating = 6
    elif eps_growth_5y > -0.05:
        rating = 5
    elif eps_growth_5y > -0.10:
        rating = 4
    elif eps_growth_5y > -0.15:
        rating = 3
    elif eps_growth_5y > -0.20:
        rating = 2
    else:
        rating = 1
    
    return rating

def getNet_profit_margin(symbol):
    stock = yf.Ticker(symbol)
    
    # Get financial data
    balance_sheet = stock.balance_sheet
    income_statement = stock.financials

    # Extract net income from the income statement
    net_income = income_statement['NetIncome'][0]
    
    # Extract revenue (total sales) from the balance sheet
    revenue = balance_sheet['TotalRevenue'][0]

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

# Streamlit interface
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

st.title("🔮 Ticker Predicter ")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Title
st.sidebar.header(' Ticker Predicter')

# Define the list of famous ticker symbols and an option to enter a custom ticker symbol
famous_ticker_symbols = all_ticker_symbols = sorted(['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA','META', 'NFLX', 'DIS', 'V', 'JPM', 'BA', 'INTC', 'GM', 'KO', 'IBM', 'AMD', 'PFE', 'UPS', 'WMT', 'SBUX'])
options = famous_ticker_symbols + ["Other option"]

# Sidebar input for selecting a stock ticker symbol
symbol = st.sidebar.selectbox("Select a Stock Ticker Symbol", options=options)


# If user selects to enter a custom ticker symbol
if symbol == "Other option":
    symbol = st.sidebar.text_input("Please input your other Ticker Symbol")
    
    # Check if the otherOption text input is empty
if not symbol:
        st.write("Please enter Ticker Symbol.")
        st.stop()
    

def get_stock_name(symbol):
    stock_info = yf.Ticker(symbol)
    return stock_info.info['longName']   

# Get the long name of the stock symbol
stock_name = get_stock_name(symbol)  
    
# Convert the input to uppercase
symbol = symbol.upper()

# User interface pages
page = st.sidebar.radio("Navigation", ["Ticker Predicter Dashboard", "Model Performance Metrics", "Stock Charts", "Additional Stock Company Info"])



# Choose the date range
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", datetime.today())
data = load_data(symbol, start_date, end_date)

# Display a loading spinner
with st.spinner('Loading...'):
    # Simulate a time-consuming task
    time.sleep(5)


# Model Training and Prediction
# Feature preparation
data['Next Close'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Prepare feature set and target variable
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Next Close']


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate polynomial features for both train and test sets
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_poly_train, y_train)



# Predict for the next day
last_known_data = data[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
last_known_data_poly = poly.transform(last_known_data)
predicted_price = model.predict(last_known_data_poly)

# Make predictions
predicted_prices = model.predict(X_poly_test)


# Performance Metrics
y_pred = model.predict(X_poly_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sort the data points by date
sorted_dates = y_test.index
sorted_indices = np.argsort(sorted_dates)
sorted_dates = sorted_dates[sorted_indices]
sorted_actual_prices = y_test[sorted_indices]
sorted_predicted_prices = predicted_prices[sorted_indices]

# Create a line plot for actual vs predicted prices
# Create the interactive line plot using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=sorted_dates, y=sorted_actual_prices, mode='lines+markers', name='Actual Prices', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=sorted_dates, y=sorted_predicted_prices, mode='lines+markers', name='Predicted Prices', line=dict(color='green')))
fig.update_layout(title=f'Random Forest Actual vs Predicted Closing Stock Prices for {symbol}', xaxis_title='Date', yaxis_title='Stock Prices (USD)', autosize=True, width=800, height=500)

def calculate_parameters(symbol):
    stock = yf.Ticker(symbol)

    # Financial Parameters Calculation
    #balance_sheet = stock.balance_sheet
    #income_statement = stock.financials
    #cashflow_statement = stock.cashflow
    
    #net_income = float(income_statement.loc['Net Income'].iloc[0])
    #total_assets = float(balance_sheet.loc['Total Assets'].iloc[0])
    #total_liabilities = float(balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0])
    #shareholders_equity = total_assets - total_liabilities
    
    # ROE (Return on Equity)
    roe_rating = getROE(symbol)
    roe = roe_rating
    
    # 1-Year Return
    one_year_return = get1YearReturnRating(symbol)
    
    # Dividend Yield
    dividend_yield = getDividends(symbol)
    
    # 5-Year Dividend Growth
    five_year_div_growth = get5_year_dividend_growth(symbol)
    
    # P/E Ratio
    #pe_ratio = getpe_ratio(symbol)

    # P/FCF Ratio (Price to Free Cash Flow)
    #p_fcf_ratio = getpfcf_ratio(symbol)
    
    # 5-Year EPS Growth
    #five_year_eps_growth = get5year_eps_growth(symbol)
    # 12-Month EPS Growth
    #twelve_month_eps_growth = getactual_eps_growth(symbol)
    
    # Debt to Equity Ratio
    #debt_equity_ratio = getdebt_equity_ratio(symbol)
    
    # Net Profit Margin
    net_profit_margin = getnet_profit_margin(symbol)
    
    # ROA (Return on Assets)
    roa = getroa(symbol)

    # Define the parameters and their rating calculation functions
    parameters = {
        'ROE': roe,
        '1-Year Return': one_year_return,
        'Dividend Yield': dividend_yield,
        '5-Year Dividend Growth': five_year_div_growth,
        'Net Profit Margin': net_profit_margin,
        #'P/E Ratio': pe_ratio,
        #'P/FCF Ratio': p_fcf_ratio,
        #'5-Year EPS Growth': five_year_eps_growth,
        #'12-Month EPS Growth': twelve_month_eps_growth,
        #'Debt/Equity': debt_equity_ratio,
        'ROA': roa,
    }
    
    
    return parameters

# Get ratings for a specific stock symbol (e.g., 'AAPL')
def create_radar_chart(symbol):
    # Get ratings for a specific stock symbol (e.g., 'AAPL')
    ratings = calculate_parameters(symbol)

    # Create a DataFrame from the ratings data
    df = pd.DataFrame(list(ratings.items()), columns=['Parameter', 'Rating'])

    # Create a radar chart to visualize ratings for each parameter
    fig2 = px.line_polar(df, r='Rating', theta='Parameter', line_close=True, title=f"Financial Ratings for {stock_name}")
    fig2.update_traces(fill='toself')
    return fig2



    # Define a layout with two columns

# Page: Ticker Predictor Dashboard
if page == "Ticker Predictor Dashboard":   
    col11, col12 = st.columns([1, 1])

    # Display content in the first column (two rows)
    with col11:
        con11 = st.container(border=True)
        # Add space between rows using markdown with line breaks
        st.markdown("<br>", unsafe_allow_html=True)
        con12 = st.container(border=True)
        with con11:
            st.header(f"📈 You have selected: {stock_name}")

        
        with con12:
            st.header(f"🔭 Next Closing Price Prediction: ${predicted_price[0]:.2f}")

    # Display content in the second column (one row)
    with col12:
        # Display the radar chart in Streamlit
        radar_chart = create_radar_chart(symbol)
        st.plotly_chart(radar_chart)

    # Display the interactive plot using st.plotly_chart
    st.plotly_chart(fig)



# Page: Stock Charts
elif page == "Model Performance Metrics":   
    st.markdown('### 📊 Random Forest Model Performance Metrics:')
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



# Page: Stock Charts
elif page == "Stock Charts":
    # Interactive Plotly chart for stock closing price
    fig_interactive = px.line(data, x=data.index, y='Close', title=f'{symbol} Stock Closing Price', labels={'x': 'Date', 'y': 'Close'})
    fig_interactive.update_layout(autosize=True, width=550, height=450)
    #st.plotly_chart(fig_interactive)

    # Additional visualizations: Moving Average
    fig_moving_avg = px.line(data, x=data.index, y='Close', title=f'Stock Closing Price with Moving Average for {symbol}')
    fig_moving_avg.add_scatter(x=data.index, y=data['Close'].rolling(window=30).mean(), mode='lines', name='30-Day Moving Average')
    fig_moving_avg.update_layout(autosize=True, width=550, height=450)

    #st.plotly_chart(fig_moving_avg)

    # Candlestick Chart
    fig1 = go.Figure()
    fig1.add_traces([
        go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candle Stick Chart'),
        go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean(), line=dict(color='orange', width=1), name='Middle Band'),
        go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean() + 1.96*data['Close'].rolling(window=20).std(), line=dict(color='gray', width=1), name='Upper Band'),
        go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean() - 1.96*data['Close'].rolling(window=20).std(), line=dict(color='gray', width=1), name='Lower Band')
    ])
    fig1.update_layout(title_text=f'🕯️ {symbol} Candlestick Chart with Bollinger Bands', autosize=True, width=550, height=450)
    #st.plotly_chart(fig1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Stock Charts")

    # Organize the charts into a dashboard layout using st.columns and st.plotly_chart
    c1, c2 = st.columns([1, 1])
    co1 = c1.container(border=True)
    co2 = c2.container(border=True)

    with co1:
        st.plotly_chart(fig_interactive, use_container_width=True)  # Interactive Closing Price figure
        
    with co2:
        st.plotly_chart(fig_moving_avg, use_container_width=True)  # Stock Closing Price with Moving Average

        

    # Create a heatmap of the stock data

    plt.figure(figsize=(8, 5.5))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.gcf().patch.set_alpha(0.4)
    plt.title('Correlation Heatmap of Stock Data')

    c3, c4 = st.columns([1, 1])
    co3 = c3.container(border=True)
    co4 = c4.container(border=True)

    with co3:
        st.plotly_chart(fig1, use_container_width=True)  # Candlestick Chart
        
    with co4:
        st.pyplot(use_container_width=True)

  
# Page: Additional Stock Company Info        
# Tabs for Historical data, fundamental data and News on relative Stock
elif page == "Additional Stock Company Info":
    historical_pricing_data, fundamental_data, news = st.tabs(["Historical Pricing Data", "Fundamental Data", "Top 10 News"])

    # Define the content for each tab
    with historical_pricing_data:  # Historical Pricing Data tab
        st.header(f'💸 {stock_name} Price Movements')
        st.write(data)

    with fundamental_data:  # Fundamental Data tab
        stock = yf.Ticker(symbol)
        st.header("🧾 Balance Sheet")
        balance_sheet = stock.balance_sheet
        st.write(balance_sheet)
        st.header("🤑 Income Statement")
        income_statement = stock.financials
        st.write(income_statement)
        st.header("💵 Cash Flow Statement")
        cash_flow = stock.cashflow
        st.write(cash_flow)

    with news:  # Top 10 News tab
        st.header(f'📢 News for {stock_name}')
        sn = StockNews(symbol, save_news=False)
        df_news = sn.read_rss()
        
        for i in range(10):
            st.subheader(f'News {i+1}:')
            st.write(df_news['published'][i])
            st.write(df_news['title'][i])
            st.write(df_news['summary'][i])
            title_sentiment = df_news['sentiment_title'][i]
            st.write(f'Title Sentiment {title_sentiment}')
            news_sentiment = df_news['sentiment_summary'][i]
            st.write(f'News Sentiment {news_sentiment}')
        


# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.title("About")
st.sidebar.info("This app uses a Random Forest model to predict future stock prices. Select a stock symbol as well as the start and end date above to get started.")
st.sidebar.info("Created by Dakarai Lewis and Asukele Harewood.")

