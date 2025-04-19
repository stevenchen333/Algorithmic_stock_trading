'''
Run the code below to create data for stock prices
'''
#-------------------------------------------------------------------------------
import requests
import json
import pandas as pd
from datetime import datetime

def create_stock_df(stocks, tickers):
    """
    Prompts the user to enter one or more stock tickers and retrieves their data.

    Returns:
        pd.DataFrame: DataFrame with 'Time' as the index and the selected stock ticker(s) as columns.
    """
    stocks = pd.DataFrame(stocks)
    available_tickers = list(stocks.columns)
    available_tickers.remove("Time")

    print(f"Available tickers: {', '.join(available_tickers)}")

    tickers = [ticker.strip() for ticker in tickers]

    missing_tickers = [ticker for ticker in tickers if ticker not in available_tickers]
    if missing_tickers:
        raise ValueError(f"Stock tickers {missing_tickers} not found in the stocks data.")

    stock_df = stocks[['Time'] + tickers].copy()
    stock_df.set_index('Time', inplace=True)

    return stock_df

def retrieve_stock(tickers,  start_date, end_date , token, save_file = False, dataframe = True):
    #start_date and end_date is in format yyyy-mm-dd
    
    tickers = [ticker.lower() for ticker in tickers]      
         
    # Define API Token
    _token = token # put tokens here
    headers = {
            'Content-Type': 'application/json'
            }
    requestResponse = requests.get(f"https://api.tiingo.com/tiingo/daily/{tickers[0]}/prices?startDate={start_date}&endDate={end_date}&token={_token}", 
                                headers=headers)
    data = requestResponse.json()
    time_list = []

    # Loop through the data and extract the 'date' field
    for j in range(len(data)):
    # Extract the date string
        date_string = data[j]['date']
        
        # Convert to datetime object and format it
        dt = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
        formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Append the formatted date to the time_list
        time_list.append(formatted_date)

    # Initialize a dictionary to store stock data
    stock_dict = {"Time": time_list}


    for i in range(len(tickers)):
            ticker = tickers[i]
            requestResponse = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={_token}", 
                                headers=headers)
            data = requestResponse.json()
            stock_dict[ticker] = []
            for j in range(len(data)):
                    stock_dict[ticker].append(data[j]['close'])
    if (save_file):
        file_name = input("Input your file name: ")
        with open(f"{file_name} {stock_dict['Time'][0]} - {stock_dict['Time'][-1]}.json", "w") as f:
            json.dump(stock_dict, f)

    if (dataframe):
         stock_dict = create_stock_df(stock_dict, tickers = tickers)



    return stock_dict

#-------------------------------------------------------------------------------




if __name__ == "__main__":
    # This will only run if you execute utils.py directly
    print(retrieve_stock(['AAPL']))