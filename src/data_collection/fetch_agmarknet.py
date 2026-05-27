import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# Agmarknet URL
URL = "https://agmarknet.gov.in/SearchCmmMkt.aspx"

def fetch_agmarknet_data(commodity_code, state_code, start_date, end_date):
    """
    Template function to scrape Agmarknet.
    Note: Agmarknet uses ASP.NET with VIEWSTATE. To properly scrape it,
    you usually need to make an initial GET request to extract the __VIEWSTATE,
    __VIEWSTATEGENERATOR, and __EVENTVALIDATION tokens, then pass them in the POST request.
    
    For a production Data Science project, it is highly recommended to use existing 
    Kaggle datasets for Agmarknet if the portal blocks automated requests.
    """
    
    print(f"Starting scraping for Commodity: {commodity_code}, State: {state_code} from {start_date} to {end_date}")
    
    # 1. Get the initial session and hidden form tokens (pseudo-code structure)
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    try:
        response = session.get(URL, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract hidden fields required for ASP.NET postbacks
        viewstate = soup.find('input', {'id': '__VIEWSTATE'})
        viewstate = viewstate['value'] if viewstate else ''
        
        viewstategenerator = soup.find('input', {'id': '__VIEWSTATEGENERATOR'})
        viewstategenerator = viewstategenerator['value'] if viewstategenerator else ''
        
        eventvalidation = soup.find('input', {'id': '__EVENTVALIDATION'})
        eventvalidation = eventvalidation['value'] if eventvalidation else ''
        
        # 2. Prepare POST payload
        payload = {
            '__VIEWSTATE': viewstate,
            '__VIEWSTATEGENERATOR': viewstategenerator,
            '__EVENTVALIDATION': eventvalidation,
            'ddlArrivalPrice': '0',
            'ddlCommodity': commodity_code, 
            'ddlState': state_code,
            'txtDate': start_date,
            'txtDateTo': end_date,
            'btnGo': 'Go'
        }
        
        # 3. Submit POST request
        post_response = session.post(URL, data=payload, timeout=20)
        post_soup = BeautifulSoup(post_response.text, 'html.parser')
        
        # 4. Find the data table (id usually gridview or similar)
        table = post_soup.find('table', {'id': 'cphBody_GridArrivalData'})
        if not table:
            print("No data table found in response. The site might have blocked the request or no data exists.")
            return pd.DataFrame()
            
        # Parse table using pandas
        dfs = pd.read_html(str(table))
        if dfs:
            df = dfs[0]
            print(f"Successfully fetched {len(df)} rows.")
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example codes: Tomato=78, Tamil Nadu=TN
    commodity = "78" 
    state = "TN"
    
    # Let's try to fetch the last 7 days of data
    end_date = datetime.now().strftime("%d-%b-%Y")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")
    
    df = fetch_agmarknet_data(commodity, state, start_date, end_date)
    
    if not df.empty:
        os.makedirs("../../data/raw", exist_ok=True)
        filename = f"../../data/raw/agmarknet_{commodity}_{state}.csv"
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print("Failed to fetch data. Consider using a pre-scraped Kaggle dataset.")
