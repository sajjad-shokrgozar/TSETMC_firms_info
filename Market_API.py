import pandas as pd
import datetime, time, jdatetime
import numpy as np
from scipy.stats import norm
import requests
import pandas as pd
import datetime, time
# from pandas_jalali.converter import get_gregorian_date_from_jalali_date
import math
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import warnings
warnings.filterwarnings('ignore')



class Market:

    def __init__(self, risk_free_rate=.3):
        self.headers = {
            'Host': 'cdn.tsetmc.com',
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/110.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.risk_free_rate = risk_free_rate
        self.c_date = datetime.date.today()

    def to_gregorian_date(self, maturity):
        try:
            maturity = str(maturity)
            maturity = maturity.replace('/', '')
            if maturity[:2] != '14':
                maturity = '14' + maturity
            year = maturity[:4]
            month = maturity[4:6]
            day = maturity[6:8]
            gregorian_date = jdatetime.date(int(year),int(month),int(day)).togregorian()

            return gregorian_date
        except:
            return None

    def to_jalali(self, date):
        date = str(date)
        if date.strip() == '':
            return None
        date = str(date).replace('-', '')
        if len(date.strip()) != 8:
            print(date, 'is not a standard date!')
            return None

        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])

        jdate_list = jdatetime.GregorianToJalali(year, month, day)
        j_year, j_month, j_day = str(jdate_list.jyear), str(jdate_list.jmonth), str(jdate_list.jday)

        if len(j_month) == 1:
            j_month = '0' + j_month
        if len(j_day) == 1:
            j_day = '0' + j_day

        return j_year + j_month + j_day

    def maturity_validation(self, x):
        try:
            if len(x) < 6:
                return False
            else:
                return True
        except:
            return True
        
    def cal_ttm(self, maturity_date):
        maturity_date = datetime.date.fromisoformat(str(maturity_date).replace(' 00:00:00', ''))
        c_date = datetime.date.fromisoformat(str(self.c_date).replace(' 00:00:00', ''))
        return int((maturity_date - c_date).days)
    
    def characters_modifier(self, text):
        return text.replace('ي', 'ی').replace('ك', 'ک').replace('پ', 'پ').strip()

    # Define a function to extract the required values
    def extract_values(self, row):
        first_order = row[0]
        return (first_order['pmd'], first_order['qmd'], first_order['pmo'], first_order['qmo'])
    
    # FIXME: some options have different size from 1000. so add size of option.
    def init_margin(self, type, underlying, s_close, s_last, k, c, contract_size, A=.2, B=.1):
        if type == 'call':
            loss = abs(min([s_close - k, 0])) * contract_size
        elif type == 'put':
            loss = abs(min([k - s_close, 0])) * contract_size

        val1 = (contract_size * A * s_close - loss)
        val2 = B * k * contract_size
        IM = max([val1, val2])

        # FIXME: I found an exeption. for 'موج' it is 100000 !
        if underlying in ['موج']:
            round_multiplier = 100000
        else:
            round_multiplier = 10000

        val = ((math.floor(IM/round_multiplier)+1)*round_multiplier) + c * contract_size
        return val/contract_size
    
    def get_last_price(self, x, total_df):
        x = x.strip()
        try:
            ua_last_price = total_df[total_df['lva'] == x]['pdv'].values[0]
        except:
            ua_last_price = None
        return ua_last_price

    def get_close_price(self, x, total_df):
        x = x.strip()
        try:
            ua_close_price = total_df[total_df['lva'] == x]['pcl'].values[0]
        except:
            ua_close_price = None
        return ua_close_price
    
    def black_scholes(self, s, k, ttm, r, sigma, option_type):
        T = ttm / 365
        d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return s * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            return k * np.exp(-r * T) * norm.cdf(-d2) - s * norm.cdf(-d1)
            
    def implied_volatility(self, S, K, market_price, ttm, r, option_type, tol=1e-5, max_iter=60):
        low_vol = 1e-5
        high_vol = 5.0
        
        for i in range(max_iter):
            mid_vol = (low_vol + high_vol) / 2.0
            price = self.black_scholes(S, K, ttm, r, mid_vol, option_type)
            
            if abs(price - market_price) < tol:
                return mid_vol
            
            if price > market_price:
                high_vol = mid_vol
            else:
                low_vol = mid_vol
        return 0

    def delta(self, s, k, ttm, r, sigma, option_type):
        T = ttm / 365
        d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        if option_type == 'call':
            return norm.cdf(d1)
        elif option_type == 'put':
            return 1 - norm.cdf(d1)

    def gamma(self, s, k, ttm, r, sigma):
        T = ttm / 365
        d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        n_prime_d1 = np.exp(-.5*(d1**2)) / np.sqrt(2*np.pi)
        gamma = n_prime_d1 / (s * sigma * np.sqrt(T))

        return gamma
    
    def get_option_market_watch(self):
        url = 'https://cdn.tsetmc.com/api/ClosingPrice/GetMarketWatch?market=0&industrialGroup=&paperTypes%5B0%5D=6&paperTypes%5B1%5D=2&paperTypes%5B2%5D=1&paperTypes%5B3%5D=8&showTraded=false&withBestLimits=true&hEven=0&RefID=0'
        res = requests.get(url, headers=self.headers)
        market_watch = res.json()['marketwatch']
        df = pd.DataFrame(market_watch)
        temp = df['blDs'].apply(lambda x: self.extract_values(x))
        extracted_df = pd.DataFrame(temp.tolist(), columns=['pmd1', 'qmd1', 'pmo1', 'qmo1'])
        # Concatenate the original DataFrame with the new columns
        temp_df = pd.concat([df, extracted_df], axis=1)
        temp_df = temp_df[['insCode', 'insID', 'lva', 'lvc', 'pdv', 'qtc', 'pMax', 'pMin', 'py', 'pcl', 'pmd1', 'qmd1', 'pmo1', 'qmo1', 'ztd']]
        total_df = temp_df.copy()
        temp_df = temp_df[temp_df['lvc'].str.contains('اختيار')]
        temp_df['type'] = 'call'
        temp_df.loc[temp_df['lvc'].str.contains('اختيارف'), 'type'] = 'put'
        temp_df['lva'] = temp_df['lva'].apply(lambda x: self.characters_modifier(x))
        temp_df[['underlying', 'strike', 'maturity']] = temp_df['lvc'].str.split('-', expand=True)
        temp_df['underlying'] = temp_df['underlying'].replace(r'اختيارف', '', regex=True).replace(r'اختيارخ', '', regex=True)
        temp_df = temp_df[['insCode', 'insID', 'lva', 'type', 'underlying', 'strike', 'maturity', 'pmd1', 'qmd1', 'pmo1', 'qmo1', 'pdv', 'qtc', 'pMax', 'pMin', 'py', 'pcl', 'ztd']]
        temp_df = temp_df[temp_df['maturity'].apply(lambda x: self.maturity_validation(x))]
        temp_df = temp_df[~temp_df['maturity'].isna()]
        temp_df['underlying'] = temp_df['underlying'].apply(lambda x: self.characters_modifier(x))
        temp_df['g_maturity'] = temp_df['maturity'].apply(lambda x: self.to_gregorian_date(x))
        temp_df['ttm'] = temp_df['g_maturity'].apply(lambda x: self.cal_ttm(x))
        temp_df['ua_last_price'] = temp_df['underlying'].apply(lambda x: self.get_last_price(x, total_df))
        temp_df['ua_close_price'] = temp_df['underlying'].apply(lambda x: self.get_close_price(x, total_df))
        temp_df = temp_df[~temp_df['ua_last_price'].isna()]
        temp_df['strike'] = temp_df['strike'].astype(float)
        temp_df['IV'] = temp_df.apply(lambda x: self.implied_volatility(x['ua_last_price'], x['strike'], x['pdv'], x['ttm'], self.risk_free_rate, x['type']), axis=1)
        # temp_df['bs_HV'] = temp_df.apply(lambda x: black_scholes(x['ua_last_price'], x['strike'], x['ttm'], risk_free_rate, .35, x['type']), axis=1)
        temp_df['delta'] = temp_df.apply(lambda x: self.delta(x['ua_last_price'], x['strike'], x['ttm'], self.risk_free_rate, x['IV'], x['type']), axis=1)
        temp_df['gamma'] = temp_df.apply(lambda x: self.gamma(x['ua_last_price'], x['strike'], x['ttm'], self.risk_free_rate, x['IV']), axis=1)
        temp_df['market'] = 'option'
        temp_df = temp_df[['insCode', 'insID', 'lva', 'market', 'type', 'strike', 'ttm', 'pcl', 'pdv', 'qtc', 'pmd1', 'qmd1', 'pmo1', 'qmo1', 'ua_last_price', 'ua_close_price','IV', 'delta', 'gamma']]
        temp_df.columns = ['id', 'code', 'symbol', 'market', 'type', 'strike', 'ttm', 'close', 'last', 'volume', 'bid_P', 'bid_Q', 'ask_P', 'ask_Q', 'ua_last', 'ua_close','IV', 'delta', 'gamma']

        return temp_df


    def get_stock_market_watch(self):
        url = 'https://cdn.tsetmc.com/api/ClosingPrice/GetMarketWatch?market=0&industrialGroup=&paperTypes%5B0%5D=2&paperTypes%5B1%5D=1&paperTypes%5B2%5D=8&showTraded=false&withBestLimits=true&hEven=0&RefID=0'
        res = requests.get(url, headers=self.headers)
        market_watch = res.json()['marketwatch']
        df = pd.DataFrame(market_watch)
        temp = df['blDs'].apply(lambda x: self.extract_values(x))
        extracted_df = pd.DataFrame(temp.tolist(), columns=['pmd1', 'qmd1', 'pmo1', 'qmo1'])
        # Concatenate the original DataFrame with the new columns
        temp_df = pd.concat([df, extracted_df], axis=1)
        temp_df = temp_df[['insCode', 'insID', 'lva', 'lvc', 'pdv', 'qtc', 'pMax', 'pMin', 'py', 'pcl', 'pmd1', 'qmd1', 'pmo1', 'qmo1', 'ztd']]
        # total_df = temp_df.copy()
        temp_df['lva'] = temp_df['lva'].apply(lambda x: self.characters_modifier(x))
        temp_df[['type', 'strike', 'ttm', 'ua_last_price', 'ua_close_price', 'IV', 'delta', 'gamma']] = None
        temp_df['market'] = 'stock'
        temp_df = temp_df[['insCode', 'insID', 'lva', 'market', 'type', 'strike', 'ttm', 'pcl', 'pdv', 'qtc', 'pmd1', 'qmd1', 'pmo1', 'qmo1', 'ua_last_price', 'ua_close_price']]
        temp_df.columns = ['id', 'code', 'symbol', 'market', 'type', 'strike', 'ttm', 'close', 'last', 'volume', 'bid_P', 'bid_Q', 'ask_P', 'ask_Q', 'ua_last', 'ua_close']

        return temp_df

    def get_market_watch(self):
        tse_options = self.get_option_market_watch()
        tse_stock = self.get_stock_market_watch()
        tse_data = pd.concat([tse_options, tse_stock])

        return tse_data
    
    def cumulation(self, x):
        x['weighted_price'] = x['price'] * x['vol']

        x['price'] = float(x['weighted_price'].sum() / x['vol'].sum())
        x['vol'] = x['vol'].sum()
        return x
    
    def get_total_trades(self, excel_name, sheet_name, tse_data):
        trades_df = pd.read_excel(excel_name, sheet_name=sheet_name)
        trades_df = trades_df[['date', 'fund', 'symbol', 'position', 'price', 'vol']]
        trades_df = pd.merge(trades_df, tse_data[['symbol', 'market']], how='left', on='symbol')
        trades_df['market'].fillna('option', inplace=True)
        trades_df['date'] = trades_df['date'].replace(r'- (.*)', '', regex=True)
        # find option type
        trades_df['type'] = None
        trades_df.loc[(trades_df['symbol'].str.startswith('ط')) & (trades_df['market'] == 'option'), 'type'] = 'put'
        trades_df.loc[(trades_df['symbol'].str.startswith('ض')) & (trades_df['market'] == 'option'), 'type'] = 'call'
        # find holding days
        trades_df['holding_days'] = trades_df.apply(lambda x: -self.cal_ttm(self.to_gregorian_date(x['date'])), axis=1)

        return trades_df

    def get_cumulative_trades(self, total_trades_df, tse_data):
        # match tse data with trades
        ls_cum_df = pd.merge(total_trades_df, tse_data, how='left', on='symbol')
        ls_cum_df = ls_cum_df[['date', 'fund', 'symbol', 'position', 'price', 'vol', 'type_x', 'holding_days']]
        ls_cum_df.rename({'type_x': 'type'}, axis=1, inplace=True)

        # adding opened_iv, opened_delta columns
        # ls_cum_df = ls_cum_df[~ls_cum_df['strike'].isna()]
        # ls_cum_df['opened_iv'] = ls_cum_df.apply(lambda x: implied_volatility(x['ua_price'], x['strike'], x['price'], x['ttm'], risk_free_rate, x['type']), axis=1)
        # ls_cum_df['opened_delta'] = ls_cum_df.apply(lambda x: delta(x['ua_price'], x['strike'], x['ttm'], risk_free_rate, x['opened_iv'], x['type']), axis=1)
        # ls_cum_df = ls_cum_df[['account', 'strategy_id', 'date', 'option', 'position', 'price', 'vol', 'ua_price', 'strike', 'type', 'holding_days', 'ttm', 'opened_iv', 'opened_delta']]

        # removing nans: somtimes there is nan that comes from not existence in tse_data because of not existing order in order book
        # ls_cum_df = ls_cum_df.groupby(['account', 'strategy_id']).apply(lambda x: x if len(x['option'].unique()) > 1 else None).reset_index(drop=True)

        # cumulation process
        ls_cum_df = ls_cum_df.groupby(['symbol', 'position']).apply(lambda x: self.cumulation(x)).reset_index(drop=True)
        ls_cum_df = ls_cum_df.drop_duplicates(subset=['symbol', 'position'])

        return ls_cum_df
    
    def create_portfolio(self, tdf, tse_data):
        tdf['net_vol'] = tdf.apply(lambda row: row['vol'] if row['position'] == 'long' else -row['vol'], axis=1)

        # Group by symbol and calculate the net volume and current value
        portfolio = tdf.groupby('symbol').agg(
            net_volume=('net_vol', 'sum'),
            avg_price=('price', 'mean'),  # Average price for simplicity, can be adjusted
            fund=('fund', 'first')
        ).reset_index()

        portfolio = pd.merge(portfolio, tse_data[['symbol', 'market']], how='left', on='symbol')
        portfolio['market'].fillna('option', inplace=True)

        # Calculate current value based on net volume and average price
        # portfolio['current_value'] = portfolio['net_volume'] * portfolio['avg_price']

        # Filter out symbols with zero net volume
        portfolio = portfolio[portfolio['net_volume'] != 0]
        portfolio = portfolio[['symbol', 'fund', 'market', 'net_volume']]
        portfolio.columns = ['symbol', 'fund', 'market', 'vol']

        portfolio['type'] = None
        portfolio.loc[(portfolio['symbol'].str.startswith('ط')) & (portfolio['market'] == 'option'), 'type'] = 'put'
        portfolio.loc[(portfolio['symbol'].str.startswith('ض')) & (portfolio['market'] == 'option'), 'type'] = 'call'

        portfolio['position'] = 'long'
        portfolio.loc[portfolio['vol'] < 0, 'position'] = 'short'

        portfolio.columns = ['symbol', 'fund', 'market', 'vol', 'type', 'position']
        return portfolio[['symbol', 'fund', 'market', 'type', 'position', 'vol']]
    
    # Define the NPV function
    def npv(self, cf_value_and_T, r):
        return (cf_value_and_T['value'] * (1 + r)**(cf_value_and_T['until_now'] / 30)).sum()

    # Optimized IRR function with dynamic bracket adjustment
    def irr(self, cf_value_and_T):
        # Start with an initial bracket
        lower, upper = -0.99, 10
        # Expand bracket until f(lower) and f(upper) have opposite signs
        while self.npv(cf_value_and_T, lower) * self.npv(cf_value_and_T, upper) > 0:
            upper += 10
            if upper > 1000:  # Prevent infinite loop
                return None
                # raise ValueError("IRR not found within reasonable range.")
        
        # Use root_scalar to find IRR
        result = root_scalar(lambda r: self.npv(cf_value_and_T, r), bracket=[lower, upper], method='brentq', xtol=1e-6)
        return result.root  # Return the IRR