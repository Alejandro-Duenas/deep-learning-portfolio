#------------------ Libraries----------------------
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
mlp.style.use('seaborn-darkgrid')

#--------------- Financial Data Class---------------

class FinancialData(object):

    '''This is a class stores information about financial data from ticker or 
    list of tickers, that can be prepared as a data frame and plotted.'''
    
    def __init__(self,tickers=['SPY'],period='max'):
        
        self.tickers = tickers
        self.period = period
    
    def prepare_data(self,fillna=True):
        '''This functions takes the data from Yahoo Finance from a set of tickers
        and prepares it in a dataframe
        INPUTS:
            tickers (list): a list of all the tickers that you want to analize
            period (string): a string with the period of the data you want to
                            analize
        OUTPUTS:
            df (pandas Data Frame): Data frame with the data required
        '''
        tickers = self.tickers
        period = self.period
        if len(tickers)>1:
            try:
                df_information = yf.Tickers(' '.join(tickers))
                df = pd.DataFrame(df_information.history(period=period))
                df.columns = ['_'.join(tup) for tup in df.columns.values]
                if fillna:
                    df.fillna(method='ffill',inplace=True)
                    df.fillna(method='bfill',inplace=True)
                self.df = df
                return self.df
            except:
                base_info = yf.Ticker(tickers[0])
                base_df = pd.DataFrame(base_info.history(period=period))
                base_df.columns = [col+'_'+tickers[0] for col in base_df.columns.values]

                for ticker in tickers[1:]:
                    temp_info = yf.Ticker(ticker)
                    temp_df = pd.DataFrame(temp_info.history(period=period))
                    temp_df.columns = [col+'_'+ticker for col in temp_df.columns.values]
                    base_df = base_df.join(temp_df,how='outer')
                if fillna:
                    base_df.fillna(method='ffill',inplace=True)
                    base_df.fillna(method='bfill',inplace=True)
                self.df = base_df
                return self.df

        else:
            info = yf.Ticker(tickers[0])
            df = pd.DataFrame(info.history(period=period))
            df.columns = [column+'_'+tickers[0] for column in df.columns]
            if fillna:
                    df.fillna(method='ffill',inplace=True)
                    df.fillna(method='bfill',inplace=True)
            self.df = df
            return df
        
    def plot_data(self,column = 'Close',kind='line', 
                  title='Historical Close Price Data',ylabel='Close Prices', 
                  xlabel='Date',**kwargs):
        '''This function plots the data in the dataframe according to the column given
        assuming that the index of the dataframe is the x axis of the plot
        INPUTS:
            df (pandas dataframe): dataframe with the information to be plotted
            tickers (list): list of the tickers to be plotted
            column (string): the kind of data to be plotted
        OUTPUTS:
            None
        '''
        df = self.df
        tickers = self.tickers
        columns_interest = [column+'_'+ticker for ticker in tickers]
        df[columns_interest].plot(kind=kind,title=title,**kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    
    def rolling_statistics(self,column='Close',ticker='SPY',function=None,
                      window=20,plot=False,bollinger=False,
                       rolling_color='crimson',roll_linewidth=1.5,**kwargs):
        '''This functions extracts the rolling statistics from a time series, and
        can plot the rolling window with the data, adding the Bollinger bands.
        
        INPUTS:
            df (pandas Data Frame): data frame with the financial data, that contains
                both the column and ticker specified in the function
            column (string): the column that you want to get the rolling function of
            ticker (str): the ticker from which you want to know the information
            function: function that will be rolled through the time series
            window (int): the window of the rolling data
            
        OUTPUTS:
            rolled (pandas series): a series with the rolling statistics specified

        '''
        df = self.df
        series = df[column+'_'+ticker].rolling(window).mean()
        if plot:
            ax = df[column+'_'+ticker].plot(label=column+' '+ticker,**kwargs)
            series.plot(ax=ax,color=rolling_color,linewidth=roll_linewidth)
            if bollinger:
                (series+(2*df[column+'_'+ticker]).rolling(window).std()).plot(linestyle='--',
                                                                          color='darkgreen',
                                                                           linewidth=1,
                                                                           ax=ax,
                                                                           label=
                                                                           'Bollinger')
                (series-(2*df[column+'_'+ticker]).rolling(window).std()).plot(linestyle='--',
                                                                          color='darkgreen',
                                                                           linewidth=1,
                                                                           ax=ax,
                                                                           label='')
            ax.legend();
        return series
    
    def find_returns(self,columns,tickers,return_window=1,plot=False,**kwargs):
        '''This function finds the returns for a set of tickers and prices
        INPUTS:
            df (pandas Data Frame): dataframe containing the time series information
            columns (list or string): columns to find the returns
            tickers (list or string): tickers to be analyzed
            return_window (int): the window from which to get the returns, default 
            daily

        OUTPUTS:
            returns (pandas series or data frame): pandas data structure with the 
            returns
        '''
        df = self.df
        if isinstance(columns,list) and isinstance(tickers,list):
            col_names = [column+'_'+ticker for ticker in tickers for column in columns]
        elif isinstance(columns,list):
            col_names = [column+'_'+tickers for column in columns]
        elif isinstance(tickers,list):
            col_names = [columns+'_'+ticker for ticker in tickers]
        else:
            col_names = columns+'_'+tickers
        returns = df[col_names].pct_change(return_window)
        if plot:
            returns.plot(**kwargs)
        self.returns = returns.dropna(how='all')
        return self.returns
    
    def find_beta_alpha(self,market='Close_SPY',plot=False,nrows=1,ncols=1,figsize=(10,5),**kwargs):
        '''This function finds the beta and alpha of an stock in relation to the market.
        INPUTS:
            market (string): is the name of the returns column that acts as the market for the
                analysis. It is the name of a column in the returns attribute of the object.
            plot (boolean): determines if a plot of the information is returned
        
        OUTPUTS:
            alpha_beta (dictionary): dictionary where the key is the stock and the value is 
                a tuple of (alpha, beta) values for each stock
        '''
        market_df = self.returns[market]
        alpha_beta = {}
        stocks = [stock for stock in self.returns.columns.values if stock != market]
        for stock in stocks:
            beta, alpha = np.polyfit(market_df,self.returns[stock],1)
            alpha_beta[stock] = (alpha,beta)
        if plot:
            fig = plt.figure(figsize=figsize)
            axs = {'ax'+str(i+1): fig.add_subplot(nrows,ncols,i+1) for i in range(len(stocks))}
            for i,stock in enumerate(stocks):
                alpha, beta = alpha_beta[stock]
                self.returns_df.plot(kind='scatter',ax=axs['ax'+str(i+1)],
                                    x=market,y=stock,**kwargs)
                axs['ax'+str(i+1)].plot(market_df,market_df*beta+alpha)
                axs['ax'+str(i+1)].text(
                    market_df.min(),
                    self.returns[stock].max(),
                    r'$\beta$ = {}  $\alpha$ = {}'.format(beta,alpha),
                    fontsize=15
                    )
            plt.show()

            

    def get_tickers(self):
        '''This function retreives the ticker attribute from the object instance
        '''
        return self.tickers
    
    def add_ticker(self,ticker):
        '''Adds one ticker to the list of tickers and prepares again the data
        INPUTS:
            ticker (string): name of the ticker added
        OUTPUTS:
            df (pandas DataFrame): dataframe with the information for all tickers
            
        '''
        self.tickers.append(ticker)
        self.prepare_data()

