#------------------ Libraries----------------------
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as spo
import tensorflow as tf
mlp.style.use('seaborn-darkgrid')
# sns.set_theme(color_codes=True)

#--------------- Financial Data Class---------------

class FinancialData(object):

    '''This is a class stores information about financial data from ticker or 
    list of tickers, that can be prepared as a data frame and plotted.'''
    
    def __init__(self,tickers=['SPY'],fillna=True,cols=None,  **kwargs):
        """ Initiates the FinancialData object and takes the data from Yahoo 
        Finance from a set of tickers and prepares it in a dataframe.
        
        Inputs:
        -------
        tickers (list, default=['SPY']: list with the tickers of the assets
            analized.
        fillna (Boolean, default=True): determines whether to fill the NaNs in 
            the output dataframe with the financial information. They are filled
            forwards and then backwards.
        cols (list|strin, default=None): the columns of information you 
            want to extract.
        
        **kwargs: complementary arguments for the function yfinance.download(),
            you can see documentation in https://pypi.org/project/yfinance/
        
        Ouputs:
        -------
        None
        """
        # Prepare the data- Extract data from Yahoo Finance:
        if isinstance(tickers,list):
            t = ' '.join(tickers)
            df = yf.download(t, **kwargs)
        
        elif isinstance(tickers,str):
            df = yf.download(tickers,**kwargs)

        if not isinstance(cols,type(None)):
            df = df[cols]

        if fillna:
            df.fillna(method='ffill',inplace=True)
            df.fillna(method='bfill',inplace=True)

        # Rename columns so that they are one level:
        df = one_lvl_colnames(df,cols,tickers)

        # Define attributes:
        self.tickers = tickers
        self.df = df
        self.columns = cols
        
    def plot_data(self,tickers=None, cols='Adj Close',
                  title='Historical Adj Close Price Data',
                  ylabel='Adj Close Prices', 
                  xlabel='Date',
                  fontsize=15,
                  **kwargs):
        '''This method plots the data in the dataframe according to the column
         given assuming that the index of the dataframe is the x axis of the plot.

        Inputs:
        -------
        tickers (list|str, default=None): tickers wich will be plotted
        cols (string): if there are multiple columns still, select one to plot
        title (string, default="Historical Adj Close Price Data"): title for the
            plot.
        ylabel (string, default='Adj Close Prices'): y-label name for the plot.
        xlabel (string, default='Date'): x-label name for the plot.
        **kwargs: define the particularities of the plot, which must be arguments
            for the function DataFrame.plot() of Pandas.
        
        Ouputs:
        ------
        ax (axis): axis of the plot
        '''
        # Retrieve important information:
        df = self.get_data()
        if isinstance(tickers,type(None)):
            tickers = self.get_tickers()

        # Define data column, if multiple data is in the dataset:
        cols = return_names(cols,tickers)
        df = df[cols]

        # Define the axis, and plot the data:
        ax = df.plot(fontsize=fontsize,**kwargs)
        ax.set_title(title,fontsize=fontsize*1.3)
        ax.set_xlabel(xlabel,fontsize=fontsize*1.1)
        ax.set_ylabel(ylabel,fontsize=fontsize*1.1);

        return ax
    
    def rolling_statistics(self,cols='Adj Close',tickers=None,functions=None,
                      window=20,bollinger=False,roll_linewidth=1.5,**kwargs):
        '''This method extracts the rolling statistics from a time series, and
        can plot the rolling window with the data, adding the Bollinger bands.
        
        Inputs:
        -------
        column (string, default=None): the column from which you want compute 
            the rolling function.
        tickers (str|list, default=None): the ticker(s) from which you want to know the 
            information.
        functions (function|string|list): function(s) that will be rolled through the 
            time series.
        window (int): the window of the rolling data
            
        OUTPUTS:
            rolled (pandas series): a series with the rolling statistics specified

        '''
        # Define important varibles:
        df = self.df
        if isinstance(tickers,type(None)):
            tickers = self.get_tickers()
        col_names = return_names(cols,tickers)
        if isinstance(functions,type(None)):
            functions = [momentum, simple_moving_average, bollinger_bands]
        elif not isinstance(functions,list):
            functions = [functions]

        # Define the actual dataframe analized
        df = df[col_names]

        # Compute the rolling statistics:
        rolling_stats = df.rolling(window).agg(functions)

        # Given one level names:
        rolling_stats = one_lvl_colnames(rolling_stats,col_names,functions)

        return rolling_stats
    
    def get_returns(self,cols='Adj Close',tickers=None,return_window=1,plot=False,
                    **kwargs):
        '''This method finds the returns for a set of tickers'  prices
        Inputs:
        -------
        cols (list|string, default='Adj Close'): columns to find the returns.
        tickers (list|string, default=None): tickers from which the returns will be 
            computed.
        return_window (int, default=1): the window from which to get the returns.
        plot (boolean, default=False): determines whether to plot the returns or not.

        Outputs:
        --------
        returns (pandas series or data frame): pandas data structure with the 
            returns.
        '''
        # Define important variables:
        df = self.df
        if isinstance(tickers, type(None)):
            tickers = self.get_tickers()
        col_names = return_names(cols,tickers)
        
        # Compute the returns:
        returns = df[col_names].pct_change(return_window)

        # Plot returns:
        if plot:
            returns.plot(**kwargs)
        
        # Define attributes:
        self.returns = returns.dropna(how='all')
        self.return_window = return_window

        return self.returns
    
    def find_beta_alpha(self,market=None,plot=False,nrows=1,ncols=1,
                        figsize=(10,5), fillna=True,**kwargs):
        '''This method finds the beta and alpha of an stock in relation to the 
        market.

        Inputs:
        -------
        market (pandas Series): series with the returns data from the market portfolio
            asset with which to compute the beta and alpha. 
        plot (boolean, default=False): determines if a plot of the information is returned.
        nrows (int, default=1): the number of rows in the grid of plots.
        ncols (int, default=1): the number of columns in the grid of plots.
        figsize (tuple, default=(10,5)): the figure size of the plot.
        fillna (boolean, default=True): determines whether to fill NaN values after the merge
            with the market portfolio asset. It fills forwards first and then backwards.
        **kwargs: arguments that correspond to the plot.
        
        Outputs:
        --------
        alpha_beta (dictionary): dictionary where the key is the stock and the value is 
            a tuple of (alpha, beta) values for each stock.
        '''
        # Define important variables:
        market = market.to_frame()
        market_name = market.columns[0]
        try:
            returns = self.returns
        except:
            returns = self.get_returns()
        
        # Merge data:
        df = market.merge(returns,left_index=True,right_index=True,how='left')

        # Fill NaNs
        if fillna:
            df.fillna(method='ffill',inplace=True)
            df.fillna(method='bfill',inplace=True)
        
        alpha_beta = {}
        stocks = [stock for stock in df.columns.values if stock != market_name]
        
        # Find the alpha, beta values for each stock in the object:
        for stock in stocks:
            beta, alpha = np.polyfit(df[market_name],df[stock],1)
            alpha_beta[stock] = (alpha,beta)
        if plot:
            fig = plt.figure(figsize=figsize)
            axs = {'ax'+str(i+1): fig.add_subplot(nrows,ncols,i+1) for i in range(len(stocks))}
            for i,stock in enumerate(stocks):
                alpha, beta = alpha_beta[stock]
                df.plot(kind='scatter',ax=axs['ax'+str(i+1)],
                                    x=market_name,y=stock,**kwargs)
                axs['ax'+str(i+1)].plot(df[market_name],df[market_name]*beta+alpha)
                axs['ax'+str(i+1)].text(
                    df[market_name].min(),
                    df[stock].max(),
                    r'$\beta$ = {}  $\alpha$ = {}'.format(round(beta,2),round(alpha,2)),
                    fontsize=15
                    )
            plt.show()

        return alpha_beta
    
    def get_normalized_prices(self, start_date=None,plot=False, prices_col='Adj Close',
                              title=None,x_label='Fecha',y_label='Norm P',
                              fontsize=15,**kwargs):
        """Gives the normalized prices for the assets inside the FinancialData instance

        Inputs:
        -------
        start_data (str, default=None): the date which serves as the normalization point
            for the prices. If None, the method takes the earliest date in the prices
            data.
        plot (boolean, default=False): determines if plot normalized prices data.
        prices_col (string, default='Adj Close'): the name of the prices columns.
        **kwargs: addtitional arguments that go into the Dataframe.plot() method.
        
        Ouputs:
        -------

        norm_prices (pandas Dataframe): dataframe with normalized prices. The start_date 
            prices determine where is the 100 point for each asset.
        """
        # Define important variables:
        if not isinstance(prices_col,list):
            prices_col = [prices_col]
        prices_names = return_names(prices_col,self.get_tickers())
        prices = self.get_data()[prices_names]

        if isinstance(start_date,type(None)):
            start_date = prices.index.min()

        # Compute the normalized prices:
        base = prices.loc[prices.index==start_date].values
        norm_prices = prices/base*100

        # Plot normalized prices
        if plot:
            norm_prices.plot(fontsize=fontsize,**kwargs)
            
            # Plot 100-line:
            plt.hlines(
                y = 100,
                xmin = norm_prices.index.min(),
                xmax = norm_prices.index.max(),
                color = 'black',
                linestyles = 'dashdot')

            # Define title:
            if isinstance(title,type(None)):
                title = 'Precios Normalizados (100 = {}-{}-{})'.\
                        format(start_date.year,start_date.month,
                        start_date.day)
            
            plt.title(title,fontsize=fontsize*1.3)
            plt.ylabel(y_label,fontsize=fontsize*1.1)
            plt.xlabel(x_label,fontsize=fontsize*1.1)

        return norm_prices

    def get_tickers(self):
        '''This method retreives the ticker attribute from the object instance
        '''
        return self.tickers

    def get_data(self):
        """This method retrieves the financial data"""
        return self.df


#---------------------- Portfolio Class--------------------------
class Portfolio(FinancialData):
    """This class contains the information of a portfolio, and inherits
    methods and attributes from the FinancialData class
    """

    def __init__(self,tickers=['SPY'],fillna=True,cols=None,weights=[1],
                 **kwargs):
        FinancialData.__init__(self,tickers,fillna,cols,**kwargs)
        columns = [column+'_'+ticker for ticker in tickers]
        prices = self.prepare_data(fillna=fillna)
        self.prices = prices.loc[:,columns]
        self.weights = weights
    
    def normalize_prices(self,start_date=None,end_date=None,tickers=None,column='Close'):
        '''This method normalizes prices according to the dates provides, slicing the
        information from the start date to the end date
        INPUTS:
            prices (Pandas Data frame): dataframe with the time series of prices
            column (string): the information of the column to be normalized
            start_date (string): the start date, which serves as the normalization
                denominator
            end_date (string): the end date of the period to be analized
        
        OUTPUTS:
            norm_prices (Pandas Data frame): dataframe with the normalized price data
        '''
        prices = self.prices
        if tickers == None:
            tickers = self.get_tickers()
        if start_date is None:
            start_date = prices.index.values.min()
        if end_date is None:
            end_date = prices.index.values.max()
        columns = [column+'_'+ticker for ticker in tickers]
        norm_prices = prices.loc[start_date:end_date,columns]/prices.loc[start_date,columns]
        return norm_prices
    
    def get_portfolio_values(self,start_date=None,end_date=None,tickers=None,column='Close'):
        """This method returns the daily portfolio values
        INPUTS:
            prices (Pandas Data frame): dataframe with the time series of prices
            column (string): the information of the column to be normalized
            start_date (string): the start date, which serves as the normalization
                denominator
            end_date (string): the end date of the period to be analized
            weights (list): list of same length of tickers, with the weight of each 
                asset
            tickers (list): list with the tickers of the portfolio
        
        OUTPUTS:
            portfolio_values (pandas dataframe): dataframe with the daily values of the
                protfolio
        """
        prices = self.get_prices()
        weights = self.get_weights()
        if start_date == None:
            start_date = prices.index.values.min()
        if end_date == None:
            end_date = prices.index.values.max()
        if tickers == None:
            tickers = self.get_tickers()
        norm_prices = self.normalize_prices(start_date,end_date,tickers,column)
        portfolio_values = norm_prices*weights
        portfolio_values['Portfolio'] = portfolio_values.sum(axis=1)
        return portfolio_values

    def get_prices(self):
        """This method returns the prices attribute of the Portfolio instance
        """
        return self.prices
    
    def get_weights(self):
        """This method returns the weights of the Portfolio instance"""
        return self.weights
    
    def change_weights(self,weights):
        """This method changes the weights attribute of the Portfolio instance
        INPUTS:
            weights (list): list with new weights.
        
        OUTPUTS:
            None
        """
        assert len(self.weights) == len(weights), "Wrong length of weights"
        self.weights = weights
    
    def get_returns(self,start_date=None,end_date=None,tickers=None,
                    column='Close',window=1,portfolio_returns=False):
        """This method returns the daily returns of the Portfolio instance
        Inputs:
        -------
        prices (Pandas Data frame): dataframe with the time series of prices
        column (string): the information of the column to be normalized
        start_date (string): the start date, which serves as the normalization
            denominator
        end_date (string): the end date of the period to be analized
        weights (list): list of same length of tickers, with the weight of each 
            asset
        tickers (list): list with the tickers of the portfolio
        window (int): the window of the returns, default daily
        
        Outputs:
        -------
        portfolio_values (pandas dataframe): dataframe with the daily values of the
            protfolio
        """

        # Get the returns of each asset inside the portfolio:
        prices = self.get_prices()
        if start_date is None:
            start_date = prices.index.values.min()
        if end_date is None:
            end_date = prices.index.values.max()
        if tickers is None:
            tickers = self.get_tickers()
        columns = [column+'_'+ticker for ticker in tickers]
        prices = prices.loc[start_date:end_date,columns]
        returns = prices.pct_change(window).dropna(how='all')

        # Add the portfolio returns
        if portfolio_returns:
            weights = self.get_weights()
            returns['Portfolio'] = (returns*weights).sum(axis=1)

        return returns
    
    def get_performance_metrics(self,risk_free_rate=0,start_date=None,
                                end_date=None,**kwargs):
        """This method computes the cumulative return, the average daily 
        return, the risk (the standard deviation) and the Sharpe Ratio of the
        Portfolio instance

        Inputs:
        ------
        risk_free_rate: the risk free rate of the market, which can be a 
            constant or a series of the same length as the returns series
        
        Outputs:
        -------
         metrics (pandas Data Frame): the metrics of the portfolio
        """
        if start_date == None:
            start_date = self.get_prices().index.values.min()
        if end_date == None:
            end_date = self.get_prices().index.values.max()
        portfolio_values = self.get_portfolio_values(start_date,end_date,
                                                    **kwargs)
        def compute_cum_return(series):
            """Computes the cumulative return of a series"""
            mn = series.index.values.min()
            mx = series.index.values.max()
            cum_return = (series[mx]/series[mn])-1
            return cum_return  

        cum_return = compute_cum_return(portfolio_values['Portfolio'])
        returns = self.get_returns(start_date,end_date,
                                   portfolio_returns=True,**kwargs)
        sharpe_ratio = self.get_sharpe_ratio(risk_free_rate,start_date=start_date,
                                             end_date=end_date,**kwargs)
        metrics = returns['Portfolio'].agg(['mean','std'])
        metrics.loc['Cum Return'] = cum_return
        metrics.loc['Sharpe Ratio'] = sharpe_ratio
        return metrics

    def get_sharpe_ratio(self,weights=None,rfr=0,negative=False,**kwargs):
        """Compute the Sharpe Ratio of a portfolio
        Inputs:
        -------
        rfr (numeric value or Pandas series): the risk free rate of 
            the market, which can be a constant value or a series of the same
            length than the returns of the Portfolio, with corresponding dates.
        negative (boolean): True if you want to multiply the Sharpe Ratio by -1
            this is used for optimizing the portfolio. Default False.
        **kwargs: arguments for the get_returns method
        Ouputs:
        -------
        sharpe_ratio (numeric value): the Sharpe Ratio of the portfolio given the
            characteristics of the Portfoliol instance.
        """
        # Get building blocks for the computation:
        returns = self.get_returns(**kwargs)
        if weights is None:
            weights = self.get_weights()

        # Get the portfolio returns:
        portfolio_returns = (returns*weights).sum(axis=1)
        portfolio_std = portfolio_returns.std()

        # Compute Sharpe Ratio formula:
        sharpe_ratio = (portfolio_returns-rfr).mean()/portfolio_std
        if negative:
            sharpe_ratio *= -1
        return sharpe_ratio

    def optimize_portfolio(self,guess_weights=None,short=False,rfr=0,**kwargs):
        """Optimizes the weights of the assets that compose the portfolio, such
        that it maximizes the Sharpe Ratio of the portfolio.
        Inputs:
        -------
        guess_weights (list,tuple, array): an array-like object with the length
            of the number of assets composing the portfolio, which will be used
            to start the optimization process
        rfr (numerical value): risk-free rate of the market, can be a series or
            a constant.
        **kwargs: arguments to be used in the get_returns method

        Outputs:
        --------
        opt_weights (array-like): array-like object with the weights that maximize
            the Sharpe Ratio of the portfolio.
        """
        tickers = self.get_tickers()

        if guess_weights is None:
            guess_weights = [1/len(tickers) for i in range(len(tickers))]
        
        # Determine the bounds of the optimized weights (min=0, max=1):
        if not short:
            bounds = [(0,1) for i in range(len(tickers))]
        else:
            bounds = [(-1,1) for i in range(len(tickers))]

        # Determine the restrictions:
        weights_sum_to_1 = {'type':'eq',
                            'fun':lambda weights: np.sum(np.absolute( weights))-1}
        
        # Optimize:
        opt_weights = spo.minimize(
            self.get_sharpe_ratio,guess_weights,
            args=(rfr,True),
            method='SLSQP', options={'disp':False},
            constraints=(weights_sum_to_1),
            bounds=bounds
        )

        # Update weights to optimized weights
        print(len(opt_weights.x))
        self.change_weights(opt_weights.x)

        return opt_weights.x

#-----------------------------WindowGenerator Class-------------------------------------
# This class was done using the guidance of the TensorFlow tutorial found in:
# https://www.tensorflow.org/tutorials/structured_data/time_series

class WindowGenerator():
    """This class takes time series data that is in a sequential format, transforming
    it into pairs of inputs and labels, so that the inputs are windows of consecutive
    samples from the data.
    """
    def __init__(self,input_width=5,label_width=1,shift=1, train_df=None, val_df=None,
                 test_df=None, label_columns=None,batch_size=None,shuffle=False):
        """This method initiates the WindowGenerator class.

        Inputs:
        -------
        input_width (int, default=5): the width of the window, which represents the 
            amount of time steps from the earliest input observation to the last.
        label_width (int, default=1): the width of the label. This determines the amount
             of time steps that will be predicted.
        shift (int, default=1): jump between the last input in the window and the first 
            label.
        train_df (pandas Dataframe, default=None): array-like object containing the train 
            data which comes in a time series format.
        val_df (pandas Dataframe, default=None): array-like object containing the 
            validation data.
        test_df (pandas Dataframe, default=None): array-like object containing the test 
            data.
        label_columns (list|string, default=None): name of the column(s) that are used 
            as labels.
        batch_size (int, deafault=None): the size of the batches of the tf.data.Dataset
            object (whose dimensions are (batch,input_width,features) for the input and
            (batch,label_width,label_columns) for the labels).
        shuffle (boolean, default=False): determines if the data inside the tf.data.Dataset
            is shuffled.
        
        Outputs:
        --------
        None
        """
        # Define attributes of the class:
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Define information about columns:
        if isinstance(label_columns,type(None)):
            self.label_columns_indices = {name:i for i,name in enumerate(label_columns)}
        self.column_indices = {name:i for i,name in enumerate(train_df.columns)}

        # Define window information:
        self.total_window_size = input_width+shift
        self.input_slice = slice(0,input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size-self.label_width
        self.labels_slice = slice(self.label_start,None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def __repr__(self):
        """This method determines what is returned when an instance of the object
        is called
        """
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])
    
    def split_window(self, features):
        """This method converts a list of consecutive inputs to a window of
        inputs and a window of labels.

        Inputs:
        -------
        features (pandas Dataset): features in the dataframe

        Outputs:
        --------
        inputs ()
        """
        inputs = features[:, self.input_slice,:]
        labels = features[:,self.labels_slice,:]
        if not isinstance(self.label_columns,type(None)):
            labels = tf.stack(
                [labels[:,:,self.column_indices[name]] for name in self.label_columns],
                axis = -1
            )
        
        # Set the shapes of the informaiton:
        inputs.set_shape([None,self.input_width,None])
        labels.set_shape([None,self.label_width,None])

        return inputs,labels
    
    def make_dataset(self,data):
        """This method takes a time series DataFrame and convert it to a 
        tf.data.Dataset of (input_window,label_window) pairs, using the
        tf.keras.preprocessing.timeseries_dataset_from_array function.

        Input:
        ------
        data (pandas DataFrame): dataframe containing the time series information
            of the inputs and labels, which will transformed into windows and then 
            a tf.Dataset object.
        
        Outputs:
        --------

        """
        data = np.array(data,dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data = data,
            targets = None,
            sequence_length = self.total_window_size,
            sequence_stride = 1,
            shuffle = self.shuffle,
            batch_size = self.batch_size
        )
        ds = ds.map(self.split_window)

        return ds

    # Adding properties for accessing the train, val and test as tf.data.Dataset objects
    @property
    def train(self):
        if isinstance(self.train_df,type(None)):
            return None
        else:
            return self.make_dataset(self.train_df)

    @property
    def val(self):
        if isinstance(self.val_df,type(None)):
            return None
        else:
            return self.make_dataset(self.val_df)

    @property
    def test(self):
        if isinstance(self.test_df,type(None)):
            return None
        else:
            return self.make_dataset(self.test_df)
    
#---------------------------------------------------------------------------------------
# 3. Complementary functions:

def one_lvl_colnames(df,cols,tickers):
    """This function changes a multi-level column indexation into a one level
    column indexation

    Inputs:
    -------
    df (pandas Dataframe): dataframe with the columns whose indexation will be 
        flattened.
    tickers (list|string): list/string with the tickers (s) in the data frame df.
    cols (list|string): list/string with the name of the columns (e.g. 'Adj Close',
        'High', 'Close', etc.) that are in the dataframe df.
    
    Ouputs:
    -------
    df (pandas Dataframe): dataframe with the same information as df, but 
        with one level of indexation.
    """
    # Define important variables:
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(cols, str):
        cols = [cols]

    # For multi-level column indexing:
    if isinstance(df.columns.values[0], tuple):

        # Define important varibles
        columns = df.columns.values
        new_cols = []

        # Itarate through the multi-level column names and flatten them:
        for col in columns:
            temp = []
            for name in col:
                if name != '':
                    temp.append(name)
            new_temp = '_'.join(temp)
            new_cols.append(new_temp)
        
        # Change the column names:
        df.columns = new_cols
    
    # For uni-level colum indexing:
    elif isinstance(df.columns.values[0], str):
        
        # Define new names:
        col_names = [column+'_'+ticker for column in cols\
                     for ticker in tickers]
        df.columns = col_names

    return df

def return_names(cols,tickers):
    """This function takes ticker(s) and column(s) and defines the names
    of the combination of both.

    Inputs:
    -------
    cols (list|string): column names (e.g. 'Adj Close', 'Close', 'High').
    tickers (list|string): ticker names.
    
    Outputs:
    --------
    col_names (list): names of the columns for the given cols and tickers
    """
    # Give the correct type:
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(tickers, str):
        tickers = [tickers]

    col_names = [col+'_'+ticker for col in cols for ticker in tickers]

    return col_names

def momentum(prices):
    """This function finds the momentum metric for a group of assets.
    Inputs:
    -------
    prices (pandas dataframe|series): dataframe with the information of the
        prices of a group of assets or an indivitual asset.
    
    Outputs:
    --------
    momentum_df (pandas Dataframe): dataframe with the information of the
        momentum.
    """
    # Compute components:
    first = prices.iloc[0]
    last = prices.iloc[-1]

    # Compute momentum:
    momentum_df = last/first

    return momentum_df

def simple_moving_average(prices):
    """This function computes the simple moving average for a set of prices
    given a time window.

    Inputs:
    -------
    prices (pandas Dataframe|Series): dataframe with the information of the
        prices of the analyzed assets.

    Outputs:
    --------
    sma (pandas Dataframe): dataframe of the SMA of the assets' prices in
        the prices dataframe.
    """
    # Compute the SMA of the assets:
    mean = prices.mean()
    sma = prices[-1]/mean-1

    return sma
    
def bollinger_bands(prices):
    """This function computes the bollinger bands values for a set of assets
    returned in a pandas dataframe.

    Inputs:
    -------
    prices (pandas Dataframe|Series): dataframe with the price infromation
        of the assets analyzed.
    window (numeric value, default=None): determines if compute the BBs in
        a rolling manner, or if compute them to the whole input of prices.
    
    Outputs:
    --------
    bb (pandas dataframe): dataframe with the information of the bollinger
        bands of the assets in the prices dataframe.
    """
    # Compute components:
    ma = prices.mean()
    std = prices.std()

    # Compute bollinger bands:
    bb = (prices[-1]-ma)/(2*std)

    return bb

def plot_window(window_dataset,pandas_dataset,window_size,model,
                figsize=(12,100)):
    """This function plots the observed returns inside a tf.data.Dataset and
    compares them with the predicted returns of a model.

    Inputs:
    -------
    window_dataset (tensorflow.data.Dataset object): dataset with both inputs
        and target values.
    pandas_dataset (pandas DataFrame): dataframe that was used to create the
        window_dataset.
    window_size (int): window size used in the transformation from sequential 
        time series into window time series.
    model (model object): trained model with the capability to predict in a
        similar manner than tensorflow.keras.Model object.
    figsize (tuple, default=(12,100)): tuple with the dimensions of the figure
        where the data will be plotted.
    
    Ouputs:
    -------
    None
    """
    # Determine the X-axis of the plot:
    plot_index = pandas_dataset.iloc[window_size:,:].index

    # Assign in the addecuate format the values of the observed taget variable(s):
    y = np.concatenate([targets for inputs,targets in window_dataset],axis=0)
    
    # Use the model to predict the target variable(s):
    y_hat = model.predict(window_dataset)

    # Adjust the shapes:
    y = y.reshape(y_hat.shape)

    # Plot the data:
    plt.figure(figsize=figsize)
    for n in range(y_hat.shape[1]):
        plt.subplot(y_hat.shape[1],1,n+1)
        plt.ylabel('Return')
        plt.plot(plot_index,y_hat[:,n],label='Predicted',color='maroon')
        plt.plot(plot_index,y[:,n],label='Observed',color='midnightblue',alpha=0.5)
    plt.legend()

def daily_rate(x, periods_year=252):
    """This function transforms a rate into a daily rate

    Inputs:
    -------
    x (numerical value): rate that you want to transform into a daily rate.
    periods_year (numerical value, default=252): amount of periods per year
        of the periodicity of rate x.
    
    Ouputs:
    -------
    df (numerical value): daily rate
    """
    dr = np.power(1+x,1/periods_year)-1
    return dr

def optimize_portfolio(returns,guess_weights=None,short=True,rfr=0):
    """This function optimizes the weight allocation for the assets in a
    portfolio, represented by the returns.

    Inputs:
    -------
    returns (pandas DataFrame|Series): contains the returns information.
    guess_weights (list of numerical values, default=None): guess values for
        the weights of the different assets in the portfolio.
    short (boolena, default=True): define ifshort possitions are allowed or 
        not.
    rfr (numerical value, default=0): risk free rate, could be a series of
        the same length as returns.

    Outputs:
    --------
    opt_weights (array-like object): array with the optimal weights for the
        portfolio.
    """
    # Define important variables:
    num_assets = returns.shape[1]
    if isinstance(guess_weights,type(None)):
        guess_weights = [1/num_assets for i in range(num_assets)]

    # Define bound if short possitions are allowed or not:
    if not short:
        bounds = [(0,1) for i in range(num_assets)]
    else:
        bounds = [(-1,1) for i in range(num_assets)]

    # Define constraints, if there can or not be leverage
    weights_sum_to_1 = {'type':'eq',
                        'fun':lambda weights: np.sum(np.absolute(weights))-1}
    
    # Minimize the function:
    opt_weights = spo.minimize(
        sharpe_ratio,
        guess_weights,
        args = (rfr, True, returns),
        method = 'SLSQP',
        options = {'disp':False},
        constraints = (weights_sum_to_1),
        bounds = bounds
    )

    return opt_weights 

def sharpe_ratio(weights=None, rfr=0, negative=False, returns=0):
    """Compute the Sharpe Ratio of a portfolio.

    Inputs:
    -------
    weights (list of numerical values, default=None): list with the weights
        of the assets in the portfolio.
    rfr (numerical value|array-like, default=0): risk-free rate.
    returns (pandas DataFrame|Series, default=0): returns of the assets in
        the portfolio.

    Outputs:
    --------
    sharpe_ratio (numerical value): Sharpe ratio of the portfolio.
    """
    # Define important variables:
    num_assets = returns.shape[1]
    if isinstance(weights,type(None)):
        weights = [1/num_assets for i in range(num_assets)]

    # Get portfolio returns:
    portfolio_returns = (returns*weights).sum(axis=1)
    portfolio_std = portfolio_returns.std()

    # Compute Sharpe Ratio formula:
    sharpe_ratio = (portfolio_returns-rfr).mean()/portfolio_std

    # If used in a minization process:
    if negative:
        sharpe_ratio *= -1

    return sharpe_ratio    