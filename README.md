# __*Portfolio Management with Deep Learning:* Forecasting returns and managing possitions.__
***
## 1. Introduction:
With the development of deep learning and the increasing disponibility of free high frequency financial data online, the oportunities to implement this techniques to forecast the behaviour and manage assets arise. 

In this project I use machine learning algorithms, in particular *recurrent neural networks*, to forecast the returns of assets of a defined asset universe. With this forecast, I then optimize the portfolio weights and take possitions on each asset, creating the future optimal portfolio. The forecasting algorithm used is a Recurrent Neural Network with LSTM cells, whose structure is:

![](i/21.png)
***

## 2. Data

The data used are the daily stock prices from a subset of stocks which have participated in the S&P500 index since at least january 2007. This data was taken from [Yahoo Finance](https://finance.yahoo.com) through the [`yfinance`](https://pypi.org/project/yfinance/) API. Also I use the [`quandl`](https://github.com/quandl/quandl-python) API for the money market rates, which extracts the data from the [Federal Reserve (FRED)](https://fred.stlouisfed.org).

The data used correspond to a subset of 10 stocks from the S&P500 index, with with information from at 2007-01-04 to 2021-01-04. 

The historical prices and much more are analized before modeling the returns.

![](i/23.png)

A very important file can be found in `financial_data.py` which contains most of the preprocessing code used, as well as plenty of useful functions.
***

## 3. Documentation
Beyond the code, it is very important to understand the underlying financial theory, as well as the architecures of the models used. For this it could be very helpful to take a look to the Udacity course [*Machine Learning for Trading*](https://classroom.udacity.com/courses/ud501) for the finacial basics. 

To go further, it is also very helpful to read [*Machine Learning for Algoritmic Trading*](https://www.packtpub.com/product/machine-learning-for-algorithmic-trading-second-edition/9781839217715).
***

## 4. Tools and Software Used:
Most of the project is composed of two files:
1. A jupyter notebook developed in JupyterLab (v.3.0.11), where the body and explanation of the project is presented.
2. financial_data.py, which contains key classes and functions, which are continually used through out the development of the project.

I also used Google Colab to train the RNN, taking advantage of the GPU service it provides.

The principal libraries used are:
+ [Tensorflow](https://www.tensorflow.org): v. 2.4.1
+ quandl
+ [Pandas](https://pandas.pydata.org): v.1.1.3
+ [Numpy](https://numpy.org): v.1.19.2
+ [Matplotlib](https://matplotlib.org): v.0.23.2
+ [Scikit-Learn](https://scikit-learn.org/stable/): v.3.3.2
+ [Yahoo Finance API](https://pypi.org/project/yfinance): v.0.1.54

***

## 5. License

This software is licensed under the [MIT](https://opensource.org/licenses/MIT) license. The license can be found in the `license.txt` file. 
***

## 6. Acknoledgements

This project was done as part of the [Udacity](udacity.com) Data Science Nano Degree, and with the help of the course [*Machine Learning for Trading*](https://classroom.udacity.com/courses/ud501). 

Also, I would like to thank the free online education platforms [Coursera](https://www.coursera.org), in particular [Deeplearning.ai](https://www.deeplearning.ai) and [Edx](https://www.edx.org), which helped me to get started in this path.