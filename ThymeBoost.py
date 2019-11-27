# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 08:57:28 2019

@author: Tyler Blume
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.filters.filtertools import convolution_filter
import pandas as pd 
import datetime
from sklearn.preprocessing import PolynomialFeatures
import scipy

sns.set_style("darkgrid")

class ThymeBoost():
  def __init__(self, 
               max_changepoints = 50, 
               freq = 7, 
               min_samples = 5, 
               estimators = 'ols', 
               regularization = 1, 
               ols_constant = False, 
               poly = 1, 
               positive = False, 
               l2 = 0, 
               n_steps = 1,
               seasonal_smoothing = False,
               additive = True,
               nested_seasonality = False,
               approximate_splits = False):
    
    self.bic = 10**100
    self.estimators = estimators
    self.changepoints = []
    self.max_changepoints = max_changepoints
    self.min_samples = min_samples
    self.regularization = regularization
    self.ols_constant = ols_constant
    self.poly = poly
    self.positive = positive
    self.l2 = l2
    self.freq = freq
    self.n_steps = n_steps
    self.seasonal_smoothing = seasonal_smoothing
    self.additive = additive   
    self.nested_seasonality = nested_seasonality
    self.approximate_splits = approximate_splits

    if self.freq < 30:
      self.seasonal_smoothing = False
    if estimators != 'ols':
        self.poly = 1
    
  def ridge(self, y, last_pred, ols_constant = False):
    if len(y) == 1:
      predicted = np.array([0])
    else:
      y = np.array(y - last_pred, ndmin=1).reshape((len(y), 1)) 
      X = np.array(list(range(len(y))), ndmin=1).reshape((len(y), 1))
      X = PolynomialFeatures(degree = self.poly, include_bias = False).fit(X).transform(X)    
      if ols_constant:
        X = np.append(X, np.asarray(np.ones(len(y))).reshape(len(y), 1), axis = 1)
      I = np.eye(X.shape[1])
      ThymeBoost.coef = np.linalg.inv(X.T.dot(X)+ self.l2*I).dot(X.T.dot(y))
      predicted = np.resize((X.dot(self.coef)), (len(y),)) + last_pred
    return predicted

  def seasonal(self, resids):
    if self.nested_seasonality:
      seasonality = np.array([np.mean(resids[i::self.freq], axis=0) for i in range(self.freq)])
      if self.additive:
        self.daily_seasonality = np.array([np.mean(resids.add(np.tile(seasonality,1000)[:len(resids)])[i::7], axis=0) for i in range(7)])
      else:
        self.daily_seasonality = np.array([np.mean(resids.multiply(np.tile(seasonality,1000)[:len(resids)])[i::7], axis=0) for i in range(7)])
        self.daily_seasonality = self.daily_seasonality/np.mean(self.daily_seasonality)
      self.daily_seasonality = pd.Series(np.tile(self.daily_seasonality, 1000)[:len(self.x) + self.n_steps])

    else:
      seasonality = np.array([np.mean(resids[i::self.freq], axis=0) for i in range(self.freq)])  
    if self.seasonal_smoothing:
      b, a = scipy.signal.butter(8, 0.125)
      if self.additive:
        seasonality = scipy.signal.filtfilt(b,a,seasonality)
        #seasonality = self.smooth(seasonality)
      else:
        seasonality = scipy.signal.filtfilt(b,a,seasonality)/np.mean(seasonality)
    return seasonality
            
  def calc_bic(self, N, trend, c):
    return N * np.log(np.sum((self.boosted_data - trend )**2)/len(self.x)) + ((c)**self.regularization) * np.log(N)

  def smooth(self, x,window_len=11,window='bartlett'):
      s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
      #print(len(s))
      if window == 'flat': #moving average
          w=np.ones(window_len,'d')
      else:
          w=eval('np.'+window+'(window_len)')
  
      y=np.convolve(w/w.sum(),s,mode='valid')
      return y
   
  def get_approximate_split_proposals(self):
    gradient = pd.Series(np.abs(np.gradient(self.boosted_data, edge_order=1)))
    gradient = gradient.sort_values(ascending = False)[:100]
    proposals = list(gradient.index)
    return proposals
    
    
  def get_split_proposals(self):
    if self.approximate_splits:
      return self.get_approximate_split_proposals()
    else:
      return range(1, len(self.boosted_data) - 2)
  
  
  def get_trends(self):
    N = len(self.x)
    self.boosted_data = self.x
    trends_found = np.zeros(len(self.boosted_data))
    bics = []
    for c in range(0, self.max_changepoints):
      last_round_trends = 0
      round_changepoint = []
      split_proposals = self.get_split_proposals()
      if c == 0:
        trend = self.ridge(self.boosted_data, np.mean(self.boosted_data))
        found_bic = self.calc_bic(N, trend, c)
        if self.bic > found_bic:
          self.bic = found_bic
          round_changepoint.append(0)
          found_trends = trend + last_round_trends
      else:
        for i in split_proposals:
          split1_len = len(self.x[:i + 1])
          split2_len = N - split1_len        
          if split1_len < self.min_samples or split2_len < self.min_samples:
            pass            
          else:
            split1_len = 0
            split2_len = 0
            predi1 = self.ridge(self.boosted_data[:i + 1],
                              np.mean(self.boosted_data[:i + 1]),
                              ols_constant = False
                              )
            predi2 = self.ridge(self.boosted_data[i + 1:],
                              predi1[-1],
                              ols_constant = self.ols_constant
                              )
            trend = np.hstack([predi1, predi2])
            found_bic = self.calc_bic(N, trend, c)
            if self.bic > found_bic:
              self.bic = found_bic
              round_changepoint.append(i)
              found_trends = trend + last_round_trends
      bics.append(self.bic)
      self.bics = bics
      try:
        if self.changepoints:
          self.changepoints.append(max(round_changepoint))
          self.boosted_data = self.boosted_data - found_trends
        else:
          self.changepoints.append(0)
          self.boosted_data = self.boosted_data - found_trends
      except:
        break
      trends_found = trends_found + found_trends      
    self.trends_found = trends_found
  
  def get_anomalies(self):
      predicted = pd.DataFrame(np.column_stack([self.x, self.full_changepoint[:len(self.x)]]))
      all_points = [0] + self.changepoints + [len(predicted)]
      all_points.sort()
      predicted['Mean Error'] = 1
      predicted['Standard Deviations'] = 1
      for i in range(len(all_points) - 1):
        refined_df = predicted.iloc[all_points[i]: all_points[i+1], :]
        if i == 0:
          predicted.iloc[all_points[i]: all_points[i+1] + 1, 2] = np.mean(np.abs(refined_df.iloc[:, 0] - refined_df.iloc[:, 1]))
          predicted.iloc[all_points[i]: all_points[i+1] + 1, 3] = np.std(np.abs(refined_df.iloc[:, 0] - refined_df.iloc[:, 1]))
        else:
          predicted.iloc[all_points[i] + 1: all_points[i + 1] + 1, 2] = np.mean(np.abs(refined_df.iloc[:, 0] - refined_df.iloc[:, 1]))
          predicted.iloc[all_points[i] + 1: all_points[i + 1] + 1, 3] = np.std(np.abs(refined_df.iloc[:, 0] - refined_df.iloc[:, 1]))   
      predicted['Abs Error'] =  np.abs(predicted.iloc[:, 0] - predicted.iloc[:, 1])
      anomalies = predicted[predicted['Abs Error']  > predicted['Mean Error'] + 2*predicted['Standard Deviations']]
      anomalies = pd.concat([anomalies, predicted[predicted['Abs Error']  < predicted['Mean Error'] - 2*predicted['Standard Deviations']]])
      self.standard_deviations = predicted['Standard Deviations']
      self.standard_deviations.iloc[0] = self.standard_deviations.iloc[1]
      self.standard_deviations = self.standard_deviations.append(pd.Series(np.tile(self.standard_deviations.iloc[-1], self.n_steps)))
      self.standard_deviations.index = self.full_changepoint.index
      return anomalies
    
  def extrapolate(self):
      if self.changepoints:
        last_segment = self.trends_found[np.max(self.changepoints) + 1:]
      else:
        last_segment = self.trends_found
      y = last_segment - last_segment[0]
      X = np.array(list(range(len(last_segment))), ndmin=1).reshape((len(last_segment), 1))
      X_future = np.array(list(range(len(X), len(X) + self.n_steps + 1)), ndmin=1).reshape((self.n_steps + 1, 1))
      X = PolynomialFeatures(degree = self.poly, include_bias = False).fit(X).transform(X)  
      X_future = PolynomialFeatures(degree = self.poly, include_bias = False).fit(X_future).transform(X_future)  
      I = np.eye(X.shape[1])
      last_segment_coefs = np.linalg.inv(X.T.dot(X)+ self.l2*I).dot(X.T.dot(y))
      season_extrap = self.full_seasonal[len(self.x):len(self.x) + self.n_steps + 1]
      extrapolation = pd.Series(np.resize(pd.Series(X_future.dot(last_segment_coefs)) + last_segment[0], (len(X_future),)), index = pd.date_range(self.x.index.values[-1], periods=self.n_steps + 1))
      extrapolation = extrapolation[1:]

      if self.positive:
          extrapolation[extrapolation < 0] = 0
      return extrapolation
    
  def fit(self, x):
      self.x = x
      self.get_trends()
      if self.additive:
        self.seasonality = pd.Series(self.seasonal(self.boosted_data))
      else:
        self.seasonality = pd.Series(self.seasonal(self.x/self.trends_found))
        
      if self.nested_seasonality:
              day_of_week = [pd.to_datetime(str(i)).strftime("%A") for i in pd.Series(self.x).index.values[:self.freq]]
              self.seasonality.index = self.x.index[:self.freq]
              self.graph_daily_seasonality = self.daily_seasonality[:7]
              self.graph_daily_seasonality.index = day_of_week[:7]

      else:
        try:
          if self.freq ==  7:
              day_of_week = [pd.to_datetime(str(i)).strftime("%A") for i in pd.Series(self.boosted_data).index.values[:self.freq]]
              self.seasonality.index = day_of_week
          else:
              day_of_week = [pd.to_datetime(str(i)) for i in pd.Series(self.boosted_data).index.values[:self.freq]]
              self.seasonality.index = day_of_week
        except:
          pass
                
      self.full_seasonal = np.tile(self.seasonality, 1000)
      self.trends = pd.Series(self.trends_found, index = pd.Series(self.x).index.values).append(self.extrapolate())
      if self.nested_seasonality:
        if self.additive:
          self.full_changepoint = self.trends.add(self.full_seasonal[:len(self.trends)])
          d_seasonality = pd.Series(self.daily_seasonality[:len(self.trends)])
          d_seasonality.index = self.full_changepoint.index
          self.full_changepoint = self.full_changepoint.add(d_seasonality)
          
        else:
          self.full_changepoint = self.trends.multiply(self.full_seasonal[:len(self.trends)])
          d_seasonality = pd.Series(self.daily_seasonality[:len(self.trends)])
          d_seasonality.index = self.full_changepoint.index
          self.full_changepoint = self.full_changepoint.multiply(d_seasonality)
      else:
        if self.additive:
          self.full_changepoint = self.trends.add(self.full_seasonal[:len(self.trends)])
        else:
          self.full_changepoint = self.trends.multiply(self.full_seasonal[:len(self.trends)])
      
        
      if self.positive:
        self.full_changepoint[self.full_changepoint < 0] = 0
      self.full_changepoint.index = self.trends.index
      anomalies = self.get_anomalies()
      anomalies = pd.DataFrame(anomalies)
      anomalies.index = self.x.index.values[anomalies.index.values]
      self.anomalies = anomalies.iloc[:, 0]
      self.changepoints = list(set(self.changepoints))
      self.changepoints.sort()
      residuals = self.x.subtract(self.full_changepoint[:len(self.x)])
      filt = np.repeat(1./int(len(residuals)/10), int(len(residuals)/10))
      self.trend = convolution_filter(residuals, filt, 2)
      ThymeBoost.bic = self.bics[-1]
      self.residuals = residuals
      fitted = self.full_changepoint[:len(self.x)]
      predicted = self.full_changepoint[len(self.x):]
      output_dict = {'Changepoints': self.changepoints,
                     'Fitted': fitted,
                     'Predicted': predicted,
                     'Full Output': self.full_changepoint,
                     'Anomalies': anomalies,
                     'Seasonality': self.seasonality,   
                     'Trend': self.trends
          }
      
      return output_dict
    
  def plot_results(self):
    if self.nested_seasonality:
      fig, axarr = plt.subplots(4, 1)    
    else:
      fig, axarr = plt.subplots(3, 1)    
    fig.set_figheight(10)
    fig.set_figwidth(12)
    plt.tight_layout()
    plt.sca(axarr[0])     
    axarr[0].set_title('TsBoost Output')
    plt.plot(self.x)
    plt.plot(self.trends)
    plt.plot(self.full_changepoint, linestyle = 'dashed', color = 'black', alpha = .7)
    plt.scatter(self.anomalies.index.values, self.anomalies , marker='o', color="red")
    plt.fill_between(self.full_changepoint.index, 
                     self.full_changepoint.add(self.standard_deviations*2)[:len(self.full_changepoint)], 
                     self.full_changepoint.subtract(self.standard_deviations*2)[:len(self.full_changepoint)],
                     alpha = .2, color = 'black'
                     )
    if self.changepoints:
      for change in self.changepoints:
        plt.axvline(x=list(pd.Series(self.x).index.values)[change], color='r', linestyle='--')
      plt.axvline(x=list(pd.Series(self.x).index.values)[-1], color='black', linestyle='--')
      plt.axvspan(list(pd.Series(self.x).index.values)[-1], list(self.full_changepoint.index.values)[-1], alpha = .2, color = 'black')
    plt.sca(axarr[1]) 
    axarr[1].set_title('Seasonality')
    plt.plot(self.seasonality)
    plt.sca(axarr[2]) 
    if self.nested_seasonality:
      axarr[2].set_title('Weekly Seasonality')
      plt.plot(self.graph_daily_seasonality[:7]) 
      plt.sca(axarr[3])        
      axarr[3].set_title('Residuals')
      plt.plot(self.residuals)
      plt.plot(self.trend, color = 'black')    
      plt.show()    
    else:
      axarr[2].set_title('Residuals')
      plt.plot(self.residuals)
      plt.plot(self.trend, color = 'black')    
      plt.show()

    return
    


  

