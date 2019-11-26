# ThymeBoost
Modified Gradient Boosted Trees for the use of spicy Time Series prediction and decomposition

Input: pandas series with a DateTime index

Output: Seasonal and trend decomposition along with anomalies and forward looking predictions.  All in a dictionary returned from the fit method.  To graph results nicely, call the plot_results method using the Thyme object.

Example using Bitcoin price vs. Prophet:
```python
import quandl
import fbprophet
import ThymeBoost as tb

data = quandl.get("BITSTAMP/USD")
y = data['High']
y = y[-730:]
df = pd.DataFrame(y)
df['ds'] = y.index
df.columns = ['y', 'ds']
model = fbprophet.Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=20)
forecast = model.predict(future)
model.plot(forecast)
boosted_model = tb.ThymeBoost(freq = 365,
                                regularization = 1.2, 
                                ols_constant = False, 
                                poly = 1, 
                                n_steps = 20,
                                seasonal_smoothing = True)
output = boosted_model.fit(y)
boosted_model.plot_results()
tsboosted_ = output['Full Output']
proph = forecast['yhat']
plt.plot(tsboosted_, label = 'Thyme Boosted', color = 'black')
proph.index = tsboosted_.index
plt.plot(y, label = 'Actual')
plt.plot(proph, label = 'Prophet')
plt.legend()
plt.show()
```
Brief explanation of parameters:

freq: The number of time periods that make up the most significant cycle in your series.  7 for weekly, 365/366 for yearly.

regularization: Penalty for the number of boosting iterations.  1.2 works well.

ols_constant: False enforces connectivity constraints at the splits.  True will probably overfit but will react better to short term shocks.

poly: The degree of the polynomial expansion.  1 means no expansion.  2 or 3 will overfit in most cases but can be used as a smoother.  As you increase this, it will misinterpret seasonality as trend since you allow non-linear trends.   

n_steps: Number of steps forward to extrapolate the trend and seasonality.

seasonal_smoothing: Smooths out the seasonality shape.  Don't use with small freq.

l2: The lambda for l2 regularization.  Pretty sure this breaks the n_steps extrapolation.

max_changepoints: The max number of boosting rounds.

positive:  Whether the output should be contrained to be >= 0.

min_samples: The minumum number of samples to consider a split.  Too low will allow the model to cheat at the beginning and end of the series. I recommend putting in between 1 and 2 times the number of data points in your smallest seasonal cycle. So far daily data your smallest cycle would be weekly, so make min_samples between 7 and 14.  For weekly data your smallest cycle would be a month, so I recommend between 4 and 8 for min_samples.  

