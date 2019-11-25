# ThymeBoost
Modified Gradient Boosted Trees for the use of Time Series prediction and decomposition
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
