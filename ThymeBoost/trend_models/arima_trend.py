# -*- coding: utf-8 -*-
import numpy as np
import statsmodels as sm
from ThymeBoost.trend_models.trend_base_class import TrendBaseModel


class ArimaModel(TrendBaseModel):
    """ARIMA Model from Statsmodels"""
    model = 'arima'

    def __init__(self):
        self.model_params = None
        self.fitted = None

    def __str__(self):
        return f'{self.model}({self.kwargs["arima_order"]})'

    def fit(self, y, **kwargs):
        """
        Fit the trend component in the boosting loop for a ewm model using alpha.

        Parameters
        ----------
        time_series : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.kwargs = kwargs
        order = kwargs['arima_order']
        bias = kwargs['bias']
        ar_model = sm.tsa.arima.model.ARIMA(y - bias, order=order).fit()
        self.fitted = ar_model.predict(start=0, end=len(y) - 1) + bias
        self.model_params = (ar_model, bias, len(y))
        return self.fitted

    def predict(self, forecast_horizon, model_params):
        last_point = model_params[2] + forecast_horizon
        prediction = model_params[0].predict(start=model_params[2] + 1, end=last_point) + \
                     model_params[1]
        return prediction
