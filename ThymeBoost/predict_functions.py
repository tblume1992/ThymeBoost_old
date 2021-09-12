# -*- coding: utf-8 -*-

import numpy as np


def predict_trend(booster_obj, boosting_round, forecast_horizon):
    """
    Predict the trend component using the booster

    Parameters
    ----------
    boosting_round : int
        The round to reference when getting model params.
    forecast_horizon : int
        Number of periods to forecast.

    Returns
    -------
    trend_round : np.array
        That boosting round's predicted trend component.

    """
    trend_param = booster_obj.trend_pred_params[boosting_round]
    trend_model = booster_obj.trend_objs[boosting_round].model_obj
    trend_round = trend_model.predict(forecast_horizon, trend_param)
    return trend_round


def predict_seasonality(booster_obj, boosting_round, forecast_horizon):
    """
    Predict the seasonality component using the booster.

    Parameters
    ----------
    boosting_round : int
        The round to reference when getting model params.
    forecast_horizon : int
        Number of periods to forecast.

    Returns
    -------
    seas_round : np.array
        That boosting round's predicted seasonal component.

    """
    seas_param = booster_obj.seasonal_pred_params[boosting_round]
    seas_model = booster_obj.seasonal_objs[boosting_round].model_obj
    if seas_model is None:
        seas_round = np.zeros(forecast_horizon)
    else:
        seas_round = seas_model.predict(forecast_horizon, seas_param)
    return seas_round

def predict_exogenous(booster_obj,
                      future_exo,
                      boosting_round,
                      forecast_horizon):
    """
    Predict the exogenous component using the booster.

    Parameters
    ----------
    boosting_round : int
        The round to reference when getting model params.
    forecast_horizon : int
        Number of periods to forecast.

    Returns
    -------
    seas_round : np.array
        That boosting round's predicted seasonal component.

    """
    if future_exo is None:
        exo_round = np.zeros(forecast_horizon)
    else:
        exo_model = booster_obj.exo_objs[boosting_round].model_obj
        exo_round = exo_model.predict(future_exo)
    return exo_round


def predict_rounds(booster_obj,
                   forecast_horizon,
                   future_exo=None):
    """
    Predict all the rounds from a booster

    Parameters
    ----------
    fitted_output : pd.DataFrame
        Output from fit method.
    forecast_horizon : int
        Number of periods to forecast.

    Returns
    -------
    trend_predictions : np.array
        Trend component.
    seasonal_predictions : np.array
        seasonal component.
    predictions : np.array
        Predictions.

    """
    trend_predictions = np.zeros(forecast_horizon)
    seasonal_predictions = np.zeros(forecast_horizon)
    exo_predictions = np.zeros(forecast_horizon)
    for boosting_round in range(booster_obj.i):
        trend_predictions += predict_trend(booster_obj,
                                           boosting_round,
                                           forecast_horizon)
        seasonal_predictions += predict_seasonality(booster_obj,
                                                    boosting_round,
                                                    forecast_horizon)
        exo_predictions += predict_exogenous(booster_obj,
                                             future_exo,
                                             boosting_round,
                                             forecast_horizon)
    predictions = (trend_predictions +
                   seasonal_predictions +
                   exo_predictions)
    return trend_predictions, seasonal_predictions, exo_predictions, predictions
