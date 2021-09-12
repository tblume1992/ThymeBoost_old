# -*- coding: utf-8 -*-
r"""
ThymeBoost combines time series decomposition with gradient boosting to 
provide a flexible mix-and-match time series framework. At the most granular 
level are the trend/level (going forward this is just referred to as 'trend') 
models, seasonal models,  and edogenous models. These are used to approximate 
the respective components at each 'boosting round' and concurrent rounds are 
fit on residuals in usual boosting fashion.

The boosting procedure is heavily influenced by traditional boosting theory [1]_ 
where the initial round's trend estimation is a simple median, although this 
can be changed to other similarly 'weak' trend estimation methods. Each round 
involves approximating each component and passing the 'psuedo' residuals to 
the next boosting round.

Gradient boosting allows us to use a single procedure to mix-and-match different 
types of models. A common question when decomposing time series is the order of 
decomposition. Some methods require approximating trend after seasonality or vice 
versa due to a underlying model eating too much of the other's component. This can 
be overcome by using the respective component's learning rate parameter to penalize 
it at each round.

References
----------
.. [1] Jerome H. Friedman. 2002. Stochastic gradient boosting. Comput. Stat. Data Anal. 38, 4
   (28 February 2002), 367–378. DOI:https://doi.org/10.1016/S0167-9473(01)00065-2

"""

from itertools import cycle
import warnings
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from ThymeBoost.fitter.booster import booster
from ThymeBoost.optimizer import Optimizer
from ThymeBoost.ensemble import Ensemble
from ThymeBoost.utils import plotting
from ThymeBoost.utils import build_output
from ThymeBoost.cost_functions import calc_cost
from ThymeBoost.predict_functions import predict_rounds
warnings.filterwarnings("ignore")


class ThymeBoost:
    """
    ThymeBoost main class which wraps around the optimizer and ensemble classes.

    Parameters
    ----------
    verbose : bool, optional
        Truey/Falsey for printing some info, logging TODO. The default is 0.
    n_split_proposals : int, optional
        Number of splits to propose for changepoints. If fit_type in the fit 
        method is 'global' then this parameter is ignored. The default is 10.
    approximate_splits : bool, optional
        Whether to reduce the search space of changepoints. If False then the 
        booster will exhaustively try each point as a changepoint. If fit_type 
        in the fit method is 'global' then this parameter is ignored.
        The default is True.
    exclude_splits : list, optional
        List of indices to exclude when searching for changepoints. The default is None.
    given_splits : list, optional
        List of indices to use when searching for changepoints. The default is None.
    cost_penalty : float, optional
        A penalty which is applied at each boosting round. This keeps the 
        booster from making too many miniscule improvements. 
        The default is .001.
    normalize_seasonality : bool, optional
        Whether to enforce seasonality average to be 0 for add or 1 for mult. 
        The default is True.
    additive: bool, optional
        FIXME
        Whether the whole process is additive or multiplicative. Definitely 
        unstable in certain parameter combinations. User beware! 
    regularization : float, optional
        A parameter which further penalizes the global cost at each boosting round. 
        The default is 1.2.
    n_rounds : int, optional
        A set number of boosting rounds until termination. If not set then 
        boosting terminates when the current round does not improve enough over 
        the previous round. The default is None.
    smoothed_trend : bool, optional
        FIXME
        Whether to apply some smoothing to the trend compoennt. 
        The default is False.
    scale_type : str, optional
        FIXME
        The type of scaling to apply. Options are ['standard', 'log'] for classic
        standardization or taking the log. The default is None.

    """
    __framework__ = 'main'
    version = '0.1.0'
    author 'Tyler Blume'

    def __init__(self,
                 verbose=0,
                 n_split_proposals=10,
                 approximate_splits=True,
                 exclude_splits=None,
                 given_splits=None,
                 cost_penalty=.001,
                 normalize_seasonality=True,
                 additive=True,
                 regularization=1.2,
                 n_rounds=None,
                 smoothed_trend=False,
                 scale_type=None):
        self.verbose = verbose
        self.n_split_proposals = n_split_proposals
        self.approximate_splits = approximate_splits
        self.additive = additive
        if exclude_splits is None:
            exclude_splits = []
        self.exclude_splits = exclude_splits
        if given_splits is None:
            given_splits = []
        self.given_splits = given_splits
        self.cost_penalty = cost_penalty
        self.scale_type = scale_type
        if not additive:
            self.scale_type = 'log'
        self.normalize_seasonality = normalize_seasonality
        self.regularization = regularization
        if n_rounds is None:
            n_rounds = -1
        self.n_rounds = n_rounds
        self.smoothed_trend = smoothed_trend
        self.trend_cap_target = None
        self.ensemble_boosters = None

    def scale_input(self, time_series):
        """
        Simple scaler method to scale and unscale the time series.  
        Used if 'additive' is False.

        Parameters
        ----------
        time_series : pd.Series
            The time series to scale.

        Raises
        ------
        ValueError
            Thrown if unsupport scale type is provided.

        Returns
        -------
        time_series : pd.Series
            Scale time series.

        """
        #FIXME
        #seems unstable with different combinations
        if self.scale_type == 'standard':
            self.time_series_mean = time_series.mean()
            self.time_series_std = time_series.std()
            time_series = (time_series - self.time_series_mean) / \
                           self.time_series_std
        elif self.scale_type == 'log':
            assert time_series.all(), 'Series can not contain 0 for mult. fit or log scaling'
            assert (time_series > 0).all(), 'Series can not contain neg. values for mult. fit or log scaling'
            time_series = np.log(time_series)
        elif self.scale_type is None:
            pass
        else:
            raise ValueError('Scaler not recognized!')
        return time_series

    def unscale_input(self, scaled_series):
        """
        Unscale the time series to return it to the OG scale.

        Parameters
        ----------
        scaled_series : pd.Series
            The previously scaled sereis that needs to be rescaled before returning.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.scale_type == 'standard':
            return scaled_series * self.time_series_std + self.time_series_mean
        elif self.scale_type == 'log':
            return np.exp(scaled_series)
        else:
            return scaled_series

    @staticmethod
    def create_iterated_features(variable):
        """
        This function creates the 'generator' features which are cycled through each boosting round.

        Parameters
        ----------
        variable : TYPE
            The variable to convert into a 'generator' feature.

        Returns
        -------
        variable : it.cycle
            The 'generator' to cycle through.

        """
        if not isinstance(variable, list):
            variable = [variable]
        variable = cycle(variable)
        return variable

    @staticmethod
    def combine(param_list: list):
        """
        A function used to denote ensembled parameters for optimization

        Parameters
        ----------
        param_list : list
            list of param values to ensemble.

        Returns
        -------
        funcion
            Returns a function to call and pass to ensembling.

        """
        def combiner():
            return param_list
        return combiner

    def fit(self,
            time_series,
            seasonal_period=0,
            trend_estimator='mean',
            seasonal_estimator='fourier',
            exogenous_estimator='ols',
            l2=None,
            poly=1,
            arima_order=(1, 0, 1),
            connectivity_constraint=True,
            fourier_order=10,
            fit_type='local',
            window_size=3,
            trend_weights=None,
            seasonality_weights=None,
            trend_lr=1,
            seasonality_lr=1,
            exogenous_lr=1,
            min_sample_pct=.01,
            split_cost='mse',
            global_cost='maic',
            exogenous=None,
            damp_factor=None,
            ewm_alpha=.5,
            alpha=None,
            beta=None,
            ransac_trials=100,
            ransac_min_samples=10,
            tree_depth=1):

        if seasonal_period is None:
            seasonal_period = 0
        #grab all variables to create 'generator' variables
        _params = locals()
        _params.pop('self', None)
        _params.pop('time_series', None)
        _params = {k: ThymeBoost.create_iterated_features(v) for k, v in _params.items()}
        time_series = pd.Series(time_series)
        self.time_series_index = time_series.index
        self.time_series = time_series.values
        assert not all([i == 0 for i in time_series]), 'All inputs are 0'
        assert len(time_series) > 1, 'ThymeBoost requires at least 2 data points'
        time_series = self.scale_input(time_series)
        self.booster_obj = booster(time_series=time_series,
                                   given_splits=self.given_splits,
                                   verbose=self.verbose,
                                   n_split_proposals=self.n_split_proposals,
                                   approximate_splits=self.approximate_splits,
                                   exclude_splits=self.exclude_splits,
                                   cost_penalty=self.cost_penalty,
                                   normalize_seasonality=self.normalize_seasonality,
                                   regularization=self.regularization,
                                   n_rounds=self.n_rounds,
                                   smoothed_trend=self.smoothed_trend,
                                   additive=self.additive,
                                   **_params)
        booster_results = self.booster_obj.boost()
        fitted_trend = booster_results[0]
        fitted_seasonality = booster_results[1]
        fitted_exogenous = booster_results[2]
        self.c = self.booster_obj.c
        self.builder = build_output.BuildOutput(self.time_series,
                                                self.time_series_index,
                                                self.unscale_input,
                                                self.c)
        output = self.builder.build_fitted_df(fitted_trend,
                                              fitted_seasonality,
                                              fitted_exogenous)
        #ensure we do not fall into ensemble prediction for normal fit
        self.ensemble_boosters = None
        return output

    def predict(self,
                fitted_output,
                forecast_horizon,
                future_exogenous=None,
                damp_factor=None,
                trend_cap_target=None) -> pd.DataFrame:
        """
        ThymeBoost predict method which uses the booster to generate 
        predictions that are a sum of each component's round.

        Parameters
        ----------
        fitted_output : pd.DataFrame
            The output from the ThymeBoost.fit method.
        forecast_horizon : int
            The number of periods to forecast.
        damp_factor : float, optional
            Damp factor to apply, constrained to (0, 1) where .5 is 50% of the 
            current predicted trend.
            The default is None.
        trend_cap_target : float, optional
            Instead of a predetermined damp_factor, this will only dampen the 
            trend to a certain % growth if it exceeds that growth. 
            The default is None.

        Returns
        -------
        predicted_output : pd.DataFrame
            The predicted output dataframe.

        """
        if future_exogenous is not None:
            assert len(future_exogenous) == forecast_horizon, 'Given future exogenous not equal to forecast horizon'
        if self.ensemble_boosters is None:
            trend, seas, exo, predictions = predict_rounds(self.booster_obj,
                                                           forecast_horizon,
                                                           future_exogenous)
            fitted_output = copy.deepcopy(fitted_output)
            predicted_output = self.builder.build_predicted_df(fitted_output,
                                                               forecast_horizon,
                                                               trend,
                                                               seas,
                                                               exo,
                                                               predictions,
                                                               trend_cap_target,
                                                               damp_factor)
        else:
            ensemble_predictions = []
            for booster_obj in self.ensemble_boosters:
                self.booster_obj = booster_obj
                trend, seas, exo, predictions = predict_rounds(self.booster_obj,
                                                               forecast_horizon,
                                                               future_exogenous)
                fitted_output = copy.deepcopy(fitted_output)
                predicted_output = self.builder.build_predicted_df(fitted_output,
                                                                   forecast_horizon,
                                                                   trend,
                                                                   seas,
                                                                   exo,
                                                                   predictions,
                                                                   trend_cap_target,
                                                                   damp_factor)
                ensemble_predictions.append(predicted_output)
            predicted_output = pd.concat(ensemble_predictions)
            predicted_output = predicted_output.groupby(predicted_output.index).mean()
        return predicted_output

    def optimize(self,
                 time_series,
                 optimization_type='grid_search',
                 optimization_strategy='rolling',
                 optimization_steps=3,
                 lag=2,
                 optimization_metric='smape',
                 test_set='all',
                 verbose=1,
                 **kwargs):
        """
        Grid search lazily through search space in roder to find the params
        which result in the 'best' forecast depending on the given optimization
        parameters.

        Parameters
        ----------
        time_series : pd.Series
            The time series.
        optimization_type : str, optional
            How to search the space, only 'grid_search' is implemented. 
            TODO: add bayesian optimization.
            The default is 'grid_search'.
        optimization_strategy : str, optional
            The strategy emplyed when determing the 'best' params. The options
            are ['rolling', 'holdout'] where rolling uses a cross validation 
            strategy to 'roll' through the test set. Holdout simply hold out 
            the last 'lag' data points for testing.
            The default is 'rolling'.
        optimization_steps : int, optional
            When performing 'rolling' optimization_strategy the number of steps
            to test on. The default is 3.
        lag : int, optional
            How many data points to use as the test set. When using 'rolling',
            this parameter and optimization_steps determines the total number of
            testing points. Fore example, a lag of 2 with 3 steps means 3 * 2
            or 6 total points.  In step one we holdout the last 6 and test only 
            using the first 3 periods of the test set. In step two we include 
            the last step's test set in the train to test the final 3 periods.
            The default is 2.
        optimization_metric : str, optional
            The metric to judge the test forecast by. Options are :
            ['smape', 'mape', 'mse', 'mae'].
            The default is 'smape'.
        test_set : TYPE, optional
            DESCRIPTION. The default is 'all'.
        verbose : TYPE, optional
            DESCRIPTION. The default is 1.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        optimized_params : pd.DataFrame
            The predicted output dataframe of the optimal parameters.

        """
        time_series = pd.Series(time_series)
        self.time_series = time_series
        optimizer = Optimizer(self,
                              time_series,
                              optimization_type,
                              optimization_strategy,
                              optimization_steps,
                              lag,
                              optimization_metric,
                              test_set,
                              verbose,
                              **kwargs)
        optimized = optimizer.optimize()
        self.optimized_params = optimizer.run_settings
        return optimized

    def ensemble(self,
                 time_series,
                 verbose=1,
                 **kwargs):
        """
        Perform ensembling aka a simple average of each combination of inputs.
        For example: passing trend_estimator=['mean', 'linear'] will fit using 
        BOTH mean and linear then average the results. We can use generator 
        features here as well such as: tren_estimator=['mean', ['linear', 'mean']].
        Notice that now we pass a list of lists.

        Parameters
        ----------
        time_series : pd.Series
            The time series.
        verbose : bool, optional
            Print statments. The default is 1.
        **kwargs : list
            list of features that are typically passed to fit that you want to
            ensemble.

        Returns
        -------
        output : : pd.DataFrame
            The predicted output dataframe fro mthe ensembled params.

        """
        ensembler = Ensemble(model_object=self,
                             y=time_series,
                             verbose=verbose,
                             **kwargs)
        output, ensemble_params = ensembler.ensemble_fit()
        self.ensemble_boosters = ensemble_params
        return output

    def detect_outliers(self,
                        time_series,
                        trend_estimator='ransac',
                        fit_type='local',
                        **kwargs):
        """
        This is an off-the-cuff helper method for outlier detection. 
        Definitely do not use ETS, ARIMA, or Loess estimators.
        User beware!

        Parameters
        ----------
        time_series : pd.Series
            the time series.
        trend_estimator : str, optional
            'Approved' options are ['mean', 'median', 'linear', 'ransac']. 
            The default is 'ransac'.
        fit_type : str, optional
            Whether to use global or local fitting.
            Options are ['local', 'global']. The default is 'local'.
        seasonal_estimator : str, optional
            The method to approximate the seasonal component. 
            The default is 'fourier'.
        seasonal_period : int, optional
            The seasonal frequency. The default is None.

        Returns
        -------
        fitted_results : pd.DataFrame
            Output from booster with a new 'outliers' column added with 
            True/False denoting outlier classification.

        """
        fitted_results = self.fit(time_series=time_series,
                                  trend_estimator=trend_estimator,
                                  fit_type=fit_type,
                                  **kwargs)
        fitted_results['outliers'] = (fitted_results['y'].gt(fitted_results['yhat_upper'])) | \
                                     (fitted_results['y'].lt(fitted_results['yhat_lower']))
        return fitted_results

    @staticmethod
    def plot_results(fitted, predicted=None, figsize=None):
        """
        Plotter helper function to plot the results. 
        Plot adapts depending on the inputs.

        Parameters
        ----------
        fitted : pd.DataFrame
            Output df from either fit, optimize, or ensemble method.
        predicted : pd.DataFrame, optional
            Dataframe from predict method. The default is None.
        figsize : tuple, optional
            Matplotlib's figsize. The default is None.

        Returns
        -------
        None.

        """
        plotting.plot_results(fitted, predicted, figsize)

    @staticmethod
    def plot_components(fitted, predicted=None, figsize=None):
        """
        Plotter helper function to plot each component. 
        Plot adapts depending on the inputs.

        Parameters
        ----------
        fitted : pd.DataFrame
            Output df from either fit, optimize, or ensemble method.
        predicted : pd.DataFrame, optional
            Dataframe from predict method. The default is None.
        figsize : tuple, optional
            Matplotlib's figsize. The default is None.

        Returns
        -------
        None.

        """
        plotting.plot_components(fitted, predicted, figsize)

    #TODO: Fix update methods for online learning
    # def update_booster_params(self,
    #                           residual_component,
    #                           new_series,
    #                           current_prediction,
    #                           predicted_trend,
    #                           predicted_seasonality,
    #                           ):
    #     self.booster_obj.i = self.booster_obj.i - 1
    #     self.num_rounds = self.booster_obj.i
    #     self.booster_obj.boosted_data = residual_component.values
    #     self.booster_obj.time_series = self.time_series
    #     self.booster_obj.time_series_index = self.time_series_index
    #     self.booster_obj.trends = [predicted_trend]
    #     self.booster_obj.seasonalities = [predicted_seasonality]
    #     self.booster_obj.fitted_exogenous = [np.append(i, np.zeros(len(new_series))) for i in self.booster_obj.fitted_exogenous]
    #     updated_cost = calc_cost(self.time_series,
    #                              current_prediction,
    #                              self.booster_obj.c,
    #                              self.booster_obj.regularization,
    #                              self.booster_obj.boosting_params['global_cost'])
    #     self.booster_obj.cost = updated_cost

    # def update(self, fitted_output, new_series, **kwargs):
    #     predicted_output = self.predict(fitted_output, len(new_series))
    #     total_trend = fitted_output['trend'].append(predicted_output['predicted_trend'])
    #     total_prediction = fitted_output['yhat'].append(predicted_output['predictions'])
    #     total_seasonality = fitted_output['seasonality'].append(predicted_output['predicted_seasonality'])
    #     if not isinstance(new_series, pd.Series):
    #         new_series = pd.Series(new_series)
    #     new_series.index = predicted_output.index
    #     full_series = fitted_output['y'].append(new_series)
    #     self.time_series = full_series.values
    #     self.time_series_index = full_series.index
    #     residual_component = fitted_output['y'] - fitted_output['yhat']
    #     residual_component = residual_component.append(new_series - predicted_output['predictions'])
    #     self.update_booster_params(residual_component,
    #                                new_series,
    #                                current_prediction=total_prediction.values,
    #                                predicted_trend=total_trend.values,
    #                                predicted_seasonality=total_seasonality.values,
    #                                )
    #     updated_results = self.booster_obj.boost()
    #     fitted_trend = updated_results[0]
    #     fitted_seasonality = updated_results[1]
    #     fitted_exogenous = updated_results[2]
    #     self.c = self.booster_obj.c
    #     self.builder = build_output.BuildOutput(self.time_series,
    #                                             self.time_series_index,
    #                                             self.unscale_input,
    #                                             self.c)
    #     output = self.builder.build_fitted_df(fitted_trend,
    #                                           fitted_seasonality,
    #                                           fitted_exogenous)
    #     return output





#%%
if __name__ == '__main__':
    import time
    import quandl
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')
    #Get bitcoin data
    data = quandl.get("BITSTAMP/USD")
    data = data.iloc[:-200, :]
    full_index = pd.date_range(data.index[0], end=data.index[-1], freq='D')
    data = data.reindex(full_index).fillna(7000)
    #let's get our X matrix with the new variables to use
    X = data.drop('Low', axis = 1)
    X_train = X.iloc[-930:-50,:]
    X_test = X.iloc[-50:,:]
    y = data['Low']
    y_train = y[-930:-50,]
    y_test = y[-50:,]
    ssw = np.ones(len(y_train))
    ssw[:200] = .00001
    freq = pd.infer_freq(data.index)

    last_date = full_index[-1]
    forecast_end_date = pd.date_range(last_date, periods = 300 + 1, freq=freq)[1:]
    #from ThymeBoost import ThymeBoost as tb
    boosted_model = ThymeBoost(
                                approximate_splits=True,
                                n_split_proposals=25,
                                #n_rounds=30,
                                verbose=0,
                                cost_penalty=.001,
                                )
    opt = boosted_model.optimize(y_train,
                                 seasonal_period=[365],
                                 trend_estimator=['mean', boosted_model.combine(['linear', 'mean'])],
                                 exogenous=[X_train],
                                 exogenous_estimator=['ols'],
                                 fit_type=['local', 'global'],
                                 seasonal_estimator=['fourier'],
                                 )
    ts = time.time()
    outliers = boosted_model.detect_outliers(y_train,
                                             exogenous=X_train,
                                             exogenous_estimator='decision_tree')
    output = boosted_model.ensemble(y_train,
                                    verbose=1,
                                        trend_estimator=['mean', 'linear', 'median'],
                                        fit_type=['local', 'global'],
                                        seasonal_period=[365, None],
                                        exogenous=[X_train],
                                        exogenous_estimator=['ols', 'decision_tree'],
                                        tree_depth=[1,2,3])
    output = boosted_model.fit(y_train,
                                trend_estimator='linear',
                                seasonal_estimator='classic',
                                split_cost='mse',
                                global_cost='maicc',
                                exogenous_estimator='decision_tree',
                                tree_depth=[1,2,3],
                               # exogenous=X_train,
                                fit_type='local',
                                seasonal_period=365,
                                poly=1,
                                connectivity_constraint=True,
                                trend_lr=1,
                                seasonality_lr=1,
                                exogenous_lr=1,
                                alpha=.9,
                                window_size=111,
                                arima_order=[(2, 1, 2)],
                                min_sample_pct=.01,
                                ewm_alpha=.3)
    output = boosted_model.update(output, y_test[:100])
    look = boosted_model.booster_obj.seasonal_pred_params
    predicted_output = boosted_model.predict(fitted_output=opt,
                                             #future_exogenous=X_test,
                                             forecast_horizon=50)
    predicted_trend = output['trend'].append(predicted_output['predicted_trend'])
    te = time.time()
    print(te - ts)
    plt.plot(np.append(output['seasonality'].values, predicted_output['predicted_seasonality'].values))
    boosted_model.plot_results(output, predicted_output)
    boosted_model.plot_components(output, predicted_output)
    plt.plot(data['Low'].iloc[-930:].values)
    plt.plot(np.append(output['trend'], predicted_output['predicted_trend']))
    plt.plot(np.append(output['yhat'], predicted_output['predictions']))
    plt.show()

# # %%
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from tqdm import tqdm
#     import pandas as pd
#     def smape(A, F):
#         return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
#     def MASE(training_series, prediction_series, testing_series):
#         """
#         Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.
        
#         See "Another look at measures of forecast accuracy", Rob J Hyndman
        
#         parameters:
#             training_series: the series used to train the model, 1d numpy array
#             testing_series: the test series to predict, 1d numpy array or float
#             prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
#             absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
        
#         """
#         n = training_series.shape[0]
#         d = np.abs(  np.diff( training_series) ).sum()/(n-1)
        
#         errors = np.abs(testing_series - prediction_series )
#         return errors.mean()/d
    
#     train_df = pd.read_csv(r'C:\Users\er90614\Downloads\m4-monthly-train.csv')
#     test_df = pd.read_csv(r'C:\Users\er90614\Downloads\m4-monthly-test.csv')
#     train_df.index = train_df['V1']
#     train_df = train_df.drop('V1', axis = 1)
#     test_df.index = test_df['V1']
#     test_df = test_df.drop('V1', axis = 1)
#     mases = []
#     smapes = []
#     naive_smape = []
#     naive_mases = []
#     for row in tqdm(range(len(train_df))):
#         try:
#             #row = 1219
#             y = train_df.iloc[row, :].dropna()
#             y_test = test_df.iloc[row, :].dropna()
#             y = y[-36:]
#             #scaler = MinMaxScaler()
#             #y = scaler.fit_transform(np.array(y).reshape(-1,1))
#             ssw = np.ones(len(y))
#             ssw[-12:] = 2
#             boosted_model = ThymeBoost(seasonal_period = [12],
#                                   fit_type = 'local',
#                                   trend_estimator = 'ransac',
#                                   connectivity_constraint = True,
#                                   seasonal_estimator = 'harmonic',
#                                   additive = True,
#                                   global_cost = 'maic',
#                                   split_cost= 'mse',
#                                   cost_penalty = .001,
#                                   verbose = 0,
#                                   min_sample_pct = 0.2,
#                                   n_rounds = None,               
#                                   trend_lr = 1,  
#                                   approximate_splits = True,
#                                   n_split_proposals = 20,
#                                   #arima_order = (0, 1, 1),
#                                   fourier_order = 4,
#                                   trend_cap_target = .1,
#                                   #seasonal_sample_weight = ssw
#                                   )
             
#             output = boosted_model.fit(y.values, forecast_horizon = 18)
#             # output = boosted_model.optimize(y.values, 
#             #                                 forecast_horizon = 18,
#             #                                 optimization_strategy = 'holdout',
#             #                                 optimization_steps = 6,
#             #                                 lag = 6
#             #                                 #3
#             #                                 #9
#             #                                 # optimization_strategy = 'rolling',
#             #                                 # optimization_steps = 3,
#             #                                 # lag = 9
#             #                                 )
#             # plt.plot(np.append(predicted, forecast))
#             # boosted_model.plot_results()
#             # boosted_model.plot_components()
#             # plt.plot(np.append(y.values, y_test.values))
#             # smapes.append(smape(y_test.values, pd.Series(forecast).clip(lower=0)))
#             smapes.append(smape(y_test.values, output['predicted'].clip(lower=0)))
#             naive_smape.append(smape(y_test.values, np.tile(y.iloc[-1], len(y_test))))  
#             # mases.append(MASE(y.values, output['predicted'].clip(lower=0), y_test.values))
#             # naive_mases.append(MASE(y.values, np.tile(y.iloc[-1], len(y_test)), y_test.values))  
                        
#         except Exception as e:
#             print(e)
#             pass
    
#     print(np.mean(smapes))
#     print(np.mean(naive_smape))

#     smapes_series = pd.Series(smapes)
#     smapes_series = smapes_series.sort_values()
# # %%
#     import numpy as np
#     from ThymeBoost import ThymeBoost as tb
#     import pandas as pd
#     #simple cosine which changes
#     y = np.append(((np.cos(np.arange(1, 25)) + 50) *10), (np.cos(np.arange(1, 13))/10 + 50) *10) 
#     #bias some random element
#     y[10] = 480
#     #creating df for the extra features
#     exogenous = pd.DataFrame(np.zeros(len(y)))
#     exogenous.iloc[10, 0] = 1
#     #create df for future features
#     forecast_horizon = 18
#     future_exogenous = pd.DataFrame(np.zeros(forecast_horizon))
#     future_exogenous.iloc[3, 0] = 1    
#     #create an array for the sample weighting 
#     ssw = np.ones(len(y))
#     ssw[-12:] = 500
#     #create obj
#     boosted_model = tb.ThymeBoost(seasonal_period = [12],
#                           fit_type = 'local',
#                           trend_estimator = 'linear',
#                           connectivity_constraint = True,
#                           seasonal_estimator = 'harmonic',
#                           verbose = 1,
#                           #additive = False,
#                           min_sample_pct = 0.15,
#                           n_rounds = None,                
#                           approximate_splits = True,
#                           n_split_proposals = 20,
#                           fourier_order = 4,
#                           trend_cap_target = .1,
#                           seasonal_sample_weight = ssw,
#                           exogenous = exogenous,
#                           future_exogenous = future_exogenous
#                           )
#     #output is a dict of results
#     output = boosted_model.fit(y, forecast_horizon = forecast_horizon)    
#     #plot
#     boosted_model.plot_results()




