import math, pickle, subprocess, time
import numpy as np
import sklearn
from sklearn.linear_model import HuberRegressor
from core import Submission

"""
PYTHON submission

Implement the model below

##################################################### OVERVIEW ######################################################

1. Use get_next_data_as_string() OR get_next_data_as_list() OR get_next_data_as_numpy_array() to recieve the next row of data
2. Use the predict method to write the prediction logic, and return a float representing your prediction
3. Submit a prediction using self.submit_prediction(...)

################################################# OVERVIEW OF DATA ##################################################

1. get_next_data_as_string() accepts no input and returns a String representing a row of data extracted from data.csv
     Example output: '1619.5,1620.0,1621.0,,,,,,,,,,,,,1.0,10.0,24.0,,,,,,,,,,,,,1615.0,1614.0,1613.0,1612.0,1611.0,
     1610.0,1607.0,1606.0,1605.0,1604.0,1603.0,1602.0,1601.5,1601.0,1600.0,7.0,10.0,1.0,10.0,20.0,3.0,20.0,27.0,11.0,
     14.0,35.0,10.0,1.0,10.0,13.0'

2. get_next_data_as_list() accepts no input and returns a List representing a row of data extracted from data.csv,
   missing data is represented as NaN (math.nan)
     Example output: [1619.5, 1620.0, 1621.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 1.0, 10.0,
     24.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 1615.0, 1614.0, 1613.0, 1612.0, 1611.0, 1610.0,
     1607.0, 1606.0, 1605.0, 1604.0, 1603.0, 1602.0, 1601.5, 1601.0, 1600.0, 7.0, 10.0, 1.0, 10.0, 20.0, 3.0, 20.0,
     27.0, 11.0, 14.0, 35.0, 10.0, 1.0, 10.0, 13.0]

3. get_next_data_as_numpy_array() accepts no input and returns a Numpy Array representing a row of data extracted from
   data.csv, missing data is represented as NaN (math.nan)
   Example output: [1.6195e+03 1.6200e+03 1.6210e+03 nan nan nan nan nan nan nan nan nan nan nan nan 1.0000e+00
    1.0000e+01 2.4000e+01 nan nan nan nan nan nan nan nan nan nan nan nan 1.6150e+03 1.6140e+03 1.6130e+03 1.6120e+03
     1.6110e+03 1.6100e+03 1.6070e+03 1.6060e+03 1.6050e+03 1.6040e+03 1.6030e+03 1.6020e+03 1.6015e+03 1.6010e+03
      1.6000e+03 7.0000e+00 1.0000e+01 1.0000e+00 1.0000e+01 2.0000e+01 3.0000e+00 2.0000e+01 2.7000e+01 1.1000e+01
       1.4000e+01 3.5000e+01 1.0000e+01 1.0000e+00 1.0000e+01 1.3000e+01]

##################################################### IMPORTANT ######################################################

1. One of the methods get_next_data_as_string(), get_next_data_as_list(), or get_next_data_as_numpy_array() MUST be used and
   _prediction(pred) MUST be used to submit below in the solution implementation for the submission to work correctly.
2. get_next_data_as_string(), get_next_data_as_list(), or get_next_data_as_numpy_array() CANNOT be called more then once in a
   row without calling self.submit_prediction(pred).
3. In order to debug by printing do NOT call the default method `print(...)`, rather call self.debug_print(...)

"""


class MySubmission(Submission):

    def __init__(self):
        self.turn = 0
        self.ARRAY_SIZE = 5000000
        self.running_model_first_fit_turn = 100000

        self.alpha_10 = 2. / (10. + 1.)
        self.alpha_12 = 2. / (12. + 1.)
        self.alpha_15 = 2. / (15. + 1.)
        self.alpha_20 = 2. / (20. + 1.)
        self.alpha_50 = 2. / (50. + 1.)
        self.alpha_500 = 2. / (500. + 1.)
        self.alpha_1500 = 2. / (1500. + 1.)

        self.bias_10 = (2. - self.alpha_10) / 2. / (1. - self.alpha_10)
        self.bias_12 = (2. - self.alpha_12) / 2. / (1. - self.alpha_12)
        self.bias_15 = (2. - self.alpha_15) / 2. / (1. - self.alpha_15)
        self.bias_20 = (2. - self.alpha_20) / 2. / (1. - self.alpha_20)
        self.bias_50 = (2. - self.alpha_50) / 2. / (1. - self.alpha_50)
        self.bias_500 = (2. - self.alpha_500) / 2. / (1. - self.alpha_500)
        self.bias_1500 = (2. - self.alpha_1500) / 2. / (1. - self.alpha_1500)

        self.model_static = pickle.load(open('model.sav', 'rb'))
        self.model_running = HuberRegressor(fit_intercept=False, epsilon=1.35)
        self.model_expanding = HuberRegressor(fit_intercept=False, epsilon=1.35)
        self.model_bagging = pickle.load(open('model_bagging.sav', 'rb'))

        self.mids = np.zeros(self.ARRAY_SIZE)
        self.y = np.zeros(self.ARRAY_SIZE)
        self.y_pred = np.zeros(self.ARRAY_SIZE)
        self.signals = np.zeros((self.ARRAY_SIZE, len(self.model_static.coef_)))

        super().__init__()


    """
    update_data(data) appends new row to existing dataframe
    """
    def update_data(self):

        data = self.get_next_data_as_string()
        data = [float(x) if x else 0 for x in data.split(',')]
        self.x = data

        if self.turn == self.ARRAY_SIZE - 10:
            self.mids.resize(2 * len(self.mids))
            self.y.resize(2 * len(self.y))
            self.y_pred.resize(2 * len(self.y_pred))
            self.signals.resize(2 * len(self.signals))
            self.ARRAY_SIZE = 2 * self.ARRAY_SIZE


    """
    is_new_day(self) tries to guess when it's a new session and rolling averages should be restarted.
    The best I could find for now is that ask depth seems to be thin at start of sessions.
    """
    def is_new_day(self):

        ask_depth = np.sum(self.x[0:15] != 0.)

        if not hasattr(self, 'ask_depth_prev'):
            self.ask_depth_prev = ask_depth

        is_reset = ask_depth - self.ask_depth_prev < -3.
        self.ask_depth_prev = ask_depth

        return is_reset


    """
    update_features(self) update features after each new line is added
    """
    def update_features(self):

        x = self.x
        turn = self.turn
        turn_prev = max(turn - 87, 0)

        askRate0 = x[0] if x[0] != 0. else np.nan
        askRate1 = x[1] if x[1] != 0. else np.nan
        askRate2 = x[2] if x[2] != 0. else np.nan
        bidRate0 = x[30] if x[30] != 0. else np.nan
        bidRate1 = x[31] if x[31] != 0. else np.nan
        bidRate2 = x[32] if x[32] != 0. else np.nan

        askSize0 = x[15]
        askSize1 = x[16]
        askSize2 = x[17]
        bidSize0 = x[45]
        bidSize1 = x[46]
        bidSize2 = x[47]

        askSize012 = askSize0 + askSize1 + askSize2
        bidSize012 = bidSize0 + bidSize1 + bidSize2
        askSizeTotal = np.sum(x[15:25])
        bidSizeTotal = np.sum(x[45:55])

        mid = 0.5 * (bidRate0 + askRate0)
        mid_mic = (askSize0 * bidRate0 + bidSize0 * askRate0) / (askSize0 + bidSize0)
        y = mid - self.mids[turn_prev]
        self.mids[turn] = mid
        self.y[turn_prev] = y

        if ((self.turn + 1) % self.running_model_first_fit_turn) == 0:
            self.model_expanding.fit(self.signals[0:turn_prev], self.y[0:turn_prev])
            self.model_running.fit(self.signals[max(turn_prev - self.running_model_first_fit_turn + 1, 0):turn_prev], self.y[max(turn_prev - self.running_model_first_fit_turn + 1, 0):turn_prev])

        if (self.turn == 0) or self.is_new_day():
            self.y_ewma500 = y
            self.y_var_ewma500 = 0.0001
            self.y_vol_ewma500 = math.sqrt(self.y_var_ewma500)

            self.bidSize0_var_ewma50 = 0.0001
            self.askSize0_var_ewma50 = 0.0001
            self.bidSize0_vol_ewma50 = math.sqrt(self.bidSize0_var_ewma50)
            self.askSize0_vol_ewma50 = math.sqrt(self.askSize0_var_ewma50)

            self.bidSize0_ewma50 = bidSize0
            self.askSize0_ewma50 = askSize0
            self.bidSize1_ewma50 = bidSize1
            self.askSize1_ewma50 = askSize1

            self.askSize012_ewma50 = askSize012
            self.bidSize012_ewma50 = bidSize012
            self.askSizeTotal_ewma20 = askSizeTotal
            self.bidSizeTotal_ewma20 = bidSizeTotal

            self.askSize012_var_ewma50 = 0.0001
            self.bidSize012_var_ewma50 = 0.0001
            self.askSize012_vol_ewma50 = math.sqrt(self.askSize012_var_ewma50)
            self.bidSize012_vol_ewma50 = math.sqrt(self.bidSize012_var_ewma50)
            self.askSizeTotal_var_ewma20 = 0.0001
            self.bidSizeTotal_var_ewma20 = 0.0001
            self.askSizeTotal_vol_ewma20 = math.sqrt(self.askSizeTotal_var_ewma20)
            self.bidSizeTotal_vol_ewma20 = math.sqrt(self.bidSizeTotal_var_ewma20)

            self.midMic_ewma10 = mid_mic
            self.midMic_var_ewma10 = 0.0001
            self.midMic_vol_ewma10 = math.sqrt(self.midMic_var_ewma10)

            self.askRate0_ewma15 = askRate0
            self.bidRate0_ewma15 = bidRate0

        else:
            # y
            self.y_var_ewma500 = (1. - self.alpha_500) * (self.y_var_ewma500 + self.bias_500 * self.alpha_500 * (y - self.y_ewma500) * (y - self.y_ewma500))
            self.y_vol_ewma500 = math.sqrt(self.y_var_ewma500)

            self.y_ewma500 = (1. - self.alpha_500) * self.y_ewma500 + self.alpha_500 * y

            # bidSize0 and askSize0 ewma
            self.bidSize0_var_ewma50 = (1. - self.alpha_50) * (self.bidSize0_var_ewma50 + self.bias_50 * self.alpha_50 * (bidSize0 - self.bidSize0_ewma50) * (bidSize0 - self.bidSize0_ewma50))
            self.askSize0_var_ewma50 = (1. - self.alpha_50) * (self.askSize0_var_ewma50 + self.bias_50 * self.alpha_50 * (askSize0 - self.askSize0_ewma50) * (askSize0 - self.askSize0_ewma50))
            self.bidSize0_vol_ewma50 = math.sqrt(self.bidSize0_var_ewma50)
            self.askSize0_vol_ewma50 = math.sqrt(self.askSize0_var_ewma50)

            self.bidSize0_ewma50 = (1. - self.alpha_50) * self.bidSize0_ewma50 + self.alpha_50 * bidSize0
            self.askSize0_ewma50 = (1. - self.alpha_50) * self.askSize0_ewma50 + self.alpha_50 * askSize0

            # bidSize1 and askSize1 ewma
            self.bidSize1_ewma50 = (1. - self.alpha_50) * self.bidSize1_ewma50 + self.alpha_50 * bidSize1
            self.askSize1_ewma50 = (1. - self.alpha_50) * self.askSize1_ewma50 + self.alpha_50 * askSize1

            # bidSizeTotal and askSizeTotal ewma
            self.askSize012_var_ewma50 = (1. - self.alpha_50) * (self.askSize012_var_ewma50 + self.bias_50 * self.alpha_50 * (askSize012 - self.askSize012_ewma50) * (askSize012 - self.askSize012_ewma50))
            self.bidSize012_var_ewma50 = (1. - self.alpha_50) * (self.bidSize012_var_ewma50 + self.bias_50 * self.alpha_50 * (bidSize012 - self.bidSize012_ewma50) * (bidSize012 - self.bidSize012_ewma50))
            self.askSize012_vol_ewma50 = math.sqrt(self.askSize012_var_ewma50)
            self.bidSize012_vol_ewma50 = math.sqrt(self.bidSize012_var_ewma50)
            self.askSizeTotal_var_ewma20 = (1. - self.alpha_20) * (self.askSizeTotal_var_ewma20 + self.bias_20 * self.alpha_20 * (askSizeTotal - self.askSizeTotal_ewma20) * (askSizeTotal - self.askSizeTotal_ewma20))
            self.bidSizeTotal_var_ewma20 = (1. - self.alpha_20) * (self.bidSizeTotal_var_ewma20 + self.bias_20 * self.alpha_20 * (bidSizeTotal - self.bidSizeTotal_ewma20) * (bidSizeTotal - self.bidSizeTotal_ewma20))
            self.askSizeTotal_vol_ewma20 = math.sqrt(self.askSizeTotal_var_ewma20)
            self.bidSizeTotal_vol_ewma20 = math.sqrt(self.bidSizeTotal_var_ewma20)

            self.askSize012_ewma50 = (1. - self.alpha_50) * self.askSize012_ewma50 + self.alpha_50 * askSize012
            self.bidSize012_ewma50 = (1. - self.alpha_50) * self.bidSize012_ewma50 + self.alpha_50 * bidSize012
            self.askSizeTotal_ewma20 = (1. - self.alpha_20) * self.askSizeTotal_ewma20 + self.alpha_20 * askSizeTotal
            self.bidSizeTotal_ewma20 = (1. - self.alpha_20) * self.bidSizeTotal_ewma20 + self.alpha_20 * bidSizeTotal

            # Micro mid zscore
            self.midMic_var_ewma10 = (1. - self.alpha_10) * (self.midMic_var_ewma10 + self.bias_10 * self.alpha_10 * (mid_mic - self.midMic_ewma10) * (mid_mic - self.midMic_ewma10))
            self.midMic_vol_ewma10 = math.sqrt(self.midMic_var_ewma10)

            self.midMic_ewma10 = (1. - self.alpha_10) * self.midMic_ewma10 + self.alpha_10 * mid_mic

            # AskRate and bidRate
            self.askRate0_ewma15 = (1. - self.alpha_15) * self.askRate0_ewma15 + self.alpha_15 * askRate0
            self.bidRate0_ewma15 = (1. - self.alpha_15) * self.bidRate0_ewma15 + self.alpha_15 * bidRate0

        #### Signals ####
        self.sig1 = (bidSize0 - askSize0) / (bidSize0 + askSize0)
        self.sig2 = (bidSize1 - askSize1) / (bidSize1 + askSize1)
        self.sig3 = (bidSize012 - self.bidSize012_ewma50) / self.bidSize012_vol_ewma50 - (askSize012 - self.askSize012_ewma50) / self.askSize012_vol_ewma50
        self.sig4 = bidSize1 / bidSize0 - askSize1 / askSize0
        self.sig5 = (bidSize0 - self.bidSize0_ewma50) / self.bidSize0_vol_ewma50 - (askSize0 - self.askSize0_ewma50) / self.askSize0_vol_ewma50
        self.sig6 = (bidSizeTotal - self.bidSizeTotal_ewma20) / self.bidSizeTotal_vol_ewma20 - (askSizeTotal - self.askSizeTotal_ewma20) / self.askSizeTotal_vol_ewma20
        self.sig7 = (askRate1 - askRate0) - (bidRate0 - bidRate1)
        self.sig8 = ((bidRate1 - bidRate2) - (askRate2 - askRate1)) / ((bidRate1 - bidRate2) + (askRate2 - askRate1))
        self.sig9 = (mid_mic - self.midMic_ewma10) / self.midMic_vol_ewma10
        self.sig10 = bidRate0 - self.bidRate0_ewma15 + askRate0 - self.askRate0_ewma15
        #################

        signals = np.array([self.sig1, self.sig2, self.sig3, self.sig4, self.sig5, self.sig6, self.sig7, self.sig8])
        signals[np.isinf(signals)] = 0.
        signals[np.isnan(signals)] = 0.
        self.signals[turn, :] = signals

        return

    """
    get_prediction(data) expects a row of data from data.csv as input and should return a float that represents a prediction for the supplied row of data
    """
    def get_prediction(self):

        signals = self.signals[self.turn:self.turn + 1, :]
        prediction_static = self.model_static.predict(signals)[0]
        prediction_bagging = self.model_bagging.predict(signals)[0]

        if self.turn >= self.running_model_first_fit_turn:
            prediction_expanding = self.model_expanding.predict(signals)[0]
            prediction_running = self.model_running.predict(signals)[0]
            prediction = 0.25 * (prediction_static + prediction_expanding + prediction_running + prediction_bagging)
        else:
            prediction = 0.5 * (prediction_static + prediction_bagging)

        if not np.isfinite(prediction):
            prediction = 0.

        prediction = np.clip(prediction, -5., 5.)
        self.y_pred[self.turn] = prediction
        return prediction


    """
    run_submission() will iteratively fetch the next row of data in the format specified for every prediction submitted to self.submit_prediction()
    """
    def run_submission(self):

        while(True):
            #start = time.time()
            self.update_data()
            self.update_features()
            prediction = self.get_prediction()

            #prediction = (time.time() - start) * 1000
            self.submit_prediction(prediction)

            self.turn += 1


if __name__ == "__main__":
    MySubmission()
