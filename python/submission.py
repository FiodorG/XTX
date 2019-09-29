import math, pickle, subprocess, time
import numpy as np
import sklearn
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import HuberRegressor
from core import Submission


class MySubmission(Submission):

    def __init__(self):
        self.turn = 0
        self.ARRAY_SIZE = 5000000

        self.model_static = pickle.load(open('model.sav', 'rb'))
        self.model_xgb = pickle.load(open('modelxgb.sav', 'rb'))
        self.model_running = HuberRegressor(fit_intercept=False, epsilon=1.35)
        self.model_expanding = HuberRegressor(fit_intercept=False, epsilon=1.35)

        self.mids = np.zeros(self.ARRAY_SIZE)
        self.y = np.zeros(self.ARRAY_SIZE)
        self.y_pred = np.zeros(self.ARRAY_SIZE)
        self.signals = np.zeros((self.ARRAY_SIZE, len(self.model_static.coef_)))
        self.bid_nbr_trades = np.zeros(self.ARRAY_SIZE)
        self.ask_nbr_trades = np.zeros(self.ARRAY_SIZE)

        self.posting_bid = np.zeros(self.ARRAY_SIZE)
        self.posting_bid_sizes = np.zeros(self.ARRAY_SIZE)
        self.posting_bid_cross = np.zeros(self.ARRAY_SIZE)
        self.posting_bid_cross_sizes = np.zeros(self.ARRAY_SIZE)

        self.posting_ask = np.zeros(self.ARRAY_SIZE)
        self.posting_ask_sizes = np.zeros(self.ARRAY_SIZE)
        self.posting_ask_cross = np.zeros(self.ARRAY_SIZE)
        self.posting_ask_cross_sizes = np.zeros(self.ARRAY_SIZE)

        self.cancellations_bid = np.zeros(self.ARRAY_SIZE)
        self.cancellations_bid_sizes = np.zeros(self.ARRAY_SIZE)

        self.cancellations_ask = np.zeros(self.ARRAY_SIZE)
        self.cancellations_ask_sizes = np.zeros(self.ARRAY_SIZE)

        self.trades_buy = np.zeros(self.ARRAY_SIZE)
        self.trades_buy_sizes = np.zeros(self.ARRAY_SIZE)
        self.trades_sell = np.zeros(self.ARRAY_SIZE)
        self.trades_sell_sizes = np.zeros(self.ARRAY_SIZE)

        self.prev_row = None

        super().__init__()



    def update_data(self):
        data = self.get_next_data_as_string()
        data = [float(x) if x else 0 for x in data.split(',')]
        self.x = data

        if not self.prev_row:
            self.prev_row = data

        if self.turn == self.ARRAY_SIZE - 10:
            self.mids.resize(2 * len(self.mids))
            self.y.resize(2 * len(self.y))
            self.y_pred.resize(2 * len(self.y_pred))
            self.self.bid_nbr_trades.resize(2 * len(self.signals))
            self.self.ask_nbr_trades.resize(2 * len(self.signals))
            self.signals.resize(2 * len(self.signals))

            self.bid_nbr_trades.resize(2 * len(self.signals))
            self.ask_nbr_trades.resize(2 * len(self.signals))

            self.posting_bid.resize(2 * len(self.signals))
            self.posting_bid_sizes.resize(2 * len(self.signals))
            self.posting_bid_cross.resize(2 * len(self.signals))
            self.posting_bid_cross_sizes.resize(2 * len(self.signals))

            self.posting_ask.resize(2 * len(self.signals))
            self.posting_ask_sizes.resize(2 * len(self.signals))
            self.posting_ask_cross.resize(2 * len(self.signals))
            self.posting_ask_cross_sizes.resize(2 * len(self.signals))

            self.cancellations_bid.resize(2 * len(self.signals))
            self.cancellations_bid_sizes.resize(2 * len(self.signals))

            self.cancellations_ask.resize(2 * len(self.signals))
            self.cancellations_ask_sizes.resize(2 * len(self.signals))

            self.trades_buy.resize(2 * len(self.signals))
            self.trades_buy_sizes.resize(2 * len(self.signals))
            self.trades_sell.resize(2 * len(self.signals))
            self.trades_sell_sizes.resize(2 * len(self.signals))

            self.ARRAY_SIZE = 2 * self.ARRAY_SIZE



    def is_new_day(self):
        ask_depth = np.sum(self.x[0:15] != 0.)

        if not hasattr(self, 'ask_depth_prev'):
            self.ask_depth_prev = ask_depth

        is_reset = ask_depth - self.ask_depth_prev < -3.
        self.ask_depth_prev = ask_depth

        return is_reset



    def compute_z_score(self, x, name, n, reset):
        var_name = name + '_var_ewm'
        ewma_name = name + '_ewma'

        if reset:
            setattr(self, var_name, 0.0001)
            setattr(self, ewma_name, x)

            return 0.
        else:
            alpha = 2. / (n + 1.)
            bias = (2. - alpha) / 2. / (1. - alpha)

            avg_prev = getattr(self, ewma_name)
            variance = (1. - alpha) * (getattr(self, var_name) + bias * alpha * (x - avg_prev)**2)
            volatility = math.sqrt(variance)
            average = (1. - alpha) * avg_prev + alpha * x

            setattr(self, var_name, variance)
            setattr(self, ewma_name, average)

            return (x - average) / volatility



    def get_average_price_depth(self, q, side):

        if side:
            sizes = np.array(self.x[15:30])
            rates = np.array(self.x[0:15])
        else:
            sizes = np.array(self.x[45:60])
            rates = np.array(self.x[30:45])

        sizes_cumsum = np.cumsum(sizes)
        indices = sizes_cumsum <= q
        last_index = np.sum(indices)
        if last_index == 0:
            q_last = q
        else:
            q_last = q - sizes_cumsum[max(last_index - 1, 0)]
        res = np.dot(sizes[indices], rates[indices])
        if last_index < 15:
            res += q_last * rates[min(14, last_index)]
        res /= q

        return res



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
        askRate0_prev = self.prev_row[0] if self.prev_row[0] != 0. else np.nan
        bidRate0_prev = self.prev_row[30] if self.prev_row[30] != 0. else np.nan

        askSize0 = x[15]
        askSize1 = x[16]
        askSize2 = x[17]
        bidSize0 = x[45]
        bidSize1 = x[46]
        bidSize2 = x[47]
        askSize0_prev = self.prev_row[15]
        bidSize0_prev = self.prev_row[45]

        askSize012 = askSize0 + askSize1 + askSize2
        bidSize012 = bidSize0 + bidSize1 + bidSize2
        askSizeTotal = np.sum(x[15:25])
        bidSizeTotal = np.sum(x[45:55])

        mid = 0.5 * (bidRate0 + askRate0)
        mid_mic = (askSize0 * bidRate0 + bidSize0 * askRate0) / (askSize0 + bidSize0)
        y = mid - self.mids[turn_prev]
        self.mids[turn] = mid
        self.y[turn_prev] = y

        ################################################
        ################################################
        if bidRate0 == bidRate0_prev:
            if bidSize0 > bidSize0_prev:
                self.bid_nbr_trades[turn] = self.bid_nbr_trades[turn-1] + 1
            elif bidSize0 < bidSize0_prev:
                self.bid_nbr_trades[turn] = self.bid_nbr_trades[turn-1] - 1
            else:
                self.bid_nbr_trades[turn] = self.bid_nbr_trades[turn-1]
        else:
            self.bid_nbr_trades[turn] = 1

        if askRate0 == askRate0_prev:
            if askSize0 > askSize0_prev:
                self.ask_nbr_trades[turn] = self.ask_nbr_trades[turn-1] + 1
            elif askSize0 < askSize0_prev:
                self.ask_nbr_trades[turn] = self.ask_nbr_trades[turn-1] - 1
            else:
                self.ask_nbr_trades[turn] = self.ask_nbr_trades[turn-1]
        else:
            self.ask_nbr_trades[turn] = 1

        ################################################
        ################################################
        prev_bid_size = np.array(self.prev_row)[45:60]
        prev_ask_size = np.array(self.prev_row)[15:30]
        prev_bid_rate = np.array(self.prev_row)[30:45]
        prev_ask_rate = np.array(self.prev_row)[0:15]

        #### BID ####
        if bidRate0 == bidRate0_prev:
            # New trades added
            if bidSize0 > bidSize0_prev:
                self.posting_bid[turn] = 1
                self.posting_bid_sizes[turn] = bidSize0 - bidSize0_prev
            # Traded at bid
            elif bidSize0 < bidSize0_prev:
                self.cancellations_bid[turn] = 1
                self.cancellations_bid_sizes[turn] = bidSize0_prev - bidSize0

        elif bidRate0 < bidRate0_prev:
            # someone traded and consumed some orders up to new best bid
            self.trades_sell[turn] = 1
            self.trades_sell_sizes[turn] = np.sum(prev_bid_size[prev_bid_rate >= bidRate0]) - bidSize0

        elif bidRate0 > bidRate0_prev:
            # Post but above best bid
            self.posting_bid_cross[turn] = 1
            self.posting_bid_cross_sizes[turn] = np.sum(prev_ask_size[prev_ask_rate <= bidRate0]) + bidSize0
        #############

        #### ASK ####
        if askRate0 == askRate0_prev:
            # New trades added
            if askSize0 > askSize0_prev:
                self.posting_ask[turn] = 1
                self.posting_ask_sizes[turn] = askSize0 - askSize0_prev
            # Traded at ask
            elif askSize0 < askSize0_prev:
                self.cancellations_ask[turn] = 1
                self.cancellations_ask_sizes[turn] = askSize0_prev - askSize0

        elif askRate0 > askRate0_prev:
            # someone traded and consumed some orders up to new best ask
            self.trades_buy[turn] = 1
            self.trades_buy_sizes[turn] = np.sum(prev_ask_size[prev_ask_rate <= askRate0]) - askSize0

        elif askRate0 < askRate0_prev:
            # Post but below best ask
            self.posting_ask_cross[turn] = 1
            self.posting_ask_cross_sizes[turn] = np.sum(prev_bid_size[prev_bid_rate >= askRate0]) + askSize0

        ################################################
        ################################################

        if ((self.turn + 1) % 50000) == 0:
            self.model_expanding.fit(self.signals[0:turn_prev], self.y[0:turn_prev])

        if ((self.turn + 1) % 100000) == 0:
            self.model_running.fit(self.signals[max(turn_prev - 100000 + 1, 0):turn_prev], self.y[max(turn_prev - 100000 + 1, 0):turn_prev])

        is_reset = (self.turn == 0) or self.is_new_day()

        bidSize0_zscore = self.compute_z_score(bidSize0, 'bidSize0', 50, is_reset)
        askSize0_zscore = self.compute_z_score(askSize0, 'askSize0', 50, is_reset)

        bidSize012_zscore = self.compute_z_score(bidSize012, 'bidSize012', 50, is_reset)
        askSize012_zscore = self.compute_z_score(askSize012, 'askSize012', 50, is_reset)

        bidSizeTotal_zscore = self.compute_z_score(bidSizeTotal, 'bidSizeTotal', 20, is_reset)
        askSizeTotal_zscore = self.compute_z_score(askSizeTotal, 'askSizeTotal', 20, is_reset)

        midMic_zscore = self.compute_z_score(mid_mic, 'mid_mic', 10, is_reset)

        nbrTradesBid_zscore = self.compute_z_score(self.bid_nbr_trades[turn], 'nbrTradesBid', 10, is_reset)
        nbrTradesAsk_zscore = self.compute_z_score(self.ask_nbr_trades[turn], 'nbrTradesAsk', 10, is_reset)

        #average_price_bid = self.get_average_price_depth(20, False)
        #average_price_ask = self.get_average_price_depth(20, True)

        sig13_bid = self.posting_bid_sizes[turn] + self.posting_bid_cross_sizes[turn] - self.cancellations_bid_sizes[turn] - self.trades_sell_sizes[turn]
        sig13_ask = self.posting_ask_sizes[turn] + self.posting_ask_cross_sizes[turn] - self.cancellations_ask_sizes[turn] - self.trades_buy_sizes[turn]

        sig14_bid = self.posting_bid[turn] + self.posting_bid_cross[turn] - self.cancellations_bid[turn] - self.trades_sell[turn]
        sig14_ask = self.posting_ask[turn] + self.posting_ask_cross[turn] - self.cancellations_ask[turn] - self.trades_buy[turn]

        alpha = 2. / (10. + 1.)
        bias = (2. - alpha) / 2. / (1. - alpha)
        if is_reset:
            self.sig13_ewma = sig13_bid - sig13_ask
            self.sig14_ewma = sig14_bid - sig14_ask
            self.sig15_ewma = self.cancellations_bid_sizes[turn] - self.cancellations_ask_sizes[turn]
        else:
            self.sig13_ewma = (1. - alpha) * self.sig13_ewma + alpha * (sig13_bid - sig13_ask)
            self.sig14_ewma = (1. - alpha) * self.sig14_ewma + alpha * (sig14_bid - sig14_ask)
            self.sig15_ewma = (1. - alpha) * self.sig15_ewma + alpha * (self.cancellations_bid_sizes[turn] - self.cancellations_ask_sizes[turn])

        #### Signals ####
        self.sig1 = (bidSize0 - askSize0) / (bidSize0 + askSize0)
        self.sig2 = (bidSize1 - askSize1) / (bidSize1 + askSize1)
        self.sig3 = bidSize012_zscore - askSize012_zscore
        self.sig4 = bidSize1 / bidSize0 - askSize1 / askSize0
        self.sig5 = bidSize0_zscore - askSize0_zscore
        self.sig6 = bidSizeTotal_zscore - askSizeTotal_zscore
        self.sig7 = (askRate1 - askRate0) - (bidRate0 - bidRate1)
        self.sig8 = ((bidRate1 - bidRate2) - (askRate2 - askRate1)) / ((bidRate1 - bidRate2) + (askRate2 - askRate1))
        self.sig9 = midMic_zscore
        self.sig11 = nbrTradesBid_zscore - nbrTradesAsk_zscore
        #self.sig12 = ((mid - average_price_bid) - (average_price_ask - mid)) / ((mid - average_price_bid) + (average_price_ask - mid))
        self.sig13 = np.clip(self.sig13_ewma, -4., 4.)
        self.sig14 = self.sig14_ewma
        self.sig15 = np.clip(self.sig15_ewma, -4., 4.)
        #################


        signals = np.array([self.sig1, self.sig2, self.sig3, self.sig4, self.sig5, self.sig6, self.sig7, self.sig8, self.sig11, self.sig13, self.sig15])
        signals[np.isinf(signals)] = 0.
        signals[np.isnan(signals)] = 0.
        self.signals[turn, :] = signals
        self.prev_row = self.x

        return


    def get_prediction(self):
        signals = self.signals[self.turn:self.turn + 1, :]
        prediction_static = self.model_static.predict(signals)[0]
        #prediction_xgb = self.model_xgb.predict(signals)[0]

        if self.turn >= 50000:
            prediction_expanding = self.model_expanding.predict(signals)[0]
            prediction = 0.5 * (prediction_static + prediction_expanding)
        if self.turn >= 100000:
            prediction_expanding = self.model_expanding.predict(signals)[0]
            prediction_running = self.model_running.predict(signals)[0]
            #prediction = 0.3333 * (prediction_static + prediction_expanding + prediction_running)
            prediction = 0.5 * (prediction_static + prediction_expanding)
        else:
            prediction = prediction_static

        if not np.isfinite(prediction):
            prediction = 0.

        prediction = np.clip(prediction, -5., 5.)
        self.y_pred[self.turn] = prediction
        return prediction



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
