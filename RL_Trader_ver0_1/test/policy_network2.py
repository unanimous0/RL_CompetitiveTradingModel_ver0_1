import numpy
from keras.optimizers import sgd
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization

class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, l_rate=0.01):
        self.input_dim = input_dim
        self.l_rate = l_rate

        #LSTM
        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=(1,input_dim), return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5), BatchNormalization())
        # self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))

        self.model.compile(optimizer=sgd(lr=l_rate), loss='mse')
        self.prob = None