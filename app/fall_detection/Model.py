from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed

def create_model(timesteps=75, n_features=38):
    model = Sequential()
   
    # ENCODER
    model.add(LSTM(128, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    # -------
    # BOTTLENECK
    model.add(LSTM(64, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    #model.add(RepeatVector(timesteps))
    # -------
    # DECODER
    model.add(LSTM(128, activation='relu', return_sequences=True))
    # -------
    
    # Adjusting data shape
    model.add(TimeDistributed(Dense(n_features)))
    # -------TimeDistributed

    #optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    return model
    
