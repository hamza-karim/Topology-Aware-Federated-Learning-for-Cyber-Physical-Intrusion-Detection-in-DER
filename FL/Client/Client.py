# Run on each client device:
# docker exec -it <client_container> bash
# cd /app/src && python3 Client.py

import os
import re
import numpy as np
import pandas as pd
import joblib
import flwr as fl
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

NUM_FEATURES = 36

ZONE_BUSES = {
    'zone1': range(1, 9),    # buses 1-8,   32 features
    'zone2': range(9, 18),   # buses 9-17,  36 features
    'zone3': range(18, 25),  # buses 18-24, 28 features
    'zone4': range(25, 33),  # buses 25-32, 32 features
}


def prompt(question, default=None, cast=str, choices=None):
    hint = f" [{default}]" if default is not None else ""
    if choices:
        hint = f" {choices}{hint}"
    while True:
        raw = input(f"{question}{hint}: ").strip()
        value = raw if raw else (str(default) if default is not None else "")
        if not value:
            print("  This field is required.")
            continue
        try:
            value = cast(value)
        except (ValueError, TypeError):
            print(f"  Expected {cast.__name__}.")
            continue
        if choices and value not in choices:
            print(f"  Choose one of {choices}.")
            continue
        return value


def get_user_config():
    print("=" * 50)
    print("  FL Client Configuration")
    print("=" * 50)
    zone_id        = prompt("Zone ID", choices=list(ZONE_BUSES.keys()))
    server_address = prompt("Server address (host:port)", default="127.0.0.1:8080")
    data_path      = prompt("Data path", default="/app/src/data/centralized_train_combined.csv")
    model_dir      = prompt("Model dir", default="/app/src/models")
    local_epochs   = prompt("Local epochs per round", default=5, cast=int)
    batch_size     = prompt("Batch size", default=32, cast=int)
    window_size    = prompt("Window size", default=30, cast=int)
    print("=" * 50)
    print()
    return {
        'zone_id':        zone_id,
        'server_address': server_address,
        'data_path':      data_path,
        'model_dir':      model_dir,
        'local_epochs':   local_epochs,
        'batch_size':     batch_size,
        'window_size':    window_size,
    }


def get_zone_columns(columns, zone_id):
    buses = ZONE_BUSES[zone_id]
    pattern = re.compile(r'_bus(' + '|'.join(str(b) for b in buses) + r')$')
    return [col for col in columns if pattern.search(col)]


def pad_to_n(data, n=NUM_FEATURES):
    if data.shape[1] < n:
        pad = np.zeros((data.shape[0], n - data.shape[1]), dtype=data.dtype)
        return np.concatenate([data, pad], axis=1)
    return data


def create_windows(data, window_size):
    return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)])


def build_model(window_size):
    inp = Input(shape=(window_size, NUM_FEATURES))
    x = LSTM(32, activation='tanh', return_sequences=True)(inp)
    x = LSTM(64, activation='tanh', return_sequences=False)(x)
    x = RepeatVector(window_size)(x)
    x = LSTM(64, activation='tanh', return_sequences=True)(x)
    x = LSTM(32, activation='tanh', return_sequences=True)(x)
    x = TimeDistributed(Dense(NUM_FEATURES))(x)
    model = Model(inp, x)
    model.compile(optimizer=Adam(learning_rate=0.0011), loss='mse')
    return model


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cfg, x_train):
        self.zone_id      = cfg['zone_id']
        self.batch_size   = cfg['batch_size']
        self.local_epochs = cfg['local_epochs']
        self.x_train      = x_train
        self.model        = build_model(cfg['window_size'])
        self.current_round = 0

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.current_round = config.get('server_round', self.current_round + 1)
        local_epochs = config.get('local_epochs', self.local_epochs)

        history = self.model.fit(
            self.x_train, self.x_train,
            epochs=local_epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        loss = float(history.history['loss'][-1])
        print(f"Round {self.current_round} | Zone {self.zone_id} | Loss: {loss:.6f}", flush=True)
        return self.get_parameters(config={}), len(self.x_train), {'loss': loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = float(self.model.evaluate(
            self.x_train, self.x_train, batch_size=self.batch_size, verbose=0
        ))
        return loss, len(self.x_train), {'loss': loss}


def load_data(cfg):
    df = pd.read_csv(cfg['data_path'])
    drop_cols = [c for c in df.columns if 'timestamp' in c.lower() or c.lower() == 'time']
    if drop_cols:
        df = df.drop(columns=drop_cols)

    zone_cols = get_zone_columns(list(df.columns), cfg['zone_id'])
    data = df[zone_cols].values.astype(np.float32)

    os.makedirs(cfg['model_dir'], exist_ok=True)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    joblib.dump(scaler, os.path.join(cfg['model_dir'], f"fedavg_{cfg['zone_id']}_scaler.pkl"))

    data = pad_to_n(data)
    return create_windows(data, cfg['window_size'])


def main():
    cfg = get_user_config()

    print(f"Loading data for {cfg['zone_id']}...", flush=True)
    x_train = load_data(cfg)
    print(f"Ready: {len(x_train)} windows, shape {x_train.shape}", flush=True)
    print()

    fl.client.start_numpy_client(
        server_address=cfg['server_address'],
        client=FlowerClient(cfg=cfg, x_train=x_train),
    )


if __name__ == '__main__':
    main()
