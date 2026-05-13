# Automated (via start_clients.sh — set env vars, no prompts):
#   docker run -d -e ZONE_ID=zone1 -e SERVER_ADDRESS=192.168.1.100:8080 ...
#
# Manual (interactive fallback):
#   docker exec -it flwr-client-zone1 bash
#   cd /app/src && python3 Client.py

import os
import re
import sys
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


def env_or_prompt(env_key, question, default=None, cast=str, choices=None):
    """Return env var if set and valid, else prompt (or use default if non-interactive)."""
    raw = os.environ.get(env_key)
    if raw is not None:
        try:
            val = cast(raw)
            if choices and val not in choices:
                raise ValueError(f"'{raw}' not in {choices}")
            print(f"  {question}: {val}  (env ${env_key})", flush=True)
            return val
        except (ValueError, TypeError) as e:
            print(f"  WARNING: invalid ${env_key}='{raw}' ({e}), ignoring.", flush=True)

    # Non-interactive (detached container): use default or exit
    if not sys.stdin.isatty():
        if default is not None:
            return default
        sys.exit(f"ERROR: ${env_key} must be set when running non-interactively (no TTY).")

    return prompt(question, default=default, cast=cast, choices=choices)


def get_user_config():
    print("=" * 50, flush=True)
    print("  FL Client Configuration", flush=True)
    print("=" * 50, flush=True)

    cfg = {
        'zone_id':        env_or_prompt('ZONE_ID',        'Zone ID',
                                         choices=list(ZONE_BUSES.keys())),
        'server_address': env_or_prompt('SERVER_ADDRESS', 'Server address (host:port)',
                                         default='127.0.0.1:8080'),
        'data_path':      env_or_prompt('DATA_PATH',      'Data path',
                                         default='/app/src/data/centralized_train_combined.csv'),
        'model_dir':      env_or_prompt('MODEL_DIR',      'Model dir',
                                         default='/app/src/models'),
        'local_epochs':   env_or_prompt('LOCAL_EPOCHS',   'Local epochs per round',
                                         default=5, cast=int),
        'batch_size':     env_or_prompt('BATCH_SIZE',     'Batch size',
                                         default=32, cast=int),
        'window_size':    env_or_prompt('WINDOW_SIZE',    'Window size',
                                         default=30, cast=int),
    }

    print("=" * 50, flush=True)
    print(flush=True)
    return cfg


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
    model.compile(optimizer=Adam(learning_rate=0.005730), loss='mse')
    return model


def fit_with_proximal(model, x_train, global_weights, proximal_mu, local_epochs, batch_size):
    """FedProx local training: MSE + (mu/2)*||w - w_global||^2.
    Uses model.fit() with a custom loss so memory usage matches FedAvg.
    """
    global_weights_t = [tf.constant(w, dtype=tf.float32) for w in global_weights]

    def proximal_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        prox = tf.add_n([
            tf.reduce_sum(tf.square(w - g))
            for w, g in zip(model.trainable_variables, global_weights_t)
        ])
        return mse + (proximal_mu / 2.0) * prox

    model.compile(optimizer=model.optimizer, loss=proximal_loss)
    model.fit(x_train, x_train, epochs=local_epochs, batch_size=batch_size, verbose=0)

    # Recompile with standard MSE and return pure MSE loss for consistent reporting
    model.compile(optimizer=model.optimizer, loss='mse')
    loss = float(model.evaluate(x_train, x_train, batch_size=batch_size, verbose=0))
    return loss


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cfg, x_train):
        self.zone_id       = cfg['zone_id']
        self.batch_size    = cfg['batch_size']
        self.local_epochs  = cfg['local_epochs']
        self.x_train       = x_train
        self.model         = build_model(cfg['window_size'])
        self.current_round = 0

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.current_round = config.get('server_round', self.current_round + 1)
        local_epochs  = config.get('local_epochs', self.local_epochs)
        proximal_mu   = float(config.get('proximal_mu', 0.0))

        if proximal_mu > 0.0:
            global_weights = [w.copy() for w in self.model.get_weights()]
            loss = fit_with_proximal(
                self.model, self.x_train, global_weights,
                proximal_mu, local_epochs, self.batch_size,
            )
            print(f"Round {self.current_round} | Zone {self.zone_id} | "
                  f"FedProx (mu={proximal_mu}) Loss: {loss:.6f}", flush=True)
        else:
            history = self.model.fit(
                self.x_train, self.x_train,
                epochs=local_epochs,
                batch_size=self.batch_size,
                verbose=0,
            )
            loss = float(history.history['loss'][-1])
            print(f"Round {self.current_round} | Zone {self.zone_id} | "
                  f"FedAvg Loss: {loss:.6f}", flush=True)

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
    joblib.dump(scaler, os.path.join(cfg['model_dir'], f"fl_{cfg['zone_id']}_scaler.pkl"))

    data = pad_to_n(data)
    return create_windows(data, cfg['window_size'])


def main():
    cfg = get_user_config()

    print(f"Loading data for {cfg['zone_id']}...", flush=True)
    x_train = load_data(cfg)
    print(f"Ready: {len(x_train)} windows, shape {x_train.shape}", flush=True)
    print(flush=True)

    fl.client.start_numpy_client(
        server_address=cfg['server_address'],
        client=FlowerClient(cfg=cfg, x_train=x_train),
    )


if __name__ == '__main__':
    main()
