# Run on server device:
# docker exec -it <server_container> bash
# cd /app/src && python3 Server.py

import os
import csv
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

WINDOW_SIZE  = 30
NUM_FEATURES = 36


def prompt(question, default=None, cast=str):
    hint = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{question}{hint}: ").strip()
        value = raw if raw else (str(default) if default is not None else "")
        if not value:
            print("  This field is required.")
            continue
        try:
            return cast(value)
        except (ValueError, TypeError):
            print(f"  Expected {cast.__name__}.")


def get_user_config():
    print("=" * 50)
    print("  FL Server Configuration")
    print("=" * 50)
    port         = prompt("Port", default=8080, cast=int)
    rounds       = prompt("FL rounds", default=20, cast=int)
    min_clients  = prompt("Min clients per round", default=4, cast=int)
    local_epochs = prompt("Local epochs (sent to clients)", default=3, cast=int)

    while True:
        strategy = prompt("Strategy [fedavg/fedprox/fedadam]", default="fedavg").lower()
        if strategy in ("fedavg", "fedprox", "fedadam"):
            break
        print("  Choose 'fedavg', 'fedprox', or 'fedadam'.")

    proximal_mu = 0.0
    server_eta  = 0.0
    if strategy == "fedprox":
        proximal_mu = prompt("Proximal mu", default=0.01, cast=float)
    elif strategy == "fedadam":
        server_eta = prompt("Server learning rate (eta)", default=0.01, cast=float)

    model_dir = prompt("Model dir", default="/app/src/models")
    print("=" * 50)
    if strategy == "fedprox":
        print(f"  Strategy : FedProx  (mu={proximal_mu})")
    elif strategy == "fedadam":
        print(f"  Strategy : FedAdam  (eta={server_eta})")
    else:
        print(f"  Strategy : FedAvg")
    print()
    return {
        'port':         port,
        'rounds':       rounds,
        'min_clients':  min_clients,
        'local_epochs': local_epochs,
        'strategy':     strategy,
        'proximal_mu':  proximal_mu,
        'server_eta':   server_eta,
        'model_dir':    model_dir,
    }


def build_initial_parameters() -> Parameters:
    """Build model with the same architecture as clients and return initial weights."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
    from tensorflow.keras.optimizers import Adam
    inp = Input(shape=(WINDOW_SIZE, NUM_FEATURES))
    x   = LSTM(32, activation='tanh', return_sequences=True)(inp)
    x   = LSTM(64, activation='tanh', return_sequences=False)(x)
    x   = RepeatVector(WINDOW_SIZE)(x)
    x   = LSTM(64, activation='tanh', return_sequences=True)(x)
    x   = LSTM(32, activation='tanh', return_sequences=True)(x)
    x   = TimeDistributed(Dense(NUM_FEATURES))(x)
    model = Model(inp, x)
    model.compile(optimizer=Adam(learning_rate=0.005730), loss='mse')
    return ndarrays_to_parameters(model.get_weights())


def make_save_strategy(base_cls):
    """Wrap any Flower strategy base class with per-round weight saving and loss logging."""
    class SaveModelStrategy(base_cls):
        def __init__(self, model_dir, local_epochs, strategy='fedavg', **kwargs):
            super().__init__(**kwargs)
            self.model_dir    = model_dir
            self.local_epochs = local_epochs
            self.prefix       = strategy
            self.round_losses = []

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )

            if aggregated_parameters is not None:
                os.makedirs(self.model_dir, exist_ok=True)
                weights = parameters_to_ndarrays(aggregated_parameters)

                np.savez(
                    os.path.join(self.model_dir,
                                 f'{self.prefix}_round_{server_round}_weights.npz'),
                    *weights,
                )
                np.savez(
                    os.path.join(self.model_dir, f'{self.prefix}_final_weights.npz'),
                    *weights,
                )

                total_examples = sum(r.num_examples for _, r in results)
                avg_loss = (
                    sum(r.metrics.get('loss', 0.0) * r.num_examples for _, r in results)
                    / total_examples if total_examples > 0 else 0.0
                )
                self.round_losses.append((server_round, avg_loss))
                print(f"Round {server_round:>3} | Clients: {len(results)} | "
                      f"Aggregated Loss: {avg_loss:.6f}", flush=True)

            return aggregated_parameters, aggregated_metrics

    return SaveModelStrategy


def main():
    cfg = get_user_config()

    if cfg['strategy'] == 'fedavg':
        prefix = 'fedavg'
    elif cfg['strategy'] == 'fedprox':
        prefix = f"fedprox_{cfg['proximal_mu']}"
    else:
        prefix = f"fedadam_{cfg['server_eta']}"

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        return {
            'server_round': server_round,
            'local_epochs': cfg['local_epochs'],
            'proximal_mu':  cfg['proximal_mu'],
        }

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cfg['min_clients'],
        min_evaluate_clients=cfg['min_clients'],
        min_available_clients=cfg['min_clients'],
        on_fit_config_fn=fit_config,
    )

    if cfg['strategy'] == 'fedadam':
        print("Building initial model parameters for FedAdam...", flush=True)
        SaveModelStrategy = make_save_strategy(fl.server.strategy.FedAdam)
        strategy = SaveModelStrategy(
            model_dir=cfg['model_dir'],
            local_epochs=cfg['local_epochs'],
            strategy=prefix,
            initial_parameters=build_initial_parameters(),
            eta=cfg['server_eta'],
            eta_l=0.005730,
            beta_1=0.9,
            beta_2=0.999,
            tau=1e-9,
            **common_kwargs,
        )
    else:
        SaveModelStrategy = make_save_strategy(fl.server.strategy.FedAvg)
        strategy = SaveModelStrategy(
            model_dir=cfg['model_dir'],
            local_epochs=cfg['local_epochs'],
            strategy=prefix,
            **common_kwargs,
        )

    fl.server.start_server(
        server_address=f"0.0.0.0:{cfg['port']}",
        config=fl.server.ServerConfig(num_rounds=cfg['rounds']),
        strategy=strategy,
    )

    if strategy.round_losses:
        os.makedirs(cfg['model_dir'], exist_ok=True)

        log_path = os.path.join(cfg['model_dir'], f'{prefix}_training_log.csv')
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'loss'])
            writer.writerows(strategy.round_losses)
        print(f"Training log saved to {log_path}", flush=True)

        run_cfg_path = os.path.join(cfg['model_dir'], f'{prefix}_run_config.json')
        with open(run_cfg_path, 'w') as f:
            json.dump({
                'strategy':     cfg['strategy'],
                'fl_rounds':    cfg['rounds'],
                'local_epochs': cfg['local_epochs'],
                'num_clients':  cfg['min_clients'],
                'proximal_mu':  cfg['proximal_mu'],
                'server_eta':   cfg['server_eta'],
            }, f, indent=2)
        print(f"Run config saved to {run_cfg_path}", flush=True)


if __name__ == '__main__':
    main()
