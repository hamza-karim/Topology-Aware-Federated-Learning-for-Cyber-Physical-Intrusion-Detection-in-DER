# Run on server device:
# docker exec -it <server_container> bash
# cd /app/src && python3 Server.py

import os
import csv
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


def prompt(question, default=None, cast=int):
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
    port         = prompt("Port", default=8080)
    rounds       = prompt("FL rounds", default=10)
    min_clients  = prompt("Min clients per round", default=4)
    local_epochs = prompt("Local epochs (sent to clients)", default=5)
    model_dir    = prompt("Model dir", default="/app/src/models", cast=str)
    print("=" * 50)
    print()
    return {
        'port':         port,
        'rounds':       rounds,
        'min_clients':  min_clients,
        'local_epochs': local_epochs,
        'model_dir':    model_dir,
    }


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_dir, local_epochs, **kwargs):
        super().__init__(**kwargs)
        self.model_dir    = model_dir
        self.local_epochs = local_epochs
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
                os.path.join(self.model_dir, f'fedavg_round_{server_round}_weights.npz'),
                *weights,
            )
            np.savez(os.path.join(self.model_dir, 'fedavg_final_weights.npz'), *weights)

            total_examples = sum(r.num_examples for _, r in results)
            avg_loss = (
                sum(r.metrics.get('loss', 0.0) * r.num_examples for _, r in results)
                / total_examples if total_examples > 0 else 0.0
            )
            self.round_losses.append((server_round, avg_loss))
            print(f"Round {server_round:>3} | Clients: {len(results)} | "
                  f"Aggregated Loss: {avg_loss:.6f}", flush=True)

        return aggregated_parameters, aggregated_metrics


def main():
    cfg = get_user_config()

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        return {'server_round': server_round, 'local_epochs': cfg['local_epochs']}

    strategy = SaveModelStrategy(
        model_dir=cfg['model_dir'],
        local_epochs=cfg['local_epochs'],
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cfg['min_clients'],
        min_evaluate_clients=cfg['min_clients'],
        min_available_clients=cfg['min_clients'],
        on_fit_config_fn=fit_config,
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{cfg['port']}",
        config=fl.server.ServerConfig(num_rounds=cfg['rounds']),
        strategy=strategy,
    )

    if strategy.round_losses:
        os.makedirs(cfg['model_dir'], exist_ok=True)
        log_path = os.path.join(cfg['model_dir'], 'fedavg_training_log.csv')
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'loss'])
            writer.writerows(strategy.round_losses)
        print(f"Training log saved to {log_path}", flush=True)


if __name__ == '__main__':
    main()
