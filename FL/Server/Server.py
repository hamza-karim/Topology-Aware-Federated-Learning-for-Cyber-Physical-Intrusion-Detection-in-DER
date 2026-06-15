# Run on server device:
# docker exec -it <server_container> bash
# cd /app/src && python3 Server.py

import os
import csv
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import FitIns, FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

WINDOW_SIZE  = 30
NUM_FEATURES = 36

ZONE_NAMES = ['zone1', 'zone2', 'zone3', 'zone4']
ZONE_BUSES = {
    'zone1': set(range(1, 9)),
    'zone2': set(range(9, 18)),
    'zone3': set(range(18, 25)),
    'zone4': set(range(25, 33)),
}
ADMITTANCE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'zone_admittance.csv')

# Zones that host DER resources — server knows this from grid topology
DER_ZONES     = {'zone2', 'zone3', 'zone4'}   # zone1 has no DER (buses 1-8)
NON_DER_SCALE = 0.0    # non-DER zone 1 excluded from DER zone aggregations; DER zones share only with each other


def load_admittance_matrix(csv_path=ADMITTANCE_PATH):
    """Read zone_admittance.csv and return row-normalised inter-zone weight dict.
    W[zone_i][zone_j] = admittance(i,j) / sum_k admittance(i,k)  for j != i
    """
    df = pd.read_csv(csv_path)
    raw = {z: {z2: 0.0 for z2 in ZONE_NAMES} for z in ZONE_NAMES}
    for _, row in df.iterrows():
        fb  = int(row['from_bus'])
        tb  = int(row['to_bus'])
        adm = float(row['admittance'])
        fz  = next((z for z, buses in ZONE_BUSES.items() if fb in buses), None)
        tz  = next((z for z, buses in ZONE_BUSES.items() if tb in buses), None)
        if fz and tz and fz != tz:
            raw[fz][tz] += adm
            raw[tz][fz] += adm
    W = {}
    for z in ZONE_NAMES:
        total = sum(raw[z][z2] for z2 in ZONE_NAMES if z2 != z)
        W[z]  = {z2: raw[z][z2] / total for z2 in ZONE_NAMES if z2 != z}
    return W


def compute_der_aware_weights(W, der_zones=DER_ZONES, non_der_scale=NON_DER_SCALE):
    """Re-weight W for model aggregation: non-DER neighbours get a small fraction
    of their admittance-based weight, then rows are renormalised to sum to 1.

    This keeps the physical topology intact for inference (test_intact.py uses
    the original W) while reducing the FL influence of non-DER zones during
    model mixing.
    """
    W_der = {}
    for zi in W:
        scaled = {zj: w * (1.0 if zj in der_zones else non_der_scale)
                  for zj, w in W[zi].items()}
        total  = sum(scaled.values())
        W_der[zi] = {zj: v / total for zj, v in scaled.items()} if total > 0 else scaled
    return W_der


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
        strategy = prompt("Strategy [fedavg/fedprox/fedadam/intact]", default="fedavg").lower()
        if strategy in ("fedavg", "fedprox", "fedadam", "intact"):
            break
        print("  Choose 'fedavg', 'fedprox', 'fedadam', or 'intact'.")

    proximal_mu = 0.0
    server_eta  = 0.0
    alpha       = 0.5
    gamma       = 0.3
    if strategy == "fedprox":
        proximal_mu = prompt("Proximal mu", default=0.01, cast=float)
    elif strategy == "fedadam":
        server_eta = prompt("Server learning rate (eta)", default=0.01, cast=float)
    elif strategy == "intact":
        alpha = prompt("Self-retention alpha (0=full neighbour mix, 1=local only)", default=0.5, cast=float)
        gamma = prompt("Consistency penalty gamma (inference)", default=0.3, cast=float)

    model_dir = prompt("Model dir", default="/app/src/models")
    print("=" * 50)
    if strategy == "fedprox":
        print(f"  Strategy : FedProx  (mu={proximal_mu})")
    elif strategy == "fedadam":
        print(f"  Strategy : FedAdam  (eta={server_eta})")
    elif strategy == "intact":
        print(f"  Strategy : INTACT   (alpha={alpha}, gamma={gamma})")
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
        'alpha':        alpha,
        'gamma':        gamma,
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


class IntactStrategy(fl.server.strategy.FedAvg):
    """Admittance-weighted personalised federated aggregation (INTACT).

    Aggregation uses DER-aware weights (W_der) so non-DER zones contribute
    minimally to DER zone models.  Each zone also has its own alpha:
        theta_i_new = alpha_i * theta_i + (1-alpha_i) * sum_j( W_der[i,j] * theta_j )

    The original admittance W is saved to run_config for test_intact.py (inference
    consistency check uses physical topology, not the DER-aware variant).
    """

    def __init__(self, model_dir, local_epochs, alpha, gamma, W, W_der,
                 zone_alpha, **kwargs):
        super().__init__(**kwargs)
        self.model_dir    = model_dir
        self.local_epochs = local_epochs
        self.alpha        = alpha        # DER zone self-retention (for config saving)
        self.gamma        = gamma
        self.W            = W            # original admittance weights (inference use)
        self.W_der        = W_der        # DER-aware aggregation weights
        self.zone_alpha   = zone_alpha   # {zone_id: alpha_i}
        self.zone_weights = {}
        self.cid_to_zone  = {}
        self.round_losses = []

    # ── Round start: send each zone its own personalised model ────────────────
    def configure_fit(self, server_round, parameters, client_manager):
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)

        fit_ins_list = []
        for client in clients:
            zone_id = self.cid_to_zone.get(client.cid)
            if zone_id and zone_id in self.zone_weights:
                # From round 2 onwards send personalised model for this zone
                params = ndarrays_to_parameters(self.zone_weights[zone_id])
            else:
                # Round 1: everyone starts from the same initial parameters
                params = parameters
            fit_ins_list.append((client, FitIns(params, config)))
        return fit_ins_list

    # ── Round end: admittance-weighted mixing ─────────────────────────────────
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Collect raw weights and zone ids from each client
        zone_raw  = {}
        zone_loss = {}
        for proxy, fit_res in results:
            zid = fit_res.metrics.get('zone_id')
            if not zid:
                continue
            self.cid_to_zone[proxy.cid] = zid
            zone_raw[zid]  = parameters_to_ndarrays(fit_res.parameters)
            zone_loss[zid] = float(fit_res.metrics.get('loss', 0.0))

        # DER-aware mixing: theta_i_new = alpha_i*theta_i + (1-alpha_i)*sum_j(W_der[i,j]*theta_j)
        # W_der down-weights non-DER neighbours; zone_alpha is per-zone.
        new_weights = {}
        for zid, own in zone_raw.items():
            a_i = self.zone_alpha.get(zid, self.alpha)
            neighbour_mix = None
            for other_zone, w in self.W_der.get(zid, {}).items():
                if other_zone not in zone_raw:
                    continue
                scaled = [w * layer for layer in zone_raw[other_zone]]
                neighbour_mix = (scaled if neighbour_mix is None
                                 else [m + s for m, s in zip(neighbour_mix, scaled)])

            if neighbour_mix is None:
                new_weights[zid] = own
            else:
                new_weights[zid] = [
                    a_i * o + (1.0 - a_i) * m
                    for o, m in zip(own, neighbour_mix)
                ]

        self.zone_weights = new_weights

        # Save per-round and latest final weights for each zone
        os.makedirs(self.model_dir, exist_ok=True)
        for zid, weights in new_weights.items():
            np.savez(os.path.join(self.model_dir,
                                  f'intact_{zid}_round_{server_round}_weights.npz'), *weights)
            np.savez(os.path.join(self.model_dir,
                                  f'intact_{zid}_final_weights.npz'), *weights)

        avg_loss = sum(zone_loss.values()) / len(zone_loss) if zone_loss else 0.0
        self.round_losses.append((server_round, avg_loss))
        print(f"Round {server_round:>3} | Zones: {sorted(new_weights)} | "
              f"Avg Loss: {avg_loss:.6f}", flush=True)

        # Return global average so Flower's evaluate phase still works
        all_w    = list(new_weights.values())
        global_w = [np.mean([w[i] for w in all_w], axis=0) for i in range(len(all_w[0]))]
        return ndarrays_to_parameters(global_w), {'avg_loss': avg_loss}


def main():
    cfg = get_user_config()

    if cfg['strategy'] == 'fedavg':
        prefix = 'fedavg'
    elif cfg['strategy'] == 'fedprox':
        prefix = f"fedprox_{cfg['proximal_mu']}"
    elif cfg['strategy'] == 'fedadam':
        prefix = f"fedadam_{cfg['server_eta']}"
    else:
        prefix = 'intact'

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

    if cfg['strategy'] == 'intact':
        print("Loading admittance matrix...", flush=True)
        W     = load_admittance_matrix()
        W_der = compute_der_aware_weights(W)

        # Zone-specific alpha: non-DER zone 1 stays more local
        der_alpha      = cfg['alpha']
        non_der_alpha  = min(der_alpha + 0.15, 0.99)
        zone_alpha     = {z: (der_alpha if z in DER_ZONES else non_der_alpha)
                          for z in ZONE_NAMES}

        print("  Admittance weights (physical, used at inference):", flush=True)
        for z, neighbours in W.items():
            print(f"    {z}: " + ", ".join(f"{n}={v:.3f}" for n, v in neighbours.items()),
                  flush=True)
        print("  DER-aware aggregation weights (non-DER scaled by "
              f"{NON_DER_SCALE}):", flush=True)
        for z, neighbours in W_der.items():
            print(f"    {z}: " + ", ".join(f"{n}={v:.3f}" for n, v in neighbours.items()),
                  flush=True)
        print("  Zone alpha:", flush=True)
        for z, a in zone_alpha.items():
            der_tag = "(DER)" if z in DER_ZONES else "(non-DER)"
            print(f"    {z} {der_tag}: alpha={a}", flush=True)

        strategy = IntactStrategy(
            model_dir=cfg['model_dir'],
            local_epochs=cfg['local_epochs'],
            alpha=cfg['alpha'],
            gamma=cfg['gamma'],
            W=W,
            W_der=W_der,
            zone_alpha=zone_alpha,
            **common_kwargs,
        )
    elif cfg['strategy'] == 'fedadam':
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
        run_cfg = {
            'strategy':     cfg['strategy'],
            'fl_rounds':    cfg['rounds'],
            'local_epochs': cfg['local_epochs'],
            'num_clients':  cfg['min_clients'],
            'proximal_mu':  cfg['proximal_mu'],
            'server_eta':   cfg['server_eta'],
            'alpha':        cfg['alpha'],
            'gamma':        cfg['gamma'],
        }
        if cfg['strategy'] == 'intact':
            run_cfg['non_der_scale'] = NON_DER_SCALE
            run_cfg['zone_alpha']    = zone_alpha
        with open(run_cfg_path, 'w') as f:
            json.dump(run_cfg, f, indent=2)
        print(f"Run config saved to {run_cfg_path}", flush=True)


if __name__ == '__main__':
    main()
