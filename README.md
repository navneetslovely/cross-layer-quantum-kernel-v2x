# A Cross-Layer Quantum Kernel for Misbehaviour Detection in Vehicular Networks

Code, configuration and result files for the paper **"A Cross-Layer Quantum Kernel for
Misbehaviour Detection in Vehicular Networks"** (under review): the SUMO + ns-3.42 802.11p
co-simulation testbed, the cross-layer quantum-kernel detector and the classical baselines,
the experiment and analysis scripts, and the CSV/JSON result files behind the tables and
figures of the paper.

No credentials are included. The optional IBM Quantum hardware runs need your own token
([Section 7](#7-ibm-quantum-hardware-runs)).

---

## 1. Repository layout

```
.
├── src/qkv2x/              library: co-simulation driver, features, feature maps, kernel training, baselines, CLI
├── configs/                experiment, simulation, feature and model YAML files
├── cosim/                  SUMO scenario + ns-3.42 802.11p module (the testbed)
├── scripts/                dataset fetchers, experiment drivers, figure scripts
│   ├── studies/            additional studies (Section 5)
│   └── gpu_engine/         torch statevector engine for the large Gram batches and qubit scaling
├── full_paper_data/        result files behind the main tables and figures (Section 6)
├── results/                result files of the additional studies, trained parameters, hardware runs
├── requirements.txt        pinned Python dependencies (full lock: requirements-lock.txt; conda: environment.yml)
├── env.sh                  conda env + SUMO_HOME / NS3_DIR for the co-simulation (edit the paths)
└── run_paper.sh            regenerates the data figures from full_paper_data/ and results/
```

---

## 2. Setup

Python 3.11; the quantum kernels are statevector-simulated with PennyLane on CPU (no GPU needed,
except optionally for the qubit-scaling study).

```bash
conda create -n qkv2x python=3.11 -y
conda activate qkv2x
pip install -r requirements.txt      # or requirements-lock.txt for the exact lock
pip install -e .
```

**Regenerate the paper figures** (no datasets needed; output in `figures/`):

```bash
bash run_paper.sh     # Fig. 2 (ROC/PR), Fig. 3 (CAN datasets), Fig. 4 (separability ordering)
```

Fig. 1 is a schematic and has no data.

---

## 3. Datasets

The external datasets are not redistributed. Download each into the target directory shown.

| Dataset | Used for | Source | Fetch | Target |
|---|---|---|---|---|
| `dev_grid` (primary) | all main results | **generated** by the co-simulation (Section 4.1) | — | `runs/dev_grid/` |
| VeReMi (van der Heijden et al., 2018) | external V2X evaluation, separability | https://github.com/VeReMi-dataset/VeReMi (release tarballs) | `bash scripts/fetch_veremi.sh` | `data/veremi/original/` |
| VeReMi NextGen (Hermann et al., 2026) | external V2X evaluation, mimicry, hardware | Zenodo record 19665762 (v1.0.0) | `bash scripts/fetch_veremi_nextgen.sh` | `data/veremi_nextgen/` |
| ROAD (Verma et al., 2024) | CAN transfer (Sec. VII-C) | Zenodo record 10462796 (`road.zip`, MD5 `cab184cfc2fe12c0834bc46188c0f330`) | `bash scripts/datasets/fetch_road.sh` | `data/road/` |
| HCRL Car-Hacking (Seo et al., 2018) | CAN transfer (Sec. VII-C) | https://ocslab.hksecurity.net/Datasets/car-hacking-dataset | `bash scripts/datasets/fetch_hcrl_carhacking.sh` | `data/hcrl_carhacking/` |
| CIC-IoV-2024 (Neto et al., 2024) | CAN transfer (Sec. VII-C) | https://www.unb.ca/cic/datasets/iov-dataset-2024.html (free, form-gated) | manual; `bash scripts/datasets/fetch_cic_iov_2024.sh` checks the files | `data/cic_iov_2024/` |

Notes:
- **VeReMi**: extract the tarballs so that every simulation directory (with a `GroundTruth*.json`)
  sits under `data/veremi/original/`; the loader `src/qkv2x/data/veremi.py` discovers them. The
  per-receiver logs carry RSSI, which the network register uses.
- **VeReMi NextGen**: the fetch script downloads the default scenario/attack subset (~1.7 GB) into
  `data/veremi_nextgen/dl/`; unzip into `data/veremi_nextgen/extracted/<scenario>/<subset>/<Split>/`.
  NextGen has no RSSI, so the pipeline runs in `network_mode: timing`.
- **HCRL**: place `DoS_dataset.csv`, `Fuzzy_dataset.csv`, `gear_dataset.csv`, `RPM_dataset.csv` and
  `normal_run_data.txt` directly in the target directory.
- **CIC-IoV-2024**: keep the `decimal/`, `hexadecimal/` and `binary/` subfolders.

The windowed model inputs of `dev_grid` and NextGen are included in
`full_paper_data/sequences/` (Section 6), so the feature tables can be inspected without running
the co-simulation.

---

## 4. Reproducing the experiments

### 4.1 Co-simulation toolchain and `dev_grid`

`scripts/install_linux.sh` (Ubuntu / WSL) or `scripts/install_macos.sh` performs the following;
the manual steps are:

```bash
# system packages (Ubuntu)
sudo apt-get install -y git cmake g++ ninja-build python3-dev wget libxerces-c-dev \
  libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev libeigen3-dev swig pkg-config
# SUMO 1.20 from source (the pip package lacks the libtraci C++ library the ns-3 module links against)
git clone --depth 1 --branch v1_20_0 https://github.com/eclipse-sumo/sumo.git && cd sumo
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_DISABLE_FIND_PACKAGE_GDAL=TRUE
cmake --build build -j"$(nproc)"; export SUMO_HOME="$PWD"; export PATH="$SUMO_HOME/bin:$PATH"; cd ..
# ns-3.42 with this repository's module attached
wget https://www.nsnam.org/releases/ns-allinone-3.42.tar.bz2 && tar -xjf ns-allinone-3.42.tar.bz2
export NS3_DIR="$PWD/ns-allinone-3.42/ns-3.42"
ln -s "<this repo>/cosim/ns3/contrib/qkv2x" "$NS3_DIR/contrib/qkv2x"
cd "$NS3_DIR" && ./ns3 configure --enable-examples --build-profile=optimized && ./ns3 build qkv2x-cosim
./ns3 run "qkv2x-cosim -- --help"      # should list --sumoCfg, --txPowerDbm, --mixSpec, ...
```

Put the two paths into `env.sh`, then:

```bash
source env.sh
python -m qkv2x.cli cosim    --config configs/sim/dev_grid.yaml   # SUMO + ns-3 -> runs/dev_grid/
python -m qkv2x.cli features --config configs/sim/dev_grid.yaml   # -> windowed cross-layer features
```

The attack-magnitude, Sybil transmit-power, flooding and end-to-end DoS-mimicry variants use
`configs/sim/intensity_*.yaml`, `sybil_ablation.yaml`, `dev_grid_flooding.yaml` and
`dev_grid_dos_mimicry.yaml` (see `scripts/intensity_run.sh` for the cosim -> features ->
experiment sequence).

### 4.2 Main experiments

Each command writes to `results/<name>/`.

```bash
python -m qkv2x.cli experiment --config configs/experiments/packetloss.yaml      # Table III (loss 0), packet loss
python -m qkv2x.cli experiment --config configs/experiments/crossattack.yaml     # held-out-attack detectability Y (Sec. VII-D)
python -m qkv2x.cli experiment --config configs/experiments/sybil_ablation.yaml  # Sybil transmit-power modes
python -m qkv2x.cli experiment --config configs/experiments/intensity.yaml       # attack magnitude (also _drift, _datafalsification)
python -m qkv2x.cli experiment --config configs/experiments/veremi.yaml          # VeReMi
python -m qkv2x.cli experiment --config configs/experiments/veremi_nextgen.yaml  # VeReMi NextGen
python scripts/dev_multiseed.py           # three leakage-safe splits (checkpoints reused by Sec. 5 studies)
python scripts/multiseed_baselines.py     # NextGen secondary baselines over three seeds
python scripts/feature_ablation.py        # mobility-only / network-only blocks (Table IV)
python scripts/controllability_tags.py    # controllable vs non-controllable feature tags
python scripts/controllability.py         # D_full and Y per attack family (Fig. 4)
python scripts/separability_grouped.py    # transmitter-grouped divergence
python scripts/magnitude_axis.py          # magnitude monotonicity over three attack families
python scripts/crossreceiver_fusion.py    # cross-receiver fusion
python scripts/mimicry_evasion.py         # white-box mimicry (Sec. VIII-A)
python scripts/nystrom_scale.py           # Nystrom landmarks, N up to 20,000 (Sec. VII-B)
python scripts/qubit_scaling.py           # n = 4 ... 20 qubits (Sec. VII-B; large n via scripts/gpu_engine/)
python scripts/road_ids.py                # CAN: ROAD
python scripts/can_ids.py                 # CAN: HCRL Car-Hacking, CIC-IoV-2024
python scripts/roc_pr_curves.py           # Fig. 2 curve data (dev_grid packetloss + NextGen predictions)
python scripts/export_sequences.py        # windowed model inputs -> full_paper_data/sequences/
```

### 4.3 Hyperparameters

The quantum feature map (8 qubits as 4+4, L = 2, bipartite couplings), the alignment schedule and the
SVM grid are in `configs/quantum.yaml`; the classical baseline grids in `configs/baselines.yaml`; the
feature definitions in `configs/features*.yaml`. The CAN experiments read the same files.

---

## 5. Additional studies (`scripts/studies/`, results in `results/studies/`)

These run on the outputs of Section 4. Each writes a `*_summary.json` whose fields are the numbers
quoted in the paper, plus the data CSVs.

| Script | What it computes | Paper | Result files |
|---|---|---|---|
| `divergence.py`, `correlations.py`, `per_model_correlation.py`, `veremi_separability.py` | grouped cross-fitted divergence, per-dataset and per-model correlations, VeReMi as third dataset, Stouffer combination | Sec. VII-D | `divergence_*`, `correlations*`, `per_model_*`, `veremi_*`, `three_dataset_correlations.json` |
| `entanglement_ablation.py`, `kta_restarts.py`, `domain_matched_kernel.py` | topology x training ablation, five alignment restarts, classical block kernels on identical inputs | Sec. VII-E, Table IV | `entanglement_ablation_*`, `kta_restarts*`, `domain_kernel*`, `results/trained_parameters/` |
| `three_splits.py` | three-split reporting | Table V | `three_splits*` |
| `kta_equivalence.py`, `baseline_fairness.py`, `tuning_budget.py`, `param_sensitivity.py` | centred alignment, hyperparameter inventory, tuning budgets, one-at-a-time parameter sensitivity | Sec. IV-C, VII-A | `kta_equivalence*`, `baseline_fairness*`, `baseline_comparisons.csv`, `hyperparams.csv`, `tuning_budget*`, `param_sensitivity*`, `pca_variance.csv` |
| `threshold_protocol.py` | validation-selected thresholds | Sec. VII-A | `threshold_protocol_summary.json`, `thresholds.csv` |
| `inference_latency.py` | per-window inference cost | Sec. VII-B | `inference_latency.json` |
| `can_seeds.py`, `fig_crossdataset.py` | CAN transfer over three seeds; Fig. 3 | Sec. VII-C, Fig. 3 | `can_seeds*`, `can_seeds/` |
| `label_rule_sensitivity.py`, `packetloss_retention.py`, `fixed_span_loss.py` | label rule, packet-loss retention, fixed one-second spans | Sec. VII-F | `label_rule*`, `packetloss_*`, `fixed_span_*` |
| `ensemble_fpr.py` | matched-FPR detector ensembles | Sec. VII-G | `ensemble_fpr*` |
| `mimicry_ksweep.py`, `mimicry_ci.py`, `constrained_mimicry.py` | mimicry candidate budget, grouped CIs, temporally constrained attacker | Sec. VIII-A | `mimicry_*`, `constrained_mimicry*` |
| `hardware_check.py`, `hardware_intervals.py` | 50-window hardware execution check, per-run intervals | Sec. VIII-B | `hardware_*` |
| `gram_inputs_prep.py` + `../gpu_engine/gram_gpu.py` | Gram batches for the ablation and label-rule studies | Sec. VII-E, VII-F | inputs of `entanglement_ablation.py`, `label_rule_sensitivity.py` |

`results/trained_parameters/` holds the trained angles (`w`, `u`, `gamma`, one row per re-uploading
layer) of every alignment run in the restart study: the bipartite map on splits 7, 17 and 27 and the
random-sparse, no-entanglement and all-pairs arms on split 7, five restarts each, under the three
selection rules (`best_batch`, used for the reported tables; `best_val`; `final`). Restart 0 is the
default initialisation behind the reported results.

---

## 6. Where each result comes from

| Paper item | Result file(s) | Script |
|---|---|---|
| Fig. 2 (ROC/PR: (a) `dev_grid` matched N, (b) VeReMi NextGen) | `full_paper_data/roc_pr/` | `roc_pr_curves.py` -> `render_figures.py` |
| Fig. 3 (CAN datasets) | `results/studies/can_seeds_per_dataset.csv`, `full_paper_data/frontier/veremi__veremi_binary_results.csv` | `studies/can_seeds.py` -> `studies/fig_crossdataset.py` |
| Fig. 4 (separability ordering) | `full_paper_data/frontier/frontier__controllability.csv` | `controllability.py` -> `render_figures.py` |
| Table II (simulation parameters) | — | `configs/sim/dev_grid.yaml` |
| Table III (main detection) | `full_paper_data/frontier/packetloss_dev__packetloss_results.csv` (`packet_loss == 0`), `packetloss_dev__significance.csv` | `configs/experiments/packetloss.yaml` |
| Table IV (attribution) | `full_paper_data/frontier/feature_ablation__*.csv`, `results/studies/{entanglement_ablation,kta_restarts,domain_kernel}*` | Section 5 |
| Table V (three splits) | `results/studies/three_splits*`, `kta_restarts*` | `dev_multiseed.py`, `studies/three_splits.py` |
| Sec. III (physical-layer attacks, Sybil transmit power) | `full_paper_data/frontier/sybil_ablation__ablation_results.csv` | `configs/experiments/sybil_ablation.yaml` |
| Sec. VII-B (scalability, cost) | `results/nystrom_scale/`, `results/qubit_scaling/`, `results/studies/inference_latency.json` | `nystrom_scale.py`, `qubit_scaling.py`, `studies/inference_latency.py` |
| Sec. VII-C (CAN) | `results/studies/can_seeds*` | `road_ids.py`, `can_ids.py`, `studies/can_seeds.py` |
| Sec. VII-D (separability, external V2X data) | `full_paper_data/frontier/{frontier__*,frontier_grouped.json,crossattack_dev__*,veremi__*}`, `full_paper_data/veremi_nextgen/`, `full_paper_data/frontier/nextgen_multiseed__*`, `results/studies/` (Section 5) | Sections 4.2 and 5 |
| Sec. VII-F (label rule, packet loss, attack magnitude) | `full_paper_data/frontier/{packetloss_dev__*,intensity_*,magnitude_axis.json}`, `results/studies/{label_rule,packetloss_*,fixed_span_*}*` | Sections 4.2 and 5 |
| Sec. VII-G (operating points) | `results/studies/ensemble_fpr*`, `full_paper_data/frontier/crossreceiver__fusion_results.csv` | `studies/ensemble_fpr.py`, `crossreceiver_fusion.py` |
| Table II flooding / DoS rows | `full_paper_data/frontier/flooding__*.csv` | `configs/sim/dev_grid_flooding.yaml` |
| Sec. VIII-A (adaptive mimicry) | `full_paper_data/frontier/{evasion_results.csv,headline.json}`, `results/studies/{mimicry,constrained_mimicry}*` | `mimicry_evasion.py`, Section 5 |
| Sec. VIII-B (hardware) | `results/dev_grid_multiseed/ibm_hardware/`, `results/veremi_nextgen/ibm_hardware/`, `results/studies/hardware_*` | `ibm_devgrid_kernel.py`, Section 5 |

`full_paper_data/sequences/` holds the labelled, windowed model inputs (see `MANIFEST.json`):
`dev_grid_sequences.csv.gz` (90,553 windows, 19 cross-layer features), `veremi_nextgen_sequences.csv`
(5,999 windows, 23 features) and `veremi_original_windows.csv` (VeReMi per-window labels and model
scores).

Standard settings unless noted: windows of 10 messages with stride 5; 8-qubit cross-layer map trained by
kernel-target alignment; PR-AUC and ROC-AUC with transmitter-grouped bootstrap CIs.

---

## 7. IBM Quantum hardware runs

The measured kernels of all hardware runs are included as JSON (per-entry zero-outcome frequencies plus
the exact statevector reference), so the hardware statistics recompute without a QPU:

```bash
python scripts/studies/hardware_check.py      # needs the dev_grid feature table (Section 4.1) for the transmitter groups
python scripts/studies/hardware_intervals.py
```

To re-measure on hardware you need **your own** IBM Quantum account. Provide the token as an
environment variable (or save it once with `QiskitRuntimeService.save_account(...)`):

```bash
export QISKIT_IBM_TOKEN="<YOUR_IBM_QUANTUM_TOKEN>"
export QISKIT_IBM_INSTANCE="<YOUR_INSTANCE_CRN>"      # optional
python scripts/ibm_devgrid_kernel.py --backend <your_backend> --n-test 50 --include-train-gram --confirm
```

The paper used `ibm_marrakesh` (Heron r2) with a pinned 8-qubit placement at 1,024 shots. Results are
written to `results/dev_grid_multiseed/ibm_hardware/`.

---

## License

MIT (see `LICENSE`). If you use this code or data, please cite the paper.
