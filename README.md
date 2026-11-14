# Can neurosymbolic RL generalize?

Code of the paper *Can neurosymbolic RL generalize?*
(Simone Murari, Celeste Veronese, Daniele Meli, University of Verona).

The paper compares four ways of integrating a symbolic policy (a set of logical rules that suggest actions) into
PPO during training, and three ways of using the same rules when a trained policy is evaluated. Policies are trained
on the simplest task of three domains and evaluated zero-shot on harder target tasks.

## Contents

| Folder | Content |
| --- | --- |
| `minigrid/` | *DoorKey* (MiniGrid): training scripts and evaluation protocols. |
| `gym-subgoal-automata/` | *OfficeWorld* and *WaterWorld*: training scripts, evaluation protocols and environments. |
| `scripts/` | `train.sh` and `eval.sh`, which launch the experiments in all three domains. |

## Names in the paper and in the code

| Paper | `scripts/` name | Training script | `--model_name` |
| --- | --- | --- | --- |
| PPO | `ppo` | `ppo.py` | `ppo` |
| H-PPO-Product | `product` | `h_ppo_product.py` | `h_ppo_product` |
| H-PPO-SymLoss↓ (decaying weight of the symbolic term) | `symloss_dec` | `h_ppo_symloss_eps.py` | `h_ppo_symloss_eps` |
| H-PPO-SymLoss= (constant weight of the symbolic term) | `symloss_const` | `h_ppo_symloss.py` | `h_ppo_symloss` |
| H-PPO-PSM | `psm` | `h_ppo_psm_sym.py` | `h_ppo_psm_sym` |
| PPO-RM | `rm` | `ppo_reward_machine.py` | `ppo_RM` |

| Evaluation protocol | `scripts/` name | `aggregate_eval.py` arguments |
| --- | --- | --- |
| Standard | `standard` | `--eval_type standard` |
| Symbolic Reweighting (SR) | `sr` | `--eval_type h_ppo --epsilon <ε>` |
| Value-Triggered Override (VTO) | `vto` | `--eval_type check_values --rolling_window_size <w> --value_skip_threshold <τ>` |
| Symbolic Override (SO) | `so` | `--eval_type random_rules` |

The symbolic policies are implemented in `minigrid/doorkey_helpers.py` (`apply_rules_batch`) and in the
`guide_agent` methods of the environments in `gym-subgoal-automata/gym_subgoal_automata/envs/`.

## Installation

The two folders use separate Python environments, because the OfficeWorld and WaterWorld environments require
Python 3.7. Both are managed with [uv](https://docs.astral.sh/uv/); `requirements.txt` files are provided for pip.

**DoorKey** (Python 3.12):

```bash
cd minigrid
uv sync
# DoorKey with more than one key: replace the DoorKey environment of the minigrid package with the one in this folder
cp doorkey.py "$(uv run python -c 'import minigrid, os; print(os.path.dirname(minigrid.__file__))')/envs/doorkey.py"
```

**OfficeWorld and WaterWorld** (Python 3.7; recent versions of uv no longer install Python 3.7):

```bash
cd gym-subgoal-automata
uvx uv@0.6.17 python install 3.7
uv sync
uv pip install -r requirements.txt
```

Training logs to TensorBoard (`runs/`). Weights & Biases tracking is off by default; `--track` turns it on and reads
`WANDB_PROJECT_NAME` and `WANDB_ENTITY` from a `.env` file in the folder.

## Running the experiments

`scripts/train.sh` and `scripts/eval.sh` run any combination of source task, method, target task and evaluation
protocol with the hyperparameters of the paper (Table 2). They use the Python environment of the folder
(`.venv/bin/python`, or the one given in `PYTHON`); `DRY_RUN=1` prints the commands without running them.
No trained models are included: train a method first, then evaluate it.

| Argument | Values |
| --- | --- |
| source task | `doorkey` (8×8, 1 key), `deliver_coffee`, `patrol_ab` (OfficeWorld), `red_green` (WaterWorld) |
| method | `ppo`, `product`, `symloss_dec`, `symloss_const`, `psm`, `rm` (see the table above) |
| target task | `doorkey`: `8x8_2keys`, `8x8_4keys`, `16x16_1key`, `16x16_2keys`, `16x16_4keys`; `deliver_coffee`: `DeliverCoffeeAndMail`; `patrol_ab`: `PatrolABC`; `red_green`: `RedGreenAndBlueCyan`, `RedGreenAndBlueCyanAndMagentaYellow` |
| protocol | `standard`, `sr` (ε = 1), `vto` (w = 3, τ = 0), `so` |

**Training.** One run per seed (by default the five seeds of the paper, listed in `seeds.txt`):

```bash
bash scripts/train.sh doorkey symloss_const            # H-PPO-SymLoss= on DoorKey 8×8 with 1 key, five seeds
bash scripts/train.sh deliver_coffee psm 70 75         # H-PPO-PSM on DeliverCoffee, seeds 70 and 75
```

A model is saved in `<folder>/models/<model_name>_<task>/<model_name>_seed=<seed>.pt` and training logs go to
TensorBoard (`runs/`). The trained models of the paper are available from the authors on request.

**Evaluation.** The five seeds of a trained model, 1000 episodes each, on one target task under one protocol:

```bash
bash scripts/eval.sh doorkey symloss_const 16x16_2keys so
bash scripts/eval.sh red_green ppo RedGreenAndBlueCyan vto
bash scripts/eval.sh doorkey ppo 8x8_4keys sr --epsilon 0.5     # extra arguments go to aggregate_eval.py
```

The return of every episode is written to a CSV in `minigrid/aggregate_evals/` or
`gym-subgoal-automata/aggregate_evals_multilayout/`; an episode is successful when its return is positive. As in the
paper, the OfficeWorld tasks are evaluated on the 8 layouts of training and the WaterWorld tasks on 100
configurations never seen in training. All the arguments of the evaluation, with their defaults, are in
`config_eval.py`.

**Without the scripts.** The scripts call the Python files directly, from their folder, for example:

```bash
cd minigrid
python h_ppo_symloss.py --seed 50 --size_env 8 --n_keys 1
python aggregate_eval.py --master_seeds 50 70 95 200 700 --size_env_model 8 --n_keys_model 1 \
    --size_env 16 --n_keys 2 --model_name h_ppo_symloss --run_code '' --eval_type random_rules
```

## Licence

MIT (see `LICENSE`). The OfficeWorld and WaterWorld environments come from gym-subgoal-automata by Daniel
Furelos-Blanco and keep their own MIT licence (`gym-subgoal-automata/LICENSE`). The PPO implementation is adapted
from [CleanRL](https://github.com/vwxyzjn/cleanrl).
