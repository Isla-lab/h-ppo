#!/bin/bash
# Trains one method on one source task with the hyperparameters of the paper, one run per seed.
#
#   bash scripts/train.sh <source> <method> [seed ...]
#
#   <source>  doorkey | deliver_coffee | patrol_ab | red_green
#   <method>  ppo | product | symloss_dec | symloss_const | psm | rm
#   seeds     default: the five seeds of the paper (minigrid/seeds.txt, gym-subgoal-automata/seeds.txt)
#
# The models are saved in <folder>/models/<model_name>_<task>/<model_name>_seed=<seed>.pt, where scripts/eval.sh
# looks for them. Environment variables: PYTHON (default: the .venv of the folder, otherwise python),
# DRY_RUN=1 (print the commands without running them).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ $# -ge 2 ] || usage
source_task=$1; method=$2; shift 2

case $method in
  ppo) script=ppo.py ;;
  product) script=h_ppo_product.py ;;
  symloss_dec) script=h_ppo_symloss_eps.py ;;
  symloss_const) script=h_ppo_symloss.py ;;
  psm) script=h_ppo_psm_sym.py ;;
  rm) script=ppo_reward_machine.py ;;
  *) echo "unknown method: $method"; usage ;;
esac

# Hyperparameters of Table 2 of the paper that differ from the defaults of config.py
case $source_task in
  doorkey)        folder=minigrid; seeds=(50 70 95 200 700)
                  args=(--size_env 8 --n_keys 1 --num_envs 4 --total_timesteps 5_000_000) ;;
  deliver_coffee) folder=gym-subgoal-automata; seeds=(70 75 95 200 700)
                  args=(--task DeliverCoffee --num_envs 8 --total_timesteps 1_000_000)
                  [ "$method" = psm ] && args+=(--psm_aux_coef 0.3) ;;
  patrol_ab)      folder=gym-subgoal-automata; seeds=(70 75 95 200 700)
                  args=(--task PatrolAB --num_envs 8 --total_timesteps 5_000_000) ;;
  red_green)      folder=gym-subgoal-automata; seeds=(70 75 95 200 700)
                  args=(--task RedGreen --num_envs 8 --total_timesteps 5_000_000) ;;
  *) echo "unknown source task: $source_task"; usage ;;
esac
[ $# -gt 0 ] && seeds=("$@")

cd "$ROOT/$folder"
if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python; fi
fi
for seed in "${seeds[@]}"; do
  cmd=("$PYTHON" "$script" --seed "$seed" "${args[@]}")
  echo "[$folder] ${cmd[*]}"
  [ "${DRY_RUN:-0}" = 1 ] || "${cmd[@]}"
done
