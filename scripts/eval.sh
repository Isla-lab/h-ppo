#!/bin/bash
# Evaluates the five seeds of a trained model on one target task under one evaluation protocol.
#
#   bash scripts/eval.sh <source> <method> <target> <protocol> [extra arguments of aggregate_eval.py]
#
#   <source>    doorkey | deliver_coffee | patrol_ab | red_green   (the task the model was trained on)
#   <method>    ppo | product | symloss_dec | symloss_const | psm | rm
#   <target>    doorkey:        8x8_2keys | 8x8_4keys | 16x16_1key | 16x16_2keys | 16x16_4keys
#               deliver_coffee: DeliverCoffeeAndMail          patrol_ab: PatrolABC
#               red_green:      RedGreenAndBlueCyan | RedGreenAndBlueCyanAndMagentaYellow
#   <protocol>  standard | sr | vto | so
#               (Standard, Symbolic Reweighting with eps = 1, Value-Triggered Override with w = 3 and tau = 0,
#               Symbolic Override; e.g. "--epsilon 0.5" or "--rolling_window_size 5" as extra arguments change them)
#
# The model must have been trained with scripts/train.sh. The returns of the 5 x 1000 episodes are written to
# minigrid/aggregate_evals/ or gym-subgoal-automata/aggregate_evals_multilayout/.
# Environment variables: SEEDS (default: the five seeds of the paper), PYTHON (default: the .venv of the folder,
# otherwise python), DRY_RUN=1 (print the command without running it).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ $# -ge 4 ] || usage
source_task=$1; method=$2; target=$3; protocol=$4; shift 4

case $method in
  ppo) model=ppo ;; product) model=h_ppo_product ;; symloss_dec) model=h_ppo_symloss_eps ;;
  symloss_const) model=h_ppo_symloss ;; psm) model=h_ppo_psm_sym ;; rm) model=ppo_RM ;;
  *) echo "unknown method: $method"; usage ;;
esac

case $protocol in
  standard) proto=(--eval_type standard) ;;
  sr)  proto=(--eval_type h_ppo --epsilon 1.0) ;;
  vto) proto=(--eval_type check_values --rolling_window_size 3 --value_skip_threshold 0.0) ;;
  so)  proto=(--eval_type random_rules) ;;
  *) echo "unknown protocol: $protocol"; usage ;;
esac

case $source_task in
  doorkey)
    folder=minigrid; seeds=${SEEDS:-50 70 95 200 700}
    case $target in
      8x8_2keys) t=(--size_env 8 --n_keys 2) ;;    8x8_4keys) t=(--size_env 8 --n_keys 4) ;;
      16x16_1key) t=(--size_env 16 --n_keys 1) ;;  16x16_2keys) t=(--size_env 16 --n_keys 2) ;;
      16x16_4keys) t=(--size_env 16 --n_keys 4) ;;
      *) echo "unknown DoorKey target: $target"; usage ;;
    esac
    args=(--size_env_model 8 --n_keys_model 1 "${t[@]}") ;;
  deliver_coffee|patrol_ab|red_green)
    folder=gym-subgoal-automata; seeds=${SEEDS:-70 75 95 200 700}
    case $source_task:$target in
      deliver_coffee:DeliverCoffeeAndMail) src=DeliverCoffee ;;
      patrol_ab:PatrolABC) src=PatrolAB ;;
      red_green:RedGreenAndBlueCyan|red_green:RedGreenAndBlueCyanAndMagentaYellow) src=RedGreen ;;
      *) echo "unknown target $target for source $source_task"; usage ;;
    esac
    # OfficeWorld: the 8 layouts of training, 125 episodes each; WaterWorld: 100 configurations never seen in
    # training, 10 episodes each
    if [ "$src" = RedGreen ]; then layouts=(--num_layouts 100 --layout_mode unseen); else layouts=(--num_layouts 8 --layout_mode train); fi
    args=(--task_model "$src" --task "$target" --eval_episodes 1000 "${layouts[@]}") ;;
  *) echo "unknown source task: $source_task"; usage ;;
esac

cd "$ROOT/$folder"
if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python; fi
fi
read -r -a seed_list <<< "$seeds"
cmd=("$PYTHON" aggregate_eval.py --master_seeds "${seed_list[@]}" "${args[@]}" --model_name "$model" --run_code "" "${proto[@]}" "$@")
echo "[$folder] ${cmd[*]}"
[ "${DRY_RUN:-0}" = 1 ] || "${cmd[@]}"
