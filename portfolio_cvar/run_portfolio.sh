#!/bin/bash

# Defaults
SEEDS=(4)
ADAPT_STATE="False"
ADAPT_REWARD="False"
CVAR_NUM_BINS=201
TRAIN_OBJECTIVE="cvar"

usage() {
    echo "Usage: $0 [--seed SEED] [--adapt_state BOOL] [--adapt_reward BOOL] [--bins N] [--train_objective OBJ]"
    echo "  --seed             Random seed (default: 4)"
    echo "  --adapt_state      Adapt state flag (default: False)"
    echo "  --adapt_reward     Adapt reward flag (default: False)"
    echo "  --bins             Number of CVaR bins, only used when train_objective=cvar (default: 201)"
    echo "  --train_objective  Training objective: cvar, sharpe, mean_return (default: cvar)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)           SEEDS=($2);         shift 2 ;;
        --adapt_state)    ADAPT_STATE="$2";   shift 2 ;;
        --adapt_reward)   ADAPT_REWARD="$2";  shift 2 ;;
        --bins)           CVAR_NUM_BINS="$2"; shift 2 ;;
        --train_objective) TRAIN_OBJECTIVE="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
RUNNER="${SCRIPT_DIR}/runner_full_exp_fin_env.py"

directory_has_contents() {
    local target_dir="$1"
    [[ -d "$target_dir" ]] && [[ -n "$(find "$target_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}

started_count=0

for SEED in "${SEEDS[@]}"; do
    if [ "$TRAIN_OBJECTIVE" = "cvar" ]; then
        BINS_SUFFIX="_bins${CVAR_NUM_BINS}"
    else
        BINS_SUFFIX=""
    fi
    SESSION_NAME="PPO_Portfolio_cvar_ours_multi_env_seed${SEED}_${TRAIN_OBJECTIVE}${BINS_SUFFIX}"
    LOG_DIR="${SCRIPT_DIR}/workspace/seed${SEED}_${TRAIN_OBJECTIVE}${BINS_SUFFIX}"

    if directory_has_contents "$LOG_DIR"; then
        read -r -p "Existing data found. Do you want to continue anyway? [y/N] " answer
        case "$answer" in
            [yY]|[yY][eE][sS])
                ;;
            *)
                echo "[INFO] Skipped: $LOG_DIR"
                continue
                ;;
        esac
    fi

    mkdir -p "$LOG_DIR"

    tmux has-session -t $SESSION_NAME 2>/dev/null

    if [ $? != 0 ]; then
        tmux new-session -d -s $SESSION_NAME
        tmux send-keys -t $SESSION_NAME "source ~/your_env/bin/activate" C-m

        tmux send-keys -t $SESSION_NAME "python ${RUNNER} --seed_start $SEED --adapt_state $ADAPT_STATE --adapt_reward $ADAPT_REWARD --result_dir ${LOG_DIR} --cvar_num_bins $CVAR_NUM_BINS --train_objective $TRAIN_OBJECTIVE | tee ${LOG_DIR}/ppo_state${ADAPT_STATE}_reward${ADAPT_REWARD}_seed${SEED}_${TRAIN_OBJECTIVE}.log" C-m
        echo "[INFO] Started training in tmux session: $SESSION_NAME"
        started_count=$((started_count + 1))
    fi
    sleep 2
done

if [ "$started_count" -gt 0 ]; then
    echo "Started $started_count tmux session(s). Use 'tmux ls' to see running sessions."
else
    echo "No new tmux sessions were started."
fi



