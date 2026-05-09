#!/bin/bash


SEEDS=(${1:-4})
ADAPT_STATE=(${2:-"False"})
ADAPT_REWARD=(${3:-"False"})
CVAR_NUM_BINS=(${4:-201})

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
RUNNER="${SCRIPT_DIR}/runner_full_exp_fin_env.py"

directory_has_contents() {
    local target_dir="$1"
    [[ -d "$target_dir" ]] && [[ -n "$(find "$target_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}

started_count=0

for SEED in "${SEEDS[@]}"; do
    SESSION_NAME="PPO_Portfolio_cvar_ours_multi_env_seed${SEED}_bins${CVAR_NUM_BINS}"
    LOG_DIR="${SCRIPT_DIR}/workspace/seed${SEED}_bins${CVAR_NUM_BINS}"

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

        tmux send-keys -t $SESSION_NAME "python ${RUNNER} --seed_start $SEED --adapt_state $ADAPT_STATE --adapt_reward $ADAPT_REWARD --result_dir ${LOG_DIR} --cvar_num_bins $CVAR_NUM_BINS | tee ${LOG_DIR}/ppo_state${ADAPT_STATE}_reward${ADAPT_REWARD}_seed${SEED}.log" C-m
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



