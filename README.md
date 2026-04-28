# Recursive Reward Aggregation

Yuting Tang, Yivan Zhang, Johannes Ackermann, Yu-Jie Zhang, Soichiro Nishimori, Masashi Sugiyama  
Reinforcement Learning Conference 2025  
[[OpenReview]](https://openreview.net/forum?id=13lUcKpWy8)

## Docker

You can build and run the Docker container as follows.

### Step 1: Build the Docker image

```sh
cd docker
docker build -t rra_image . -f Dockerfile 
```

### Step 2: Run the Docker container

```sh
docker run -dit -p 8888:22 --mount type=bind,source=/path/to/your/RRA,destination=/workspace/RRA --name rra_container -m 16g --gpus all rra_image /bin/bash
```

Replace `/path/to/your/RRA` with the absolute path to your local RRA directory.

### Step 3: Enter the running container

Once the container is up and running, you can access its shell with:

```sh
docker exec -it rra_container bash
```

### Additional Setup for Experiments

To run the portfolio experiment inside the Docker container, you will also need to manually install:

```sh
pip install torch stable-baselines3 empyrical tensorboard distrax IPython
```

Make sure to run this inside the container after building it.

### Tips

If you encounter issues with **Cython**, try the following:

```sh
pip uninstall Cython
pip install Cython==3.0.0a10
```

This can resolve version conflicts or compatibility issues with certain dependencies.

## Portfolio Experiment

This repository is focused on the portfolio experiment. The training script uses a recursive PPO variant with portfolio-specific statistics.

```sh
/workspace/RRA/portfolio/run_portfolio.sh
```

If you are already inside the container (prompt like `root@...:/workspace/RRA#`), the training launched by `run_portfolio.sh` runs under `tmux`. Use these commands inside the container:

```sh
# List tmux sessions
tmux ls

# Attach to a session (replace <SESSION_NAME> with the name shown by tmux ls)
tmux attach -t <SESSION_NAME>

# Example: attach to the PPO session
tmux attach -t PPO_Portfolio_ours_multi_env_seed4
```

Detach without stopping the session using `Ctrl+b` then `d`.


Notes:
- `run_portfolio.sh` uses `tmux` and activates `~/your_env/bin/activate`. Update that path for your environment.
- Outputs are written under `/workspace/RRA/portfolio/workspace/`.
- The experiment expects data in `portfolio/preproc_data/`.