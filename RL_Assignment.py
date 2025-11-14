
import os
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt


from env import create_standard_grid, create_four_room


# Configuration 

MASTER_SEED = 42

# hyperparameter grids to search
TUNE_ALPHA = [0.001, 0.01, 0.1, 1.0]
TUNE_GAMMA = [0.7, 0.8, 0.9, 1.0]
TUNE_EPS = [0.001, 0.01, 0.05, 0.1]
TUNE_TAU = [0.01, 0.1, 1.0, 2.0]


TUNE_SEEDS = [MASTER_SEED + i for i in range(5)]   


EVAL_SEEDS = [MASTER_SEED + i for i in range(100)]  

TUNE_EPISODES = 200
TUNE_MAX_STEPS = 500

EVAL_EPISODES = 1000
EVAL_MAX_STEPS = 500

EXTRA_EPISODES = 500
EXTRA_MAX_STEPS = 500


RESULTS_DIR = "results"
BEST_PARAMS_FILE = "best_hyperparams.json"
BEST_PARTIAL_FILE = "best_hyperparams_partial.json"

# which tasks to run 
RUN_TASK1 = True
RUN_TASK2A = True
RUN_TASK2B = True
RUN_EXTRA_METRICS = True


# Utility functions


def set_global_seed(seed):
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def epsilon_greedy_action(Q, state, n_actions, eps):

    if np.random.rand() < eps:
        return int(np.random.randint(n_actions))
    return int(np.argmax(Q[int(state)]))

def softmax_action(Q, state, tau):
    """Return action sampled from softmax over Q-values (temperature tau)."""
    q = np.array(Q[int(state)], dtype=float)
    q = q - np.max(q)
    prefs = np.exp(q / max(tau, 1e-8))
    s = prefs.sum()
    if s == 0 or np.isnan(s):
        probs = np.ones(len(q)) / len(q)
    else:
        probs = prefs / s
    return int(np.random.choice(len(q), p=probs))

def select_action(Q, state, n_actions, explore, param):
    """Dispatch to exploration strategy."""
    if explore == "epsilon":
        return epsilon_greedy_action(Q, state, n_actions, param)
    else:
        return softmax_action(Q, state, param)

def plot_and_save(xs, title, xlabel, ylabel, filename):
    plt.figure(figsize=(6,4))
    plt.plot(xs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_npy(arr, fname):
    np.save(fname, np.array(arr))


# Task 1: SARSA baseline

def run_task1():
    print("Running Task 1: SARSA on standard grid")
    set_global_seed(MASTER_SEED)
    env = create_standard_grid()
    n_states = env.num_states
    n_actions = env.num_actions

    Q = np.zeros((n_states, n_actions))
    alpha = 0.1
    gamma = 0.9
    eps = 0.1
    episodes = 500
    max_steps = 300

    rewards_per_episode = []

    for ep in range(episodes):
        state = int(env.reset())
        action = select_action(Q, state, n_actions, "epsilon", eps)
        total_reward = 0.0
        for t in range(max_steps):
            next_state, reward = env.step(state, action)
            next_state = int(next_state)
            next_action = select_action(Q, next_state, n_actions, "epsilon", eps)
            # SARSA update
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            total_reward += float(reward)
            if reward == 10:  # goal reached
                break
        rewards_per_episode.append(total_reward)

    ensure_dir(RESULTS_DIR)
    plot_and_save(rewards_per_episode, "Task1 SARSA - Reward per Episode", "Episode", "Reward", os.path.join(RESULTS_DIR, "task1_sarsa_learning_curve.png"))
    save_npy(rewards_per_episode, os.path.join(RESULTS_DIR, "task1_sarsa_rewards.npy"))
    print("Task 1 complete. Plot and rewards saved.")


# Config list (20 configs)

def all_configs():
    configs = {}

    standard_list = [
        ("standard_qlearning_eps_p1.0_nowind", ("standard", "qlearning", "epsilon", 1.0, False)),
        ("standard_qlearning_eps_p0.7_nowind", ("standard", "qlearning", "epsilon", 0.7, False)),
        ("standard_qlearning_softmax_p1.0_nowind", ("standard", "qlearning", "softmax", 1.0, False)),
        ("standard_qlearning_softmax_p0.7_nowind", ("standard", "qlearning", "softmax", 0.7, False)),
        ("standard_qlearning_eps_p1.0_wind", ("standard", "qlearning", "epsilon", 1.0, True)),
        ("standard_sarsa_eps_p1.0_nowind", ("standard", "sarsa", "epsilon", 1.0, False)),
        ("standard_sarsa_eps_p0.7_nowind", ("standard", "sarsa", "epsilon", 0.7, False)),
        ("standard_sarsa_softmax_p1.0_nowind", ("standard", "sarsa", "softmax", 1.0, False)),
        ("standard_sarsa_softmax_p0.7_wind", ("standard", "sarsa", "softmax", 0.7, True)),
    ]

    for k, v in standard_list:
        configs[k] = v

    fourroom_list = [
        ("fourroom_qlearning_eps_p1.0_goalchange", ("fourroom", "qlearning", "epsilon", 1.0, "goalchange")),
        ("fourroom_qlearning_eps_p0.7_goalchange", ("fourroom", "qlearning", "epsilon", 0.7, "goalchange")),
        ("fourroom_qlearning_softmax_p1.0_goalchange", ("fourroom", "qlearning", "softmax", 1.0, "goalchange")),
        ("fourroom_qlearning_softmax_p0.7_goalchange", ("fourroom", "qlearning", "softmax", 0.7, "goalchange")),
        ("fourroom_qlearning_eps_p1.0_fixedgoal", ("fourroom", "qlearning", "epsilon", 1.0, "fixedgoal")),
        ("fourroom_qlearning_softmax_p1.0_fixedgoal", ("fourroom", "qlearning", "softmax", 1.0, "fixedgoal")),
        ("fourroom_sarsa_eps_p1.0_goalchange", ("fourroom", "sarsa", "epsilon", 1.0, "goalchange")),
        ("fourroom_sarsa_eps_p0.7_goalchange", ("fourroom", "sarsa", "epsilon", 0.7, "goalchange")),
        ("fourroom_sarsa_softmax_p1.0_goalchange", ("fourroom", "sarsa", "softmax", 1.0, "goalchange")),
        ("fourroom_sarsa_softmax_p0.7_fixedgoal", ("fourroom", "sarsa", "softmax", 0.7, "fixedgoal")),
        ("fourroom_sarsa_softmax_p1.0_fixedgoal", ("fourroom", "sarsa", "softmax", 1.0, "fixedgoal")),
    ]

    for k, v in fourroom_list:
        configs[k] = v

    return configs

# Task 2A - Tuning logic 

def tune_single_config(cfg_name, cfg):
    """Tune one configuration by grid search. Returns best params dictionary."""
    print(f" Tuning {cfg_name} ...")
    env_type, algo, _, pval, extra = cfg

    
    if env_type == "fourroom":
        goal_change = (extra == "goalchange")
        env = create_four_room(goal_change=goal_change)
    else:
        wind_flag = bool(extra)
        env = create_standard_grid(wind=wind_flag)

    n_states = env.num_states
    n_actions = env.num_actions

    best_avg = -1e15
    best_params = None

    # grid search
    for alpha in TUNE_ALPHA:
        for gamma in TUNE_GAMMA:
            for explore_type in ["epsilon", "softmax"]:
                params_to_try = TUNE_EPS if explore_type == "epsilon" else TUNE_TAU

                for param in params_to_try:
                    seed_totals = []
                    for seed in TUNE_SEEDS:
                        set_global_seed(seed)
                        Q = np.zeros((n_states, n_actions))
                        total_for_seed = 0.0

                        for ep in range(TUNE_EPISODES):
                            state = int(env.reset())
                            for t in range(TUNE_MAX_STEPS):
                                action = select_action(Q, state, n_actions, explore_type, param)
                                next_state, reward = env.step(state, action)
                                next_state = int(next_state)
                                if algo == "qlearning":
                                    Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
                                else:  # sarsa
                                    next_action = select_action(Q, next_state, n_actions, explore_type, param)
                                    Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
                                state = next_state
                                total_for_seed += float(reward)
                                if reward == 10:
                                    break
                        seed_totals.append(total_for_seed)

                    avg_reward = float(np.mean(seed_totals))

                    if avg_reward > best_avg:
                        best_avg = avg_reward
                        best_params = {
                            "alpha": float(alpha),
                            "gamma": float(gamma),
                            "explore": explore_type,
                            "param": float(param)
                        }

    print(f"  Best for {cfg_name}: {best_params} (avg reward {best_avg:.2f})")
    return best_params

def run_task2A():
    print("\nStarting Task 2A: Hyperparameter tuning for all configs")
    ensure_dir(RESULTS_DIR)
    cfgs = all_configs()

    best_params = {}
    if os.path.exists(BEST_PARTIAL_FILE):
        try:
            with open(BEST_PARTIAL_FILE, "r") as f:
                best_params = json.load(f)
            print(" Loaded partial best params; will resume tuning missing configs.")
        except Exception:
            best_params = {}

    for name, cfg in cfgs.items():
        if name in best_params:
            print(f" Skipping {name} (already tuned in partial results).")
            continue
        bp = tune_single_config(name, cfg)
        best_params[name] = bp
        # save partial progress after every config
        with open(BEST_PARTIAL_FILE, "w") as f:
            json.dump(best_params, f, indent=2)


    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)

    print("Task 2A complete. Best hyperparameters written to", BEST_PARAMS_FILE)

# Task 2B - Evaluation and extra metrics 

def evaluate_config(cfg_name, cfg, params, seeds, episodes, max_steps):
    print(f" Evaluating {cfg_name} ...")
    env_type, algo, _, _, extra = cfg

    if env_type == "fourroom":
        goal_change = (extra == "goalchange")
        env = create_four_room(goal_change=goal_change)
    else:
        wind_flag = bool(extra)
        env = create_standard_grid(wind=wind_flag)

    n_states = env.num_states
    n_actions = env.num_actions

    all_seed_rewards = []
    all_seed_steps = []

    for s in seeds:
        set_global_seed(s)
        Q = np.zeros((n_states, n_actions))
        seed_episode_rewards = []
        seed_steps = []

        for ep in range(episodes):
            state = int(env.reset())
            total_reward = 0.0
            for t in range(max_steps):
                action = select_action(Q, state, n_actions, params["explore"], params["param"])
                next_state, reward = env.step(state, action)
                next_state = int(next_state)
                if algo == "qlearning":
                    Q[state, action] += params["alpha"] * (reward + params["gamma"] * np.max(Q[next_state]) - Q[state, action])
                else:
                    next_action = select_action(Q, next_state, n_actions, params["explore"], params["param"])
                    Q[state, action] += params["alpha"] * (reward + params["gamma"] * Q[next_state, next_action] - Q[state, action])
                state = next_state
                total_reward += float(reward)
                if reward == 10:
                    break
            seed_episode_rewards.append(total_reward)
            seed_steps.append(t + 1)  # 

        all_seed_rewards.append(seed_episode_rewards)
        all_seed_steps.append(seed_steps)

    # convert to numpy arrays
    rewards_arr = np.array(all_seed_rewards)
    steps_arr = np.array(all_seed_steps)

    mean_rewards = np.mean(rewards_arr, axis=0)
    std_rewards = np.std(rewards_arr, axis=0)
    mean_steps = np.mean(steps_arr, axis=0)

    ensure_dir(RESULTS_DIR)
    save_npy(mean_rewards, os.path.join(RESULTS_DIR, f"{cfg_name}_mean_rewards.npy"))
    save_npy(std_rewards, os.path.join(RESULTS_DIR, f"{cfg_name}_std_rewards.npy"))
    save_npy(rewards_arr, os.path.join(RESULTS_DIR, f"{cfg_name}_per_seed_rewards.npy"))

  
    plt.figure(figsize=(7,4))
    plt.plot(mean_rewards, label="mean reward")
    plt.fill_between(range(len(mean_rewards)),
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards, alpha=0.2)
    plt.title(f"{cfg_name} - Avg Reward (over seeds)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{cfg_name}_avg_reward.png"))
    plt.close()


    plot_and_save(mean_steps, f"{cfg_name} - Steps per Episode (mean)", "Episode", "Steps", os.path.join(RESULTS_DIR, f"{cfg_name}_steps.png"))


    set_global_seed(MASTER_SEED)
    Q = np.zeros((n_states, n_actions))
    visits = np.zeros(n_states)

    for ep in range(EXTRA_EPISODES):
        state = int(env.reset())
        for t in range(EXTRA_MAX_STEPS):
            action = select_action(Q, state, n_actions, params["explore"], params["param"])
            next_state, reward = env.step(state, action)
            next_state = int(next_state)
            visits[next_state] += 1
            if algo == "qlearning":
                Q[state, action] += params["alpha"] * (reward + params["gamma"] * np.max(Q[next_state]) - Q[state, action])
            else:
                next_action = select_action(Q, next_state, n_actions, params["explore"], params["param"])
                Q[state, action] += params["alpha"] * (reward + params["gamma"] * Q[next_state, next_action] - Q[state, action])
            state = next_state
            if reward == 10:
                break

    # reshape visits and q-values to grid when possible
    try:
        visits_grid = visits.reshape(env.num_rows, env.num_cols)
    except Exception:
        visits_grid = visits.copy()

    qvals_max = np.max(Q, axis=1)
    try:
        qvals_grid = qvals_max.reshape(env.num_rows, env.num_cols)
    except Exception:
        qvals_grid = qvals_max.copy()

    # save heatmaps and combined image
    plt.figure(figsize=(6,5))
    plt.imshow(visits_grid, cmap="YlGnBu")
    plt.title(f"{cfg_name} - State visit frequency")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{cfg_name}_heatmap_visits.png"))
    plt.close()

    plt.figure(figsize=(6,5))
    plt.imshow(qvals_grid, cmap="inferno")
    plt.title(f"{cfg_name} - Max Q-value heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{cfg_name}_heatmap_qvals.png"))
    plt.close()

    
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].imshow(visits_grid, cmap="YlGnBu")
    axes[0].set_title("State visits")
    axes[1].imshow(qvals_grid, cmap="inferno")
    axes[1].set_title("Max Q-values")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{cfg_name}_heatmap.png"))
    plt.close()

    overall_mean = float(np.mean(rewards_arr))
    overall_std = float(np.std(rewards_arr))
    return overall_mean, overall_std

def run_task2B():
    print("\nStarting Task 2B: final evaluation (100 seeds) and metrics")
    if not os.path.exists(BEST_PARAMS_FILE):
        raise FileNotFoundError(f"{BEST_PARAMS_FILE} not found. Run Task 2A first.")

    with open(BEST_PARAMS_FILE, "r") as f:
        best_params = json.load(f)

    cfgs = all_configs()
    summary = {}

    for cfg_name, cfg in cfgs.items():
        if cfg_name not in best_params:
            print(f" Skipping {cfg_name} — no tuned parameters found.")
            continue
        params = best_params[cfg_name]
        mean_val, std_val = evaluate_config(cfg_name, cfg, params, EVAL_SEEDS, EVAL_EPISODES, EVAL_MAX_STEPS)
        summary[cfg_name] = {"mean": mean_val, "std": std_val}
        print(f" {cfg_name}: mean={mean_val:.2f}, std={std_val:.2f}")

    # save summary
    with open(os.path.join(RESULTS_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Task 2B complete. Evaluation results saved.")


# Minimal extra-metrics driver (placeholder)
def run_extra_metrics():
    print("\nExtra metrics were generated and saved per configuration during evaluation (heatmaps and steps).")



def main():
    start = time.time()
    print("\n=== Reinforcement Learning Assignment Execution Started ===\n")
    print("Master seed:", MASTER_SEED)
    ensure_dir(RESULTS_DIR)
    set_global_seed(MASTER_SEED)

    if RUN_TASK1:
        run_task1()

    if RUN_TASK2A:
        run_task2A()

    if RUN_TASK2B:
        run_task2B()

    if RUN_EXTRA_METRICS:
        run_extra_metrics()

    elapsed = time.time() - start
    print(f"\nAll enabled tasks completed. Total time: {elapsed:.2f} seconds")
    print("Results are in the 'results' folder.")

if __name__ == "__main__":
    main()
