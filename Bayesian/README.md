# Bayesian MAMAB Simulation

Simulation code for the Bayesian-rational agent experiments in *"Rational Social Learning Makes Group Hiring More Efficient and Biased"*.

Agents are Bayesian learners who maintain Beta-distributed beliefs over arm success probabilities and select arms via Thompson sampling (or ε-greedy / UCB). The key manipulation is whether agents share their choices and outcomes after each round (social learning) or act in isolation (asocial learning).


## Code Structure

| File | Description |
|------|-------------|
| `mamab_state.py` | Core simulation: 10 agents × 10 arms, fixed initial states, with all other settings, saves one log file per run |
| `mamab_state_AgentArmChange_batch.py` | Batch simulation for scaling experiments (variable agent/arm counts, parallel via joblib) |
| `process_state_identical.py` | Analysis for the **equal-productivity** condition: entropy trend plot, final-round scatter, OLS regression |
| `process_state_different.py` | Analysis for the **unequal-productivity** condition: optimal arm rate trend plot, reward improvement statistics |
| `initial_states.json` | 42 pre-defined initial arm assignments used to seed all simulations (ensures reproducibility across conditions) |
| `data/` | data for four conditions |


## Dependencies

```
numpy
scipy
matplotlib
pandas
statsmodels
seaborn
joblib
```

Install with:

```bash
pip install numpy scipy matplotlib pandas statsmodels seaborn joblib
```

## Usage

### 1. Run the core simulation (`mamab_state.py`)

Simulates 10 agents × 10 arms for 1 000 rounds across all 42 initial states, with 100 repetitions per state, under both social and asocial conditions.

```bash
python mamab_state.py
```

Key parameters at the top of `main()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ground_truth` | `"different"` | `"identical"` (equal productivity) or `"different"` (unequal) |
| `strategy` | `"Thompson"` | `"Thompson"`, `"UCB"`, or `"greedy"` |
| `rounds` | `1000` | Number of decision rounds per run |
| `experiment_repeats` | `100` | Repetitions per initial state |

Output is written to `experiment_logs_<timestamp>/state_<id>/{communication,non-communication}/<repeat>.log`.


### 2. Run scaling experiments (`mamab_state_AgentArmChange_batch.py`)

Tests how the social/asocial efficiency and bias gap changes as the number of agents and arms varies. Runs in parallel using all available CPU cores.

```bash
python mamab_state_AgentArmChange_batch.py
```

Edit the `settings` list in `main()` to select agent–arm configurations, e.g.:

```python
settings = [
    (10, 10),
    (100, 10),
    (1000, 10),
]
```

Output mirrors the structure of `mamab_state.py`, nested under a per-configuration subfolder.


### 3. Analyze equal-productivity results (`process_state_identical.py`)

Computes Shannon entropy of the arm-selection distribution over rounds and runs an OLS regression comparing final-round entropy between conditions.

```bash
python process_state_identical.py <exp_dir> [<num_states>] [<num_arms>]
```

| Argument | Description |
|----------|-------------|
| `exp_dir` | Root folder produced by `mamab_state.py` |
| `num_states` | Number of initial states (default: 1) |
| `num_arms` | Number of arms (default: 10) |

Outputs saved to `exp_dir/`:
- `entropy_trend.pdf` — mean entropy ± 95% CI over rounds
- `entropy_last_round.pdf` — final-round scatter with mean/CI
- `results.txt` — OLS coefficient, 95% CI, p-value, and % entropy reduction


### 4. Analyze unequal-productivity results (`process_state_different.py`)

Computes the optimal arm selection rate over rounds and the cumulative reward improvement of social over asocial learning.

```bash
python process_state_different.py <exp_dir>
```

Outputs saved to `exp_dir/results/`:
- `optimal_arm_rate.pdf` — mean optimal arm rate ± 95% CI over rounds
- `optimal_arm_rate_comm.csv` / `optimal_arm_rate_no_comm.csv` — underlying data
- Console: bootstrapped effect size b (%), 95% CI, and Welch's t-test p-value


## Log Format

Each `.log` file is a JSON object:

```json
{
  "results": [
    {"choices": [3, 1, 7, ...], "rewards": [1, 0, 1, ...]},
    ...
  ]
}
```

`choices[i]` is the 0-indexed arm selected by agent `i` in that round; `rewards[i]` is the binary outcome (1 = success, 0 = failure).
