# LLM MAMAB Simulation

Simulation and analysis code for the GPT-4o generative-agent experiments in *"Rational Social Learning Makes Group Hiring More Efficient and Biased"*.

Each agent is a GPT-4o instance that receives a running summary of arm outcomes as a text prompt and responds with its next arm choice. In the **social learning** condition, the summary reflects pooled outcomes from all agents; in the **asocial learning** condition, each agent sees only its own history. Simulation state is checkpointed after every round so interrupted runs can resume.


## Code Structure

| File | Description |
|------|-------------|
| `mamab_llm.py` | Simulation — **unequal-productivity** condition (arms differ in success probability) |
| `mamab_llm_identical.py` | Simulation — **equal-productivity** condition (all arms at p = 0.9) |
| `new_llm_different.py` | Analysis — efficiency: horizontal four-group bar chart (Overall / Top 20% / Middle 20% / Bottom 20%) comparing cumulative reward between conditions |
| `new_llm_identical.py` | Analysis — bias: entropy trend and final-round scatter plots comparing arm-selection entropy between conditions |
| `data/` | data for four conditions |


## Dependencies

```
numpy
openai
httpx
matplotlib
pandas
scipy
```

Install with:

```bash
pip install numpy openai httpx matplotlib pandas scipy
```

## Setup

The simulation scripts use the OpenAI API. Set the following environment variables before running:

```bash
export API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # or your custom endpoint
export HTTPS_PROXY="http://127.0.0.1:7890"           # optional; omit if not behind a proxy
```


## Usage

### 1. Run simulations

**Unequal-productivity condition** (measures efficiency):

```bash
python mamab_llm.py
```

**Equal-productivity condition** (measures bias):

```bash
python mamab_llm_identical.py
```

Key parameters at the top of `setup_experiment()` in each script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_agents` | `10` | Number of LLM agents |
| `num_arms` | `10` | Number of candidate arms |
| `ground_truth` | `"different"` / `"identical"` | Reward structure |
| `rounds` | `200` | Decision rounds per run |
| `communication` | `False` / `True` | Social vs. asocial condition |
| `experiment_repeats` | `3` / `7` | Number of independent runs |

Logs are written to `experiment_logs_<timestamp>/experiment_<n>_log.json`. Checkpoints for resuming are stored in `state/` or `state_identical/`.

> **Resuming an interrupted run**: if a checkpoint file exists, the simulation automatically resumes from the last completed round.


### 2. Analyze efficiency — unequal condition (`new_llm_different.py`)

```bash
python new_llm_different.py <comm_folder> <no_comm_folder> [--outdir <dir>] [--top 0.2] [--mid 0.2]
```

| Argument | Description |
|----------|-------------|
| `comm_folder` | Folder containing social-condition `.json` logs |
| `no_comm_folder` | Folder containing asocial-condition `.json` logs |
| `--outdir` | Output directory (default: `comm_folder`) |
| `--top` | Top/bottom stratum fraction (default: 0.2) |
| `--mid` | Middle stratum fraction (default: 0.2) |

Outputs saved to `outdir/`:
- `four_group_bars.pdf` / `four_group_bars.png` — paired bar chart with % difference annotations


### 3. Analyze bias — equal condition (`new_llm_identical.py`)

```bash
python new_llm_identical.py <comm_dir> <nocomm_dir> \
    [--num-arms 10] [--outdir <dir>] \
    [--xlim a b] [--ylim c d] \
    [--dot-size 30] [--mean-dot-size 6]
```

This script reads `*_entropy.csv` files produced during or after simulation (one per run, containing a per-round `Arm Entropy` column).

| Argument | Description |
|----------|-------------|
| `comm_dir` | Directory with social-condition entropy CSVs |
| `nocomm_dir` | Directory with asocial-condition entropy CSVs |
| `--num-arms` | Number of arms (default: 10) |
| `--outdir` | Output directory (default: parent of `comm_dir`) |
| `--xlim`/`--ylim` | Override axis limits |
| `--dot-size`/`--mean-dot-size` | Scatter point sizes |

Outputs saved to `outdir/`:
- `llm_entropy_trend.pdf` — mean entropy ± 95% CI over rounds
- `llm_entropy_last_round.pdf` — final-round jitter scatter with mean/CI


## Log Format

Each `.json` log file produced by the simulation scripts has the structure:

```json
{
  "results": [
    {
      "round": 1,
      "choices": [2, 7, 0, ...],
      "rewards": [1, 0, 1, ...],
      "responses": [
        {"agent": "Agent1", "choice": 3, "reward": 1, "response": "<full LLM output>"},
        ...
      ]
    },
    ...
  ]
}
```

`choices[i]` and `rewards[i]` are the 0-indexed arm and binary outcome for agent `i` in that round. The `responses` field stores the full chain-of-thought output from each LLM agent for qualitative analysis.
