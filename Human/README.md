# Human

Human-subjects experiment for the HiringMAMAB paper. Participants (recruited via Prolific) play a multi-agent multi-armed bandit (MAMAB) hiring game: they repeatedly choose a university from which to hire an intern and receive binary success/failure feedback. Two conditions are compared:

- **Social (Communication)**: players see cumulative success/failure counts pooled across all group members.
- **Asocial (Non-Communication)**: players see only their own private cumulative counts.

The experiment is run using the [Empirica](https://empirica.ly/) framework. Data are processed and analysed with the Python scripts in `tools/`.


## Repository Structure

```
Human/
├── TogetherHire/                    # Empirica web application (student sample)
│   ├── server/src/
│   │   ├── callbacks.js             # Game lifecycle hooks (round/stage setup, score updates)
│   │   └── index.js                 # Empirica server entry point
│   └── client/src/
│       ├── App.jsx                  # Root component; configures intro, exit, consent, lobby
│       ├── Game.jsx                 # In-game layout (profile bar + stage area)
│       ├── Profile.jsx              # Top bar: round name, stage name, timer, score
│       ├── Stage.jsx                # Routes stage name → correct stage component
│       ├── intro-exit/
│       │   ├── Introduction.jsx     # Welcome/task overview screen
│       │   └── ExitSurvey.jsx       # Exit survey (demographics, hiring background, strategy)
│       ├── components/
│       │   ├── Alert.jsx            # Shared alert/info box
│       │   ├── Avatar.jsx           # Player avatar
│       │   ├── Button.jsx           # Shared button component
│       │   └── Timer.jsx            # Stage countdown timer
│       └── examples/
│           ├── JellyBeans.jsx       # Example game task (jelly bean estimation)
│           └── MineSweeper.jsx      # Example game task (minesweeper)
├── tools/                           # Data processing and analysis scripts
│   ├── unzip.py                         # Recursively extract Empirica export ZIP files
│   ├── process_round_results.py         # Pivot playerRound.csv → playerRound_processed.csv
│   ├── process_different_to_log.py      # Convert unequal-condition rounds to JSON log
│   ├── process_all_identical.py         # Batch runner: process all equal-condition sessions
│   ├── process_all_different.py         # Batch runner: process all unequal-condition sessions
│   ├── group_allocation.py              # Parse groupAllocations and exitSurvey from player.csv
│   ├── process_group_allocation.py      # Parse group allocation per folder → group_allocation.csv (entropy)
│   ├── process_participant.py           # Merge per-player entropy with exit survey demographics
│   ├── plot_entropy_results.py          # Per-session entropy analysis and figures
│   ├── aggregate_identical.py           # Aggregate entropy across equal-condition sessions; strip plot
│   ├── aggregate_different.py           # Aggregate cumulative reward across unequal-condition sessions
│   ├── aggregate_all_participant.py     # OLS regression of individual entropy on demographics
│   ├── aggregate_exit_survey.py         # Aggregate keyFactors and strategy responses across sessions
│   ├── aggregate_group_allocation_group.py      # Group-level group allocation analysis
│   ├── aggregate_group_allocation_individual.py # Individual-level group allocation analysis
│   ├── process_highest.py               # Optimal arm selection rate across sessions
│   ├── stats_summ.py                    # Cross-session demographic summary CSV
│   ├── model_fitting_new.py             # Build per-player long-format CSV for logit modeling
│   ├── bayesian_choice_model.py         # Multinomial logistic choice model definition
│   ├── logit_per_participant.py         # Per-participant multinomial logit regression
│   ├── batch_model.py                   # Compute strategy shares (own/group/prev/other) per participant
│   ├── process_individual_learning.py   # Classify choices as individual learning strategies
│   ├── process_social_learning.py       # Classify choices as social learning strategies
│   ├── visualize_share_distribution.py  # Publication plot of strategy share distributions
│   ├── visualize_strategy_distribution.py # Publication plot of strategy proportions
│   ├── plot_wordfreq_new.py             # Word frequency bar charts from exit survey text responses
│   └── word_frequency.py               # Word frequency extraction from survey responses
└── data/                            # Raw and processed experimental data (not tracked in repo)
```


## Experiment Design

| Parameter | Value |
|-----------|-------|
| Arms (candidates) | 10 universities |
| Success probabilities | **Unequal condition**: 0.9, 0.7, 0.5, 0.3, 0.1 (two each) · **Equal condition**: 0.9 (all) |
| Players per session | 10 |
| Rounds per session | 50 (+ 1 tutorial) |
| Social info | Cumulative success/failure counts per candidate |
| Measures | Shannon entropy (bias), optimal arm rate (efficiency), cumulative reward, strategy shares |


## Data Processing Pipeline

### Equal-productivity condition (bias measure)

```
playerRound.csv
    └── process_round_results.py   →  playerRound_processed.csv
            └── plot_entropy_results.py  →  overall_entropy.csv, average_entropy.csv, figures/
                    └── aggregate_identical.py  →  aggregated CSVs, entropy comparison plots
```

### Unequal-productivity condition (efficiency measure)

```
playerRound.csv
    └── process_different_to_log.py  →  playerRound.log (JSON)
            └── aggregate_different.py  →  cumulative_reward.png, optimal arm rate plots
```

### Demographics and individual differences

```
player.csv
    └── group_allocation.py  →  exit_survey_results.csv, figures/player_*_allocations.png

exit_survey_results.csv  +  playerRound_processed.csv
    └── process_participant.py  →  merged_output.csv (entropy + demographics)
            └── aggregate_all_participant.py  →  OLS regression output
```

### Strategy modeling

```
playerRound.csv
    └── model_fitting_new.py  →  model_<session>.csv (long-format per-arm choice data)
            ├── logit_per_participant.py  →  summary_logit_per_participant.csv
            └── batch_model.py  →  processed_strategy_summary.csv (own/group/prev/other shares)
                    └── visualize_share_distribution.py  →  share distribution PDF
```


## Usage

### 1. Unzip raw Empirica export

```bash
python tools/unzip.py
# Enter path when prompted
```

### 2. Process round data

**Equal condition:**
```bash
python tools/process_round_results.py <session_dir>/playerRound.csv
# Produces: playerRound_processed.csv

python tools/process_all_identical.py <root_dir>
# Batch: processes all sessions, runs plot_entropy_results.py on each
```

**Unequal condition:**
```bash
python tools/process_different_to_log.py <session_dir>/playerRound.csv
# Produces: playerRound.log

python tools/process_all_different.py
# Batch: processes all sessions interactively
```

### 3. Extract group allocations and exit surveys

```bash
python tools/group_allocation.py <session_dir>/player.csv
# Produces: exit_survey_results.csv, figures/player_*_allocations.png
```

### 4. Merge entropy with demographics

```bash
python tools/process_participant.py <root_dir>
# Produces: merged_output.csv in each session subfolder
```

### 5. Aggregate and plot

**Equal condition (bias):**
```bash
python tools/aggregate_identical.py <social_dir> <asocial_dir>
# Produces: entropy comparison plot, strip plot, OLS output
```

**Unequal condition (efficiency):**
```bash
python tools/aggregate_different.py <social_dir> <asocial_dir>
# Produces: cumulative_reward.png, optimal arm rate plots, reward improvement stats
```

**Optimal arm rate:**
```bash
python tools/process_highest.py <social_dir> <asocial_dir>
# Produces: human-optimal_arm_rate.pdf
```

### 6. Individual-level regression

```bash
python tools/aggregate_all_participant.py <social_dir> <asocial_dir>
# Prints OLS summary: entropy ~ Communication + demographics
```

### 7. Strategy modeling

```bash
# Step 1: build long-format model input
python tools/model_fitting_new.py <root_dir>
# Produces: model_<session>.csv in each session subfolder

# Step 2a: per-participant logit regression
python tools/logit_per_participant.py <root_dir>
# Produces: summary_logit_per_participant.csv

# Step 2b: strategy share computation
python tools/batch_model.py <root_dir>
# Produces: processed_strategy_summary.csv

# Step 3: visualize strategy shares
python tools/visualize_share_distribution.py processed_strategy_summary.csv
```

### 8. Exit survey text analysis

```bash
python tools/aggregate_exit_survey.py <social_dir> [<asocial_dir> ...]
# Produces: aggregated_exit_survey_*.csv

python tools/plot_wordfreq_new.py --stacked <aggregated_exit_survey.csv> --out plots/
# Produces: word frequency bar charts (PDF)
```

### 9. Demographic summaries

```bash
python tools/stats_summ.py <dir1> [dir2 ...]
# Produces: summary_output.csv
```

## Dependencies

```
pandas
numpy
scipy
statsmodels
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn
```
