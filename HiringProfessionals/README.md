# HiringProfessionals

Human-subjects experiment for the paper. Participants play a multi-agent multi-armed bandit (MAMAB) hiring game in which they repeatedly choose a university from which to hire an intern and receive binary success/failure feedback. Two conditions are compared:

- **Social (Communication)**: players see the cumulative success/failure counts for all candidates, pooled across all group members.
- **Asocial (Non-Communication)**: players see only their own private cumulative counts.

The experiment is run using the [Empirica](https://empirica.ly/) framework. Data are then processed and analysed with the Python scripts in `tools/` and the survey plots in `Survey/`.


## Repository Structure

```
HiringProfessionals/
├── TogetherHire-Professional/   # Empirica web application
│   ├── server/src/
│   │   ├── callbacks.js         # Game lifecycle hooks (round/stage setup, score updates)
│   │   └── index.js             # Empirica server entry point
│   └── client/src/
│       ├── App.jsx              # Root component; configures intro, exit, consent, lobby
│       ├── Game.jsx             # In-game layout (profile bar + stage area)
│       ├── Profile.jsx          # Top bar: round name, stage name, timer, bonus score
│       ├── Stage.jsx            # Routes stage name → correct stage component
│       ├── stages/
│       │   ├── Choice.jsx       # Main hiring decision UI (tutorial + game rounds)
│       │   ├── Result.jsx       # Per-round result display (unused in current config)
│       │   ├── GameIntroduction.jsx  # In-game instructions screen
│       │   └── GroupAllocation.jsx   # Post-game: allocate 100 slots across universities
│       ├── intro-exit/
│       │   ├── Introduction.jsx # Welcome/task overview screen
│       │   ├── ExitSurvey.jsx   # 4-page exit survey (demographics, hiring background, game experience)
│       │   └── MyConsent.jsx    # Informed consent form
│       └── components/
│           ├── Button.jsx       # Shared button component
│           ├── Alert.jsx        # Alert/info box component
│           ├── Avatar.jsx       # Player avatar
│           ├── Timer.jsx        # Stage countdown timer
│           ├── MyLobby.jsx      # Custom lobby waiting room
│           ├── LoadingOverlay.jsx   # Connection/loading spinner
│           ├── Finished.jsx     # Game-finished screen
│           └── SelfLoading.jsx  # Interactive waiting-room activities
├── tools/                       # Data processing and analysis scripts
│   ├── process_round_results.py     # Pivot playerRound.csv → playerRound_processed.csv
│   ├── process_different_to_log.py  # Convert unequal-condition rounds to JSON log
│   ├── process_all_different.py     # Batch runner: process all unequal-condition sessions
│   ├── process_all_identical.py     # Batch runner: process all equal-condition sessions
│   ├── group_allocation.py          # Parse groupAllocations and exitSurvey from player.csv
│   ├── plot_entropy_results.py      # Per-session entropy analysis and figures
│   ├── plot_all.py                  # Arm-pull bar charts (pooled and per-session, with CI)
│   ├── aggregate_identical.py       # Aggregate entropy across equal-condition sessions; bias plots
│   ├── aggregate_different.py       # Aggregate cumulative reward across unequal-condition sessions
│   ├── aggregate_all_participant.py # OLS regression of individual entropy on demographics
│   ├── process_highest.py           # Optimal arm selection rate across sessions
│   ├── process_participant.py       # Merge per-player entropy with exit survey demographics
│   ├── analysis_human.py            # Primary analysis: entropy curves, OLS, publication figures
│   ├── analyze_demographics.py      # Demographic descriptive stats and balance tests
│   ├── stats_summ.py                # Cross-session demographic summary CSV
│   ├── process_csv.py               # Filter out specific game IDs from raw CSV
│   └── unzip.py                     # Recursively extract Empirica export ZIP files
├── Survey/
│   └── process.py               # University familiarity and quality survey plots
└── data/                        # Data and analysis results
```


## Experiment Design

| Parameter | Value |
|-----------|-------|
| Arms (candidates) | 10 universities |
| Success probabilities | **Unequal condition**: 0.9, 0.7, 0.5, 0.3, 0.1 (two each) · **Equal condition**: 0.9 (all) |
| Players per session | 10 |
| Rounds per session | 50 (+ 1 tutorial) |
| Social info | Cumulative success/failure counts per candidate |
| Measures | Shannon entropy (bias), optimal arm rate (efficiency), cumulative reward |

Arm success probabilities are hardcoded identically in both `callbacks.js` (server) and `Choice.jsx` (client); the client draws the Bernoulli trial so scores are available immediately without a server round-trip.


## Data Processing Pipeline

### Equal-productivity condition (bias measure)

```
playerRound.csv
    └── process_round_results.py   →  playerRound_processed.csv
            └── plot_entropy_results.py  →  overall_entropy.csv, average_entropy.csv, figures/
                    └── aggregate_identical.py  →  aggregated CSVs, comparison plots
```

### Unequal-productivity condition (efficiency measure)

```
playerRound.csv
    └── process_different_to_log.py  →  playerRound.log (JSON)
            └── aggregate_different.py  →  cumulative_reward.pdf
```

### Demographics and individual differences

```
player.csv
    └── group_allocation.py  →  exit_survey_results.csv, figures/

exit_survey_results.csv  +  playerRound_processed.csv
    └── process_participant.py  →  merged_output.csv (entropy + demographics)
            └── aggregate_all_participant.py  →  OLS regression output
```

---

## Usage

### 1. Unzip raw Empirica export

```bash
python tools/unzip.py
# Enter path when prompted, or edit the script to pass sys.argv
```

### 2. Filter known bad game sessions (if any)

```bash
python tools/process_csv.py --input raw/playerRound.csv --output clean/playerRound.csv
```

### 3. Process round data

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

### 4. Extract group allocations and exit surveys

```bash
python tools/group_allocation.py <session_dir>/player.csv
# Produces: exit_survey_results.csv, figures/player_*_allocations.png
```

### 5. Merge entropy with demographics

```bash
python tools/process_participant.py <root_dir>
# Produces: merged_output.csv in each session subfolder
```

### 6. Aggregate and plot

**Equal condition (bias):**
```bash
python tools/aggregate_identical.py <social_dir> <asocial_dir>
# Produces: entropy comparison plot, strip plot, OLS output
```

**Unequal condition (efficiency):**
```bash
python tools/aggregate_different.py <social_dir> <asocial_dir>
# Produces: cumulative_reward.pdf, reward improvement stats
```

**Optimal arm rate (unequal condition):**
```bash
python tools/process_highest.py <social_dir> <asocial_dir>
# Produces: human-optimal_arm_rate.pdf
```

**Arm pull distribution (bias):**
```bash
python tools/plot_all.py <base_folder> [--combined-top-k 2]
# Produces: per-session PDFs + combined pooled and mean±CI PDFs
```

### 7. Individual-level regression

```bash
python tools/aggregate_all_participant.py <social_dir> <asocial_dir>
# Prints OLS summary: entropy ~ Communication + demographics
```

### 8. Demographic summaries

```bash
python tools/stats_summ.py <dir1> [dir2 ...]
# Produces: summary_output.csv

python tools/analyze_demographics.py --social <social_dir> --asocial <asocial_dir>
# Prints: descriptive stats, balance tests, hiring-experience regression
```

### 9. University survey plots

```bash
python Survey/process.py --csv "University-Survey-data.csv" --top 20 --output-dir plots/
# Produces: familiarity and quality bar charts (PDF)
```


## Primary Analysis Script

[`tools/analysis_human.py`](tools/analysis_human.py) is the main publication analysis script. It reads all processed session folders and produces the figures and statistics reported in the paper:

```bash
python tools/analysis_human.py <social_dir> <asocial_dir> [options]
```

Key outputs: entropy trend curves with 95% CI, final-round entropy OLS, arm-pull bar charts, optimal arm rate curves.


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
