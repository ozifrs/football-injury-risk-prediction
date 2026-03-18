# Elite Football Injury Risk

Repository for exploring open football datasets to support a **14-day elite player injury-risk modeling project**, with a focus on **non-contact lower-limb soft-tissue injuries** (for example: hamstring, groin, calf, and thigh injuries).

The current objective is to understand what can realistically be built from publicly accessible data, how datasets can be joined, and which features are most promising for a first baseline model.

---

## Project Goal

We want to investigate whether open football datasets can support a model that estimates the probability that an elite male player sustains a **time-loss injury within the next 14 days**.

This repository is currently focused on:

- dataset exploration
- schema understanding
- joinability analysis
- feature ideation
- early data cleaning and preprocessing
- exploratory notebooks by dataset and by workstream

---

## Datasets Being Explored

### 1. football-datasets
**Source:** `salimt/football-datasets`  
**Link:** <https://github.com/salimt/football-datasets/tree/main>

**Why we are exploring it**
- Contains structured football data derived from Transfermarkt-like sources
- Especially useful for **injury history**
- Likely helpful for building the **target variable / injury label table**

**Potential value for the project**
- Player injury history
- Player profiles and metadata
- Player market values
- Transfer histories
- Potential season-level injury episodes

**Main questions**
- Can we isolate relevant injuries such as hamstring, groin, calf, thigh, muscle injuries?
- How noisy are the injury descriptions?
- Can we build a clean injury episode table?
- Can this dataset be linked to match-exposure data through player identifiers, names, clubs, and dates?

---

### 2. SkillCorner Open Data
**Source:** `SkillCorner/opendata`  
**Link:** <https://github.com/SkillCorner/opendata>

**Why we are exploring it**
- Provides tracking-related open data
- Useful for understanding what physical and movement-based features could look like
- Valuable for **feature prototyping**, even if it is not the final training source

**Potential value for the project**
- Tracking-derived movement features
- Distance and speed proxies
- Acceleration / deceleration profiles
- Phase-of-play context
- Physical-load style variables

**Main questions**
- What physical indicators can be engineered from the tracking files?
- Can we create player-level match summaries?
- Which movement or load proxies might be relevant for injury risk?
- What is the level of effort needed to transform raw tracking into model-ready tables?

---

### 3. StatsBomb Open Data
**Source:** `statsbomb/open-data`  
**Link:** <https://github.com/statsbomb/open-data/tree/master>

**Why we are exploring it**
- Rich event data with strong football context
- Useful for generating **per-player per-match contextual features**
- Can help approximate intensity and role-based exposure through event involvement

**Potential value for the project**
- Match events
- Lineups
- Competitions and matches
- Player roles / actions
- Tactical and event-based context
- Freeze-frame / 360 context for selected matches

**Main questions**
- Which event-based features might be relevant to injury risk?
- Can we derive player-level match features such as pressures, duels, fouls suffered, carries, defensive actions, substitutions, and position usage?
- How complete is the coverage for the competitions we care about?
- How easy is it to align this data with other datasets?

---

### 4. Kaggle Notebook: Player-Team Records
**Source:** Kaggle notebook by jockeroika  
**Link:** <https://www.kaggle.com/code/jockeroika/player-team-records>

**Why we are exploring it**
- Mainly as a **reference notebook**
- Useful for ideas on exploratory analysis, visualizations, aggregations, and presentation style
- Not necessarily treated as a distinct raw data source for the final model

**Potential value for the project**
- Inspiration for notebook structure
- Example EDA workflow
- Useful record-style summaries
- Possible ideas for derived tables and indicators

**Main questions**
- What analyses are worth reproducing in our own data pipeline?
- Are there useful chart types or aggregation patterns we should reuse?
- Can it help us standardize notebook outputs across the team?

---

### 5. transfermarkt-datasets
**Source:** `dcaribou/transfermarkt-datasets`  
**Link:** <https://github.com/dcaribou/transfermarkt-datasets>

**Why we are exploring it**
- Strong candidate for building the **player-match exposure backbone**
- Structured relational data
- Likely one of the most useful sources for match history, appearances, and joinable football entities

**Potential value for the project**
- Games
- Appearances
- Game lineups
- Club games
- Game events
- Player-match level exposure
- Rolling workload feature generation

**Main questions**
- Can we build a clean player-match table?
- Can we generate rolling exposure features such as:
  - minutes in last 7 / 14 / 28 days
  - matches in last 7 / 14 / 28 days
  - starts vs substitute appearances
  - days since last match
  - fixture congestion
- Can this be linked reliably with injury-history data?

---

## Current Working Hypothesis

The most realistic first baseline model will likely come from combining:

- **injury history data** from `football-datasets`
- **appearance / match exposure data** from `transfermarkt-datasets`

Then, depending on feasibility, we may enrich that baseline with:

- **event-context features** from `StatsBomb`
- **physical-load prototypes** from `SkillCorner`

At this stage, the goal is not to force all datasets into one pipeline, but to determine:

1. what each source contributes,
2. what can actually be joined,
3. what the minimum viable modeling dataset looks like.

---

## Suggested Data Products

As exploration progresses, we expect to produce some or all of the following tables:

### Core tables
- `players`
- `matches`
- `appearances`
- `injury_episodes`

### Derived tables
- `player_match_features`
- `player_rolling_load_features`
- `player_event_context_features`
- `player_tracking_features`
- `modeling_dataset_14d_horizon`

---

## Team Exploration Structure

Each dataset exploration should ideally answer the same questions:

1. What is the grain of the main table?
2. What are the main IDs and keys?
3. What date fields exist?
4. What can be used as labels, exposures, or features?
5. What are the major missingness / data quality issues?
6. What can be joined directly, and what requires fuzzy matching?
7. What 10 candidate features could this dataset contribute?
8. Should this source be considered:
   - **core**
   - **auxiliary**
   - **prototype-only**

---

## Repository Structure

```text
.
├── README.md
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_football_datasets_exploration.ipynb
│   ├── 02_transfermarkt_exposure_exploration.ipynb
│   ├── 03_statsbomb_event_context_exploration.ipynb
│   ├── 04_skillcorner_tracking_exploration.ipynb
│   └── 05_cross_dataset_joinability.ipynb
├── src/
│   ├── data_loading/
│   ├── preprocessing/
│   ├── feature_engineering/
│   └── modeling/
└── docs/

```

# Next Steps

- Explore each dataset independently
- Document schemas and key fields
- Build small cleaned tables for each source
- Test joinability across datasets
- Identify the best candidate for a first baseline modeling dataset
- Build an initial binary prediction target for injury risk within a 14-day horizon

# Notes

This repository is currently an exploration and data-understanding workspace.
The final modeling scope may evolve depending on:

- dataset coverage
- joinability
- label quality
- competition availability
- feasibility of feature engineering

# References

football-datasets: https://github.com/salimt/football-datasets/tree/main

SkillCorner Open Data: https://github.com/SkillCorner/opendata

StatsBomb Open Data: https://github.com/statsbomb/open-data/tree/master

Kaggle notebook: https://www.kaggle.com/code/jockeroika/player-team-records

transfermarkt-datasets: https://github.com/dcaribou/transfermarkt-datasets