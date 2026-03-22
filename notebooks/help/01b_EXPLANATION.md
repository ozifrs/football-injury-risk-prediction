# Notebook 01b — Injury Target Table: Full Explanation

## What This Notebook Does

Notebook `01b_injury_target_table.ipynb` takes the exploratory injury label table from NB01 (143,173 raw injury episodes with keep/exclude/ambiguous tags) and turns it into the **final target table for the prediction model**: **21,123 clean injury spells** with 23 columns including 7 prior-injury / recurrence features.

Everything described below was fully executed and verified — the output CSV exists at `data/transfermarkt/derived/target_injury_spells.csv`.

---

## Step-by-Step Walkthrough

### Step 1 — Load the NB01 Output (Cell 3)

We load `injury_label_table.csv` produced by NB01. This file has 143,173 rows — every injury episode from Transfermarkt's `player_injuries.csv`, each tagged with:
- `label_category`: `keep` (21,203), `exclude` (104,437), or `ambiguous` (17,533)
- `injury_subtype`: `hamstring`, `groin_adductor`, `calf`, `thigh` (only for `keep` rows)
- Enrichment columns from NB01: player demographics, market value, age at injury

The paths now use **repo-relative** format (`../data/transfermarkt/`) so the notebook works out of the box when cloned.

---

### Step 2 — Finalize the Taxonomy (Cell 5)

**The problem:** NB01 left 17,533 injuries as "ambiguous" — mostly generic labels like "Muscle injury" (6,433 occurrences), "muscular problems" (4,880), "Torn muscle fiber" (2,662). These *might* be lower-limb soft-tissue injuries, but we cannot determine the body region.

**The decision:** We adopt a **conservative approach**:

| Category | Role in this notebook | Count |
|----------|----------------------|------:|
| **keep** | → **Target injuries** (prediction targets) | 21,203 |
| **ambiguous** | → **Context only** (used for prior-injury features, NOT as targets) | 17,533 |
| **exclude** | → **Dropped entirely** | 104,437 |

**Why this matters:** If we include ambiguous injuries as targets, we'd be training a model to predict "Muscle injury" — but we can't even verify those are in our target body regions. However, a generic "Muscle injury" 3 weeks before a hamstring tear *is* useful context, so we keep ambiguous injuries in the history for feature engineering.

We also apply a **time-loss filter**: only injuries with `days_missed ≥ 1` count as true spells. Result: all 21,203 keep injuries already satisfy this (no zero-day keep injuries exist).

---

### Step 3 — One Clean Row per Injury Spell (Cells 7–8)

**The problem:** Some players have overlapping injury records for the same body region. For example, Albert Streit (player 1080) has two thigh injury rows where the second starts *during* the first:

```
Player 1080:
  2008-11-21 → 2008-12-09  (19 days, "Thigh problems")
  2008-11-27 → 2008-12-07  (11 days, "Thigh problems")  ← overlaps!
```

This likely represents the same injury episode recorded twice (perhaps updated mid-recovery), not two separate injuries.

**What we did:**

1. **Detection (Cell 7):** For each row, we check if `from_date ≤ previous_end_date` within the same player + body region. Found **80 overlapping rows** out of 21,203 (0.38%).

2. **Merging (Cell 8):** The `merge_overlapping_spells()` function walks through each player+region group chronologically:
   - If a row overlaps with the current spell, it **extends** the current spell (takes the later `end_date`, recomputes `days_missed`, sums `games_missed`)
   - If no overlap, it closes the current spell and starts a new one
   - Keeps the `injury_reason` from the first row of the merged spell

**Result:** 21,203 → **21,123** rows (80 duplicates merged).

---

### Step 4 — Prior-Injury & Recurrence Features (Cells 10–13)

This is the most substantial new work. For each of the 21,123 target spells, we compute **7 backward-looking features** from the player's full injury history.

#### 4a. Building the History Table (Cell 10)

We combine **keep** (21,203) + **ambiguous** (17,533) injuries = **38,736 rows** as the injury history. Ambiguous injuries get a generic `unknown_muscle` subtype. We deliberately exclude "exclude" injuries (illness, fractures, etc.) — a flu episode is not relevant muscle injury history.

#### 4b. Computing the Features (Cell 11)

For each target spell, the `compute_prior_features()` function looks at all of that player's injuries *before* the current spell date:

| Feature | What it measures | Example |
|---------|-----------------|---------|
| `n_prior_total` | Total prior muscle injuries (keep + ambiguous) for this player | Player has had 3 muscle injuries before this one |
| `n_prior_same_region` | Prior injuries to the **same body region** | 1 previous hamstring injury before this hamstring injury |
| `n_prior_target` | Prior **keep** injuries only (excludes ambiguous context) | 2 confirmed soft-tissue injuries before this one |
| `days_since_last_injury` | Days from the most recent prior injury (any type) to this one | Last injury was 45 days ago |
| `days_since_last_same_region` | Days from the most recent same-region injury to this one | Last hamstring injury was 120 days ago |
| `is_recurrence` | Same-region injury within **180 days** (6 months) of a previous one | True = this is a recurrence |
| `career_injury_burden_days` | Cumulative days missed across all prior injuries | Player has already missed 200 days to injuries in their career |

**Why 180 days for recurrence?** This is a standard clinical threshold in sports medicine — a same-site injury within 6 months is typically considered a recurrence or re-injury rather than a new independent injury.

**Why include ambiguous injuries in history?** A "Muscle injury" (ambiguous) recorded 3 weeks before a confirmed hamstring tear is likely the same or related incident. Excluding it would undercount the player's injury burden.

#### 4c. Feature Statistics (Cell 12)

Key findings from the computed features:
- **Recurrence rate:** ~9.7% of target spells are recurrences (same region within 180 days)
- **Recurrence by subtype:** Groin/adductor highest at ~11.1%, hamstring ~10.6%, calf ~9.8%, thigh ~6.6%
- **Median prior injuries:** ~1 prior muscle injury before each target spell
- **Median gap between injuries:** 271 days (~9 months)

#### 4d. Visualizations (Cell 13)

Four-panel figure showing:
1. **Distribution of prior injury count** — heavily right-skewed, most spells have 0–3 prior injuries
2. **Distribution of same-region prior injuries** — most spells (>15,000) have zero prior same-region injuries
3. **Days since last injury** — median 271 days, long right tail from players returning after years
4. **Recurrence rate by subtype** — bar chart showing groin/adductor and hamstring have highest recurrence

---

### Step 5 — Export (Cells 15–17)

**Final table:** 21,123 rows × 23 columns, saved to `data/transfermarkt/derived/target_injury_spells.csv`.

**Column schema (23 columns):**

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `player_id` | int | Transfermarkt player identifier |
| 2 | `player_name` | str | Player full name |
| 3 | `injury_reason` | str | Original injury reason string |
| 4 | `injury_subtype` | str | Body region: hamstring, groin_adductor, calf, thigh |
| 5 | `from_date` | date | Spell start date |
| 6 | `end_date` | date | Spell end date |
| 7 | `days_missed` | float | Duration of the spell in days |
| 8 | `games_missed` | int | Games missed during the spell |
| 9 | `season_name` | str | Season (e.g. "22/23") |
| 10 | `position` | str | Player position category |
| 11 | `main_position` | str | Specific position |
| 12 | `age_at_injury` | float | Player age at spell start (years) |
| 13 | `height` | float | Player height (cm) |
| 14 | `foot` | str | Preferred foot |
| 15 | `citizenship` | str | Player nationality |
| 16 | `market_value_eur` | float | Most recent market value before spell (EUR) |
| 17 | `n_prior_total` | int | Total prior muscle injuries |
| 18 | `n_prior_same_region` | int | Prior same-region injuries |
| 19 | `n_prior_target` | int | Prior confirmed target injuries |
| 20 | `days_since_last_injury` | float | Days since last muscle injury |
| 21 | `days_since_last_same_region` | float | Days since last same-region injury |
| 22 | `is_recurrence` | bool | Same-region injury within 180 days |
| 23 | `career_injury_burden_days` | float | Cumulative days missed before this spell |

---

## How This Connects to the Pipeline

```
NB01 (exploration)                    NB01b (this notebook)
─────────────────                    ──────────────────────
player_injuries.csv                  injury_label_table.csv
       │                                    │
  classify 349 reasons                 finalize taxonomy
  → keep/exclude/ambiguous            → keep = target
       │                              → ambiguous = context
  enrich with profiles               → exclude = drop
  + market values                          │
       │                              deduplicate spells
  injury_label_table.csv              → 21,203 → 21,123
  (143,173 rows × 18 cols)                │
                                      compute prior-injury
                                      features (7 new cols)
                                           │
                                      target_injury_spells.csv
                                      (21,123 rows × 23 cols)
                                           │
                                      ┌────┴────┐
                                      │ READY   │
                                      │ FOR     │
                                      │ MODEL   │
                                      └─────────┘
```

This table is the **Y variable** (injury outcome) plus player-level features. The downstream modelling notebooks will join this with:
- StatsBomb event data (match-level workload/actions)
- SkillCorner tracking data (physical load: distance, sprints, accelerations)
- Time-series exposure features from NB02

to build the full feature matrix for injury prediction.
