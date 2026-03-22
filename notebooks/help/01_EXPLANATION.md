# Notebook 01 — Football Datasets Exploration: Full Explanation

## Purpose

This notebook (`01_football_datasets_exploration.ipynb`) tackles the first milestone of the football injury risk prediction project: **building a clean injury label for lower-limb soft-tissue injuries** from Transfermarkt data. The core output is a one-row-per-injury-episode table where every injury is classified as `keep` (target), `exclude` (irrelevant), or `ambiguous` (uncertain).

---

## Notebook Structure (6 Sections, 29 Cells)

### Section 1 — Setup & Imports (Cells 1–3)

**What we did:** Imported the standard data science stack (`pandas`, `numpy`, `matplotlib`, `seaborn`, `re`, `pathlib`) and set up display options and the path to the Transfermarkt data folder.

**Why:** Centralizing the `DATA_ROOT` path (`../../data/data/transfermarkt`) ensures all subsequent data loads use consistent relative paths. Setting `display.max_columns = None` prevents pandas from hiding columns when we inspect DataFrames.

---

### Section 2 — Load & Inspect `player_injuries.csv` (Cells 4–7)

**What we did:**
1. **Loaded the raw CSV** → 143,195 rows × 7 columns (`player_id`, `season_name`, `injury_reason`, `from_date`, `end_date`, `days_missed`, `games_missed`).
2. **Missingness report** — checked how complete each column is. `games_missed` has some NaNs but the other columns are well-populated.
3. **Parsed dates** (`from_date`, `end_date` → datetime) and computed basic stats:
   - Date range: 1973 → 2025
   - 34,561 unique players
   - 349 unique `injury_reason` strings
   - `days_missed` distribution: median around 14–16 days

**Why:** Before any classification, we need to understand the shape, types, completeness, and quirks of the raw data. Date parsing is essential for temporal analysis and for joining with market values later.

---

### Section 3 — Explore & Classify the `injury_reason` Taxonomy (Cells 8–20)

This is the most substantial section — the core intellectual work of the notebook.

#### 3.0 — Frequency Analysis (Cells 9–10)

**What we did:** Printed the top 40 most frequent injury reasons, then listed all 349 unique reasons alphabetically.

**Why:** You can't build a classifier without first *reading* what's in the data. The alphabetic listing makes it easy to spot patterns (e.g., all "hamstring" variants cluster together).

#### 3.1 — First-Pass Classification (Cells 11–13)

**What we did:** Defined three lists of regex patterns and a `classify_injury()` function:

- **`KEEP_PATTERNS`** (14 patterns): Matches clear lower-limb soft-tissue injuries:
  - `hamstring` — any variant ("Hamstring injury", "Torn hamstring", etc.)
  - `groin`, `adductor`, `pubalgia`, `pubic bone` — groin/adductor complex
  - `calf` — calf muscle
  - `thigh problem`, `thigh muscle`, etc. — thigh (quadriceps/general)

- **`KEEP_EXCLUDE_PATTERNS`** (1 pattern): Removes a false positive — "bone bleeding in the thigh" is bone-related, not muscle.

- **`EXCLUDE_PATTERNS`** (~100 patterns): Catches everything clearly *not* a target injury:
  - Illness/virus (corona, flu, fever, etc.)
  - Upper body (shoulder, arm, hand, neck, back, head, etc.)
  - Knee/ankle ligaments (cruciate, meniscus, achilles, etc.)
  - Fractures (fracture, broken, crack)
  - Surgery/procedures
  - Non-injury (rest, fitness, quarantine, traffic accident)
  - Other anatomical areas (hip, shin, foot, toe, etc.)

- **Classification logic (order matters):**
  1. Check `KEEP_PATTERNS` first → if matched, check `KEEP_EXCLUDE_PATTERNS` for false positives
  2. Check `EXCLUDE_PATTERNS` → if matched, exclude
  3. Everything else → `ambiguous`

- **Validation cell:** Printed every unique `injury_reason` grouped by category to manually verify correctness.

**First-pass result:**
| Category | Count | % |
|----------|------:|---:|
| keep | 21,207 | 14.8% |
| exclude | 100,697 | 70.3% |
| ambiguous | 21,291 | 14.9% |

#### 3.2 — Second-Pass Refinement (Cells 14–15)

**What we did:** Reviewed the 21K ambiguous injuries and identified additional patterns that are clearly not soft-tissue muscle targets:

- **`EXCLUDE_REFINEMENTS`** (~25 patterns): knock, dead leg, bruise, ligament injury/stretching, cartilage damage, bursitis, tendon rupture/tear/irritation, lumbago, acromioclavicular, cerebral hemorrhage, nerve issues, etc.

- **`classify_injury_v2()`**: Same logic but adds the refinement check between original exclude and ambiguous.

**Why a two-pass approach?** The first pass catches the obvious cases; the second pass surgically cleans the ambiguous bucket after manual review. This prevents premature over-classification and keeps the audit trail clear.

**Refined result:**
| Category | Count | % |
|----------|------:|---:|
| keep | 21,207 | 14.8% |
| exclude | 104,455 | 72.9% |
| ambiguous | 17,533 | 12.2% |

The ambiguous bucket shrank from 21K to 17.5K. The remaining ambiguous injuries are genuinely uncertain — e.g., "Muscle injury" (6,433 occurrences) with no body part specified.

#### 3.3 — Sub-classification of Target Injuries (Cells 16–17)

**What we did:** For every `keep` injury, assigned a body region subtype using `get_injury_subtype()`:

| Subtype | Count | Regex trigger |
|---------|------:|---------------|
| hamstring | 7,730 | `hamstring` |
| groin_adductor | 6,563 | `groin\|adductor\|pubalgia\|pubic bone` |
| calf | 3,742 | `calf` |
| thigh | 3,168 | `thigh` |

**Visualizations produced:**
- **Pie chart:** Label category distribution (keep 14.8%, exclude 72.9%, ambiguous 12.2%)
- **Horizontal bar chart:** Subtype counts within keep injuries

#### 3.4 — Severity & Temporal Analysis (Cells 18–20)

**What we did:**

1. **Severity statistics:** Computed `days_missed` descriptive stats by subtype (mean, median, std, quartiles).
   - Hamstring: median 28 days
   - Groin/adductor: median 23 days
   - Calf: median 22 days
   - Thigh: median 19 days

2. **Visualizations:**
   - Box plot of days_missed by subtype (capped at 120 days for readability)
   - Histogram of days_missed with a vertical line at 14 days (our prediction horizon)
   - Result: **72.6% of target injuries last >14 days** — confirming the 14-day horizon is meaningful

3. **Temporal coverage:** Stacked bar chart of target injuries per year by subtype.
   - Data is richest from 2015 onwards: **17,751 target injuries** from **9,643 unique players**
   - This informs downstream modeling decisions (e.g., train on 2015+ data)

---

### Section 4 — Enrich with Player Profiles & Market Values (Cells 21–24)

#### Player Profiles (Cells 22–23)

**What we did:**
1. Loaded `player_profiles.csv` (92,671 players, 8 selected columns: name, DOB, height, position, main_position, foot, citizenship)
2. Checked join coverage: **100%** of injury players have profiles
3. Merged injuries with profiles via `player_id`
4. Computed `age_at_injury` = (injury_date - DOB) / 365.25

**Visualizations:**
- Target injuries by position (stacked horizontal bar, colored by subtype)
- Age at injury histogram with median line (median = 27.0 years)

**Why:** Position and age are two of the strongest injury risk factors. We need these columns in the output table for downstream modeling.

#### Market Values (Cell 24)

**What we did:**
1. Loaded `player_market_value.csv` (901,429 valuation rows, 69K unique players)
2. Converted Unix timestamps to datetime
3. Used `pd.merge_asof()` to attach the **most recent market value before each injury date** — this prevents data leakage (no future information)
4. Coverage: **95.5% overall**, **97.4% for keep injuries**

**Why `merge_asof` instead of a loop?** A naive loop over 143K injuries would take minutes. `merge_asof` is vectorized and handles the "nearest match before" logic efficiently (completed in ~2 seconds).

**Why backward direction?** Using the most recent valuation *before* the injury avoids leaking information from after the injury occurred, which would be invalid in a prediction model.

---

### Section 5 — Build the Cleaned Injury Label Table (Cells 25–27)

**What we did:**
1. **Selected 17 columns** for the final output:
   - Identifiers: `player_id`, `player_name`
   - Injury info: `injury_reason`, `label_category`, `injury_subtype`, `from_date`, `end_date`, `days_missed`, `games_missed`, `season_name`
   - Player demographics: `position`, `main_position`, `age_at_injury`, `height`, `foot`, `citizenship`
   - Economic context: `market_value_eur`

2. **Added `is_time_loss` flag**: `days_missed >= 1` (all keep injuries are time-loss injuries)

3. **Sorted** by `player_id` + `from_date` for clean chronological ordering

4. **Exported two CSV files** to `data/data/transfermarkt/derived/`:
   - `injury_label_table.csv` — full table, all 3 categories (143,173 rows)
   - `injury_label_table_keep.csv` — target injuries only (21,203 rows)

**Why two files?** The full table preserves the complete audit trail (other notebooks can see what was excluded and why). The keep-only file is the direct input for the injury prediction pipeline.

---

### Section 6 — Summary & Key Findings (Cells 28–29)

**What we did:** A markdown summary of all key results, plus a final code cell printing a formatted summary dashboard.

---

## Key Design Decisions & Rationale

### 1. Regex-based classification (not ML)
We used hand-crafted regex patterns rather than a text classifier for several reasons:
- Only 349 unique strings — small enough for manual review
- Full transparency: every classification decision is auditable
- No labeled training data needed
- Deterministic and reproducible

### 2. Three-category system (keep / exclude / ambiguous)
Rather than forcing every injury into keep/exclude, we preserve an "ambiguous" bucket for genuinely uncertain cases. This is more honest and lets downstream analysis decide how to handle them (e.g., exclude for conservative analysis, include for sensitivity analysis).

### 3. Two-pass classification
The iterative approach (broad first pass → targeted refinement) makes the logic easier to follow and debug. Each pass has a clear rationale documented in markdown.

### 4. `merge_asof` for market values
This is the correct way to attach temporal features without data leakage. It finds the latest market value observation *before* each injury date, handling the irregular time series of valuations elegantly.

### 5. Body region sub-classification
Grouping target injuries into 4 subtypes (hamstring, groin/adductor, calf, thigh) enables body-region-specific analysis and potentially separate prediction models for each injury type.

---

## Output Files

| File | Path | Rows | Columns | Description |
|------|------|-----:|--------:|-------------|
| Full table | `data/data/transfermarkt/derived/injury_label_table.csv` | 143,173 | 18 | All injuries with classification |
| Keep-only | `data/data/transfermarkt/derived/injury_label_table_keep.csv` | 21,203 | 18 | Target soft-tissue injuries only |

### Column Schema

| Column | Type | Description |
|--------|------|-------------|
| `player_id` | int | Transfermarkt player identifier |
| `player_name` | str | Player full name |
| `injury_reason` | str | Original injury reason string from Transfermarkt |
| `label_category` | str | Classification: `keep`, `exclude`, or `ambiguous` |
| `injury_subtype` | str | Body region for keep injuries: `hamstring`, `groin_adductor`, `calf`, `thigh` (null for non-keep) |
| `from_date` | date | Injury start date |
| `end_date` | date | Injury end date (return to availability) |
| `days_missed` | int | Number of days missed |
| `games_missed` | int | Number of games missed |
| `season_name` | str | Season (e.g., "2023-2024") |
| `position` | str | Player position category (Attack, Midfield, Defender, Goalkeeper) |
| `main_position` | str | Specific position (e.g., "Centre-Forward", "Central Midfield") |
| `age_at_injury` | float | Player age at injury date (years) |
| `height` | float | Player height (cm) |
| `foot` | str | Preferred foot (left, right, both) |
| `citizenship` | str | Player nationality |
| `market_value_eur` | float | Most recent market value before injury (EUR) |
| `is_time_loss` | bool | Whether the injury caused any time loss (days_missed >= 1) |

---

## Key Statistics

- **143,173** total injury episodes across **34,554** unique players
- **21,203** target injuries (14.8%) — hamstring, groin/adductor, calf, thigh
- **104,437** excluded injuries (72.9%)
- **17,533** ambiguous injuries (12.2%)
- **72.6%** of target injuries last more than 14 days
- **100%** profile join coverage
- **97.4%** market value coverage for target injuries
- Data richest from **2015 onwards** (17,751 target injuries, 9,643 unique players)

---

## Recommendations for Next Notebooks

1. **Ambiguous bucket:** The 17,533 ambiguous injuries (e.g., "Muscle injury" n=6,433) could be partially rescued using NLP or contextual rules
2. **Prospective label:** Use `from_date` to create the forward-looking 14-day injury risk binary label
3. **Recurrence flag:** Compute whether each injury is a first occurrence or recurrence for the same player + body region
4. **Cross-dataset linking:** Join with StatsBomb event data and SkillCorner tracking data using player name + team + date window (see notebook 05)
