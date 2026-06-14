# Steam Review Sentiment Analysis

Binary sentiment classification of Steam game reviews, with a focus on **how well sentiment models generalize** — across game genres, across languages, and on sarcastic reviews.

The project compares a classical baseline (**TF-IDF + Logistic Regression**) against a fine-tuned transformer (**XLM-RoBERTa**) on a balanced, genre-diverse dataset collected directly from Steam's public review API.

---

## Overview

The notebook (`Steam.ipynb`) runs end to end and answers four questions:

1. **In-genre performance** — How accurately can each model predict whether a review is *Recommended* or *Not Recommended* within a single genre?
2. **Cross-genre generalization (LOGO)** — Train on 5 genres, test on the held-out 6th. Does the model learn *general* sentiment, or just genre-specific vocabulary?
3. **Cross-lingual transfer** — Train on English, test on Russian (and vice versa). Can a multilingual transformer move sentiment knowledge between languages it was never explicitly aligned on for this task?
4. **Sarcasm failure analysis** — Using *funny* votes as a sarcasm proxy, how much worse do both models do on sarcastic reviews than on straightforward ones?

---

## Dataset

Reviews are pulled from Steam's public endpoint (`store.steampowered.com/appreviews/{appid}`), collecting an **equal number of positive and negative reviews per game** so the data is balanced from the start. The label comes from each review's `voted_up` flag (Recommended = 1, Not Recommended = 0).

**6 genres × 3 games = 18 games**, collected in two languages:

| Genre | Games |
|-------|-------|
| Shooter | CS2, Marathon, PAYDAY 3 |
| Roguelike | The Binding of Isaac: Rebirth, Risk of Rain 2, DRG Rogue Core |
| Roleplay | Baldur's Gate 3, Seven Deadly Sins Origin, Monster Hunter Wilds |
| Strategy | Dota 2, Age of Empires II DE, Cities: Skylines II |
| Simulation | The Sims 4, EA SPORTS FC 26, Russian Fishing 4 |
| Survival | Rust, Escape the Backrooms, Necesse |

**Collection targets (per game):**

- English — 750 positive + 750 negative (main experiment)
- Russian — 500 positive + 500 negative (cross-lingual test set)

**Resulting corpus (after cleaning):**

- English: ~26.3k reviews (≈49.7% positive) — **25,511** used in the English-only experiments
- Russian: ~16.6k reviews (≈50.6% positive)

Each `(genre, game, language)` triple is saved as one CSV, e.g. `shooter__CS2__english.csv`, under `steam_reviews/`.

---

## Approach

### Text cleaning — two variants

Steam reviews are messy (BBCode tags, URLs, heart-emoji spam, mixed scripts), so two cleaners are used:

- **Heavy** (`clean_review`) — lowercases, strips BBCode/URLs/hearts, and removes all non-ASCII. Used for the TF-IDF baseline.
- **Light** (`clean_review_light`) — strips only BBCode/URLs, keeps case, punctuation, and Unicode. Used for the transformer and **all** cross-lingual work (heavy cleaning would delete Cyrillic).

The same minimum-length filter (≥ 3 chars on the heavy-cleaned text) is applied across experiments so model comparisons stay fair.

### Models

- **Baseline:** `TfidfVectorizer` (unigrams + bigrams, 50k features, sublinear TF) → `LogisticRegression` (balanced class weights).
- **Transformer:** `xlm-roberta-base` fine-tuned for sequence classification — bottom 6 of 12 encoder layers frozen, `max_length=256`, 3 epochs, lr `2e-5`, fp16 on GPU.

### Evaluation splits

- **In-genre:** stratified 80/20 split within each genre.
- **LOGO (Leave-One-Genre-Out):** train on equal-sized samples from the other 5 genres, test on the held-out genre. The gap between in-genre and LOGO scores measures genre-vocabulary dependence.

---

## Results

> Metric: **macro-F1** on the test split. Numbers are from a single run (`random_state=42`); transformer runs vary slightly between executions.

### 1. In-genre vs. cross-genre (English)

| Genre | TF-IDF (in-genre) | TF-IDF (LOGO) | XLM-R (in-genre) | XLM-R (LOGO) |
|-------|:---:|:---:|:---:|:---:|
| Roguelike | 0.829 | 0.763 | 0.846 | 0.826 |
| Roleplay | 0.840 | 0.789 | 0.885 | 0.874 |
| Shooter | 0.858 | 0.721 | 0.883 | 0.796 |
| Simulation | 0.848 | 0.844 | 0.881 | 0.864 |
| Strategy | 0.820 | 0.813 | 0.863 | 0.852 |
| Survival | 0.818 | 0.813 | 0.841 | 0.844 |

**Takeaways:** Both models score in the low-to-mid 0.80s in-genre. XLM-RoBERTa is consistently stronger and — more importantly — **degrades far less** when tested on an unseen genre. The biggest generalization gap is **shooter** for both models (TF-IDF drops ~0.14, XLM-R ~0.09), suggesting shooter sentiment leans heavily on vocabulary that doesn't transfer (e.g. *cheaters*, server complaints). The TF-IDF model's most informative words confirm this: top-positive cues are *fun, great, best, love, amazing*; top-negative are *not, boring, crashes, trash, worst, cheaters*.

### 2. Cross-lingual transfer (XLM-RoBERTa)

| Direction | Pooled | Per-genre range |
|-----------|:---:|:---:|
| English → Russian | 0.773 | 0.699 (shooter) – 0.797 (roleplay) |
| Russian → English | 0.839 | 0.790 (shooter) – 0.855 (roleplay) |

Sentiment knowledge transfers across languages without any per-task alignment, and **RU → EN is consistently easier than EN → RU**. Shooter is again the hardest genre in both directions.

### 3. Sarcasm failure analysis

Reviews with `votes_funny ≥ 2` are treated as a sarcasm proxy (284 of 5,103 test reviews). Sarcastic reviews are overwhelmingly *negative-leaning* in intent — only **13.4%** are labeled positive, vs **54.6%** for straightforward reviews.

| Subset | TF-IDF F1 | XLM-R F1 |
|--------|:---:|:---:|
| All test | 0.843 | 0.873 |
| Straight (`funny = 0`) | 0.843 | 0.874 |
| Sarcastic (`funny ≥ 2`) | 0.732 | 0.784 |

Both models drop sharply on sarcasm (TF-IDF gap **+0.111**, XLM-R **+0.090**). The transformer is more robust but still struggles — surfaced misclassifications are classic sarcasm/irony cases where surface words contradict intent.

---

## Repository structure

```
.
├── Steam.ipynb                 # Full pipeline (Colab-ready)
├── steam_reviews/              # Collected CSVs: genre__game__language.csv
├── baseline_results/           # TF-IDF + LR metrics, top features, plots
├── transformer_results/        # XLM-RoBERTa metrics + plots
├── crosslingual_results/       # EN↔RU transfer metrics + plots
└── sarcasm_results/            # Sarcasm-slice metrics + misclassified examples
```

Each results folder contains a `*.json` summary and a `plots/` directory (confusion matrices, generalization-gap and comparison bar charts).

---

## Setup & usage

The notebook is built for **Google Colab with a GPU runtime** (the full transformer run takes ~15 minutes on a Colab GPU; on CPU it is very slow).

1. Open `Steam.ipynb` in Colab and set **Runtime → Change runtime type → GPU**.
2. Run the cells top to bottom. The notebook installs its own dependencies:
   ```bash
   pip install requests
   pip install -U transformers accelerate
   ```
3. **Section 1** collects the reviews into `steam_reviews/` (rate-limited; takes a while). Optionally mount Google Drive to persist the data.
4. Run the baseline cells first — the transformer and cross-lingual sections reuse helpers and config (`GENRES`, `clean_review`, split functions, etc.) defined there in the same kernel.

> ⚠️ Steam's review API has no key but is rate-limited (the collector sleeps 1.5s between calls and caps pages per class). Re-running collection re-downloads the data, which may differ slightly over time as new reviews appear.

### Running locally instead

```bash
git clone <your-username>/steam-review-sentiment.git
cd steam-review-sentiment
pip install requests transformers accelerate torch scikit-learn pandas numpy matplotlib
jupyter notebook Steam.ipynb
```

A CUDA-capable GPU is strongly recommended for the XLM-RoBERTa sections.

---

## Tech stack

- **Data collection:** `requests` + Steam public review API
- **Classical ML:** `scikit-learn` (TF-IDF, Logistic Regression)
- **Transformer:** `transformers`, `accelerate`, `torch` (XLM-RoBERTa-base)
- **Data & plotting:** `pandas`, `numpy`, `matplotlib`
- **Environment:** Google Colab (GPU)

---

## Notes & limitations

- Sarcasm is approximated by community *funny* votes, not human annotation — it captures "reviews others found funny," which correlates with but isn't identical to sarcasm.
- Results come from single runs with a fixed seed; transformer scores fluctuate between executions, so treat differences of a few points as noise.
- Labels are Steam's binary Recommended / Not Recommended flag, which is a coarse proxy for sentiment (a positive recommendation can still contain heavy criticism).
- The corpus reflects a specific snapshot of games and reviews and isn't a general-purpose sentiment dataset.

---

## License

No license is specified yet. Add one (e.g. MIT) before publishing if you want to allow reuse. Review text belongs to its original authors and is collected via Steam's public API for research/educational purposes.
