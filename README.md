## News Sentiment Analysis

1. Project Structure
The repository follows a standard data science project structure:

## Project Structure

```
news-sentiment-analysis/
├── data/
│   ├── newData/
│   │   └── raw_analyst_ratings.csv   # Financial news headlines
│   └── yfinance_data/Data/           # Historical stock prices (one CSV per ticker)
├── notebooks/
│   ├── task_1_eda.ipynb              # Exploratory data analysis
│   ├── task_2_technical_indicators.ipynb  # TA-Lib indicators & PyNance metrics
│   └── task_3_sentiment_correlation.ipynb # Sentiment vs return correlation
├── tests/
├── scripts/
├── requirements.txt
└── README.md
```

---

## Tasks

### Task 1 — Exploratory Data Analysis
- Headline length statistics and publisher activity
- Keyword and topic extraction (TF-IDF / LDA)
- Publication frequency time-series and spike analysis

### Task 2 — Technical Indicators
- SMA, EMA, RSI, and MACD computed with TA-Lib
- Additional financial metrics via PyNance
- Visualizations of price action vs indicators

### Task 3 — Sentiment vs Stock Returns
- Headline sentiment scored with **TextBlob** and **VADER**
- News dates aligned to trading days (weekends/holidays forwarded)
- Pearson correlation between avg daily sentiment and daily return
- Category analysis (Positive / Neutral / Negative days) and rolling correlation

---

## Setup

```bash
# Clone the repo and create a virtual environment
git clone https://github.com/Helina-Yilma/news-sentiment-analysis.git   
cd news-sentiment-analysis
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download VADER lexicon (required for Task 3)
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## Requirements

Key libraries used:

```
pandas
numpy
matplotlib
seaborn
scipy
textblob
vaderSentiment
ta-lib
pynance
jupyter
```

> Full list with pinned versions is in `requirements.txt`.

---

## CI/CD

GitHub Actions runs unit tests on every push via `.github/workflows/unittests.yml`.

---

## Data Sources

| Dataset | Source |
|---------|--------|
| Analyst ratings / headlines | `data/newData/raw_analyst_ratings.csv` |
| Historical prices | Yahoo Finance via `yfinance` |

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, reviewed code |
| `task-1` | EDA work |
| `task-2` | Technical indicators |
| `task-3` | Sentiment correlation |

