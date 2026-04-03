# 🏨 Booking.com — Hotel Recommendation System & Analytics Suite

A multi-module data science project that analyses hotel booking behaviour, predicts hotel service quality, and evaluates pricing strategies — combining **Python**, **Tableau**, and **machine learning** across four independent analytical workstreams.

---

## 📌 Project Overview

This repository brings together four interconnected analyses built around hotel and e-commerce datasets. The work spans exploratory data analysis, classification and regression modelling, A/B testing, word cloud generation, and interactive Tableau dashboards — all framed around real business questions a platform like Booking.com or Agoda would ask.

---

## 📁 Project Structure

```
├── Data/
│   ├── hotel_bookings.csv               # 119k hotel booking records (main cancellation dataset)
│   ├── Hotel Booking.csv                # Booking.com scraped hotel data (ratings, amenities)
│   ├── online_shoppers_intention.csv    # UCI online shoppers dataset (12k sessions)
│   ├── New_Updated_df.csv               # Processed output dataset
│   └── Word_Data.csv                    # Pre-processed amenities text data
│
├── Booking Cancellation (Done).py               # Module 1: Cancellation rate analysis
├── Booking.com -- Hotels Ratings --             
│   Classification Models (The Latest Version).py # Module 2: Hotel service classification
├── Agoda Assignment _ AB Testing +
│   Linear Regression.py                         # Module 3: Pricing & A/B test analysis
├── Online Shoppers Purchasing Intention
│   Project.py                                   # Module 4: E-commerce conversion analysis
├── Concatenate the raw datasets (Done).py       # Data pipeline: merges 5-city CSVs
├── Booking Analysis in Tableau.twbx             # Tableau dashboard workbook
├── Booking Cancellation.pdf                     # PDF report for Module 1
├── Agoda Assignment _ AB Testing +
│   Linear Regression.pdf                        # PDF report for Module 3
└── README.md
```

---

## 🔬 Modules

---

### Module 1 — Hotel Booking Cancellation Analysis
**Script:** `Booking Cancellation (Done).py`  
**Dataset:** `hotel_bookings.csv` (119,390 records, 2015–2017)  
**Source:** [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

**Business Questions Answered:**
1. What is the overall cancellation rate?
2. Which days and months see the highest cancellations?
3. Does family type (adults vs. children/babies) affect cancellation rate?
4. What are the statistical characteristics of cancellations (lead time, special requests)?
5. Which booking distribution channel drives the most cancellations?

**Key Findings:**
- Overall cancellation rate: **37.04%**
- Travel Agencies (TA) and Tour Operators (TO) account for the highest cancellation volume (~40k over 3 years)
- Longer lead time (avg. 130+ days) is positively correlated with higher cancellation likelihood
- 2016–2017 monthly breakdown reveals seasonal cancellation peaks

**Techniques:** Data wrangling, mode imputation, correlation heatmap, annotated bar charts, box plots, grouped bar charts by country and distribution channel

---

### Module 2 — Hotel Ratings Classification & Service Level Clustering
**Script:** `Booking.com -- Hotels Ratings -- Classification Models (The Latest Version).py`  
**Dataset:** `Hotel Booking.csv` (scraped Booking.com hotel data)

**Pipeline:**
1. **Data Cleaning** — dropped high-missing columns (`lat`, `long`, `address`, `pageurl`); filled nulls with zero for imputation
2. **Feature Engineering** — extracted numeric star rating from string, label-encoded hotel type, parsed crawl timestamps
3. **KNN Imputation** — predicted missing `average_rating` (k=9) and `Rating` (k=5) values using KNN classifier
4. **K-Means Clustering** (k=5, Elbow Method) — generated a `Service Level` target variable with 5 tiers:
   - Usual Service → Great Service → High Service → Very High Service → Exceptional Service
5. **AutoML with PyCaret** — compared and selected best classifier across KNN, LR, Naive Bayes, Decision Tree, and SVM
6. **Word Cloud** — extracted top amenity keywords from hotels with `average_rating ≥ 6`

**Visualisations:** Scatter plots (rating vs. reviews/photos/cleanliness/WiFi/comfort), heatmap, box plot (value-for-money by hotel type and star rating), bar chart (hotel type vs. review count), word cloud

---

### Module 3 — Agoda Pricing Analysis: A/B Testing & Regression Modelling
**Script:** `Agoda Assignment _ AB Testing + Linear Regression.py`  
**Dataset:** 5-city booking data (City A–E), merged via `Concatenate the raw datasets (Done).py`

**Business Questions Answered:**
- At which funnel stage should an urgency message be implemented?
- What factors drive hotel pricing (ADR — Average Daily Rate in USD)?
- How do booking-to-checkin lead time and stay duration affect revenue?
- Which hotels and days generate the highest cumulative revenue?

**Key Findings:**
- **Hotel star rating** and **stay duration** are the strongest positive price predictors
- **Chain hotel** status and longer **lead time** have a negative relationship with ADR
- Booking traffic is highest Mon–Fri; Saturday and Sunday see the least activity
- Hotels rated 3–4 stars are the most frequently booked category
- **1-day stays** are the most common trip duration across all cities

**Models Used:**
- **Linear Regression** — baseline price prediction
- **Gradient Boosting Regressor** — improved ADR prediction accuracy
- **Evaluation metrics:** RMSLE, RMSE, R²

**Data Pipeline:** `Concatenate the raw datasets (Done).py` standardises column names across 5 city-level CSVs and merges them into a single dataframe

---

### Module 4 — Online Shoppers Purchasing Intention
**Script:** `Online Shoppers Purchasing Intention Project.py`  
**Dataset:** `online_shoppers_intention.csv` (12,330 sessions)  
**Source:** UCI Machine Learning Repository

**Business Questions Answered:**
1. What is the page conversion rate?
2. How does time-on-page relate to the likelihood of a purchase?
3. Do special days or specific months optimise revenue?
4. What are the primary drivers of the Revenue outcome?

**Key Findings:**
- **Conversion rate: 15.47%**
- Visitors spending under 1,000 seconds on product pages are most likely to convert
- Special days (Mother's Day, Valentine's Day) show **no positive effect** on conversion — non-special days convert better
- **Top revenue months:** November, October, September, August, July
- **Top revenue drivers:** Page Values → Product Related pages → Informational pages

**Visualisations:** Scatter plot (informational duration vs. page values, sized by revenue), heatmap, bar charts by month and special day, feature-vs-revenue bar charts

---

## 📊 Tableau Dashboard
**File:** `Booking Analysis in Tableau.twbx`

Open with [Tableau Public](https://public.tableau.com/) or Tableau Desktop. The dashboard visualises key metrics from the cancellation and booking analyses including cancellation rate breakdowns, country-level booking patterns, and distribution channel performance.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation and wrangling |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | KNN imputation, K-Means, LR, GBR, label encoding |
| `pycaret` | AutoML classification benchmarking |
| `wordcloud` | Amenity keyword visualisation |
| `Tableau` | Interactive business intelligence dashboard |

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn pycaret wordcloud
```

2. Place all CSV files from the `Data/` folder in the same directory as the scripts

3. To merge the Agoda city data first:
```bash
python "Concatenate the raw datasets (Done).py"
```

4. Then run any module independently:
```bash
python "Booking Cancellation (Done).py"
python "Booking.com -- Hotels Ratings -- Classification Models (The Latest Version).py"
python "Agoda Assignment _ AB Testing + Linear Regression.py"
python "Online Shoppers Purchasing Intention Project.py"
```

5. Open `Booking Analysis in Tableau.twbx` in Tableau Desktop or Tableau Public

---

## 📝 Notes & Limitations

- The hotel ratings dataset is scraped web data — missing value rates are high for some columns (lat/lng, URLs) and were dropped rather than imputed
- The `Service Level` clustering target (Module 2) is self-generated via K-Means — findings should be interpreted as exploratory rather than ground-truth labels
- The Agoda dataset covers a limited time window (prime booking months only), which limits seasonality analysis; a full year of data would strengthen conclusions
- The 5-city dataset is **imbalanced** — two cities account for ~66% of records, so city-level comparisons should be treated with caution

---

## 👤 Author

Independent data analytics portfolio project — combining hotel industry business intelligence, machine learning, and Tableau visualisation across four analytical case studies.
