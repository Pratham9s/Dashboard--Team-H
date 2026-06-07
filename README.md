# Northcard Capital Analytics Dashboard
## Where Should Global Businesses Invest Next?

**Data Science Capstone | Northcard Capital Analytics**
Mentor: Binish Thomas | Team: Anant, Lakshmi Priya, Yash, Supriya, Pratham, Sumukh, Chinni Sumanth, Venkat, Harish

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Push this entire folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click Deploy

## Folder Structure

```
dashboard/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── README.md
└── data/
    ├── master_wide.csv             # Master dataset (194 countries, 2000–2023)
    ├── cluster_labels.csv          # Task 2: KMeans cluster labels
    ├── cluster_assignments.csv     # Task 3: Hierarchical cluster assignments
    ├── task4_importance_table.csv  # Task 4: SHAP feature importance
    ├── task4_shap_values.csv       # Task 4: Per-country SHAP values
    ├── task5_forecasts.csv         # Task 5: Prophet forecasts 2024–2027
    ├── task5_intervals.csv         # Task 5: Confidence intervals
    ├── task6_recommendation.csv    # Task 6: Investment recommendations
    ├── task6_scenario_forecasts.csv
    ├── task7_scenario_comparison.csv  # Task 7: A/B/C scenario comparison
    └── knn_investment_results.csv  # Task 1: KNN predictions (optional)
```

## Pages

| Page | Description |
|------|-------------|
| Overview | Project intro, KPIs, global map, pipeline |
| Country Explorer | Historical + forecast trends per country, radar chart |
| ML Models | KNN, KMeans, Hierarchical, XGBoost+SHAP results |
| Forecasting | Prophet 2024–2027, top 20 countries, global map |
| Policy Scenarios | A/B/C scenario comparison for India, USA, Viet Nam |
| Investment Recommendations | Final verdicts, rationale, key insights |

## Notes

- `cluster_assignments.csv` is regenerated from `master_wide.csv` using Task 3 logic (Ward linkage, k=4)
- All scores are normalised [0, 1]
- Composite = FDI×0.30 + Banking×0.25 + Manuf×0.25 + Digital×0.20
