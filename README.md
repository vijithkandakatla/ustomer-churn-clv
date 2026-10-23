# Customer Churn, Survival Analysis & Lifetime Value

Predicting **who** will churn (classification), **when** they will churn (survival analysis) and **what they are worth** (customer lifetime value). The three are combined into a value-versus-risk segmentation that tells a retention team whom to call first.

`Python` · `scikit-learn` · `XGBoost` · `lifelines` · `Survival Analysis` · `CLV` · `Customer Analytics`

## Why three models?

| Question | Model | Why it matters |
|---|---|---|
| Who is likely to leave? | Logistic Regression vs XGBoost classifier | Ranks customers for retention campaigns |
| When will they leave? | Kaplan-Meier + Cox proportional hazards | Handles **censoring**: active customers haven't churned *yet*. Plain classifiers ignore this. |
| Who is worth saving? | Survival-based 24-month discounted CLV | Spends retention budget where the most value is at risk |

This mirrors work I did at Mu Sigma, where survival and CLV models flagged at-risk customers earlier than rule-based triggers. Here it is rebuilt on a public dataset.

## Data

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d): 7,043 subscription customers with contract, services, billing, tenure (months) and churn flag. Downloaded automatically.

## Approach

1. **Churn classification:** one-hot features, class-imbalance handling, 5-fold stratified CV. Reports ROC-AUC, PR-AUC, F1 and the business metric that matters: how many churners you reach by contacting the top 10% riskiest customers.
2. **Survival analysis:** Kaplan-Meier curves by contract type with a log-rank test. A penalised Cox PH model gives hazard ratios (which factors speed up or slow down churn) and its concordance index is reported.
3. **CLV:** for every active customer, the Cox model's survival curve *conditional on their current tenure* gives expected paying months over the next 24 months. That is multiplied by monthly charges × margin and discounted.
4. **Segmentation:** value (median CLV) × risk (top quartile of 12-month churn probability) produces a ranked retention target list.

## Results

<!-- RESULTS:START -->
_Run `python run.py` to generate results. This section is filled in automatically._
<!-- RESULTS:END -->

## How to run

```bash
git clone https://github.com/vijithkandakatla/customer-churn-clv.git
cd customer-churn-clv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py                 # --margin 0.4 to change the profit-margin assumption
pytest
```

Outputs in `results/`: model scores, Cox hazard ratios, Kaplan-Meier and gains charts, CLV score for every customer, and `retention_target_list.csv`.

## Project structure

```
├── run.py
├── src/
│   ├── data.py           # download, cleaning, one-hot encoding
│   ├── churn_model.py    # classifiers, CV, lift metrics
│   ├── survival_clv.py   # Kaplan-Meier, Cox PH, CLV, segments
│   └── readme_utils.py
├── tests/
└── results/              # generated
```

## Assumptions and limitations

Margin (30%) and discount rate (10% per year) are assumptions; change them with flags. The Cox model assumes proportional hazards, and the full summary in `results/cox_full_summary.csv` can be used to check this. This is a single-snapshot dataset, so there is no out-of-time validation.

## Author

**Vijith Kandakatla** · Data Scientist · M.Sc. Autonomy Technologies, FAU Erlangen-Nürnberg · [LinkedIn](https://www.linkedin.com/in/vijith-kandakatla-8a023a111)
