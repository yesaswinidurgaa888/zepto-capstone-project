# Analytics Pipeline

## Run order

From the repository root:

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

`01_eda.py` is the **only** place that calls `sns.load_dataset("titanic")`. It immediately commits the resulting raw DataFrame to `analytics/titanic.csv`. `02_modeling.py` reads that CSV and never calls `sns.load_dataset` again.

## Outputs

- `titanic.csv`: offline fallback required by the assignment.
- `cleaned_titanic.csv`: the cleaned data used by the later modeling stage.
- `plots/`: EDA, correlation, decision-tree, ROC, confusion-matrix and residual plots.
- `EDA_REPORT.md`: generated written analysis, including missing percentages and chart interpretations.
- `MODELING_REPORT.md`: generated model metrics, imbalance comparison, GridSearchCV/OOB results, regression metrics and deployment choice.
- `best_pipeline.joblib`: complete fitted preprocessing + estimator pipeline, reloadable on raw feature columns.

## Modeling leakage control

The train/test split is stratified before modeling. Numeric imputation/scaling and categorical imputation/encoding are inside a scikit-learn `ColumnTransformer` inside a `Pipeline`, so fitting occurs on the training data and the test data is only transformed. SMOTE is inside an imbalanced-learn pipeline and therefore runs only on the training fold.
