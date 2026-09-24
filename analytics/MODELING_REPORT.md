# Modeling Report

## Why stratification is used
The train/test split uses `stratify=y` so the proportion of survived/non-survived cases is preserved in both subsets. This reduces the risk that a small test set has an unrepresentative class mix.

## Classification metrics

| model               |   accuracy |   precision |   recall |     f1 |    auc |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8090 |      0.7833 |   0.6912 | 0.7344 | 0.8610 |
| Decision Tree       |     0.7640 |      0.7600 |   0.5588 | 0.6441 | 0.8374 |
| Random Forest       |     0.8034 |      0.7619 |   0.7059 | 0.7328 | 0.8237 |

## Imbalance comparison

| variant               |   precision |   recall |     f1 |
|:----------------------|------------:|---------:|-------:|
| baseline              |      0.7619 |   0.7059 | 0.7328 |
| class_weight_balanced |      0.7742 |   0.7059 | 0.7385 |
| SMOTE_train_only      |      0.7612 |   0.7500 | 0.7556 |

The selected imbalance strategy for this report is the variant with the highest F1 in the table: **SMOTE_train_only**. The comparison is based on the held-out test set after all preprocessing is structurally confined to the training data.

## Random Forest GridSearchCV

Best parameters: `{'model__max_depth': None, 'model__max_features': 'sqrt', 'model__n_estimators': 300}`
Best cross-validation F1: **0.7449**
OOB score from the refit `RandomForestClassifier(oob_score=True, ...)`: **0.8073**

## Regression — fare

MAE: **18.3735**
RMSE: **41.2921**
R²: **0.3609**
Adjusted R²: **0.2655**

Residual interpretation: **shows possible heteroscedasticity**. This is based on the residual-vs-predicted plot and the spread of residuals, not on a formal heteroscedasticity test.

## Final model comparison

Classification metrics and regression metrics are intentionally kept as separate groups because they measure different tasks and are not directly comparable.

### Classification

| model               |   accuracy |   precision |   recall |     f1 |    auc |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8090 |      0.7833 |   0.6912 | 0.7344 | 0.8610 |
| Decision Tree       |     0.7640 |      0.7600 |   0.5588 | 0.6441 | 0.8374 |
| Random Forest       |     0.8034 |      0.7619 |   0.7059 | 0.7328 | 0.8237 |

### Regression

| model                    |     MAE |    RMSE |     R2 |   Adjusted_R2 |
|:-------------------------|--------:|--------:|-------:|--------------:|
| Linear Regression (fare) | 18.3735 | 41.2921 | 0.3609 |        0.2655 |

### Deployment choice
The complete pipeline selected for deployment is **Logistic Regression**, because it has the highest held-out F1 in the classification comparison (with AUC used as the secondary tie-breaker). Its preprocessing and estimator are saved together in `best_pipeline.joblib`, so new raw rows can be passed directly to the artifact. The other metrics should be considered alongside F1 when interpreting the model's behavior.

Reload check: predictions on three raw test rows were `[0, 0, 0]`.
