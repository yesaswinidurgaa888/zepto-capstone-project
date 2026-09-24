# Modeling Report

## Why stratification is used
The train/test split uses `stratify=y` so the proportion of survived/non-survived cases is preserved in both subsets. This reduces the risk that a small test set has an unrepresentative class mix.

## Classification metrics

| model               |   accuracy |   precision |   recall |     f1 |    auc |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8045 |      0.7931 |   0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |     0.7654 |      0.7547 |   0.5797 | 0.6557 | 0.7971 |
| Random Forest       |     0.8101 |      0.7966 |   0.6812 | 0.7344 | 0.8287 |

## Imbalance comparison

| variant               |   precision |   recall |     f1 |
|:----------------------|------------:|---------:|-------:|
| baseline              |      0.7966 |   0.6812 | 0.7344 |
| class_weight_balanced |      0.8000 |   0.6957 | 0.7442 |
| SMOTE_train_only      |      0.7727 |   0.7391 | 0.7556 |

The selected imbalance strategy for this report is the variant with the highest F1 in the table: **SMOTE_train_only**. The comparison is based on the held-out test set after all preprocessing is structurally confined to the training data.

## Random Forest GridSearchCV

Best parameters: `{'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__n_estimators': 100}`
Best cross-validation F1: **0.7459**
OOB score from the refit `RandomForestClassifier(oob_score=True, ...)`: **0.8272**

## Regression — fare

MAE: **17.2892**  
RMSE: **28.9728**  
R²: **0.4575**  
Adjusted R²: **0.3476**

Residual interpretation: **shows possible heteroscedasticity**. This is based on the residual-vs-predicted plot and the spread of residuals, not on a formal heteroscedasticity test.

## Final model comparison

Classification metrics and regression metrics are intentionally kept as separate groups because they measure different tasks and are not directly comparable.

### Classification

| model               |   accuracy |   precision |   recall |     f1 |    auc |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8045 |      0.7931 |   0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |     0.7654 |      0.7547 |   0.5797 | 0.6557 | 0.7971 |
| Random Forest       |     0.8101 |      0.7966 |   0.6812 | 0.7344 | 0.8287 |

### Regression

| model                    |     MAE |    RMSE |     R2 |   Adjusted_R2 |
|:-------------------------|--------:|--------:|-------:|--------------:|
| Linear Regression (fare) | 17.2892 | 28.9728 | 0.4575 |        0.3476 |

### Deployment choice
The complete pipeline selected for deployment is **Random Forest**, because it has the highest held-out F1 in the classification comparison (with AUC used as the secondary tie-breaker). Its preprocessing and estimator are saved together in `best_pipeline.joblib`, so new raw rows can be passed directly to the artifact. The other metrics should be considered alongside F1 when interpreting the model's behavior.

Reload check: predictions on three raw test rows were `[0, 0, 0]`.