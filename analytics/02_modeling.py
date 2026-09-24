"""Train and evaluate the Titanic classification and fare-regression pipelines."""
from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
PLOTS.mkdir(exist_ok=True)
DATA = HERE / "cleaned_titanic.csv"

FEATURES = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]
NUMERIC = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL = ["sex", "embarked"]


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


def classification_metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, prob),
    }, pred, prob


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError("Run 01_eda.py first so titanic.csv exists.")
    df = pd.read_csv(DATA)
    X = df[FEATURES].copy()
    y = df["survived"].astype(int)

    print("Class balance:")
    print(y.value_counts(normalize=False).rename("count"))
    print(y.value_counts(normalize=True).rename("proportion"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    fitted = {}
    rows = []
    roc_data = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        metrics, pred, prob = classification_metrics(pipe, X_test, y_test)
        fitted[name] = pipe
        rows.append({"model": name, **metrics})
        roc_data[name] = roc_curve(y_test, prob)

        cm = confusion_matrix(y_test, pred)
        print(f"\n{name} confusion matrix:\n{cm}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(HERE / "classification_metrics.csv", index=False)

    # Confusion matrices side by side.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (name, pipe) in zip(axes, fitted.items()):
        pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        ax.imshow(cm)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout(); plt.savefig(PLOTS / "confusion_matrices.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 5))
    for name, (fpr, tpr, _) in roc_data.items():
        auc = metrics_df.loc[metrics_df.model == name, "auc"].iloc[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves"); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "roc_curves.png", dpi=150); plt.close()

    # Decision tree visualization with transformed feature names.
    tree_pipe = fitted["Decision Tree"]
    feature_names = tree_pipe.named_steps["preprocess"].get_feature_names_out()
    plt.figure(figsize=(22, 12))
    plot_tree(tree_pipe.named_steps["model"], feature_names=feature_names, class_names=["0", "1"], filled=False, max_depth=4)
    plt.tight_layout(); plt.savefig(PLOTS / "decision_tree.png", dpi=150); plt.close()

    # Imbalance comparison using Random Forest.
    imbalance_specs = {
        "baseline": RandomForestClassifier(n_estimators=300, random_state=42),
        "class_weight_balanced": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    }
    imbalance_rows = []
    for name, estimator in imbalance_specs.items():
        pipe = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        m, _, _ = classification_metrics(pipe, X_test, y_test)
        imbalance_rows.append({"variant": name, **{k: m[k] for k in ["precision", "recall", "f1"]}})

    smote_pipe = ImbPipeline([
        ("preprocess", make_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
    ])
    smote_pipe.fit(X_train, y_train)
    smote_m, _, _ = classification_metrics(smote_pipe, X_test, y_test)
    imbalance_rows.append({"variant": "SMOTE_train_only", **{k: smote_m[k] for k in ["precision", "recall", "f1"]}})
    imbalance_df = pd.DataFrame(imbalance_rows)
    imbalance_df.to_csv(HERE / "imbalance_comparison.csv", index=False)

    # GridSearchCV with OOB enabled.
    rf_pipe = Pipeline([("preprocess", make_preprocessor()), ("model", RandomForestClassifier(oob_score=True, random_state=42))])
    grid = GridSearchCV(
        rf_pipe,
        param_grid={
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 5, 10],
            "model__max_features": ["sqrt", "log2"],
        },
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    best_oob = best_rf.named_steps["model"].oob_score_

    # Regression: predict fare using all other available cleaned features except fare.
    reg_target = df["fare"].astype(float)
    reg_features = [c for c in df.columns if c not in ["fare"]]
    X_reg = df[reg_features].copy()
    y_reg = reg_target
    # Separate numeric/categorical columns dynamically.
    reg_num = [c for c in X_reg.columns if pd.api.types.is_numeric_dtype(X_reg[c])]
    reg_cat = [c for c in X_reg.columns if c not in reg_num]
    reg_pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), reg_num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), reg_cat),
    ])
    reg_pipe = Pipeline([("preprocess", reg_pre), ("model", LinearRegression())])
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.20, random_state=42)
    reg_pipe.fit(Xr_train, yr_train)
    pred_fare = reg_pipe.predict(Xr_test)
    mae = mean_absolute_error(yr_test, pred_fare)
    rmse = mean_squared_error(yr_test, pred_fare) ** 0.5
    r2 = reg_pipe.score(Xr_test, yr_test)
    n = len(yr_test)
    p = len(reg_pipe.named_steps["preprocess"].get_feature_names_out())
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

    residuals = yr_test - pred_fare
    plt.figure(figsize=(7, 4)); plt.scatter(pred_fare, residuals, alpha=0.7); plt.axhline(0, linestyle="--"); plt.xlabel("Predicted fare"); plt.ylabel("Residual"); plt.title("Fare regression residuals"); plt.tight_layout(); plt.savefig(PLOTS / "fare_residuals.png", dpi=150); plt.close()
    hetero = "shows possible heteroscedasticity" if abs(np.corrcoef(np.abs(residuals), pred_fare)[0, 1]) > 0.25 else "does not show strong evidence of heteroscedasticity"

    # Select the classification pipeline using the highest F1, then save the complete pipeline.
    best_name = metrics_df.sort_values(["f1", "auc"], ascending=False).iloc[0]["model"]
    best_pipeline = fitted[best_name]
    joblib.dump(best_pipeline, HERE / "best_pipeline.joblib")

    # Reload and prove raw-input prediction works.
    reloaded = joblib.load(HERE / "best_pipeline.joblib")
    reload_prediction = reloaded.predict(X_test.head(3)).tolist()

    report = [
        "# Modeling Report", "",
        "## Why stratification is used", 
        "The train/test split uses `stratify=y` so the proportion of survived/non-survived cases is preserved in both subsets. This reduces the risk that a small test set has an unrepresentative class mix.", "",
        "## Classification metrics", "",
        metrics_df.to_markdown(index=False, floatfmt=".4f"), "",
        "## Imbalance comparison", "",
        imbalance_df.to_markdown(index=False, floatfmt=".4f"), "",
        f"The selected imbalance strategy for this report is the variant with the highest F1 in the table: **{imbalance_df.sort_values('f1', ascending=False).iloc[0]['variant']}**. The comparison is based on the held-out test set after all preprocessing is structurally confined to the training data.", "",
        "## Random Forest GridSearchCV", "",
        f"Best parameters: `{grid.best_params_}`",
        f"Best cross-validation F1: **{grid.best_score_:.4f}**",
        f"OOB score from the refit `RandomForestClassifier(oob_score=True, ...)`: **{best_oob:.4f}**", "",
        "## Regression — fare", "",
        f"MAE: **{mae:.4f}**  ", f"RMSE: **{rmse:.4f}**  ", f"R²: **{r2:.4f}**  ", f"Adjusted R²: **{adj_r2:.4f}**", "",
        f"Residual interpretation: **{hetero}**. This is based on the residual-vs-predicted plot and the spread of residuals, not on a formal heteroscedasticity test.", "",
        "## Final model comparison", "",
        "Classification metrics and regression metrics are intentionally kept as separate groups because they measure different tasks and are not directly comparable.", "",
        "### Classification", "",
        metrics_df.to_markdown(index=False, floatfmt=".4f"), "",
        "### Regression", "",
        pd.DataFrame([{"model": "Linear Regression (fare)", "MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted_R2": adj_r2}]).to_markdown(index=False, floatfmt=".4f"), "",
        f"### Deployment choice\nThe complete pipeline selected for deployment is **{best_name}**, because it has the highest held-out F1 in the classification comparison (with AUC used as the secondary tie-breaker). Its preprocessing and estimator are saved together in `best_pipeline.joblib`, so new raw rows can be passed directly to the artifact. The other metrics should be considered alongside F1 when interpreting the model's behavior.\n",
        f"Reload check: predictions on three raw test rows were `{reload_prediction}`.",
    ]
    (HERE / "MODELING_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"Best RF params: {grid.best_params_}; OOB={best_oob:.4f}")
    print(f"Saved {HERE / 'best_pipeline.joblib'}")


if __name__ == "__main__":
    main()
