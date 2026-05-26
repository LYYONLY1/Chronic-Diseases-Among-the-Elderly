import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from elderly_risk_system import ElderlyRiskSystem


def build_preprocessor(x_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in x_df.columns if c not in numeric_cols]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def evaluate_models(data_dir: Path, out_dir: Path, test_size: float = 0.25) -> pd.DataFrame:
    system = ElderlyRiskSystem(data_dir=data_dir, output_dir=out_dir)
    feature_df = system.build_feature_table()
    y = feature_df["target_hypertension"]
    x_full = feature_df.drop(columns=["target_hypertension"])
    user_ids = x_full["user_id"].astype(str)
    x = x_full.drop(columns=["user_id"])

    x_train, x_test, y_train, y_test, uid_train, uid_test = train_test_split(
        x, y, user_ids, test_size=test_size, random_state=42, stratify=y
    )
    preprocessor = build_preprocessor(x_train)

    models: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42
        ),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    rows: List[Dict[str, float]] = []
    pred_rows: List[pd.DataFrame] = []

    fitted_pipelines: Dict[str, Pipeline] = {}
    for name, model in models.items():
        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        clf.fit(x_train, y_train)
        fitted_pipelines[name] = clf
        pred = clf.predict(x_test)
        prob = clf.predict_proba(x_test)[:, 1]

        rows.append(
            {
                "model": name,
                "auc": round(float(roc_auc_score(y_test, prob)), 4),
                "f1": round(float(f1_score(y_test, pred)), 4),
                "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            }
        )

        tmp = uid_test.to_frame(name="user_id")
        tmp["model"] = name
        tmp["y_true"] = y_test.values
        tmp["y_pred"] = pred
        tmp["risk_prob"] = prob
        pred_rows.append(tmp)

    # Automatic parameter optimization for RandomForest
    rf_search = RandomizedSearchCV(
        estimator=Pipeline(
            steps=[("preprocess", preprocessor), ("model", RandomForestClassifier(random_state=42))]
        ),
        param_distributions={
            "model__n_estimators": [150, 250, 350, 500],
            "model__max_depth": [5, 8, 12, None],
            "model__min_samples_leaf": [1, 2, 4, 6],
            "model__min_samples_split": [2, 4, 8],
        },
        n_iter=8,
        cv=3,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc",
    )
    rf_search.fit(x_train, y_train)
    tuned_rf = rf_search.best_estimator_
    tuned_pred = tuned_rf.predict(x_test)
    tuned_prob = tuned_rf.predict_proba(x_test)[:, 1]
    rows.append(
        {
            "model": "TunedRandomForest",
            "auc": round(float(roc_auc_score(y_test, tuned_prob)), 4),
            "f1": round(float(f1_score(y_test, tuned_pred)), 4),
            "accuracy": round(float(accuracy_score(y_test, tuned_pred)), 4),
        }
    )

    tmp = uid_test.to_frame(name="user_id")
    tmp["model"] = "TunedRandomForest"
    tmp["y_true"] = y_test.values
    tmp["y_pred"] = tuned_pred
    tmp["risk_prob"] = tuned_prob
    pred_rows.append(tmp)

    et_search = RandomizedSearchCV(
        estimator=Pipeline(
            steps=[("preprocess", preprocessor), ("model", ExtraTreesClassifier(random_state=42))]
        ),
        param_distributions={
            "model__n_estimators": [150, 250, 350, 500],
            "model__max_depth": [5, 8, 12, None],
            "model__min_samples_leaf": [1, 2, 4, 6],
            "model__min_samples_split": [2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
        },
        n_iter=8,
        cv=3,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc",
    )
    et_search.fit(x_train, y_train)
    tuned_et = et_search.best_estimator_
    tuned_et_pred = tuned_et.predict(x_test)
    tuned_et_prob = tuned_et.predict_proba(x_test)[:, 1]
    rows.append(
        {
            "model": "TunedExtraTrees",
            "auc": round(float(roc_auc_score(y_test, tuned_et_prob)), 4),
            "f1": round(float(f1_score(y_test, tuned_et_pred)), 4),
            "accuracy": round(float(accuracy_score(y_test, tuned_et_pred)), 4),
        }
    )
    tmp = uid_test.to_frame(name="user_id")
    tmp["model"] = "TunedExtraTrees"
    tmp["y_true"] = y_test.values
    tmp["y_pred"] = tuned_et_pred
    tmp["risk_prob"] = tuned_et_prob
    pred_rows.append(tmp)

    gb_search = RandomizedSearchCV(
        estimator=Pipeline(
            steps=[("preprocess", preprocessor), ("model", GradientBoostingClassifier(random_state=42))]
        ),
        param_distributions={
            "model__n_estimators": [100, 150, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
            "model__min_samples_leaf": [1, 2, 5],
        },
        n_iter=8,
        cv=3,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc",
    )
    gb_search.fit(x_train, y_train)
    tuned_gb = gb_search.best_estimator_
    tuned_gb_pred = tuned_gb.predict(x_test)
    tuned_gb_prob = tuned_gb.predict_proba(x_test)[:, 1]
    rows.append(
        {
            "model": "TunedGradientBoosting",
            "auc": round(float(roc_auc_score(y_test, tuned_gb_prob)), 4),
            "f1": round(float(f1_score(y_test, tuned_gb_pred)), 4),
            "accuracy": round(float(accuracy_score(y_test, tuned_gb_pred)), 4),
        }
    )
    tmp = uid_test.to_frame(name="user_id")
    tmp["model"] = "TunedGradientBoosting"
    tmp["y_true"] = y_test.values
    tmp["y_pred"] = tuned_gb_pred
    tmp["risk_prob"] = tuned_gb_prob
    pred_rows.append(tmp)

    # Fusion model (soft averaging of tuned components)
    probs = [tuned_rf.predict_proba(x_test)[:, 1], tuned_et.predict_proba(x_test)[:, 1], tuned_gb.predict_proba(x_test)[:, 1]]
    fusion_prob = sum(probs) / len(probs)
    fusion_pred = (fusion_prob >= 0.5).astype(int)
    rows.append(
        {
            "model": "Fusion(TunedSoftVoting)",
            "auc": round(float(roc_auc_score(y_test, fusion_prob)), 4),
            "f1": round(float(f1_score(y_test, fusion_pred)), 4),
            "accuracy": round(float(accuracy_score(y_test, fusion_pred)), 4),
        }
    )
    tmp = uid_test.to_frame(name="user_id")
    tmp["model"] = "Fusion(TunedSoftVoting)"
    tmp["y_true"] = y_test.values
    tmp["y_pred"] = fusion_pred
    tmp["risk_prob"] = fusion_prob
    pred_rows.append(tmp)

    result_df = pd.DataFrame(rows).sort_values(by="auc", ascending=False).reset_index(drop=True)
    pred_df = pd.concat(pred_rows, ignore_index=True)

    result_df.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "model_test_predictions.csv", index=False, encoding="utf-8-sig")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Compare baseline models for elderly chronic risk warning.")
    parser.add_argument("--data-dir", type=str, default=".", help="Path of dataset folder.")
    parser.add_argument("--out-dir", type=str, default="./outputs", help="Output directory.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test set ratio.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result_df = evaluate_models(data_dir=data_dir, out_dir=out_dir, test_size=args.test_size)
    print("=== Model Comparison ===")
    print(result_df.to_string(index=False))
    print(f"Saved: {out_dir / 'model_comparison.csv'}")
    print(f"Saved: {out_dir / 'model_test_predictions.csv'}")


if __name__ == "__main__":
    main()
