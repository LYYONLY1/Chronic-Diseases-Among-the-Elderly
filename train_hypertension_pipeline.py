import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_age_group_to_mid(age_group: str) -> float:
    if pd.isna(age_group):
        return np.nan
    text = str(age_group).strip()
    text = text.replace("[", "").replace("]", "")
    parts = [x.strip() for x in text.split(",") if x.strip()]
    if len(parts) != 2:
        return np.nan
    try:
        low = float(parts[0])
        high = float(parts[1])
        return (low + high) / 2.0
    except ValueError:
        return np.nan


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def aggregate_scanwatch_hr(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty or "Heart Rate" not in df.columns:
        return {}
    hr = pd.to_numeric(df["Heart Rate"], errors="coerce").dropna()
    if hr.empty:
        return {}
    return {
        "sw_hr_mean": hr.mean(),
        "sw_hr_std": hr.std(ddof=0),
        "sw_hr_min": hr.min(),
        "sw_hr_max": hr.max(),
        "sw_hr_q25": hr.quantile(0.25),
        "sw_hr_q75": hr.quantile(0.75),
    }


def aggregate_scanwatch_steps(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty or "Steps" not in df.columns:
        return {}
    steps = pd.to_numeric(df["Steps"], errors="coerce").dropna()
    if steps.empty:
        return {}
    nonzero_days = (steps > 0).sum()
    return {
        "steps_mean": steps.mean(),
        "steps_std": steps.std(ddof=0),
        "steps_sum": steps.sum(),
        "steps_q75": steps.quantile(0.75),
        "steps_nonzero_ratio": nonzero_days / len(steps),
    }


def aggregate_sleep_physio(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    out = {}
    for col, prefix in [
        ("Heart Rate", "sp_hr"),
        ("Respiration Rate", "sp_rr"),
        ("Snoring", "sp_snoring"),
        ("SDNN_1", "sp_sdnn"),
    ]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                out[f"{prefix}_mean"] = values.mean()
                out[f"{prefix}_std"] = values.std(ddof=0)
                out[f"{prefix}_q75"] = values.quantile(0.75)
    return out


def aggregate_sleep_state(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty or "Sleep state" not in df.columns:
        return {}
    if "Start time" not in df.columns or "End time" not in df.columns:
        return {}

    tmp = df.copy()
    tmp["Start time"] = pd.to_datetime(tmp["Start time"], errors="coerce")
    tmp["End time"] = pd.to_datetime(tmp["End time"], errors="coerce")
    tmp = tmp.dropna(subset=["Start time", "End time", "Sleep state"])
    if tmp.empty:
        return {}

    tmp["duration_min"] = (
        (tmp["End time"] - tmp["Start time"]).dt.total_seconds() / 60.0
    ).clip(lower=0)
    total_duration = tmp["duration_min"].sum()
    if total_duration <= 0:
        return {}

    by_state = tmp.groupby("Sleep state")["duration_min"].sum().to_dict()
    wakeup_min = by_state.get("wakeup", 0.0)
    light_min = by_state.get("light", 0.0)
    rem_min = by_state.get("REM", 0.0)
    deep_min = by_state.get("deep", 0.0)

    return {
        "sleep_total_min": total_duration,
        "sleep_wakeup_ratio": wakeup_min / total_duration,
        "sleep_light_ratio": light_min / total_duration,
        "sleep_rem_ratio": rem_min / total_duration,
        "sleep_deep_ratio": deep_min / total_duration,
        "sleep_stage_count": float(tmp["Sleep state"].nunique()),
    }


def build_wearable_features(data_dir: Path) -> pd.DataFrame:
    wearable_root = data_dir / "Sleepmat_Watch_Data"
    rows: List[Dict[str, float]] = []

    if not wearable_root.exists():
        return pd.DataFrame()

    for user_dir in wearable_root.iterdir():
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name
        feats: Dict[str, float] = {"user_id": user_id}

        scanwatch_hr = safe_read_csv(user_dir / "ScanWatch_HR.csv")
        scanwatch_steps = safe_read_csv(user_dir / "ScanWatch_Steps.csv")
        sleep_physio = safe_read_csv(user_dir / "Sleep_physio.csv")
        sleep_state = safe_read_csv(user_dir / "Sleep_state.csv")

        feats.update(aggregate_scanwatch_hr(scanwatch_hr))
        feats.update(aggregate_scanwatch_steps(scanwatch_steps))
        feats.update(aggregate_sleep_physio(sleep_physio))
        feats.update(aggregate_sleep_state(sleep_state))
        rows.append(feats)

    return pd.DataFrame(rows)


def build_demo_features(data_dir: Path) -> pd.DataFrame:
    demo = pd.read_csv(data_dir / "Demographics.csv")
    demo["user_id"] = demo["user_id"].astype(str)

    feature_cols = [
        "user_id",
        "Sex",
        "Age group",
        "Essential hypertension",
        "Osteoarthritis",
        "phq_total",
        "gad_total",
        "gds_total",
        "ace_total",
        "ace_total_6months",
        "attention_subscale",
        "memory_subscale",
        "fluency_subscale",
        "language_subscale",
        "visuospatial_subscale",
    ]

    existing_cols = [c for c in feature_cols if c in demo.columns]
    df = demo[existing_cols].copy()
    df["age_mid"] = df["Age group"].apply(parse_age_group_to_mid)

    if "ace_total_6months" in df.columns and "ace_total" in df.columns:
        df["ace_decline_6m"] = pd.to_numeric(
            df["ace_total"], errors="coerce"
        ) - pd.to_numeric(df["ace_total_6months"], errors="coerce")

    df["target_hypertension"] = (
        demo["Essential hypertension"].astype(str).str.lower().isin(["true", "1"])
    ).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Train hypertension warning model with multi-source elderly data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Dataset directory containing Demographics.csv and Sleepmat_Watch_Data/",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./outputs",
        help="Output directory for merged features and predictions.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_df = build_demo_features(data_dir)
    wearable_df = build_wearable_features(data_dir)

    wearable_df["user_id"] = wearable_df["user_id"].astype(str)
    merged = demo_df.merge(wearable_df, on="user_id", how="left")
    merged.to_csv(out_dir / "merged_features.csv", index=False, encoding="utf-8-sig")

    y = merged["target_hypertension"]
    feature_df = merged.drop(columns=["target_hypertension"])

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42
    )
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    x_train, x_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.25, random_state=42, stratify=y
    )

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, pred, digits=4))
    print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))

    result = x_test[["user_id"]].copy()
    result["y_true"] = y_test.values
    result["y_pred"] = pred
    result["risk_prob"] = prob
    result["risk_level"] = pd.cut(
        result["risk_prob"],
        bins=[-1, 0.35, 0.65, 1],
        labels=["low", "medium", "high"],
    )
    result.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"Saved merged feature table: {out_dir / 'merged_features.csv'}")
    print(f"Saved test predictions: {out_dir / 'test_predictions.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
