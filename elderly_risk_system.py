from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _parse_age_group_to_mid(age_group: str) -> float:
    if pd.isna(age_group):
        return np.nan
    text = str(age_group).strip().replace("[", "").replace("]", "")
    parts = [x.strip() for x in text.split(",") if x.strip()]
    if len(parts) != 2:
        return np.nan
    try:
        return (float(parts[0]) + float(parts[1])) / 2.0
    except ValueError:
        return np.nan


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _agg_scanwatch_hr(df: pd.DataFrame) -> Dict[str, float]:
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
        "sw_hr_q75": hr.quantile(0.75),
    }


def _agg_steps(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty or "Steps" not in df.columns:
        return {}
    steps = pd.to_numeric(df["Steps"], errors="coerce").dropna()
    if steps.empty:
        return {}
    return {
        "steps_mean": steps.mean(),
        "steps_sum": steps.sum(),
        "steps_std": steps.std(ddof=0),
        "steps_nonzero_ratio": float((steps > 0).sum()) / len(steps),
    }


def _agg_sleep_physio(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    out: Dict[str, float] = {}
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


def _agg_sleep_state(df: pd.DataFrame) -> Dict[str, float]:
    required = {"Start time", "End time", "Sleep state"}
    if df.empty or not required.issubset(set(df.columns)):
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
    total = tmp["duration_min"].sum()
    if total <= 0:
        return {}
    by_state = tmp.groupby("Sleep state")["duration_min"].sum().to_dict()
    return {
        "sleep_total_min": total,
        "sleep_wakeup_ratio": by_state.get("wakeup", 0.0) / total,
        "sleep_light_ratio": by_state.get("light", 0.0) / total,
        "sleep_rem_ratio": by_state.get("REM", 0.0) / total,
        "sleep_deep_ratio": by_state.get("deep", 0.0) / total,
    }


@dataclass
class TrainResult:
    auc: float
    f1: float
    accuracy: float
    sample_count: int
    best_params: Dict[str, object]
    baseline_auc: float
    ensemble_method: str
    tuned_component_aucs: Dict[str, float]
    shap_computed: bool
    shap_method: str | None


class SoftVotingFusion:
    """Soft-voting: 平均各组件模型的正类概率。"""

    def __init__(self, pipelines: Dict[str, Pipeline]):
        self.pipelines = pipelines

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = [p.predict_proba(X)[:, 1] for p in self.pipelines.values()]
        avg = np.mean(np.vstack(probs), axis=0)
        return np.vstack([1 - avg, avg]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class ElderlyRiskSystem:
    def __init__(self, data_dir: str | Path, output_dir: str | Path = "outputs"):
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.output_dir / "risk_model.joblib"
        self.model: Pipeline | None = None
        self.features: pd.DataFrame | None = None
        self.processing_report: Dict[str, object] = {}
        self.optimization_report: Dict[str, object] = {}

    def build_demographics_features(self) -> pd.DataFrame:
        demo = pd.read_csv(self.data_dir / "Demographics.csv")
        demo["user_id"] = demo["user_id"].astype(str)

        keep_cols = [
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
        keep_cols = [c for c in keep_cols if c in demo.columns]
        out = demo[keep_cols].copy()
        out["age_mid"] = out["Age group"].apply(_parse_age_group_to_mid)
        if "ace_total_6months" in out.columns and "ace_total" in out.columns:
            out["ace_decline_6m"] = pd.to_numeric(
                out["ace_total"], errors="coerce"
            ) - pd.to_numeric(out["ace_total_6months"], errors="coerce")

        out["target_hypertension"] = (
            demo["Essential hypertension"].astype(str).str.lower().isin(["true", "1"])
        ).astype(int)
        return out

    def build_wearable_features(self) -> pd.DataFrame:
        root = self.data_dir / "Sleepmat_Watch_Data"
        rows: List[Dict[str, float]] = []
        for user_dir in root.iterdir():
            if not user_dir.is_dir():
                continue
            row: Dict[str, float] = {"user_id": user_dir.name}
            row.update(_agg_scanwatch_hr(_safe_read_csv(user_dir / "ScanWatch_HR.csv")))
            row.update(_agg_steps(_safe_read_csv(user_dir / "ScanWatch_Steps.csv")))
            row.update(_agg_sleep_physio(_safe_read_csv(user_dir / "Sleep_physio.csv")))
            row.update(_agg_sleep_state(_safe_read_csv(user_dir / "Sleep_state.csv")))
            rows.append(row)
        return pd.DataFrame(rows)

    def build_feature_table(self) -> pd.DataFrame:
        demo = self.build_demographics_features()
        wearable = self.build_wearable_features()
        wearable["user_id"] = wearable["user_id"].astype(str)
        merged = demo.merge(wearable, on="user_id", how="left")
        merged, processing_report = self._clean_feature_table(merged)
        self.processing_report = processing_report
        merged.to_csv(
            self.output_dir / "merged_features.csv", index=False, encoding="utf-8-sig"
        )
        with open(self.output_dir / "data_processing_report.json", "w", encoding="utf-8") as f:
            json.dump(self.processing_report, f, ensure_ascii=False, indent=2)
        self.features = merged
        return merged

    @staticmethod
    def _clean_feature_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
        out = df.copy()
        numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
        report = {
            "total_rows": int(len(out)),
            "total_columns": int(len(out.columns)),
            "numeric_columns": int(len(numeric_cols)),
            "missing_before": int(out.isna().sum().sum()),
            "missing_after": 0,
            "outlier_clipped_counts": {},
            "feature_engineering": [
                "age_mid from Age group",
                "ace_decline_6m from ace_total and ace_total_6months",
                "wearable aggregated features from HR/Steps/Sleep",
            ],
        }
        for col in numeric_cols:
            vals = pd.to_numeric(out[col], errors="coerce")
            if vals.dropna().empty:
                continue
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                out[col] = vals.fillna(vals.median())
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            clipped_count = int(((vals < lower) | (vals > upper)).fillna(False).sum())
            if clipped_count > 0:
                report["outlier_clipped_counts"][col] = clipped_count
            vals = vals.clip(lower=lower, upper=upper)
            out[col] = vals.fillna(vals.median())
        report["missing_after"] = int(out.isna().sum().sum())
        return out, report

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob <= 0.35:
            return "low"
        if prob <= 0.65:
            return "medium"
        return "high"

    def _build_preprocessor(self, x_df: pd.DataFrame) -> ColumnTransformer:
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

    def _make_pipeline(self, x_df: pd.DataFrame, model: object) -> Pipeline:
        preprocessor = self._build_preprocessor(x_df)
        return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    def train(
        self,
        search_method: str = "random",
        ensemble_method: str = "soft",
        random_state: int = 42,
        rf_n_iter: int = 12,
        et_n_iter: int = 8,
        gb_n_iter: int = 8,
        cv: int = 3,
        tune_et: bool = True,
        tune_gb: bool = True,
        shap_sample_size: int = 200,
        compute_shap: bool = True,
    ) -> TrainResult:
        """
        search_method: 'random' 或 'grid'
        ensemble_method: 'soft' 或 'stacking'
        """

        if self.features is None:
            self.build_feature_table()
        assert self.features is not None

        # 建模特征：去掉 user_id（避免把“标识符”当特征造成噪声/不可解释）
        x_full = self.features.drop(columns=["target_hypertension"])
        y = self.features["target_hypertension"]
        user_ids = x_full["user_id"].astype(str)
        x = x_full.drop(columns=["user_id"])

        x_train, x_test, y_train, y_test, uid_train, uid_test = train_test_split(
            x,
            y,
            user_ids,
            test_size=0.25,
            random_state=random_state,
            stratify=y,
        )

        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        def _run_search(model, pipe_model_name: str, param_distributions: dict, param_grid: dict):
            base_pipe = Pipeline(
                steps=[("preprocess", self._build_preprocessor(x_train)), ("model", model)]
            )
            if search_method == "grid":
                search = GridSearchCV(
                    estimator=base_pipe,
                    param_grid=param_grid,
                    cv=cv_split,
                    scoring="roc_auc",
                    n_jobs=-1,
                )
            else:
                search = RandomizedSearchCV(
                    estimator=base_pipe,
                    param_distributions=param_distributions,
                    n_iter=(
                        rf_n_iter
                        if model.__class__.__name__.startswith("RandomForest")
                        else (et_n_iter if model.__class__.__name__.startswith("ExtraTrees") else gb_n_iter)
                    ),
                    cv=cv_split,
                    scoring="roc_auc",
                    random_state=random_state,
                    n_jobs=-1,
                )
            search.fit(x_train, y_train)
            best_pipe = search.best_estimator_
            return best_pipe, search.best_params_, float(search.best_score_)

        # baseline：固定随机森林默认参数
        baseline_model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=random_state
        )
        baseline_pipeline = self._make_pipeline(x_train, baseline_model)
        baseline_pipeline.fit(x_train, y_train)
        baseline_prob = baseline_pipeline.predict_proba(x_test)[:, 1]
        baseline_auc = float(roc_auc_score(y_test, baseline_prob))

        # 参数空间（keep small-ish，避免网格爆炸）
        rf_param_distributions = {
            "model__n_estimators": [200, 300, 400, 600],
            "model__max_depth": [None, 6, 8, 10, 12],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__min_samples_split": [2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
        }
        rf_param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [6, None],
            "model__min_samples_leaf": [1, 2],
        }

        et_param_distributions = {
            "model__n_estimators": [200, 300, 400, 600],
            "model__max_depth": [None, 6, 8, 10, 12],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__min_samples_split": [2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
        }
        et_param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [6, None],
            "model__min_samples_leaf": [1, 2],
        }

        gb_param_distributions = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
            "model__min_samples_leaf": [1, 2, 5],
        }
        gb_param_grid = {
            "model__n_estimators": [150, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }

        tuned_rf, rf_best_params, rf_best_cv_score = _run_search(
            RandomForestClassifier(random_state=random_state),
            "model",
            rf_param_distributions,
            rf_param_grid,
        )

        if tune_et:
            tuned_et, et_best_params, et_best_cv_score = _run_search(
                ExtraTreesClassifier(random_state=random_state),
                "model",
                et_param_distributions,
                et_param_grid,
            )
        else:
            tuned_et = self._make_pipeline(
                x_train, ExtraTreesClassifier(n_estimators=400, random_state=random_state)
            )
            tuned_et.fit(x_train, y_train)
            et_best_params, et_best_cv_score = {}, float("nan")

        if tune_gb:
            tuned_gb, gb_best_params, gb_best_cv_score = _run_search(
                GradientBoostingClassifier(random_state=random_state),
                "model",
                gb_param_distributions,
                gb_param_grid,
            )
        else:
            tuned_gb = self._make_pipeline(
                x_train, GradientBoostingClassifier(n_estimators=200, random_state=random_state)
            )
            tuned_gb.fit(x_train, y_train)
            gb_best_params, gb_best_cv_score = {}, float("nan")

        # 组件模型测试集表现（用于报告/对比）
        prob_rf = tuned_rf.predict_proba(x_test)[:, 1]
        prob_et = tuned_et.predict_proba(x_test)[:, 1]
        prob_gb = tuned_gb.predict_proba(x_test)[:, 1]

        tuned_component_aucs = {
            "RandomForest": float(roc_auc_score(y_test, prob_rf)),
            "ExtraTrees": float(roc_auc_score(y_test, prob_et)),
            "GradientBoosting": float(roc_auc_score(y_test, prob_gb)),
        }

        # 组合模型
        ensemble_method = ensemble_method.lower().strip()
        if ensemble_method == "stacking":
            stack = StackingClassifier(
                estimators=[("rf", tuned_rf), ("et", tuned_et), ("gb", tuned_gb)],
                final_estimator=LogisticRegression(max_iter=1000),
                stack_method="predict_proba",
                passthrough=False,
                cv=cv,
                n_jobs=-1,
            )
            stack.fit(x_train, y_train)
            prob = stack.predict_proba(x_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            self.model = stack
        else:
            fusion = SoftVotingFusion(
                {"RandomForest": tuned_rf, "ExtraTrees": tuned_et, "GradientBoosting": tuned_gb}
            )
            prob = fusion.predict_proba(x_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            self.model = fusion

        ensemble_auc = float(roc_auc_score(y_test, prob))

        pred_df = pd.DataFrame(
            {
                "user_id": uid_test.values,
                "y_true": y_test.values,
                "y_pred": pred,
                "risk_prob": prob,
                "risk_level": pd.Series(prob).apply(self._risk_level).values,
                "fusion_prob_rf": prob_rf,
                "fusion_prob_et": prob_et,
                "fusion_prob_gb": prob_gb,
            }
        )
        pred_df.to_csv(
            self.output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
        )

        self.optimization_report = {
            "search_method": search_method,
            "ensemble_method": ensemble_method,
            "cv": cv,
            "scoring": "roc_auc",
            "baseline_auc": baseline_auc,
            "component_aucs": tuned_component_aucs,
            "best_params": {
                "RandomForest": rf_best_params,
                "ExtraTrees": et_best_params,
                "GradientBoosting": gb_best_params,
            },
            "best_cv_scores": {
                "RandomForest": rf_best_cv_score,
                "ExtraTrees": et_best_cv_score,
                "GradientBoosting": gb_best_cv_score,
            },
            "ensemble_auc": ensemble_auc,
        }
        with open(self.output_dir / "optimization_report.json", "w", encoding="utf-8") as f:
            json.dump(self.optimization_report, f, ensure_ascii=False, indent=2)

        # 生成可解释性报告（全局 SHAP + 置换重要度）
        shap_computed = False
        shap_method = None
        shap_report: dict = {"computed": False, "component_model": "RandomForest"}
        if compute_shap:
            try:
                import shap  # type: ignore

                shap_method = "TreeExplainer(RandomForest)"
                # 使用 tuned_rf 的 preprocess 和模型
                rf_pipe = tuned_rf
                rf_model = rf_pipe.named_steps["model"]
                rf_pre = rf_pipe.named_steps["preprocess"]
                feature_names = rf_pre.get_feature_names_out()

                n = min(int(shap_sample_size), len(x_test))
                x_shap = x_test.sample(n=n, random_state=random_state)
                x_shap_trans = rf_pre.transform(x_shap)

                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(x_shap_trans)
                if isinstance(shap_values, list):
                    # 二分类：取正类贡献
                    shap_values = shap_values[1]

                shap_values = np.asarray(shap_values)
                shap_abs_mean = np.abs(shap_values).mean(axis=0)

                # processed 特征 Top
                top_idx = np.argsort(-shap_abs_mean)[:20]
                top_processed = [
                    {"feature": str(feature_names[i]), "shap_abs_mean": float(shap_abs_mean[i])}
                    for i in top_idx
                ]

                # 聚合回原始特征（粗粒度：num__/cat__ 前缀归并）
                def _proc_to_base(fn: str) -> str:
                    if fn.startswith("num__"):
                        return fn[len("num__") :]
                    if fn.startswith("cat__"):
                        rem = fn[len("cat__") :]
                        return rem.split("_")[0] if "_" in rem else rem
                    return fn

                agg: dict[str, dict[str, float]] = {}
                for fn, v in zip(feature_names, shap_abs_mean):
                    base = _proc_to_base(str(fn))
                    if base not in agg:
                        agg[base] = {"shap_abs_mean": 0.0, "count": 0.0}
                    agg[base]["shap_abs_mean"] += float(v)
                    agg[base]["count"] += 1.0

                top_agg = sorted(
                    [{"feature": k, "shap_abs_mean": float(v["shap_abs_mean"]), "count": int(v["count"])} for k, v in agg.items()],
                    key=lambda x: x["shap_abs_mean"],
                    reverse=True,
                )[:20]

                shap_report = {
                    "computed": True,
                    "method": shap_method,
                    "component_model": "RandomForest",
                    "sample_size": n,
                    "top_features_processed": top_processed,
                    "top_features_aggregated": top_agg,
                }
                shap_computed = True
            except Exception:
                shap_computed = False
                shap_method = None

        # 置换重要度（用于无 SHAP 时也能解释；开销可控）
        try:
            perm = permutation_importance(
                tuned_rf,
                x_test,
                y_test,
                scoring="roc_auc",
                n_repeats=5,
                random_state=random_state,
                n_jobs=-1,
            )
            perm_df = pd.DataFrame(
                {"feature": x_test.columns, "importance": perm.importances_mean}
            ).sort_values("importance", ascending=False)
            shap_report["permutation_importance_top"] = (
                perm_df.head(20).to_dict(orient="records")
            )
        except Exception:
            pass

        with open(self.output_dir / "shap_report.json", "w", encoding="utf-8") as f:
            json.dump(shap_report, f, ensure_ascii=False, indent=2)

        joblib.dump(self.model, self.model_path)
        return TrainResult(
            auc=ensemble_auc,
            f1=float(f1_score(y_test, pred)),
            accuracy=float(accuracy_score(y_test, pred)),
            sample_count=int(len(self.features)),
            best_params={
                "RandomForest": rf_best_params,
                "ExtraTrees": et_best_params,
                "GradientBoosting": gb_best_params,
            },
            baseline_auc=baseline_auc,
            ensemble_method=ensemble_method,
            tuned_component_aucs=tuned_component_aucs,
            shap_computed=shap_computed,
            shap_method=shap_method,
        )

    def load_model(self) -> None:
        self.model = joblib.load(self.model_path)

    def predict_all_users(self) -> pd.DataFrame:
        if self.features is None:
            self.build_feature_table()
        if self.model is None:
            self.load_model()
        assert self.features is not None and self.model is not None

        # 为兼容旧版模型：旧模型训练阶段把 user_id 当作特征保留下来。
        # 这里只移除标签列，保留 user_id 作为“额外列”，让新旧模型都能正常预测。
        x = self.features.drop(columns=["target_hypertension"])
        prob = self.model.predict_proba(x)[:, 1]
        base_cols = ["user_id", "Sex", "Age group"]
        extra_cols = [
            "steps_mean",
            "sw_hr_mean",
            "sleep_total_min",
            "phq_total",
            "gad_total",
            "gds_total",
        ]
        cols = [c for c in base_cols + extra_cols if c in self.features.columns]
        out = self.features[cols].copy()
        out["risk_prob"] = prob
        out["risk_level"] = out["risk_prob"].apply(self._risk_level)
        out["window_90d"] = out["risk_level"]
        out["window_180d"] = np.where(
            out["risk_prob"] > 0.55, "high", np.where(out["risk_prob"] > 0.3, "medium", "low")
        )
        out["window_360d"] = np.where(
            out["risk_prob"] > 0.5, "high", np.where(out["risk_prob"] > 0.25, "medium", "low")
        )
        out.to_csv(self.output_dir / "all_user_risk.csv", index=False, encoding="utf-8-sig")
        return out

    def risk_overview(self) -> Dict[str, float]:
        df = self.predict_all_users()
        total = len(df)
        high = int((df["risk_level"] == "high").sum())
        medium = int((df["risk_level"] == "medium").sum())
        low = int((df["risk_level"] == "low").sum())
        return {
            "total_users": total,
            "high_risk_users": high,
            "medium_risk_users": medium,
            "low_risk_users": low,
            "high_risk_ratio": round(high / total, 4) if total else 0.0,
        }

    def get_user_detail(self, user_id: str) -> Dict[str, object]:
        if self.features is None:
            self.build_feature_table()
        if self.model is None:
            self.load_model()
        assert self.features is not None and self.model is not None

        data = self.features.copy()
        data["user_id"] = data["user_id"].astype(str)
        user_row = data[data["user_id"] == str(user_id)]
        if user_row.empty:
            raise ValueError(f"User {user_id} not found")

        # 兼容旧版模型：保留 user_id 列作为额外输入（旧模型需要它，新模型会忽略它）。
        x_user = user_row.drop(columns=["target_hypertension"])
        risk_prob = float(self.model.predict_proba(x_user)[:, 1][0])
        risk_level = self._risk_level(risk_prob)

        key_metrics = {
            "sw_hr_mean": float(user_row["sw_hr_mean"].iloc[0]) if "sw_hr_mean" in user_row.columns and pd.notna(user_row["sw_hr_mean"].iloc[0]) else np.nan,
            "steps_mean": float(user_row["steps_mean"].iloc[0]) if "steps_mean" in user_row.columns and pd.notna(user_row["steps_mean"].iloc[0]) else np.nan,
            "sleep_total_min": float(user_row["sleep_total_min"].iloc[0]) if "sleep_total_min" in user_row.columns and pd.notna(user_row["sleep_total_min"].iloc[0]) else np.nan,
            "phq_total": float(user_row["phq_total"].iloc[0]) if "phq_total" in user_row.columns and pd.notna(user_row["phq_total"].iloc[0]) else np.nan,
            "gad_total": float(user_row["gad_total"].iloc[0]) if "gad_total" in user_row.columns and pd.notna(user_row["gad_total"].iloc[0]) else np.nan,
            "gds_total": float(user_row["gds_total"].iloc[0]) if "gds_total" in user_row.columns and pd.notna(user_row["gds_total"].iloc[0]) else np.nan,
        }

        explain_method = "zscore"
        explain_summary = ""
        top_factors: List[Dict[str, float]] = []
        try:
            import shap  # type: ignore

            rf_pipe = None
            if isinstance(self.model, SoftVotingFusion):
                rf_pipe = self.model.pipelines.get("RandomForest")
            else:
                named = getattr(self.model, "named_estimators_", None) or {}
                rf_pipe = named.get("rf") or named.get("RandomForest") or named.get("RandomForestClassifier")
                if rf_pipe is None and getattr(self.model, "estimators_", None):
                    rf_pipe = self.model.estimators_[0]

            if rf_pipe is not None and hasattr(rf_pipe, "named_steps"):
                rf_model = rf_pipe.named_steps["model"]
                rf_pre = rf_pipe.named_steps["preprocess"]
                feature_names = rf_pre.get_feature_names_out()

                x_user_model = user_row.drop(columns=["target_hypertension", "user_id"])
                x_trans = rf_pre.transform(x_user_model)

                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(x_trans)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                shap_values = np.asarray(shap_values)[0]  # (n_features,)

                def _proc_to_base(fn: str) -> str:
                    if fn.startswith("num__"):
                        return fn[len("num__") :]
                    if fn.startswith("cat__"):
                        rem = fn[len("cat__") :]
                        return rem.split("_")[0] if "_" in rem else rem
                    return fn

                agg: dict[str, float] = {}
                for fn, v in zip(feature_names, shap_values):
                    base = _proc_to_base(str(fn))
                    agg[base] = agg.get(base, 0.0) + float(v)

                top_sorted = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                for feat, contrib in top_sorted:
                    raw_val = user_row[feat].iloc[0] if feat in user_row.columns else np.nan
                    top_factors.append(
                        {"feature": feat, "z_score": round(contrib, 4), "value": raw_val if pd.notna(raw_val) else np.nan}
                    )

                explain_method = "shap"

                feature_cn = {
                    "sw_hr_mean": "平均心率",
                    "steps_mean": "日均步数",
                    "sleep_total_min": "睡眠总时长(分钟)",
                    "sleep_wakeup_ratio": "清醒占比",
                    "phq_total": "PHQ抑郁总分",
                    "gad_total": "GAD焦虑总分",
                    "gds_total": "GDS抑郁总分",
                    "ace_decline_6m": "6个月认知变化值",
                    "Sex": "性别",
                    "Age group": "年龄段",
                }

                def _fmt_val(v: object) -> str:
                    if v is None:
                        return "-"
                    if isinstance(v, float) and np.isnan(v):
                        return "-"
                    return str(v)

                pos_sorted = sorted(
                    [(k, v) for k, v in agg.items() if v >= 0],
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:2]
                neg_sorted = sorted(
                    [(k, v) for k, v in agg.items() if v < 0],
                    key=lambda kv: kv[1],
                )[:2]

                pos_parts = [f"{feature_cn.get(k, k)}({_fmt_val(user_row[k].iloc[0])})" for k, _ in pos_sorted if k in user_row.columns]
                neg_parts = [f"{feature_cn.get(k, k)}({_fmt_val(user_row[k].iloc[0])})" for k, _ in neg_sorted if k in user_row.columns]

                explain_summary = (
                    f"模型解释结果显示：主要抬高风险的因素包括：{';'.join(pos_parts) if pos_parts else '无明显正贡献'}；"
                    f"主要降低风险的因素包括：{';'.join(neg_parts) if neg_parts else '无明显负贡献'}。"
                )

        except Exception:
            top_factors = self._estimate_top_factors(user_row, data)
            explain_method = "zscore"
            explain_summary = "SHAP 解释计算失败，已回退到基于群体均值与标准差的近似解释（z-score）。"

        if not top_factors:
            top_factors = self._estimate_top_factors(user_row, data)
            explain_method = "zscore"
            if not explain_summary:
                explain_summary = "已使用近似解释（z-score）。"

        suggestions = self._build_suggestions(key_metrics, risk_level)

        return {
            "user_id": str(user_id),
            "sex": user_row["Sex"].iloc[0] if "Sex" in user_row.columns else "",
            "age_group": user_row["Age group"].iloc[0] if "Age group" in user_row.columns else "",
            "risk_prob": risk_prob,
            "risk_level": risk_level,
            "window_90d": risk_level,
            "window_180d": "high" if risk_prob > 0.55 else ("medium" if risk_prob > 0.3 else "low"),
            "window_360d": "high" if risk_prob > 0.5 else ("medium" if risk_prob > 0.25 else "low"),
            "key_metrics": key_metrics,
            "top_factors": top_factors,
            "explain_method": explain_method,
            "explain_summary": explain_summary,
            "suggestions": suggestions,
        }

    def predict_from_manual_features(self, manual: Dict[str, object]) -> Dict[str, object]:
        if self.features is None:
            self.build_feature_table()
        if self.model is None:
            self.load_model()
        assert self.features is not None and self.model is not None

        # 为兼容旧版模型：旧模型训练阶段可能包含 user_id。
        # 这里仅移除标签列，保留 user_id 作为额外列。
        x_template = self.features.drop(columns=["target_hypertension"]).copy()
        one_row = {}
        for col in x_template.columns:
            if pd.api.types.is_numeric_dtype(x_template[col]):
                one_row[col] = float(pd.to_numeric(x_template[col], errors="coerce").median())
            else:
                mode = x_template[col].mode(dropna=True)
                one_row[col] = mode.iloc[0] if not mode.empty else ""

        age_mid = float(manual.get("age_mid", one_row.get("age_mid", 80)))
        sex = str(manual.get("sex", one_row.get("Sex", "Female")))
        steps_mean = float(manual.get("steps_mean", one_row.get("steps_mean", 200)))
        sw_hr_mean = float(manual.get("sw_hr_mean", one_row.get("sw_hr_mean", 80)))
        sleep_total_min = float(manual.get("sleep_total_min", one_row.get("sleep_total_min", 420)))
        phq_total = float(manual.get("phq_total", one_row.get("phq_total", 3)))
        gad_total = float(manual.get("gad_total", one_row.get("gad_total", 3)))
        gds_total = float(manual.get("gds_total", one_row.get("gds_total", 3)))

        if age_mid >= 88:
            age_group = "[88, 99]"
        elif age_mid >= 76:
            age_group = "[76, 87]"
        elif age_mid >= 72:
            age_group = "[72, 75]"
        else:
            age_group = "[60, 71]"

        one_row.update(
            {
                "user_id": "manual_input",
                "Sex": sex,
                "Age group": age_group,
                "age_mid": age_mid,
                "steps_mean": steps_mean,
                "sw_hr_mean": sw_hr_mean,
                "sleep_total_min": sleep_total_min,
                "phq_total": phq_total,
                "gad_total": gad_total,
                "gds_total": gds_total,
            }
        )

        x_manual = pd.DataFrame([one_row], columns=x_template.columns)
        risk_prob = float(self.model.predict_proba(x_manual)[:, 1][0])
        risk_level = self._risk_level(risk_prob)
        label = "高血压高风险" if risk_level == "high" else ("高血压中风险" if risk_level == "medium" else "高血压低风险")

        key_metrics = {
            "sw_hr_mean": sw_hr_mean,
            "steps_mean": steps_mean,
            "sleep_total_min": sleep_total_min,
            "phq_total": phq_total,
            "gad_total": gad_total,
            "gds_total": gds_total,
        }
        suggestions = self._build_suggestions(key_metrics, risk_level)
        return {
            "risk_prob": risk_prob,
            "risk_level": risk_level,
            "predicted_label": label,
            "window_90d": risk_level,
            "window_180d": "high" if risk_prob > 0.55 else ("medium" if risk_prob > 0.3 else "low"),
            "window_360d": "high" if risk_prob > 0.5 else ("medium" if risk_prob > 0.25 else "low"),
            "key_metrics": key_metrics,
            "suggestions": suggestions,
        }

    @staticmethod
    def _estimate_top_factors(user_row: pd.DataFrame, all_data: pd.DataFrame) -> List[Dict[str, float]]:
        candidate_cols = [
            "sw_hr_mean",
            "steps_mean",
            "sleep_total_min",
            "sleep_wakeup_ratio",
            "phq_total",
            "gad_total",
            "gds_total",
            "ace_decline_6m",
        ]
        result: List[Dict[str, float]] = []
        for col in candidate_cols:
            if col not in all_data.columns:
                continue
            series = pd.to_numeric(all_data[col], errors="coerce")
            val = pd.to_numeric(user_row[col], errors="coerce").iloc[0] if col in user_row.columns else np.nan
            if pd.isna(val):
                continue
            mean = series.mean()
            std = series.std(ddof=0)
            if pd.isna(std) or std == 0:
                continue
            z = (float(val) - float(mean)) / float(std)
            result.append({"feature": col, "z_score": round(z, 3), "value": float(val)})
        result.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return result[:5]

    @staticmethod
    def _build_suggestions(metrics: Dict[str, float], risk_level: str) -> List[str]:
        suggestions: List[str] = []
        hr = metrics.get("sw_hr_mean", np.nan)
        steps = metrics.get("steps_mean", np.nan)
        sleep_total = metrics.get("sleep_total_min", np.nan)
        phq = metrics.get("phq_total", np.nan)

        if not pd.isna(hr) and hr > 90:
            suggestions.append("日常心率偏高，建议增加静息监测并就医评估血压与心功能。")
        if not pd.isna(steps) and steps < 150:
            suggestions.append("日均步数偏低，建议在安全前提下分段步行提升活动量。")
        if not pd.isna(sleep_total) and sleep_total < 360:
            suggestions.append("睡眠时长不足，建议规律作息并减少夜间刺激性活动。")
        if not pd.isna(phq) and phq >= 10:
            suggestions.append("情绪量表分数较高，建议转介心理支持与家属陪伴干预。")
        if risk_level == "high":
            suggestions.append("建议社区医生每周随访一次，并同步照护者观察记录。")

        if not suggestions:
            suggestions.append("当前风险相对平稳，建议保持现有生活方式并持续监测。")
        return suggestions


def run_training(data_dir: str | Path, output_dir: str | Path = "outputs") -> Tuple[TrainResult, pd.DataFrame]:
    system = ElderlyRiskSystem(data_dir, output_dir)
    result = system.train()
    all_users = system.predict_all_users()
    return result, all_users
