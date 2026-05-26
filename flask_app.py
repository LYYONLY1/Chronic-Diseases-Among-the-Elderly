from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from flask import Flask, flash, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.exceptions import HTTPException

from compare_models import evaluate_models
from elderly_risk_system import ElderlyRiskSystem


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "elderly-risk-demo-secret-key"
logging.basicConfig(
    filename=str(BASE_DIR / "system.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("elderly-risk-system")

USER_DB = {
    "elder_001": {"password": "elder123", "role": "老人"},
    "care_001": {"password": "care123", "role": "照护者"},
    "doctor_001": {"password": "doctor123", "role": "医生"},
    "admin": {"password": "admin123", "role": "管理员"},
}

ALL_PAGES = {
    "dashboard",
    "data_management",
    "visualization",
    "risk_prediction",
    "model_training",
    "model_report",
    "usage_guide",
    "governance",
}
ROLE_PERMS = {
    "老人": {"dashboard", "visualization", "risk_prediction", "usage_guide"},
    "照护者": {"dashboard", "visualization", "risk_prediction", "usage_guide"},
    "医生": {"dashboard", "visualization", "risk_prediction", "model_training", "model_report", "usage_guide", "governance"},
    "管理员": set(ALL_PAGES),
}
SYSTEM = ElderlyRiskSystem(DATA_DIR, OUTPUT_DIR)
EXECUTOR = ThreadPoolExecutor(max_workers=2)
TASK_STATE = {
    "train": {"running": False, "message": "未开始"},
    "compare": {"running": False, "message": "未开始"},
}
TRAIN_CONFIG = {
    "search_method": "random",
    "ensemble_method": "soft",
}


@app.context_processor
def inject_permissions():
    role = session.get("role", "")
    return {"allowed_pages": ROLE_PERMS.get(role, set())}


def get_system() -> ElderlyRiskSystem:
    return SYSTEM


def login_required():
    return "username" in session


def role_required(page_key: str) -> bool:
    role = session.get("role", "")
    return page_key in ROLE_PERMS.get(role, set())


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _safe_counts(df: pd.DataFrame, column: str) -> dict:
    if df.empty or column not in df.columns:
        return {}
    return df[column].value_counts().to_dict()


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return pd.read_json(path, typ="series").to_dict() if path.suffix == ".json" else {}
    except Exception:
        try:
            import json

            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


def _run_train_task():
    TASK_STATE["train"] = {"running": True, "message": "训练中，请稍候..."}
    try:
        system = get_system()
        result = system.train(
            search_method=TRAIN_CONFIG.get("search_method", "random"),
            ensemble_method=TRAIN_CONFIG.get("ensemble_method", "soft"),
        )
        system.predict_all_users()
        TASK_STATE["train"] = {
            "running": False,
            "message": (
                f"训练完成({TRAIN_CONFIG.get('search_method')} / {TRAIN_CONFIG.get('ensemble_method')}) "
                f"AUC={result.auc:.4f}, F1={result.f1:.4f}, ACC={result.accuracy:.4f}"
            ),
        }
        logger.info("Training task completed")
    except Exception as exc:
        TASK_STATE["train"] = {"running": False, "message": f"训练失败: {exc}"}
        logger.error("Training task failed: %s\n%s", exc, traceback.format_exc())


def _run_compare_task():
    TASK_STATE["compare"] = {"running": True, "message": "模型对比运行中，请稍候..."}
    try:
        evaluate_models(DATA_DIR, OUTPUT_DIR, test_size=0.25)
        TASK_STATE["compare"] = {"running": False, "message": "模型对比完成"}
        logger.info("Compare task completed")
    except Exception as exc:
        TASK_STATE["compare"] = {"running": False, "message": f"模型对比失败: {exc}"}
        logger.error("Compare task failed: %s\n%s", exc, traceback.format_exc())


@app.route("/")
def index():
    if not login_required():
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = USER_DB.get(username)
        if user and user["password"] == password:
            session["username"] = username
            session["role"] = user["role"]
            logger.info("Login success: %s (%s)", username, user["role"])
            flash("登录成功", "success")
            return redirect(url_for("dashboard"))
        logger.warning("Login failed for user=%s", username)
        flash("用户名或密码错误", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    logger.info("Logout: %s", session.get("username", "unknown"))
    session.clear()
    flash("已退出登录", "info")
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))
    if not role_required("dashboard"):
        flash("当前角色无权访问该页面", "danger")
        return redirect(url_for("index"))
    system = get_system()
    overview = None
    risk_counts = {}
    window_counts = {"90d": {}, "180d": {}, "360d": {}}
    avg_prob = None
    risk_file = OUTPUT_DIR / "all_user_risk.csv"
    if risk_file.exists():
        try:
            overview = system.risk_overview()
            risk_df = pd.read_csv(risk_file)
            risk_counts = _safe_counts(risk_df, "risk_level")
            window_counts = {
                "90d": _safe_counts(risk_df, "window_90d"),
                "180d": _safe_counts(risk_df, "window_180d"),
                "360d": _safe_counts(risk_df, "window_360d"),
            }
            if "risk_prob" in risk_df.columns and not risk_df.empty:
                avg_prob = round(float(risk_df["risk_prob"].mean()), 4)
        except Exception:
            overview = None
    return render_template(
        "dashboard.html",
        overview=overview,
        risk_counts=risk_counts,
        window_counts=window_counts,
        avg_prob=avg_prob,
    )


@app.route("/data-management", methods=["GET", "POST"])
def data_management():
    if not login_required():
        return redirect(url_for("login"))
    if not role_required("data_management"):
        flash("当前角色无权访问数据管理", "danger")
        return redirect(url_for("dashboard"))

    uploaded_file = None
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            save_path = UPLOAD_DIR / file.filename
            file.save(save_path)
            uploaded_file = file.filename
            flash(f"文件已上传：{file.filename}", "success")
        else:
            flash("请选择要上传的CSV文件", "warning")

    demo_df = load_csv_if_exists(DATA_DIR / "Demographics.csv")
    merged_df = load_csv_if_exists(OUTPUT_DIR / "merged_features.csv")
    page_size = int(request.args.get("page_size", 10))
    demo_page = int(request.args.get("demo_page", 1))
    merged_page = int(request.args.get("merged_page", 1))

    demo_total_pages = max(1, (len(demo_df) + page_size - 1) // page_size) if not demo_df.empty else 1
    merged_total_pages = max(1, (len(merged_df) + page_size - 1) // page_size) if not merged_df.empty else 1
    demo_page = max(1, min(demo_page, demo_total_pages))
    merged_page = max(1, min(merged_page, merged_total_pages))

    demo_start = (demo_page - 1) * page_size
    merged_start = (merged_page - 1) * page_size
    demo_page_df = demo_df.iloc[demo_start : demo_start + page_size] if not demo_df.empty else pd.DataFrame()
    merged_page_df = merged_df.iloc[merged_start : merged_start + page_size] if not merged_df.empty else pd.DataFrame()

    return render_template(
        "data_management.html",
        demo_shape=demo_df.shape if not demo_df.empty else (0, 0),
        merged_shape=merged_df.shape if not merged_df.empty else (0, 0),
        demo_head=demo_page_df.to_dict(orient="records") if not demo_page_df.empty else [],
        merged_head=merged_page_df.to_dict(orient="records") if not merged_page_df.empty else [],
        demo_page=demo_page,
        merged_page=merged_page,
        demo_total_pages=demo_total_pages,
        merged_total_pages=merged_total_pages,
        page_size=page_size,
        uploaded_file=uploaded_file,
    )


@app.route("/visualization")
def visualization():
    if not login_required():
        return redirect(url_for("login"))
    if not role_required("visualization"):
        flash("当前角色无权访问可视化分析", "danger")
        return redirect(url_for("dashboard"))

    risk_df = load_csv_if_exists(OUTPUT_DIR / "all_user_risk.csv")
    if risk_df.empty:
        flash("请先在“模型训练”页面执行训练与预警生成", "warning")
        return render_template("visualization.html", risk_counts={}, avg_prob=None, top_users=[])

    risk_counts = risk_df["risk_level"].value_counts().to_dict()
    avg_prob = float(risk_df["risk_prob"].mean())
    by_age = (
        risk_df.groupby("Age group")["risk_prob"].mean().reset_index()
        if "Age group" in risk_df.columns
        else pd.DataFrame(columns=["Age group", "risk_prob"])
    )
    by_sex = (
        risk_df.groupby("Sex")["risk_prob"].mean().reset_index()
        if "Sex" in risk_df.columns
        else pd.DataFrame(columns=["Sex", "risk_prob"])
    )
    top_users = (
        risk_df.sort_values("risk_prob", ascending=False)
        .head(10)[["user_id", "risk_prob", "risk_level"]]
        .to_dict(orient="records")
    )
    trend_df = (
        risk_df.sort_values("risk_prob", ascending=False)[["user_id", "risk_prob"]]
        .head(20)
        .reset_index(drop=True)
    )
    scatter_rows = []
    if {"steps_mean", "sw_hr_mean", "risk_prob"}.issubset(set(risk_df.columns)):
        scatter_rows = (
            risk_df[["steps_mean", "sw_hr_mean", "risk_prob"]]
            .dropna()
            .head(120)
            .to_dict(orient="records")
        )
    if not scatter_rows:
        merged_df = load_csv_if_exists(OUTPUT_DIR / "merged_features.csv")
        if not merged_df.empty and {"user_id", "steps_mean", "sw_hr_mean"}.issubset(set(merged_df.columns)):
            merged = risk_df.merge(
                merged_df[["user_id", "steps_mean", "sw_hr_mean"]],
                on="user_id",
                how="left",
                suffixes=("", "_m"),
            )
            if "steps_mean_m" in merged.columns:
                merged["steps_mean"] = merged["steps_mean"].fillna(merged["steps_mean_m"])
            if "sw_hr_mean_m" in merged.columns:
                merged["sw_hr_mean"] = merged["sw_hr_mean"].fillna(merged["sw_hr_mean_m"])
            scatter_rows = (
                merged[["steps_mean", "sw_hr_mean", "risk_prob"]]
                .dropna()
                .head(120)
                .to_dict(orient="records")
            )
    return render_template(
        "visualization.html",
        risk_counts=risk_counts,
        avg_prob=avg_prob,
        by_age=by_age.to_dict(orient="records"),
        by_sex=by_sex.to_dict(orient="records"),
        trend_data=trend_df.to_dict(orient="records"),
        scatter_rows=scatter_rows,
        top_users=top_users,
    )


@app.route("/risk-prediction", methods=["GET", "POST"])
def risk_prediction():
    if not login_required():
        return redirect(url_for("login"))
    if not role_required("risk_prediction"):
        flash("当前角色无权访问风险预测", "danger")
        return redirect(url_for("dashboard"))

    system = get_system()
    risk_df = load_csv_if_exists(OUTPUT_DIR / "all_user_risk.csv")
    users = sorted(risk_df["user_id"].astype(str).tolist()) if not risk_df.empty else []
    detail = None
    selected_user = None

    manual_result = None
    if request.method == "POST":
        action = request.form.get("action", "query")
        selected_user = request.form.get("user_id", "")
        if action == "query" and selected_user:
            try:
                detail = system.get_user_detail(selected_user)
            except Exception as exc:
                flash(f"获取用户详情失败：{exc}", "danger")
        elif action == "manual_predict":
            try:
                def _num(name: str, default: float, min_v: float, max_v: float) -> float:
                    v = request.form.get(name, default)
                    v_f = float(v)
                    if v_f < min_v or v_f > max_v:
                        raise ValueError(f"{name} 超出范围")
                    return v_f

                manual_inputs = {
                    "age_mid": _num("age_mid", 78.0, 60.0, 110.0),
                    "sex": request.form.get("sex", "Female").strip(),
                    "steps_mean": _num("steps_mean", 200.0, 0.0, 10000.0),
                    "sw_hr_mean": _num("sw_hr_mean", 80.0, 40.0, 180.0),
                    "sleep_total_min": _num("sleep_total_min", 420.0, 0.0, 1440.0),
                    "phq_total": _num("phq_total", 3.0, 0.0, 27.0),
                    "gad_total": _num("gad_total", 3.0, 0.0, 21.0),
                    "gds_total": _num("gds_total", 3.0, 0.0, 15.0),
                }
                manual_result = system.predict_from_manual_features(manual_inputs)
            except Exception as exc:
                flash(f"手工特征预测失败：{exc}", "danger")

    user_risk = (
        risk_df[["user_id", "risk_prob", "risk_level"]].to_dict(orient="records")
        if not risk_df.empty
        else []
    )

    return render_template(
        "risk_prediction.html",
        users=users,
        detail=detail,
        selected_user=selected_user,
        user_risk=user_risk,
        manual_result=manual_result,
    )


@app.route("/model-training", methods=["GET", "POST"])
def model_training():
    if not login_required():
        return redirect(url_for("login"))
    if not role_required("model_training"):
        flash("当前角色无权访问模型训练", "danger")
        return redirect(url_for("dashboard"))

    train_result = None
    compare_rows = []
    confusion = None
    feature_importance = []

    if request.method == "POST":
        action = request.form.get("action", "")
        if action == "train":
            if TASK_STATE["train"]["running"]:
                flash("训练任务正在运行，请稍候", "warning")
            else:
                TRAIN_CONFIG["search_method"] = request.form.get("search_method", "random").strip().lower()
                TRAIN_CONFIG["ensemble_method"] = request.form.get("ensemble_method", "soft").strip().lower()
                if TRAIN_CONFIG["search_method"] not in {"random", "grid"}:
                    TRAIN_CONFIG["search_method"] = "random"
                if TRAIN_CONFIG["ensemble_method"] not in {"soft", "stacking"}:
                    TRAIN_CONFIG["ensemble_method"] = "soft"
                EXECUTOR.submit(_run_train_task)
                flash("训练任务已在后台启动，可先浏览其他页面", "info")
        if action == "compare":
            if TASK_STATE["compare"]["running"]:
                flash("模型对比任务正在运行，请稍候", "warning")
            else:
                EXECUTOR.submit(_run_compare_task)
                flash("模型对比任务已在后台启动，可先浏览其他页面", "info")

    if not compare_rows:
        stored_compare = load_csv_if_exists(OUTPUT_DIR / "model_comparison.csv")
        if not stored_compare.empty:
            compare_rows = stored_compare.to_dict(orient="records")

    if confusion is None:
        pred_df = load_csv_if_exists(OUTPUT_DIR / "test_predictions.csv")
        if not pred_df.empty and {"y_true", "y_pred"}.issubset(pred_df.columns):
            cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"], labels=[0, 1])
            confusion = cm.tolist()
            train_result = {
                "auc": round(float(roc_auc_score(pred_df["y_true"], pred_df["risk_prob"])), 4)
                if "risk_prob" in pred_df.columns
                else None,
                "f1": round(float(f1_score(pred_df["y_true"], pred_df["y_pred"])), 4),
                "accuracy": round(float((pred_df["y_true"] == pred_df["y_pred"]).mean()), 4),
                "sample_count": int(len(load_csv_if_exists(OUTPUT_DIR / "merged_features.csv"))),
            }

    feature_importance_title = "SHAP/置换重要度 Top12"
    shap_report = _read_json_if_exists(OUTPUT_DIR / "shap_report.json")
    if shap_report:
        try:
            if shap_report.get("computed") and shap_report.get("top_features_aggregated"):
                agg = shap_report.get("top_features_aggregated", [])[:12]
                feature_importance = [
                    {"feature": x.get("feature"), "importance": x.get("shap_abs_mean")} for x in agg
                ]
                feature_importance_title = "SHAP 归并贡献 Top12"
            elif shap_report.get("permutation_importance_top"):
                perm = shap_report.get("permutation_importance_top", [])[:12]
                feature_importance = [
                    {"feature": x.get("feature"), "importance": x.get("importance")} for x in perm
                ]
                feature_importance_title = "置换重要度 Top12"
        except Exception:
            feature_importance = []

    return render_template(
        "model_training.html",
        train_result=train_result,
        compare_rows=compare_rows,
        confusion=confusion,
        feature_importance=feature_importance,
        feature_importance_title=feature_importance_title,
        task_state=TASK_STATE,
    )


@app.route("/model-report")
def model_report():
    if not login_required():
        return redirect(url_for("login"))
    if not role_required("model_report"):
        flash("当前角色无权访问训练报告中心", "danger")
        return redirect(url_for("dashboard"))

    optimization_report = _read_json_if_exists(OUTPUT_DIR / "optimization_report.json")
    processing_report = _read_json_if_exists(OUTPUT_DIR / "data_processing_report.json")
    shap_report = _read_json_if_exists(OUTPUT_DIR / "shap_report.json")
    logs_tail = []
    log_path = BASE_DIR / "system.log"
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                logs_tail = [line.strip() for line in lines[-40:]]
        except Exception:
            logs_tail = []

    return render_template(
        "model_report.html",
        optimization_report=optimization_report,
        processing_report=processing_report,
        shap_report=shap_report,
        logs_tail=logs_tail,
        task_state=TASK_STATE,
    )


@app.route("/usage-guide")
def usage_guide():
    if not login_required():
        return redirect(url_for("login"))
    if not role_required("usage_guide"):
        flash("当前角色无权访问使用说明", "danger")
        return redirect(url_for("dashboard"))
    return render_template("usage_guide.html")


@app.route("/governance")
def governance():
    if not login_required():
        return redirect(url_for("login"))
    if not role_required("governance"):
        flash("当前角色无权访问系统治理", "danger")
        return redirect(url_for("dashboard"))

    matrix_rows = []
    page_keys = ["dashboard", "data_management", "visualization", "risk_prediction", "model_training", "model_report", "usage_guide", "governance"]
    for role, perms in ROLE_PERMS.items():
        row = {"role": role}
        for p in page_keys:
            row[p] = "Y" if p in perms else "N"
        matrix_rows.append(row)

    logs_tail = []
    log_path = BASE_DIR / "system.log"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            logs_tail = [line.strip() for line in f.readlines()[-50:]]

    return render_template(
        "governance.html",
        matrix_rows=matrix_rows,
        logs_tail=logs_tail,
        task_state=TASK_STATE,
    )


@app.route("/task-status")
def task_status():
    if not login_required():
        return jsonify({"error": "not_login"}), 401
    return jsonify(TASK_STATE)


@app.route("/export/<name>")
def export_file(name: str):
    if not login_required():
        return redirect(url_for("login"))
    if not role_required("data_management") and not role_required("model_training"):
        flash("当前角色无权导出数据", "danger")
        return redirect(url_for("dashboard"))
    allow = {
        "merged_features.csv",
        "all_user_risk.csv",
        "model_comparison.csv",
        "test_predictions.csv",
        "optimization_report.json",
        "data_processing_report.json",
        "shap_report.json",
    }
    if name not in allow:
        flash("不支持的导出文件", "danger")
        return redirect(url_for("dashboard"))
    path = OUTPUT_DIR / name
    if not path.exists():
        flash("文件不存在，请先训练生成", "warning")
        return redirect(url_for("dashboard"))
    return send_from_directory(str(OUTPUT_DIR), name, as_attachment=True)


@app.errorhandler(Exception)
def handle_exception(exc):
    if isinstance(exc, HTTPException):
        return exc
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return render_template("error.html", message=str(exc)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
