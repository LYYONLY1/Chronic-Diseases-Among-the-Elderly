from pathlib import Path

from fastapi import FastAPI, HTTPException

from elderly_risk_system import ElderlyRiskSystem


app = FastAPI(title="Elderly Chronic Risk Warning API", version="1.0.0")

DATA_DIR = Path(".")
OUT_DIR = Path("./outputs")
system = ElderlyRiskSystem(data_dir=DATA_DIR, output_dir=OUT_DIR)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train():
    result = system.train()
    return {
        "auc": result.auc,
        "f1": result.f1,
        "accuracy": result.accuracy,
        "sample_count": result.sample_count,
    }


@app.get("/risk/overview")
def risk_overview():
    if not system.model_path.exists():
        raise HTTPException(status_code=400, detail="Model not trained yet. Call /train first.")
    return system.risk_overview()


@app.get("/risk/users")
def risk_users():
    if not system.model_path.exists():
        raise HTTPException(status_code=400, detail="Model not trained yet. Call /train first.")
    df = system.predict_all_users()
    return {"rows": df.to_dict(orient="records")}
