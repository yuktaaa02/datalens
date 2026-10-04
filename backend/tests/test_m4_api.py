import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.analysis_service import AnalysisService

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_upload_and_full_flow(synthetic_clean_classification):
    # 1. Upload CSV in memory
    csv_bytes = io.BytesIO()
    synthetic_clean_classification.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    upload_res = client.post(
        "/api/v1/upload",
        files={"file": ("dataset.csv", csv_bytes, "text/csv")},
        data={"target_column": "target", "task_type": "classification"}
    )

    assert upload_res.status_code == 200
    data = upload_res.json()
    analysis_id = data["analysis_id"]
    assert analysis_id is not None
    assert data["status"] == "UPLOADED"

    # 2. Run sync pipeline via AnalysisService directly
    AnalysisService.run_pipeline_sync(analysis_id)

    # 3. Check status endpoint
    status_res = client.get(f"/api/v1/analysis/{analysis_id}/status")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "COMPLETED"
    assert status_res.json()["progress_percent"] == 100

    # 4. Check profile endpoint
    profile_res = client.get(f"/api/v1/analysis/{analysis_id}/profile")
    assert profile_res.status_code == 200
    assert "profile" in profile_res.json()
    assert "meta_features" in profile_res.json()

    # 5. Check benchmark endpoint
    bench_res = client.get(f"/api/v1/analysis/{analysis_id}/benchmark")
    assert bench_res.status_code == 200
    assert len(bench_res.json()["benchmarks"]) >= 1

    # 6. Check recommendation endpoint
    rec_res = client.get(f"/api/v1/analysis/{analysis_id}/recommendation")
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert "winning_pipeline_id" in rec_data["decision"]
    assert len(rec_data["shap"]["top_features"]) > 0
    assert len(rec_data["explanation"]["narrative"]) > 0

    # 7. Check Python script export endpoint
    export_res = client.get(f"/api/v1/analysis/{analysis_id}/export/script")
    assert export_res.status_code == 200
    assert "def build_datalens_pipeline()" in export_res.text