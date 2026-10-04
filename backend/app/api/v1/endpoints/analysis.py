import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Response
from app.schemas.dataset import TaskType
from app.schemas.decision import DecisionWeights
from app.services.analysis_service import AnalysisService, AnalysisState
from app.core.decision.ranker import DecisionEngine

router = APIRouter()

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: TaskType = Form(...)
):
    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported.")

    content = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    try:
        record = AnalysisService.create_analysis(
            df=df,
            target_col=target_column,
            task_type=task_type,
            filename=file.filename
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "analysis_id": record.analysis_id,
        "filename": record.filename,
        "status": record.status,
        "target_column": record.target_col,
        "task_type": record.task_type.value
    }

@router.post("/analysis/{analysis_id}/start")
async def start_analysis(
    analysis_id: str,
    background_tasks: BackgroundTasks
):
    record = AnalysisService.get_record(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")

    if record.status in [AnalysisState.PROFILING, AnalysisState.BENCHMARKING]:
        return {"message": "Analysis is already running.", "status": record.status}

    background_tasks.add_task(AnalysisService.run_pipeline_sync, analysis_id)
    return {"message": "Analysis started in background.", "analysis_id": analysis_id}

@router.get("/analysis/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    record = AnalysisService.get_record(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")

    return {
        "analysis_id": record.analysis_id,
        "status": record.status,
        "progress_percent": record.progress_percent,
        "error_message": record.error_message
    }

@router.get("/analysis/{analysis_id}/profile")
async def get_analysis_profile(analysis_id: str):
    record = AnalysisService.get_record(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")
    if not record.profile:
        raise HTTPException(status_code=425, detail="Profile is not yet ready.")

    return {
        "profile": record.profile,
        "meta_features": record.meta_features
    }

@router.get("/analysis/{analysis_id}/benchmark")
async def get_analysis_benchmark(analysis_id: str):
    record = AnalysisService.get_record(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")
    if not record.benchmarks:
        raise HTTPException(status_code=425, detail="Benchmarks are not yet ready.")

    return {
        "candidates": record.candidates,
        "benchmarks": record.benchmarks
    }

@router.get("/analysis/{analysis_id}/recommendation")
async def get_analysis_recommendation(
    analysis_id: str,
    weight_perf: float = 0.80,
    weight_time: float = 0.20
):
    record = AnalysisService.get_record(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")
    if not record.decision:
        raise HTTPException(status_code=425, detail="Recommendation is not yet ready.")

    weights = DecisionWeights(weight_performance=weight_perf, weight_runtime=weight_time)
    decision = DecisionEngine.rank_candidates(record.benchmarks, weights=weights)

    return {
        "decision": decision,
        "shap": record.shap_result,
        "explanation": record.llm_explanation
    }

@router.get("/analysis/{analysis_id}/export/script")
async def export_script(analysis_id: str):
    try:
        script_code = AnalysisService.export_python_script(analysis_id)
        return Response(
            content=script_code,
            media_type="text/x-python",
            headers={"Content-Disposition": "attachment; filename=pipeline.py"}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))