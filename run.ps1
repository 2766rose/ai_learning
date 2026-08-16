param(
    [ValidateSet("dev", "celery")]
    [string]$Target = "dev"
)

switch ($Target) {
    "dev" {
        Write-Host "🚀 Starting API server..." -ForegroundColor Green
        uvicorn ai_rag.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src/ai_rag
    }
    "celery" {
        Write-Host "🔧 Starting Celery worker..." -ForegroundColor Yellow
        celery -A ai_rag.tasks.celery_app worker --loglevel=info --pool=solo -Q celery
    }
}
