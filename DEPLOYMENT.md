# Backend Deployment (Render)

## Runtime
- Python web service using FastAPI + Uvicorn

## Render Service Setup
- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

You can use the included `render.yaml` for infrastructure-as-code deployment.

## Required Environment Variables
- `DATABASE_URL`

## Recommended Environment Variables
- `APP_ENV=production`
- `CORS_ORIGINS` and/or `CORS_ORIGIN_REGEX`
- Additional auth/email settings from `.env.example` as needed for your enabled features
