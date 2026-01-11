# Emotion Detection API

FastAPI backend for emotion detection models.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python run.py
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --reload --port 5000
```

The API will be available at http://localhost:5000

API documentation available at http://localhost:5000/docs

## Architecture

- `app/main.py` - Application entry point
- `app/core/` - Core functionality (config, model manager, exceptions)
- `app/api/v1/` - API routes
- `app/models/` - Pydantic schemas
- `app/services/` - Business logic layer
