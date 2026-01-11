# Quick Start Guide

## Prerequisites

- Python 3.8+ (with pip)
- Node.js 18+ (with npm)
- Trained models (run `python scripts/train_all.py` first)

## Running the Application

### 1. Start the Backend API

Open a terminal and run:

```bash
cd api
pip install -r requirements.txt
python run.py
```

The API will start on http://localhost:5000

You can verify it's working by visiting:
- http://localhost:5000/docs - Interactive API documentation
- http://localhost:5000/api/health - Health check endpoint

### 2. Start the Frontend

Open a **new terminal** and run:

```bash
cd frontend
npm install
npm run dev
```

The frontend will start on http://localhost:3000

Open your browser and navigate to http://localhost:3000

## Usage

1. Select a language (English or Romanian)
2. Choose a model type (lexicon, ML/TF-IDF, or hybrid)
3. Enter text in the textarea
4. Click "Detect Emotion" to get the prediction

## Troubleshooting

### Backend Issues

- **Models not found**: Make sure you've trained the models first:
  ```bash
  python scripts/train_all.py
  ```

- **Port 5000 already in use**: Change the port in `api/run.py`:
  ```python
  uvicorn.run(..., port=5001)
  ```

### Frontend Issues

- **npm install fails**: Make sure you have Node.js 18+ installed
- **Port 3000 already in use**: Vite will automatically use the next available port

### CORS Issues

If you see CORS errors, make sure the frontend port (3000) is in the `CORS_ORIGINS` list in `api/app/core/config.py`
