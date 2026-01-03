# MLOps CI/CD Demo (Student Friendly)

This repo demonstrates a complete CI/CD pipeline for ML models.

## Flow
- Train model
- Evaluate with quality gate
- Run tests
- Build Docker image

## Local Run
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
uvicorn app.main:app --reload
