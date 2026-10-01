Setup and run: the commands from Steps 1, 5 and 6 (create venv, pip install -r requirements.txt, python src/train.py, python src/predict.py sample_customer.json), plus where to put the Kaggle CSV.
What I did: 4–5 lines on dropping customerID, fixing TotalCharges, one-hot encoding, class_weight="balanced", and keeping everything in one Pipeline to avoid data leakage.
Model comparison: the table printed by train.py, with your real numbers.
Why accuracy is not enough: about 74% of customers stay, so a model that always says "No churn" gets 74% accuracy and catches no churners. ROC-AUC, PR-AUC and F1 show how well the model finds the churners.
With 2 more days, top 3 improvements:
Serve it as a FastAPI app in Docker, with input validation and tests.
Add experiment tracking (MLflow) and a CI/CD pipeline for retraining.
Add monitoring for data drift, and tune the decision threshold to the business cost of losing a customer.