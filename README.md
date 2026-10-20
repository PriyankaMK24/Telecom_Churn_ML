# Customer Churn Prediction Using Machine Learning

A beginner-friendly Streamlit dashboard that explores a processed IBM Telco Customer Churn dataset, compares three scikit-learn classifiers, and predicts whether an entered customer is likely to churn or stay.

## Dataset

The project uses the existing file `Data/processed_data.csv` directly. The current file contains 7,043 customer rows. The training script detects the target column (it supports common names such as `Churn Value`, `Churn`, and `Churn Label`) and converts its two classes to 0 = stayed and 1 = churned.

Put your processed CSV at `Data/processed_data.csv`, or edit `DATA_PATH` near the top of `train_model.py` and `app.py`. The target must be a binary customer outcome. Columns whose names indicate churn/target information, customer identifiers, row indexes, and constant columns are removed from model inputs. This prevents duplicate labels such as `Churn Label_Yes` from leaking the answer into the features.

## Project structure

```text
Telecom-Churn-Prediction/
├── app.py                      # Streamlit dashboard
├── train_model.py              # Dataset inspection, training and evaluation
├── requirements.txt            # Python dependencies used by this project
├── Data/
│   └── processed_data.csv       # Existing processed IBM Telco dataset
├── Models/
│   ├── churn_models.joblib      # Saved fitted pipelines and evaluation data
│   └── model_metrics.json       # Test-set metrics for the three models
└── README.md
```

## Setup in VS Code (Windows PowerShell)

1. Open this project folder in VS Code. Open **Terminal → New Terminal** and confirm the terminal is in the project folder.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. You can also use `.venv\Scripts\python.exe` without activating.

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Train and compare the models (run again only after changing the dataset or training code):

   ```powershell
   python train_model.py
   ```

   The script uses an 80/20 stratified split. Imputation, scaling, and one-hot encoding are fitted inside each pipeline on the training partition only. It trains Logistic Regression, Decision Tree, and Random Forest classifiers with class weighting, evaluates each on the held-out test partition, and selects the highest F1-score model (ROC-AUC breaks a tie). It saves fitted pipelines to `Models/churn_models.joblib` and the measured test metrics to `Models/model_metrics.json`.

5. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

   Streamlit prints a local address, usually `http://localhost:8501`, and opens it in your browser. Stop the server with **Ctrl+C** in the terminal.

## Dashboard sections

- **Home Dashboard:** customer counts, churn rate, target detection, and whether the trained artifact is available.
- **Customer Churn Prediction:** generates a form from the actual model input columns and dataset values. Predictions use the saved best pipeline and show a probability when the classifier supports it.
- **Upload a dataset:** upload a CSV with the required model input columns, score every row, review the predictions, and download the original rows with prediction and probability columns added. A blank CSV template is available in the app. Extra columns such as IDs and existing churn labels are preserved in the output but are never used as model inputs.
- **Model Performance:** compares accuracy, precision, recall, F1-score and ROC-AUC; shows test confusion matrices and held-out permutation feature importance.
- **Data Analysis:** plots churn distribution, churn by contract type, tenure and monthly charges when those columns are present. Encoded contract indicator columns are recognized.

The model is not retrained when Streamlit starts. If no artifact exists, train it first with `python train_model.py`.

## Interpreting predictions

“Likely to Churn” and “Likely to Stay” are statistical estimates learned from past records. A probability is the model's estimated churn likelihood, not a guarantee. The model may perform differently when applied to new populations, changed services, or incomplete and differently encoded data. Feature importance describes predictive association and does not show that a feature causes churn.

Model metrics are calculated from the dataset's held-out test partition; they are not promised in advance and can change when the dataset changes. Review recall and precision alongside accuracy because churn classes may be imbalanced.
