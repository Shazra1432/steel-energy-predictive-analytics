# Steel Energy standalone Streamlit app

Start with `START_HERE_WINDOWS.md`. This project is a classroom prototype.

- `app.py`: user interface, predictions, SAS comparison and Python validation tables
- `regression_model.py`: exactly implements the coefficients from the included SAS PROC GLMSELECT score code
- `features.py`: common calendar and lag features
- `train_model.py`: reproducibly fits the **independent Python** classifier on the included raw CSV
- `models/python_random_forest.joblib`: ready-to-use independently trained Python Random Forest
- `models/python_validation_metrics.json`: independently generated Python test results
- `data/demo_examples.csv`: examples from the holdout period
- `reference/Forward_Regression_Score_FROM_SAS.sas`: provided SAS reference score code
- `data/Steel_industry_data.csv`: originally uploaded UCI raw data

Windows commands: `py -m venv .venv`, `.venv\Scripts\activate.bat`, `python -m pip install -r requirements.txt`, `python -m streamlit run app.py`.

**Non-equivalence warning:** Python Random Forest is NOT the SAS Viya trained Random Forest. It is a separate reproducible demonstrator. The SAS Model Studio comparison figures displayed in the app are labelled as such.
