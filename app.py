"""Standalone Steel Energy prediction dashboard; no network or SAS required."""
from pathlib import Path
from datetime import datetime, date, time
import json
import pandas as pd
import joblib
import streamlit as st
from features import manual_features, FEATURES
from regression_model import predict_kwh

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Steel Energy | Predictive Analytics', page_icon='⚡', layout='wide')
st.title('Steel Energy | Predictive Analytics')
st.caption('Standalone academic demonstrator · Regression: SAS-exported coefficients · Classification: independently trained Python Random Forest')

@st.cache_resource
def load_classifier():
    obj = joblib.load(ROOT/'models'/'python_random_forest.joblib')
    if obj['features'] != FEATURES:
        raise ValueError('Feature order mismatch. Run train_model.py again.')
    return obj

@st.cache_data
def load_examples():
    return pd.read_csv(ROOT/'data'/'demo_examples.csv')

try:
    bundle = load_classifier()
    examples = load_examples()
    py_metrics = json.loads((ROOT/'models'/'python_validation_metrics.json').read_text(encoding='utf-8'))
except (FileNotFoundError, ValueError) as exc:
    st.error(f'Application files are missing or inconsistent: {exc}. Run: python train_model.py')
    st.stop()

with st.sidebar:
    st.header('Project details')
    st.markdown('**Source:** UCI Steel Industry Energy Consumption (2018)')
    st.markdown('**Prediction interval:** 15 minutes')
    st.markdown('**Regression:** SAS Studio — Forward Selection')
    st.markdown('**Classification:** Python scikit-learn — separately fitted RF')
    st.info('This app works without SAS or internet once installed. It is a classroom prototype, not an industrial control system.')

predict_tab, results_tab, methods_tab = st.tabs(['Make a prediction', 'Model comparison', 'How the app works'])

with predict_tab:
    st.subheader('Predict a 15-minute operating interval')
    st.write('Choose a sample observation or enter a date/time and four known historical consumption readings. The model estimates the consumption and load category for that interval.')
    choice = st.selectbox('Input example', ['Enter values manually'] + [f"SAS test-period example {i+1} — {row['timestamp']}" for i, row in examples.iterrows()])
    if choice == 'Enter values manually':
        dt = datetime(2018, 11, 12, 10, 0)
        defaults = {'usage_lag1': 40.0, 'usage_lag4': 35.0, 'usage_lag96': 32.0, 'usage_lag672': 30.0}
        truth = None
    else:
        ix = int(choice.split('example ')[1].split(' — ')[0]) - 1
        record = examples.iloc[ix]
        dt = pd.Timestamp(record['timestamp']).to_pydatetime()
        defaults = {k: float(record[k]) for k in ['usage_lag1','usage_lag4','usage_lag96','usage_lag672']}
        truth = {'Usage_kWh': float(record['Usage_kWh']), 'Load_Type': record['Load_Type']}

    with st.form('prediction_input'):
        col1,col2 = st.columns(2)
        with col1:
            selected_date = st.date_input('Prediction date', value=dt.date())
            selected_time = st.time_input('Prediction time', value=dt.time(), step=900)
        with col2:
            st.caption('Use measurements known before the prediction interval.')
            lag1 = st.number_input('15 minutes earlier (kWh)', min_value=0.0, max_value=1000.0, value=defaults['usage_lag1'], step=0.5)
            lag4 = st.number_input('1 hour earlier (kWh)', min_value=0.0, max_value=1000.0, value=defaults['usage_lag4'], step=0.5)
            lag96 = st.number_input('24 hours earlier (kWh)', min_value=0.0, max_value=1000.0, value=defaults['usage_lag96'], step=0.5)
            lag672 = st.number_input('7 days earlier (kWh)', min_value=0.0, max_value=1000.0, value=defaults['usage_lag672'], step=0.5)
        submitted = st.form_submit_button('Generate predictions', type='primary', use_container_width=True)
    if submitted:
        when = datetime.combine(selected_date, selected_time)
        feat = manual_features(when, lag1, lag4, lag96, lag672)
        pred_reg = predict_kwh(feat)
        row = pd.DataFrame([{k: feat[k] for k in FEATURES}], columns=FEATURES)
        model = bundle['model']
        pred_cls = model.predict(row)[0]
        probs = model.predict_proba(row)[0]
        a,b = st.columns(2)
        a.metric('Predicted electricity use', f'{pred_reg:,.2f} kWh')
        b.metric('Predicted operating load', pred_cls.replace('_',' '))
        if pred_reg < 0:
            st.warning('The unconstrained SAS linear equation returned a negative value. This is a modelling limitation; no artificial clipping has been applied.')
        st.subheader('Predicted classification probabilities')
        pp = pd.DataFrame({'Load category': [x.replace('_',' ') for x in model.classes_], 'Probability': probs})
        st.bar_chart(pp.set_index('Load category')['Probability'])
        st.dataframe(pp.style.format({'Probability':'{:.1%}'}), hide_index=True, use_container_width=True)
        if truth is not None:
            st.caption(f"Reference test row: observed consumption {truth['Usage_kWh']:.2f} kWh; observed category {truth['Load_Type'].replace('_',' ')}. This compares with one historical observation, not a new evaluation metric.")
        with st.expander('See the engineered features used'):
            st.dataframe(pd.DataFrame([feat]).T.rename(columns={0:'Value'}))
        st.download_button('Download this prediction as CSV', pd.DataFrame([{**{'prediction_time':when.isoformat(),'predicted_kwh':pred_reg,'predicted_load':pred_cls},**{f'prob_{c}':float(p) for c,p in zip(model.classes_,probs)}}]).to_csv(index=False), file_name='steel_prediction.csv', mime='text/csv')

with results_tab:
    st.subheader('Results reported by SAS')
    st.caption('The figures in these tables originated in your SAS Studio and SAS Viya output reports. The app does not recalculate them.')
    reg = pd.DataFrame([
        ['Forward',11.24823,6.06005,0.86807],['Stepwise',11.24823,6.06005,0.86807],
        ['Backward',11.25896,6.11670,0.86782],['LASSO',11.34070,6.12184,0.86589],
        ['Elastic Net',11.34070,6.12184,0.86589],['LAR',11.34070,6.12184,0.86589]
    ], columns=['Regression model','Test RMSE (kWh)','Test MAE (kWh)','Test R²'])
    st.dataframe(reg, hide_index=True, use_container_width=True)
    st.subheader('SAS Viya classification comparison')
    sas_cls = pd.DataFrame([
        ['Full Tree',0.7779,0.6149,0.8133],['Pruned Tree',0.7807,0.6180,0.8110],
        ['Gradient Boosting',0.7656,0.6086,0.7937],['Random Forest',0.7930,0.6333,0.8371],
        ['Multinomial Logistic',0.7610,0.5722,0.8194]
    ], columns=['SAS Viya model','Reported accuracy','Reported event F1','Reported event AUC'])
    st.dataframe(sas_cls.style.format({'Reported accuracy':'{:.2%}','Reported event F1':'{:.4f}','Reported event AUC':'{:.4f}'}),hide_index=True,use_container_width=True)
    st.caption('SAS event F1 / AUC are not necessarily macro-averaged multiclass metrics. SAS-selected KS champion: Full Decision Tree; preferred on displayed accuracy/F1/AUC: Random Forest.')
    st.divider()
    st.subheader('Separate Python classifier: held-out test metrics')
    c1,c2,c3=st.columns(3)
    c1.metric('Test accuracy',f"{py_metrics['accuracy_test']:.2%}")
    c2.metric('Macro F1',f"{py_metrics['macro_f1_test']:.3f}")
    c3.metric('Weighted F1',f"{py_metrics['weighted_f1_test']:.3f}")
    matrix=pd.DataFrame(py_metrics['confusion_matrix'],index=[f'Actual {c}' for c in py_metrics['classes']],columns=[f'Predicted {c}' for c in py_metrics['classes']])
    st.write('Python classifier confusion matrix (10,311 test observations)')
    st.dataframe(matrix, use_container_width=True)
    st.warning('The Python classification metrics are NOT SAS Viya results and must not be presented as such in the assignment.')

with methods_tab:
    st.subheader('Application architecture')
    st.markdown('**User inputs → Python feature engineering → two scoring paths → displayed outputs**')
    st.markdown('**Path 1:** Original SAS Studio Forward Selection coefficients → local numeric kWh prediction. No model retraining.')
    st.markdown('**Path 2:** Independently fitted scikit-learn Random Forest → load class and class probabilities. No live SAS connection.')
    st.markdown('**Data split:** 34,368 eligible records after the first 672 history-building rows; first 24,057 train and remaining 10,311 test, chronologically.')
    st.markdown('**Predictors:** Calendar variables plus consumption at 15 minutes, 1 hour, 24 hours and 7 days earlier; interaction/polynomial terms.')
    st.markdown('**Note:** Load_Type labels represent the original dataset categories, which strongly reflect operating schedules. They are not defined simply by kWh thresholds.')
    st.markdown('**Limitations:** Historical 2018 single-facility data; generalisation may be weak outside its observed operating conditions. The two models may disagree. This is an academic proof of concept.')
