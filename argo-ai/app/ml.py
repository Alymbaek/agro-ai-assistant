import joblib
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
path_yield = base_dir / 'models' / 'yield_model.joblib'
path_risk = base_dir / 'models' / 'risk_model.joblib'

yield_model = joblib.load(path_yield)
risk_model = joblib.load(path_risk)

def prepare_features(data):
    df = pd.DataFrame([data])

    # one-hot как при обучении
    df['region_Osh'] = (df['region'] == 'Osh').astype(int)
    df = df.drop(columns=['region'])

    return df


def predict(data):
    features = prepare_features(data)

    yield_pred = yield_model.predict(features)[0]
    risk_pred = risk_model.predict(features)[0]

    return round(float(yield_pred), 2), risk_pred





