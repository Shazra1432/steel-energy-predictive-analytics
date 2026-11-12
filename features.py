"""Feature definitions matching the user's SAS feature-engineering workflow."""
import math
import pandas as pd

FEATURES = [
    'hour_cos', 'hour_lag1', 'hour_of_day', 'hour_sin', 'is_weekend',
    'month_no', 'usage_lag1', 'usage_lag1_sq', 'usage_lag4',
    'usage_lag672', 'usage_lag96', 'weekday_no'
]

def calendar_features(when):
    """SAS WEEKDAY convention: Sunday=1, Monday=2, ..., Saturday=7."""
    hour = when.hour
    weekday_no = ((when.weekday() + 1) % 7) + 1
    return {
        'hour_of_day': hour,
        'hour_sin': math.sin(2 * math.pi * hour / 24),
        'hour_cos': math.cos(2 * math.pi * hour / 24),
        'month_no': when.month,
        'weekday_no': weekday_no,
        'is_weekend': int(weekday_no in (1, 7)),
    }

def manual_features(when, lag1, lag4, lag96, lag672):
    f = calendar_features(when)
    f.update({
        'usage_lag1': float(lag1),
        'usage_lag4': float(lag4),
        'usage_lag96': float(lag96),
        'usage_lag672': float(lag672),
        'usage_lag1_sq': float(lag1)**2,
        'hour_lag1': when.hour * float(lag1),
    })
    return f

def prepare(raw):
    d = raw.copy()
    d['timestamp'] = pd.to_datetime(d['date'], dayfirst=True, errors='raise')
    d = d.sort_values('timestamp', kind='stable').reset_index(drop=True)
    if d['timestamp'].duplicated().any():
        raise ValueError('Duplicate timestamps found; inspect the source CSV.')
    if d['timestamp'].diff().dropna().ne(pd.Timedelta(minutes=15)).any():
        raise ValueError('Timestamp intervals are not consistently 15 minutes.')
    t = d['timestamp']
    d['hour_of_day'] = t.dt.hour
    d['hour_sin'] = (2 * math.pi * d['hour_of_day'] / 24).map(math.sin)
    d['hour_cos'] = (2 * math.pi * d['hour_of_day'] / 24).map(math.cos)
    d['month_no'] = t.dt.month
    d['weekday_no'] = ((t.dt.dayofweek + 1) % 7) + 1
    d['is_weekend'] = d['WeekStatus'].eq('Weekend').astype(int)
    for lag in (1, 4, 96, 672):
        d[f'usage_lag{lag}'] = d['Usage_kWh'].shift(lag)
    d['usage_lag1_sq'] = d['usage_lag1'] ** 2
    d['hour_lag1'] = d['hour_of_day'] * d['usage_lag1']
    d = d.iloc[672:].reset_index(drop=True)
    assert len(d) == 34368, 'Unexpected eligible record count'
    return d
