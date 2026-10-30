import pandas as pd

ORDER = ['minor', 'major', 'fatal']
CAT = ['road_type', 'weather', 'visibility', 'lighting', 'traffic_density', 'cause', 'city', 'day_of_week', 'festival']
NUM = ['hour', 'lanes', 'temperature', 'traffic_signal', 'is_peak_hour', 'is_weekend', 'month']
A = CAT + NUM
B = A + ['vehicles_involved', 'casualties']
C = B + ['risk_score']
SETS = {'A: Conditions': A, 'B: A + crash outcome': B, 'C: B + risk_score': C}
TREE_PARAMS = {'A: Conditions': {'max_depth': 12, 'min_samples_leaf': 1},
               'B: A + crash outcome': {'max_depth': 12, 'min_samples_leaf': 100},
               'C: B + risk_score': {'max_depth': 12, 'min_samples_leaf': 20}}


def lighting(h):
    if 7 <= h <= 17: return 'daylight'
    if h in (5, 6, 18, 19): return 'dawn/dusk'
    return 'dark'


def load_data(path='indian_roads_dataset.csv'):
    # keep_default_na=False stops the festival value "None" being read as missing
    df = pd.read_csv(path, keep_default_na=False)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['lighting'] = df['hour'].map(lighting)
    return df


def cols_for(columns, feats):
    return [c for c in columns if any(c == f or (f in CAT and c.startswith(f + '_')) for f in feats)]
