import json
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from common import load_data, lighting, ORDER, CAT, A, SETS, TREE_PARAMS

st.set_page_config(page_title='Road Accident Severity', page_icon='🚦', layout='wide')

SEV_COL = {'minor': '#2a78d6', 'major': '#eb6834', 'fatal': '#1baf7a'}
MODEL_COL = {'Decision Tree': '#2a78d6', 'SVM': '#eb6834'}
LABELS = {'weather': 'Weather', 'road_type': 'Road type', 'lighting': 'Lighting (from hour)', 'traffic_density': 'Traffic density',
          'cause': 'Cause', 'vehicles_involved': 'Vehicles involved', 'visibility': 'Visibility', 'city': 'City',
          'festival': 'Festival', 'day_of_week': 'Day of week', 'lanes': 'Lanes', 'traffic_signal': 'Traffic signal',
          'is_peak_hour': 'Peak hour', 'is_weekend': 'Weekend', 'casualties': 'Casualties', 'hour': 'Hour',
          'accident_severity': 'Actual severity'}


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_results():
    with open('results.json') as f:
        return json.load(f)


@st.cache_resource
def get_predictor():
    df = get_data()
    X = pd.get_dummies(df[A], columns=CAT, dtype=int)
    tr, _ = train_test_split(df.index, test_size=.2, stratify=df.accident_severity, random_state=42)
    model = DecisionTreeClassifier(class_weight='balanced', random_state=42, **TREE_PARAMS['A: Conditions'])
    return model.fit(X.loc[tr], df.accident_severity.loc[tr]), list(X.columns)


def severity_mix(data, col):
    t = pd.crosstab(data[col], data.accident_severity, normalize='index').reindex(columns=ORDER, fill_value=0) * 100
    fig = px.bar(t.reset_index().melt(id_vars=col, var_name='Severity', value_name='Share'), y=col, x='Share', color='Severity',
                 orientation='h', color_discrete_map=SEV_COL, category_orders={'Severity': ORDER}, text=None,
                 labels={col: LABELS.get(col, col), 'Share': '% of accidents'}, title=LABELS.get(col, col))
    fig.update_traces(texttemplate='%{x:.0f}%', textposition='inside', marker_line_color='white', marker_line_width=1,
                      hovertemplate='%{y}: %{x:.1f}%<extra>%{fullData.name}</extra>')
    fig.update_layout(barmode='stack', height=300, xaxis_range=[0, 100], yaxis_type='category', margin=dict(t=45, b=10), legend_title_text='')
    return fig


df = get_data()
res = get_results()
metrics = pd.DataFrame(res['metrics'])

st.sidebar.title('🚦 Road Accident Severity')
page = st.sidebar.radio('Section', ['Overview', 'Data & Cleaning', 'Severity Explorer', 'Map', 'Model Comparison',
                                    'Accuracy by Type', 'Major Factors', 'Try a Prediction', 'Recommendations'])
st.sidebar.caption('20,000 accidents across 8 Indian cities, 2022 to 2025. Models: Decision Tree and SVM.')

if page == 'Overview':
    st.title('Predicting Road Accident Severity')
    st.write('A transportation department wants to identify accidents likely to be severe, so high-risk roads and conditions can be addressed.')
    c = st.columns(4)
    c[0].metric('Accidents', f'{len(df):,}')
    c[1].metric('Fatal share', f'{(df.accident_severity == "fatal").mean():.1%}')
    c[2].metric('Severe share (major or fatal)', f'{(df.accident_severity != "minor").mean():.1%}')
    c[3].metric('Best model', 'Decision Tree')
    st.warning('**Main finding:** conditions known before a crash barely predict severity in this dataset. '
               'The Decision Tree beats the SVM narrowly, but on before-crash conditions neither has real skill.')
    left, right = st.columns(2)
    counts = df.accident_severity.value_counts().reindex(ORDER).reset_index()
    fig = px.bar(counts, x='accident_severity', y='count', color='accident_severity', color_discrete_map=SEV_COL, text='count',
                 labels={'accident_severity': 'Severity', 'count': 'Accidents'}, title='Accidents by severity')
    left.plotly_chart(fig.update_layout(showlegend=False), width='stretch')
    show = metrics[metrics['Feature set'].isin(['Baseline', 'A: Conditions'])][['Model', 'Accuracy', 'Severe recall', 'Severe precision']].set_index('Model')
    right.subheader('Before-crash conditions vs baseline')
    right.dataframe(show.style.format('{:.1%}'), width='stretch')
    right.caption('The share of severe accidents is 44.9%, so a severe precision near that figure means the flags are no better than chance.')

elif page == 'Data & Cleaning':
    st.title('Data & Cleaning')
    c = st.columns(4)
    c[0].metric('Rows', f'{len(df):,}')
    c[1].metric('Original columns', 24)
    c[2].metric('Missing values', 0)
    c[3].metric('Duplicate rows', 0)
    st.subheader('Steps')
    st.markdown("""
- Read the file so the festival value `None` stays a category instead of becoming missing.
- Parsed `date` and added `month`.
- Added `lighting` from `hour`: daylight 7 to 17, dawn/dusk 5, 6, 18 and 19, otherwise dark.
- Removed identifiers and duplicates from the model inputs: `accident_id`, `date`, `time`, `state`, `latitude`, `longitude`.
- One-hot encoded categories, scaled inputs for the SVM, and made a stratified 80/20 train/test split.
""")
    st.subheader('Requested variables that are missing')
    st.table(pd.DataFrame({'Requested': ['Lighting', 'Speed', 'Vehicle type'],
                           'Used instead': ['Derived from hour (an approximation)', 'cause = overspeeding', 'vehicles_involved (a count)']}))
    st.subheader('Sample')
    st.dataframe(df.drop(columns=['date']).head(200), width='stretch')

elif page == 'Severity Explorer':
    st.title('Severity Explorer')
    st.write('Filter the accidents to see whether the severity mix changes.')
    f = st.columns(4)
    cities = f[0].multiselect('City', sorted(df.city.unique()))
    weather = f[1].multiselect('Weather', sorted(df.weather.unique()))
    road = f[2].multiselect('Road type', sorted(df.road_type.unique()))
    light = f[3].multiselect('Lighting', sorted(df.lighting.unique()))
    hours = st.slider('Hour of day', 0, 23, (0, 23))
    d = df[df.hour.between(*hours)]
    for col, val in [('city', cities), ('weather', weather), ('road_type', road), ('lighting', light)]:
        if val:
            d = d[d[col].isin(val)]
    if d.empty:
        st.info('No accidents match these filters.')
        st.stop()
    c = st.columns(3)
    c[0].metric('Accidents', f'{len(d):,}')
    c[1].metric('Fatal share', f'{(d.accident_severity == "fatal").mean():.1%}', f'{(d.accident_severity == "fatal").mean() - (df.accident_severity == "fatal").mean():+.1%} vs all', delta_color='off')
    c[2].metric('Severe share', f'{(d.accident_severity != "minor").mean():.1%}', f'{(d.accident_severity != "minor").mean() - (df.accident_severity != "minor").mean():+.1%} vs all', delta_color='off')
    chosen = st.multiselect('Charts', list(LABELS)[:9], default=['weather', 'road_type', 'lighting', 'traffic_density', 'cause', 'vehicles_involved'], format_func=LABELS.get)
    cols = st.columns(2)
    for i, col in enumerate(chosen):
        cols[i % 2].plotly_chart(severity_mix(d, col), width='stretch')
    t = d.groupby('hour').accident_severity.apply(lambda s: (s != 'minor').mean() * 100).reset_index(name='Severe %')
    fig = px.line(t, x='hour', y='Severe %', markers=True, title='Severe share by hour', labels={'hour': 'Hour of day'})
    st.plotly_chart(fig.update_yaxes(range=[0, 100]), width='stretch')
    t = pd.crosstab(d.weather, d.road_type, values=(d.accident_severity != 'minor') * 100, aggfunc='mean').round(1)
    st.plotly_chart(px.imshow(t, text_auto=True, color_continuous_scale='Blues', zmin=0, zmax=100, aspect='auto',
                              title='Severe % by weather and road type', labels={'x': 'Road type', 'y': 'Weather', 'color': 'Severe %'}), width='stretch')

elif page == 'Map':
    st.title('Accident Map')
    sev = st.multiselect('Severity', ORDER, default=['fatal'])
    d = df[df.accident_severity.isin(sev)]
    if d.empty:
        st.info('Choose at least one severity.')
        st.stop()
    d = d.sample(min(len(d), 3000), random_state=42)
    fig = px.scatter_geo(d, lat='latitude', lon='longitude', color='accident_severity', color_discrete_map=SEV_COL, hover_name='city',
                         hover_data={'road_type': True, 'weather': True, 'latitude': False, 'longitude': False}, category_orders={'accident_severity': ORDER}, labels={'accident_severity': 'Severity'})
    fig.update_geos(fitbounds='locations', showcountries=True)
    st.plotly_chart(fig.update_layout(height=600, margin=dict(t=10, b=10)), width='stretch')
    st.caption('Shows up to 3,000 accidents. Coordinates cluster around the 8 cities.')
    t = df.groupby('city').accident_severity.apply(lambda s: (s == 'fatal').mean() * 100).sort_values().reset_index(name='Fatal %')
    st.plotly_chart(px.bar(t, y='city', x='Fatal %', orientation='h', text_auto='.1f', title='Fatal share by city', labels={'city': 'City'}), width='stretch')

elif page == 'Model Comparison':
    st.title('Model Comparison: Decision Tree vs SVM')
    s = st.radio('Feature set', list(SETS), horizontal=True)
    st.info({'A: Conditions': 'Known before a crash. This is the realistic prevention model.',
             'B: A + crash outcome': 'Adds vehicles and casualties, which are only known after a crash. High fatal recall comes from a data pattern: every fatal record has casualties equal to vehicles.',
             'C: B + risk_score': 'Leaky: risk_score appears to be calculated using severity, so these scores are not real predictive power.'}[s])
    cols = ['Accuracy', 'Balanced accuracy', 'Macro F1', 'Minor recall', 'Major recall', 'Fatal recall', 'Severe recall', 'Severe precision']
    m = metrics[metrics['Feature set'] == s]
    long = m.melt(id_vars='Model', value_vars=cols, var_name='Metric', value_name='Score')
    long['Score'] *= 100
    fig = px.bar(long, x='Metric', y='Score', color='Model', barmode='group', color_discrete_map=MODEL_COL, text_auto='.0f', labels={'Score': '%'})
    fig.add_hline(y=55.1, line_dash='dash', annotation_text='Always-minor accuracy 55.1%', annotation_position='top left')
    st.plotly_chart(fig.update_yaxes(range=[0, 110]), width='stretch')
    st.subheader('All results')
    st.dataframe(metrics.style.format({c: '{:.1%}' for c in cols}), width='stretch', hide_index=True)
    st.subheader('Confusion matrices')
    grid = st.columns(2)
    for g, model in zip(grid, MODEL_COL):
        fig = px.imshow(res['confusion'][f'{s}|{model}'], x=ORDER, y=ORDER, text_auto=True, color_continuous_scale='Blues', title=model,
                        labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Accidents'})
        g.plotly_chart(fig, width='stretch')

elif page == 'Accuracy by Type':
    st.title('Prediction Accuracy by Type')
    bt = pd.DataFrame(res['by_type'])
    c = st.columns(3)
    var = c[0].selectbox('Variable', bt.Variable.unique(), format_func=lambda v: LABELS.get(v, v))
    s = c[1].selectbox('Feature set', list(SETS))
    d = bt[bt.Variable == var]
    long = d.melt(id_vars=['Type', 'n'], value_vars=[f'{s}|Decision Tree', f'{s}|SVM'], var_name='Model', value_name='Accuracy')
    long['Model'] = long.Model.str.split('|').str[1]
    long['Accuracy'] *= 100
    fig = px.bar(long, x='Type', y='Accuracy', color='Model', barmode='group', color_discrete_map=MODEL_COL, text_auto='.0f', hover_data=['n'], labels={'Accuracy': 'Accuracy %', 'Type': LABELS.get(var, var)})
    st.plotly_chart(fig.update_xaxes(type='category').update_yaxes(range=[0, 110]), width='stretch')
    if d.n.min() < 30:
        st.warning('Some groups have fewer than 30 test accidents, so their scores are unreliable.')
    table = long.pivot(index=['Type', 'n'], columns='Model', values='Accuracy').reset_index()
    st.dataframe(table.style.format({'Decision Tree': '{:.1f}%', 'SVM': '{:.1f}%'}), width='stretch', hide_index=True)

elif page == 'Major Factors':
    st.title('Major Factors')
    rows = []
    for col in ['road_type', 'weather', 'visibility', 'lighting', 'traffic_density', 'cause', 'city', 'festival', 'traffic_signal', 'is_peak_hour', 'is_weekend', 'lanes', 'vehicles_involved', 'casualties', 'risk_score']:
        t = pd.crosstab(df[col], df.accident_severity)
        chi2, p, _, _ = chi2_contingency(t)
        rows.append({'Variable': col, "Cramér's V": np.sqrt(chi2 / (t.values.sum() * min(t.shape[0] - 1, t.shape[1] - 1))), 'p-value': p, 'Significant': 'Yes' if p < 0.05 / 15 else 'No'})
    a = pd.DataFrame(rows).sort_values("Cramér's V")
    st.plotly_chart(px.bar(a, y='Variable', x="Cramér's V", color='Significant', orientation='h', text_auto='.3f', title='Strength of link with severity (0 = none)',
                           color_discrete_map={'Yes': '#eb6834', 'No': '#2a78d6'}).update_layout(height=520), width='stretch')
    st.write('Only `casualties` and `risk_score` show a real link, and both are outcomes or derived from the label rather than causes. Significance uses a corrected cutoff of 0.05 / 15.')
    s = st.selectbox('Decision Tree importance for', list(SETS))
    imp = pd.Series(res['importance'][s]).sort_values().tail(10).reset_index()
    imp.columns = ['Variable', 'Importance']
    st.plotly_chart(px.bar(imp, y='Variable', x='Importance', orientation='h', text_auto='.2f'), width='stretch')
    st.caption("Set A's importances are mostly noise. In Set B, vehicles_involved ranks high only because the tree compares it with casualties.")

elif page == 'Try a Prediction':
    st.title('Try a Prediction')
    st.warning('This uses the before-crash Decision Tree (33.9% test accuracy). Treat it as a demo of the model, not as a reliable risk estimate.')
    model, columns = get_predictor()
    c = st.columns(3)
    inp = {}
    for i, col in enumerate(['road_type', 'weather', 'traffic_density', 'cause', 'city', 'day_of_week', 'festival']):
        inp[col] = c[i % 3].selectbox(LABELS.get(col, col), sorted(df[col].unique()))
    inp['hour'] = c[1].slider('Hour', 0, 23, 12)
    inp['lanes'] = c[2].slider('Lanes', int(df.lanes.min()), int(df.lanes.max()), 3)
    inp['temperature'] = c[0].slider('Temperature (°C)', int(df.temperature.min()), int(df.temperature.max()), 28)
    inp['traffic_signal'] = int(c[1].checkbox('Traffic signal present'))
    inp['month'] = c[2].slider('Month', 1, 12, 6)
    vis = sorted(df[df.weather == inp['weather']].visibility.unique())
    inp['visibility'] = c[0].selectbox('Visibility', vis)
    inp['is_peak_hour'] = int(df[df.hour == inp['hour']].is_peak_hour.mode()[0])
    inp['is_weekend'] = int(inp['day_of_week'] in ('Saturday', 'Sunday'))
    inp['lighting'] = lighting(inp['hour'])
    row = pd.get_dummies(pd.DataFrame([inp])[A], columns=CAT, dtype=int).reindex(columns=columns, fill_value=0)
    pred = model.predict(row)[0]
    st.metric('Predicted severity', pred.title())
    proba = pd.DataFrame({'Severity': model.classes_, 'Probability': model.predict_proba(row)[0] * 100})
    st.plotly_chart(px.bar(proba, x='Severity', y='Probability', color='Severity', color_discrete_map=SEV_COL, category_orders={'Severity': ORDER}, text_auto='.0f',
                           labels={'Probability': 'Tree leaf share % (class-balanced)'}).update_layout(showlegend=False, yaxis_range=[0, 105]), width='stretch')

else:
    st.title('Recommendations')
    st.info('The data does not show that any road or condition is riskier, so these are general road-safety practice rather than findings from this analysis.')
    st.markdown("""
1. **Fix the data first.** Record speed, vehicle type, actual lighting, helmet and seatbelt use, and exact road segment.
2. **Don't target by weather or road type yet.** Severity is about the same across them here.
3. **Apply proven measures everywhere:** speed enforcement, drunk-driving checks, and better road maintenance and signage.
4. **Improve emergency response,** since the severity mix is similar across all conditions.
5. **Review festival-period enforcement** with more years of data, starting with Holi (34% fatal, but only 38 accidents).
6. **Rebuild `risk_score`** using only information available before a crash, then validate it on new data.
""")
