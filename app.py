"""
CPS Anomaly Detection - Final Iteration Streamlit App
Features:
- Read CSV from file or URL
- Data exploration & visualization
- Choose analysis method: Z-score, IsolationForest, PCA reconstruction error
- Produce anomaly scores, mark top anomalies
- Export cleaned CSV and downloadable HTML report
- Basic input validation, file-size limit, help page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO, BytesIO
import requests
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import base64
import json
import time
import uuid
import html
import os

# -----------------------
# Configuration
# -----------------------
MAX_FILE_MB = 10  # max upload size
ALLOWED_EXTENSIONS = ('.csv',)
st.set_page_config(page_title="CPS Anomaly Detection", layout="wide")


# -----------------------
# Helper utilities
# -----------------------
def allowed_file(filename):
    return filename.lower().endswith(ALLOWED_EXTENSIONS)

def safe_read_csv_bytes(bytes_data):
    return pd.read_csv(StringIO(bytes_data.decode('utf-8')))

def read_csv_from_url(url):
    # Basic safety: only http(s)
    if not url.lower().startswith(('http://', 'https://')):
        raise ValueError("URL must begin with http:// or https://")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    # limit size
    if int(r.headers.get('Content-Length', 0)) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError("Remote file too large")
    return pd.read_csv(StringIO(r.text))

def sanitize_column_names(df):
    df = df.copy()
    df.columns = [c.strip().replace(' ', '_').replace('/', '_') for c in df.columns]
    return df

def compute_zscore_anomaly(df_numeric):
    # compute per-row multivariate z-score (Euclidean of z-scores)
    z = (df_numeric - df_numeric.mean()) / df_numeric.std(ddof=0)
    score = np.sqrt((z**2).sum(axis=1))
    return score

def compute_isolation_forest(df_numeric, contamination=0.01, random_state=42):
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(df_numeric.values)
    scores = -clf.score_samples(df_numeric.values)  # higher => more anomalous
    return scores

def compute_pca_recon_error(df_numeric, n_components=0.95):
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(df_numeric.values)
    transformed = pca.transform(df_numeric.values)
    recon = pca.inverse_transform(transformed)
    errors = np.mean((df_numeric.values - recon)**2, axis=1)
    return errors

def make_html_report(title, metadata, df_sample, plots_png, anomaly_table_html, pipeline_log):
    # Simple HTML report builder
    html_parts = []
    html_parts.append(f"<h1>{html.escape(title)}</h1>")
    html_parts.append("<h2>Metadata</h2><pre>{}</pre>".format(html.escape(json.dumps(metadata, indent=2))))
    html_parts.append("<h2>Pipeline Log</h2><pre>{}</pre>".format(html.escape(json.dumps(pipeline_log, indent=2))))
    html_parts.append("<h2>Data sample</h2>")
    html_parts.append(df_sample.to_html(index=False))
    for i, img_b64 in enumerate(plots_png):
        html_parts.append(f"<h3>Plot {i+1}</h3>")
        html_parts.append(f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;">')
    html_parts.append("<h2>Top anomalies</h2>")
    html_parts.append(anomaly_table_html)
    return "<html><body>{}</body></html>".format("\n".join(html_parts))

def fig_to_base64(fig):
    # fig is a plotly figure; convert to png image bytes via to_image
    try:
        img_bytes = fig.to_image(format="png", width=800, height=400, scale=1)
        return base64.b64encode(img_bytes).decode('ascii')
    except Exception:
        return ""

# -----------------------
# UI Layout
# -----------------------
st.title("CPS Anomaly Detection — Final Iteration")
st.markdown("Interactive prototype: load data (file or URL), explore, choose anomaly method, visualize, and export a report.")

menu = st.sidebar.selectbox("Navigation", ["Run Analysis", "Help / Documentation", "Implementation Notes"])

# -----------------------
# Help Page
# -----------------------
if menu == "Help / Documentation":
    st.header("Help & How to use this product")
    st.markdown("""
    **Quick start**
    1. Upload a CSV (must contain numeric sensor columns) or enter a direct CSV URL.
    2. On the Run Analysis page choose cleaning options then an analysis method.
    3. Press **Run Analysis**. View interactive plots and the anomalies table.
    4. Export cleaned CSV or generate a downloadable HTML report.
    
    **Notes**
    - File size limit: {} MB.
    - Required columns: timestamp (optional but recommended) and numeric sensor columns.
    - Methods: Z-score (simple), IsolationForest (ensemble), PCA reconstruction error (proxy for autoencoder).
    """.format(MAX_FILE_MB))
    st.markdown("**Security & Privacy**: Do not upload PII to the demo. In production, use encrypted storage and access controls.")
    st.stop()

# -----------------------
# Implementation Notes
# -----------------------
if menu == "Implementation Notes":
    st.header("Implementation Notes (Milestone deliverables)")
    st.markdown("""
    This prototype demonstrates:
    - Interactive data ingestion (file + URL)
    - Data exploration and cleaning choices
    - Three anomaly detection methods
    - Report generation (HTML), downloadable
    - Basic input validation and file size limits
    - Export cleaned data
    """)
    st.markdown("**To run locally**: `streamlit run app.py`")
    st.stop()

# -----------------------
# Run Analysis Page
# -----------------------
# --- Data input
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
url_input = st.sidebar.text_input("Or enter CSV URL (http/https)")

df = None
source = None
pipeline_log = []

if uploaded:
    if not allowed_file(uploaded.name):
        st.sidebar.error("Only CSV files are supported.")
    else:
        size_mb = uploaded.size / (1024*1024)
        if size_mb > MAX_FILE_MB:
            st.sidebar.error(f"File too large ({size_mb:.1f} MB). Max {MAX_FILE_MB} MB.")
        else:
            try:
                df = pd.read_csv(uploaded)
                source = f"upload:{uploaded.name}"
                pipeline_log.append({"event":"file_uploaded","name":uploaded.name,"size_mb":size_mb})
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")

elif url_input:
    if st.sidebar.button("Fetch URL"):
        try:
            df = read_csv_from_url(url_input.strip())
            source = f"url:{url_input}"
            pipeline_log.append({"event":"url_fetched","url":url_input})
            st.sidebar.success("URL loaded")
        except Exception as e:
            st.sidebar.error(f"URL error: {e}")

if df is None:
    st.info("Upload or fetch a CSV to begin. Example columns: timestamp, sensor1, sensor2, ...")
    st.stop()

# sanitize
df = sanitize_column_names(df)
st.subheader("Data preview")
st.dataframe(df.head())

# --- Data exploration
st.subheader("Exploration")
st.write("Basic info:")
st.write({"rows": df.shape[0], "columns": df.shape[1]})
st.write(df.describe(include='all'))

numeric = df.select_dtypes(include=[np.number])
if numeric.shape[1] == 0:
    st.error("No numeric columns found — at least one numeric sensor column is required.")
    st.stop()

st.markdown("**Correlation matrix**")
corr = numeric.corr()
st.dataframe(corr.round(2))

# --- Visualization pick
st.subheader("Visualizations")
vis_choice = st.selectbox("Plot type", ["Time-series (timestamp vs sensor)", "Scatter (2 sensors)", "Histogram (sensor)"])
if vis_choice.startswith("Time") and 'timestamp' in df.columns:
    ts_col = 'timestamp'
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception:
        st.warning("Could not parse timestamp column, using row index instead.")
        df['timestamp'] = pd.RangeIndex(len(df))
    sensor_ts = st.selectbox("Sensor", numeric.columns.tolist())
    fig_ts = px.line(df, x='timestamp', y=sensor_ts, title=f"{sensor_ts} over time")
    st.plotly_chart(fig_ts, use_container_width=True)
elif vis_choice.startswith("Scatter"):
    s1 = st.selectbox("X sensor", numeric.columns.tolist(), index=0, key="x")
    s2 = st.selectbox("Y sensor", numeric.columns.tolist(), index=1 if numeric.shape[1]>1 else 0, key="y")
    fig_sc = px.scatter(df, x=s1, y=s2, title=f"{s2} vs {s1}")
    st.plotly_chart(fig_sc, use_container_width=True)
else:
    s = st.selectbox("Histogram sensor", numeric.columns.tolist(), key="hist")
    fig_hist = px.histogram(df, x=s, nbins=50, title=f"Histogram of {s}")
    st.plotly_chart(fig_hist, use_container_width=True)

# --- Cleaning and preprocessing
st.subheader("Cleaning & Preprocessing")
strategy = st.selectbox("Missing value strategy", options=["drop", "mean", "median", "interpolate"])
scale_opt = st.checkbox("Apply standard scaling")

if st.button("Apply cleaning (preview)"):
    dclean = df.copy()
    if strategy == "drop":
        dclean = dclean.dropna()
    elif strategy in ["mean", "median"]:
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy=strategy)
        dclean[numeric.columns] = imp.fit_transform(dclean[numeric.columns])
    else:
        dclean[numeric.columns] = dclean[numeric.columns].interpolate().fillna(method="bfill").fillna(method="ffill")
    if scale_opt:
        scaler = StandardScaler()
        dclean[numeric.columns] = scaler.fit_transform(dclean[numeric.columns])
    st.write(dclean.head(10))
    pipeline_log.append({"event":"clean_preview","strategy":strategy,"scaling":bool(scale_opt)})
    st.success("Preview generated. Use Export to save cleaned CSV.")

# Export cleaned data
if st.button("Export cleaned CSV"):
    dclean = df.copy()
    if strategy == "drop":
        dclean = dclean.dropna()
    elif strategy in ["mean", "median"]:
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy=strategy)
        dclean[numeric.columns] = imp.fit_transform(dclean[numeric.columns])
    else:
        dclean[numeric.columns] = dclean[numeric.columns].interpolate().fillna(method="bfill").fillna(method="ffill")
    if scale_opt:
        scaler = StandardScaler()
        dclean[numeric.columns] = scaler.fit_transform(dclean[numeric.columns])
    csv_bytes = dclean.to_csv(index=False).encode('utf-8')
    st.download_button("Download cleaned CSV", csv_bytes, file_name="cleaned_data.csv", mime="text/csv")

# --- Choose analysis method
st.subheader("Anomaly Detection")
method = st.selectbox("Method", ["Z-score", "IsolationForest", "PCA_reconstruction"])
contamination = st.slider("Contamination (IsolationForest)", min_value=0.001, max_value=0.2, value=0.01, step=0.001)
pca_var = st.slider("PCA retained variance (PCA_reconstruction)", min_value=0.5, max_value=0.999, value=0.95)

if st.button("Run Analysis"):
    t0 = time.time()
    numeric = df.select_dtypes(include=[np.number]).copy().dropna()
    pipeline_log.append({"event":"analysis_start","method":method})
    try:
        if method == "Z-score":
            scores = compute_zscore_anomaly(numeric)
        elif method == "IsolationForest":
            scores = compute_isolation_forest(numeric, contamination=contamination)
        else:
            scores = compute_pca_recon_error(numeric, n_components=pca_var)
        # attach scores
        result = df.loc[numeric.index].copy()
        result['anomaly_score'] = scores
        # normalized score
        result['anomaly_rank'] = result['anomaly_score'].rank(method='first', ascending=False)
        n_top = min(20, len(result))
        top_anoms = result.sort_values('anomaly_score', ascending=False).head(n_top)
        pipeline_log.append({"event":"analysis_end","time_s": round(time.time()-t0,2),"n_rows": len(result)})
        st.success(f"Analysis complete in {round(time.time()-t0,2)}s; found {len(top_anoms)} top anomalies (top {n_top}).")

        # plots: time series with anomalies (if timestamp)
        plots_png = []
        if 'timestamp' in result.columns:
            try:
                result['timestamp'] = pd.to_datetime(result['timestamp'])
                sensor = numeric.columns[0]
                fig = px.line(result, x='timestamp', y=sensor, title=f"{sensor} with anomaly score")
                fig.add_scatter(x=result['timestamp'], y=result[sensor], mode='markers',
                                marker=dict(size=6, color=result['anomaly_score'], colorscale='reds'), name='anomaly_score')
                st.plotly_chart(fig, use_container_width=True)
                plots_png.append(fig_to_base64(fig))
            except Exception:
                pass

        # histogram of scores
        fig2 = px.histogram(result, x='anomaly_score', nbins=50, title='Anomaly score distribution')
        st.plotly_chart(fig2, use_container_width=True)
        plots_png.append(fig_to_base64(fig2))

        # top anomalies table
        st.subheader("Top anomalies")
        st.dataframe(top_anoms.head(50))

        # Offer to generate HTML report
        if st.button("Generate downloadable HTML report"):
            metadata = {"source": source, "method": method, "timestamp": time.ctime()}
            report_html = make_html_report("CPS Anomaly Report", metadata, result.head(20), plots_png,
                                           top_anoms.head(50).to_html(index=False), pipeline_log)
            b = report_html.encode('utf-8')
            st.download_button("Download HTML report", b, file_name="anomaly_report.html", mime="text/html")
            pipeline_log.append({"event":"report_generated","report_type":"html"})

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        pipeline_log.append({"event":"analysis_error","error":str(e)})

# End of app
