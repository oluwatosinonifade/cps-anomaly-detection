"""
CPS Anomaly Detection Data Product (Final Iteration)

This Streamlit app is part of my capstone project and it:

- Loads CPS time-series data from a CSV file (upload) or CSV URL
- Explores data (summary statistics, correlation matrix, scatter plot)
- Cleans and optionally scales numeric features
- Runs one of several unsupervised anomaly detection methods:
    * Z-score thresholding
    * Isolation Forest
    * PCA reconstruction error
- Visualizes results (time series with anomalies, anomaly-score histogram, anomaly table)
- Exports a cleaned CSV (with anomaly flags) and a human-readable HTML report
- Provides basic security (size limits, URL validation) and an in-app Help page

This is the updated version that incorporates and extends the original prototype.
"""

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import requests
from io import BytesIO, StringIO

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from typing import Tuple, Dict, Optional, List


# ---
# Page configuration
# ---
st.set_page_config(
    page_title="CPS Anomaly Detection - Data Product",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---
# Utility functions for data loading
# ---
def load_csv_from_upload(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Safely load a CSV file from a Streamlit file_uploader object.
    Includes a simple size limit for security/performance.
    """
    if uploaded_file is None:
        return None

    try:
        # Check file size (~10 MB limit)
        uploaded_file.seek(0, 2)
        size_mb = uploaded_file.tell() / (1024 * 1024)
        uploaded_file.seek(0)

        if size_mb > 10:
            st.error("Uploaded file is too large (> 10 MB). Please provide a smaller sample.")
            return None

        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"The file could not read the uploaded file as CSV: {e}")
        return None


def load_csv_from_url(url: str, timeout: int = 10) -> Optional[pd.DataFrame]:
    """
    Safely load a CSV file from a URL.

    - Only allows http/https
    - Enforces a size limit and a timeout
    """
    if not url:
        return None

    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        st.error("URL must start with http:// or https://")
        return None

    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            st.error(f"Error fetching URL: HTTP {response.status_code}")
            return None

        # Size check (~10 MB)
        content_size_mb = len(response.content) / (1024 * 1024)
        if content_size_mb > 10:
            st.error("Remote file is too large (> 10 MB). Please provide a smaller sample.")
            return None

        # Use BytesIO for general robustness
        df = pd.read_csv(BytesIO(response.content))
        return df
    except Exception as e:
        st.error(f"Could not download or parse CSV from URL: {e}")
        return None


# ---
# Data preprocessing for anomaly detection
# ---
def preprocess_data(
    df: pd.DataFrame,
    missing_strategy: str,
    scale_numeric: bool,
) -> Tuple[pd.DataFrame, List[str], Optional[StandardScaler]]:
    """
    Apply missing-value handling and optional scaling to numeric features.

    Args:
        df: input dataframe (original data)
        missing_strategy: textual strategy name
        scale_numeric: whether to apply StandardScaler

    Returns:
        processed_df: cleaned dataframe
        numeric_cols: list of numeric column names used for analysis
        scaler: fitted StandardScaler (or None)
    """
    processed_df = df.copy()

    # Identify numeric columns for anomaly detection
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns detected. Anomaly detection will need at least one numeric column.")
        return processed_df, [], None

    # Handle missing values in numeric columns
    if missing_strategy == "Drop rows with missing values":
        processed_df = processed_df.dropna()
    else:
        numeric_data = processed_df[numeric_cols]
        if missing_strategy == "Mean imputation":
            processed_df[numeric_cols] = numeric_data.fillna(numeric_data.mean())
        elif missing_strategy == "Median imputation":
            processed_df[numeric_cols] = numeric_data.fillna(numeric_data.median())
        elif missing_strategy == "Interpolate (numeric only)":
            processed_df[numeric_cols] = numeric_data.interpolate(method="linear")
        else:
            # Fallback: leave as-is
            pass

    # Optional scaling
    scaler = None
    if scale_numeric:
        scaler = StandardScaler()
        processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])

    return processed_df, numeric_cols, scaler


# ---
# Anomaly detection methods
# ---
def run_zscore_anomaly(
    df: pd.DataFrame,
    numeric_cols: List[str],
    threshold: float,
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    """
    Z-score-based anomaly detection.

    - Computes z-scores for each numeric column
    - Uses the maximum absolute z-score across features as the anomaly score
    - Flags rows where this score exceeds a given threshold
    """
    numeric_data = df[numeric_cols]
    means = numeric_data.mean()
    stds = numeric_data.std(ddof=0).replace(0, np.nan)

    z_scores = (numeric_data - means) / stds
    z_scores = z_scores.fillna(0.0)

    scores = z_scores.abs().max(axis=1)
    is_anomaly = scores > threshold
    params = {"threshold_abs_z": threshold}

    return scores, is_anomaly, params


def run_isolation_forest_anomaly(
    df: pd.DataFrame,
    numeric_cols: List[str],
    contamination: float,
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    """
    Anomaly detection in isolation forest.

    - IsolationForest fitted on the numeric columns.
    - Transforms output of decision function to positive anomaly scores.
      (higher - more anomalous)
    """
    X = df[numeric_cols].values
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X)

    raw_scores = model.decision_function(X)
    scores = -pd.Series(raw_scores, index=df.index)
    labels = model.predict(X)
    is_anomaly = labels == -1

    params = {"contamination": contamination}

    return scores, is_anomaly, params


def run_pca_reconstruction_anomaly(
    df: pd.DataFrame,
    numeric_cols: List[str],
    variance_ratio: float,
    percentile_threshold: float,
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    """
    Abnormality in PCA reconstruction error.

    - Fits PCA retaining a proportion of variance
- Fits a PCA retaining proportion of variance
    - Calculates reconstruction error in each row.
    - Flags has an error more than a percentile.
    """
    X = df[numeric_cols].values
    pca = PCA(n_components=variance_ratio, svd_solver="full")
    X_trans = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_trans)

    recon_error = ((X - X_recon) ** 2).mean(axis=1)
    scores = pd.Series(recon_error, index=df.index)

    threshold = np.percentile(scores, percentile_threshold)
    is_anomaly = scores > threshold

    params = {
        "variance_ratio": variance_ratio,
        "percentile_threshold": percentile_threshold,
        "error_threshold": float(threshold),
    }

    return scores, is_anomaly, params


# ---
# Reporting
# ---
def generate_html_report(
    df_with_flags: pd.DataFrame,
    method_name: str,
    method_params: Dict[str, float],
    numeric_cols: List[str],
    time_col: Optional[str],
    source_label: str,
) -> str:
    """
    Generate a simple HTML report summarizing the anomaly detection run.
    """
    total_rows = len(df_with_flags)
    anomaly_rows = df_with_flags["is_anomaly"].sum()
    anomaly_rate = (anomaly_rows / total_rows * 100.0) if total_rows > 0 else 0.0

    params_html = "".join(
        f"<li><b>{k}</b>: {v}</li>" for k, v in method_params.items()
    )

    if anomaly_rows > 0:
        # Show up to 50 anomalies in an HTML table
        anomaly_sample = df_with_flags[df_with_flags["is_anomaly"]].copy()
        anomaly_sample = anomaly_sample.sort_values("anomaly_score", ascending=False)
        anomaly_sample = anomaly_sample.head(50)
        anomaly_table_html = anomaly_sample.to_html(
            index=False,
            escape=True,
            border=1,
            justify="center",
        )
    else:
        anomaly_table_html = "<p>No anomalies were detected with the current configuration.</p>"

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CPS Anomaly Detection Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 1.5rem;
        }}
        h1, h2, h3 {{
            color: #1b4f72;
        }}
        .summary-box {{
            background-color: #f4f6f6;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 0.9rem;
        }}
        th, td {{
            padding: 6px;
            border: 1px solid #ccc;
        }}
        th {{
            background-color: #d6eaf8;
        }}
        .meta {{
            font-size: 0.85rem;
            color: #555;
        }}
    </style>
</head>
<body>
    <h1>CPS Anomaly Detection Report</h1>
    <p class="meta">Source: {source_label}</p>

    <div class="summary-box">
        <h2>Run Summary</h2>
        <p><b>Analysis method:</b> {method_name}</p>
        <p><b>Numeric features used:</b> {", ".join(numeric_cols)}</p>
        <p><b>Total records:</b> {total_rows}</p>
        <p><b>Anomalous records:</b> {anomaly_rows} ({anomaly_rate:.2f}% of data)</p>
        <p><b>Time column:</b> {time_col if time_col else "Index used as time axis"}</p>
        <h3>Method Parameters</h3>
        <ul>
            {params_html}
        </ul>
    </div>

    <h2>Top Anomalies</h2>
    {anomaly_table_html}

    <p class="meta">
        The CPS Anomaly Detection prototype data product was used to generate this report.
        It is aimed at field testing and learning. Do not use this report as
        the only foundation of safety critical decisions devoid of supplementary validation and
        domain expert review.
    </p>
</body>
</html>
"""
    return html


# ---
# Visualization helpers
# ---
def plot_time_series_with_anomalies(
    df_with_flags: pd.DataFrame,
    time_col: Optional[str],
    value_col: str,
):
    """
    Plot a time series (or index-based series) with anomalies highlighted.
    """
    plot_df = df_with_flags.copy()

    if time_col and time_col in plot_df.columns:
        x = plot_df[time_col]
    else:
        x = plot_df.index

    fig = go.Figure()

    # Full series line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=plot_df[value_col],
            mode="lines",
            name="Value",
        )
    )

    # Overlay anomalies as markers
    anomalies = plot_df[plot_df["is_anomaly"]]
    if not anomalies.empty:
        if time_col and time_col in anomalies.columns:
            ax = anomalies[time_col]
        else:
            ax = anomalies.index

        fig.add_trace(
            go.Scatter(
                x=ax,
                y=anomalies[value_col],
                mode="markers",
                name="Anomaly",
                marker=dict(size=8, symbol="circle-open"),
        ))

    fig.update_layout(
        title=f"Time Series: {value_col} (Anomalies Highlighted)",
        xaxis_title=time_col if time_col else "Index",
        yaxis_title=value_col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_anomaly_score_histogram(df_with_flags: pd.DataFrame):
    """
    Plot a histogram of anomaly scores.
    """
    fig = px.histogram(
        df_with_flags,
        x="anomaly_score",
        nbins=40,
    )
    fig.update_layout(
        title="Anomaly Score Distribution",
        xaxis_title="Anomaly score",
        yaxis_title="Count",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---
# Help / Documentation page
# ---
def render_help_page():
    """
    Render the in-app Help / Documentation page.
    """
    st.title("Help & Documentation: CPS Anomaly Detection Data Product")

    st.markdown(
        """
This prototype data product assists in exploring time-series data of a cyber-physical system (CPS).
and identify possible anomalies through a number of unsupervised approaches.

### 1. Recommended workflow

1. **Load data**
 - A CSV file can be uploaded off of your computer.
 - YOu can akso just paste a straight CSV link, e.g. a GitHub raw URL.
 -  - CSV files are the only accepted format of the program.
 - There is a size restriction in this demo, which makes it illegal to use large files..

2. **Explore data**
   - Examine the snapshot and summary statistics of the raw data.
   - Check the correlation table between numerical features.
   - Inspections. Use the scatter plot to check associations in between two numerical variables.

3. **Configure preprocessing**
 - Decision as to the treatment of missing values:
      - Do away with rows that have no values in them.
      -  Imputation of the mean and Imputation of the median.
      -  Interpolate (simply on the numerical values only)
    -  Numeric feature standardization is an optional process that is to be undertaken before an anomaly detection is conducted.
4. **Choose anomaly detection method**
   - **Z-score thresholding**
 - Uses the z-score that is absolute and maximum of all numeric features.
      Higher values greater than the threshold are taken to be abnormalities.
   - **Isolation Forest**
 - Isolating the data items that are not common and which make up the model based on trees.
  - You control the normal percentage of irregularities, or, put another way, contamination.
   - **PCA reconstruction error**
 -  - Uses a representation which has fewer dimensions which retain a ratio of variance which has been chosen.
      Such points with a high error in reconstruction (which is more than a defined percentile) can be regarded as anomalies.

5. **Run analysis & review results**
   - Click **Run analysis** in the sidebar.
   - Inspect:
     - A **time-series plot** with anomalies highlighted.
     - A **histogram of anomaly scores**.
     - A **table of anomalous records**, sorted by severity.

6. **Export outputs**
 - The initial data, (`anomaly_score` + `is_anomaly`) must be downloaded as an amended CSV.
  - Please download an HTML summary report which contains a summary of the settings and the most interesting anomalies.

### 2. Security & responsible use

 - This application is local based and sends no data of yours to any server.
  - The inputs are validated (excluding CSV files, size restriction, HTTP/ HTTPS URLs and network timeouts).
 - The tool is intended to be used both in research and teaching; it is very essential to always combine the findings with professional knowledge about the subject and the safety guidelines before any judgment is made about the operations.

### 3. CPS-specific tips

 - The index will be used as the time axis in case your time column, e.g., timestamp, is not properly parsing. - Be sure that your time column is properly managed.
 - Start with Z-score because it is the fastest way to conduct sanity tests, and then compare it against Isolation Forest and principal component analysis to decide whether or not they agree to the most alarming things.

 - It is suggested that lower contamination and larger thresholds should be used on systems which are very stable, and higher rates of anomaly should be used in the process of exploring noisy data.
 
"""
    )


# ---
# Main app
# ---
def main():
    st.sidebar.title("CPS Anomaly Detection")
    page = st.sidebar.radio("Navigation", ["Data & Analysis", "Help Section"])

    if page == "Help Section":
        render_help_page()
        return

    # -------------------------
    # Data & Analysis page
    # -------------------------
    st.title("CPS Anomaly Detection - Prototype (Final Iteration)")

    st.markdown(
        """
This data product prototype can be used to load CPS time-series data, interactively view it, and run various anomaly.
detection methods, and exporting of cleaned data and HTML summary report.
"""
    )

    # 1) Data input (file or URL)
    st.sidebar.subheader("1. Data Input")

    upload = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    url_input = st.sidebar.text_input("Or enter CSV URL (direct CSV link)")

    df = None
    source_label = ""

    if upload is not None:
        df = load_csv_from_upload(upload)
        source_label = f"Uploaded file: {upload.name}"
        if df is not None:
            st.sidebar.success("File loaded successfully.")
    elif url_input.strip():
        df = load_csv_from_url(url_input.strip())
        source_label = f"URL: {url_input.strip()}"
        if df is not None:
            st.sidebar.success("URL data loaded successfully.")

    if df is None:
        st.info(
            "Upload a CSV file or provide a CSV URL to begin. "
            "Example columns: timestamp, sensor1, sensor2, ..."
        )
        return

    # Optional overall size limit for responsiveness
    max_rows = 200_000
    max_cols = 200
    if df.shape[0] > max_rows or df.shape[1] > max_cols:
        st.error(
            f"Dataset too large for this demo (rows > {max_rows} or columns > {max_cols}). "
            "Please upload a smaller sample."
        )
        return

    # 2) Raw data snapshot
    st.subheader("Raw Data Snapshot")
    st.write(f"Loaded dataset with **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
    st.dataframe(df.head())

    # Try to find and convert a time column if present (like original 'timestamp' logic)
    time_col = None
    candidate_time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if candidate_time_cols:
        time_col_guess = candidate_time_cols[0]
    else:
        time_col_guess = None

    with st.expander("Time column configuration", expanded=False):
        st.markdown(
            "If your data contains a timestamp/datetime column, select it below "
            "so plots can use a proper time axis."
        )
        time_col = st.selectbox(
            "Time column (optional)",
            options=["<None (use index)>"] + df.columns.tolist(),
            index=(df.columns.tolist().index(time_col_guess) + 1) if time_col_guess in df.columns else 0,
        )
        if time_col == "<None (use index)>":
            time_col = None
        else:
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            except Exception:
                st.warning(f"Could not convert column '{time_col}' to datetime. It will be used as-is.")

    # 3) Data exploration (preserves original functionality)
    st.subheader("Data Exploration")

    st.write("Summary statistics:")
    st.write(df.describe(include="all"))

    numeric_original = df.select_dtypes(include=[np.number])

    if numeric_original.shape[1] > 0:
        # Correlation matrix
        corr = numeric_original.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Matrix (Numeric Features)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Scatter plot for two numeric columns
        st.markdown("#### Scatter Plot (Numeric vs Numeric)")
        cols = numeric_original.columns.tolist()
        x_axis = st.selectbox("X axis", options=cols)
        y_axis = st.selectbox(
            "Y axis",
            options=cols,
            index=1 if len(cols) > 1 else 0,
        )
        fig_scatter = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No numeric columns available for correlation matrix or scatter plot.")

    # 4) Preprocessing configuration (used for anomaly engine)
    st.sidebar.subheader("2. Preprocessing")

    missing_strategy = st.sidebar.selectbox(
        "Missing-value handling",
        [
            "Drop rows with missing values",
            "Mean imputation",
            "Median imputation",
            "Interpolate (numeric only)",
        ],
    )

    scale_numeric = st.sidebar.checkbox(
        "Standardize numeric features (recommended for PCA / IsolationForest)",
        value=True,
    )

    # 5) Anomaly method selection
    st.sidebar.subheader("3. Anomaly Method")

    method = st.sidebar.selectbox(
        "Select analysis method",
        ["Z-score thresholding", "Isolation Forest", "PCA reconstruction error"],
    )

    # Method-specific hyperparameters
    if method == "Z-score thresholding":
        z_threshold = st.sidebar.slider(
            "Absolute Z-score threshold",
            min_value=1.5,
            max_value=6.0,
            value=3.0,
            step=0.1,
        )
    elif method == "Isolation Forest":
        contamination = st.sidebar.slider(
            "Expected fraction of anomalies (contamination)",
            min_value=0.001,
            max_value=0.20,
            value=0.02,
            step=0.001,
        )
    else:  # PCA reconstruction error
        variance_ratio = st.sidebar.slider(
            "Variance explained by PCA components",
            min_value=0.50,
            max_value=0.99,
            value=0.95,
            step=0.01,
        )
        percentile_threshold = st.sidebar.slider(
            "Percentile threshold for reconstruction error",
            min_value=80.0,
            max_value=99.9,
            value=95.0,
            step=0.1,
        )

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Run analysis", type="primary")

    # 6) Preprocess data for anomaly detection
    processed_df, numeric_cols, scaler = preprocess_data(df, missing_strategy, scale_numeric)

    if not numeric_cols:
        # Error already displayed by preprocess_data()
        return

    st.subheader("Processed Data Summary (for Anomaly Detection)")
    st.markdown(
        f"Using **{len(numeric_cols)} numeric feature(s)** for anomaly detection: "
        f"`{', '.join(numeric_cols)}`"
    )
    st.dataframe(processed_df.head())

    if not run_button:
        st.info("Select the parameters in the sidebar (Left and press Run analysis in order to calculate anomaly scores.")
        return

    # 7) Run selected anomaly detection method
    try:
        if method == "Z-score thresholding":
            scores, is_anomaly, params = run_zscore_anomaly(processed_df, numeric_cols, z_threshold)
        elif method == "Isolation Forest":
            scores, is_anomaly, params = run_isolation_forest_anomaly(
                processed_df,
                numeric_cols,
                contamination,
            )
        else:  # PCA reconstruction error
            scores, is_anomaly, params = run_pca_reconstruction_anomaly(
                processed_df,
                numeric_cols,
                variance_ratio,
                percentile_threshold,
            )
    except Exception as e:
        st.error(f"An error occurred while running the anomaly detection method: {e}")
        return

    # Attach anomaly scores/flags to processed data
    processed_df_with_flags = processed_df.copy()
    processed_df_with_flags["anomaly_score"] = scores
    processed_df_with_flags["is_anomaly"] = is_anomaly

    # Merge flags back onto original df (for reporting with original values)
    df_with_flags = df.copy()
    df_with_flags["anomaly_score"] = scores.reindex(df_with_flags.index)
    df_with_flags["is_anomaly"] = is_anomaly.reindex(df_with_flags.index)

    # 8) Summary metrics
    n_total = len(df_with_flags)
    n_anom = int(df_with_flags["is_anomaly"].sum())
    rate = (n_anom / n_total * 100.0) if n_total > 0 else 0.0

    st.subheader("Anomaly Detection Summary")
    st.markdown(
        f"""
- **Method:** `{method}`
- **Total records analyzed:** `{n_total}`
- **Anomalous records:** `{n_anom}` (**{rate:.2f}%** of data)
"""
    )

    # 9) Visualizations for anomalies
    st.subheader("Anomaly Visualizations")

    numeric_for_plot = st.selectbox(
        "Select numeric column to plot as time series",
        options=numeric_cols,
        help="This column will be plotted over time with anomalies highlighted.",
    )

    plot_time_series_with_anomalies(df_with_flags, time_col, numeric_for_plot)
    plot_anomaly_score_histogram(df_with_flags)

    # 10) Anomaly table
    st.subheader("Anomaly Table")

    if n_anom > 0:
        max_rows_display = st.slider(
            "Maximum number of anomalous rows to display",
            min_value=10,
            max_value=min(200, n_anom),
            value=min(50, n_anom),
            step=10,
        )
        anomalies_df = df_with_flags[df_with_flags["is_anomaly"]].copy()
        anomalies_df = anomalies_df.sort_values("anomaly_score", ascending=False)
        st.dataframe(anomalies_df.head(max_rows_display))
    else:
        st.info("No anomalies were detected with the current configuration.")

    # 11) Export cleaned CSV and HTML report
    st.subheader("Export Results")

    cleaned_csv = df_with_flags.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download cleaned CSV (with anomaly flags)",
        data=cleaned_csv,
        file_name="cps_cleaned_with_anomalies.csv",
        mime="text/csv",
    )

    html_report = generate_html_report(
        df_with_flags=df_with_flags,
        method_name=method,
        method_params=params,
        numeric_cols=numeric_cols,
        time_col=time_col,
        source_label=source_label,
    )
    st.download_button(
        label="Download HTML summary report",
        data=html_report.encode("utf-8"),
        file_name="cps_anomaly_report.html",
        mime="text/html",
    )


if __name__ == "__main__":
    main()
