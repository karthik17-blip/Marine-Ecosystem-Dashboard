
import os
import glob
import numpy as np
import joblib
import json

import streamlit as st
from datetime import date
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Marine Ecosystem Health Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

VIZ_DIR = os.path.join(
    RESULTS_DIR,
    "phase14_visualizations"
)


# =============================================================================
# CSS
# =============================================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_csv(filename):

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception as e:

        st.error(
            f"Error reading {filename}: {e}"
        )

        return pd.DataFrame()


# =============================================================================
# LOAD DATASETS
# =============================================================================

yearly = load_csv(
    "stage10_yearly_conditions.csv"
)

depth_health = load_csv(
    "stage10_depth_health.csv"
)

spatial_health = load_csv(
    "stage10_spatial_health.csv"
)

anomalies_year = load_csv(
    "stage10_anomalies_by_year.csv"
)

mehi = load_csv(
    "stage9_mehi_region_year.csv"
)

cluster_profile = load_csv(
    "phase11_oceanographic_cluster_profiles.csv"
)

cluster_interpretation = load_csv(
    "phase12_cluster_interpretation.csv"
)

cluster_anomaly = load_csv(
    "phase12_cluster_anomaly_analysis.csv"
)

cluster_year = load_csv(
    "phase12_cluster_year_distribution.csv"
)

cluster_depth = load_csv(
    "phase12_cluster_depth_distribution.csv"
)

cluster_integrated = load_csv(
    "phase13_cluster_integrated_summary.csv"
)

cluster_year_integrated = load_csv(
    "phase13_cluster_year_analysis.csv"
)

cluster_depth_integrated = load_csv(
    "phase13_cluster_depth_analysis.csv"
)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🌊 Marine Ecosystem")

st.sidebar.markdown(
    "### Dashboard Navigation"
)

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Overview",
        "🌡️ Ocean Conditions",
        "⚠️ Anomaly Analysis",
        "🤖 ML Oceanographic Regimes",
        "🗺️ Geographic Analysis",
        "💚 MEHI & Ecosystem Health",
        "📊 Data Explorer",

        "🔮 Suitability Predictor",
        "🤖 Suitability Model Comparison",
        "🧠 Suitability Explainability",
        "🌍 Geographic Suitability",
        "📈 Year-wise Suitability",
        "🌊 Depth-wise Suitability",
        "🔗 Suitability ↔ MEHI",
        "📥 Suitability Download Center",

        "📚 Methodology"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Distributed Oceanographic Data Analytics "
    "for Marine Ecosystem Health Assessment"
)


# =============================================================================

# =============================================================================

# =============================================================================
# PHASE 16C — SUITABILITY MODELS
# =============================================================================

SUITABILITY_MODEL_DIR = os.path.join(
    BASE_DIR,
    "results",
    "phase16c_models"
)

SUITABILITY_METADATA_PATH = os.path.join(
    SUITABILITY_MODEL_DIR,
    "phase16c_model_metadata.json"
)


@st.cache_resource
def load_suitability_models():

    with open(
        SUITABILITY_METADATA_PATH,
        "r"
    ) as f:

        metadata = json.load(f)

    model_files = {
        "Random Forest":
            "random_forest_model.pkl",

        "XGBoost":
            "xgboost_model.pkl",

        "LightGBM":
            "lightgbm_model.pkl"
    }

    loaded_models = {}

    for model_name, filename in model_files.items():

        model_path = os.path.join(
            SUITABILITY_MODEL_DIR,
            filename
        )

        loaded_models[
            model_name
        ] = joblib.load(
            model_path
        )

    return loaded_models, metadata


# PAGE 1 — OVERVIEW
# =============================================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">'
        '🌊 Marine Ecosystem Health Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Distributed Oceanographic Data Analytics for '
        'Marine Ecosystem Health Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # KPIs

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Raw Observations",
        "5,898,956"
    )

    c2.metric(
        "Clean Observations",
        "5,898,883"
    )

    c3.metric(
        "Feature Observations",
        "5,898,883"
    )

    c4.metric(
        "Anomalous",
        "74,968"
    )

    c5.metric(
        "ML Regimes",
        "5"
    )

    c6.metric(
        "MEHI Regions",
        "328"
    )

    st.markdown("---")

    st.subheader("Project Overview")

    st.write(
        "This system analyzes large-scale oceanographic CTD "
        "observations using distributed storage and processing. "
        "The pipeline combines HDFS, PySpark, anomaly detection, "
        "Spark ML, PCA, K-Means, MEHI and interactive visualization."
    )

    if not yearly.empty:

        available = [
            c
            for c in [
                "average_temperature",
                "average_salinity",
                "average_oxygen"
            ]
            if c in yearly.columns
        ]

        if available:

            st.subheader(
                "Oceanographic Conditions Across Years"
            )

            fig = px.line(
                yearly,
                x="year",
                y=available,
                markers=True,
                title="Yearly Oceanographic Conditions"
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =============================================================================
# PAGE 2 — OCEAN CONDITIONS
# =============================================================================

elif page == "🌡️ Ocean Conditions":

    st.title(
        "🌡️ Oceanographic Conditions"
    )

    if yearly.empty:

        st.warning(
            "Yearly dataset unavailable."
        )

    else:

        if "average_temperature" in yearly.columns:

            st.subheader(
                "Average Temperature"
            )

            fig = px.line(
                yearly,
                x="year",
                y="average_temperature",
                markers=True,
                title="Average Temperature by Year"
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if (
            "average_salinity" in yearly.columns
            and
            "average_oxygen" in yearly.columns
        ):

            st.subheader(
                "Salinity and Dissolved Oxygen"
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=yearly["year"],
                    y=yearly["average_salinity"],
                    mode="lines+markers",
                    name="Salinity"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=yearly["year"],
                    y=yearly["average_oxygen"],
                    mode="lines+markers",
                    name="Oxygen",
                    yaxis="y2"
                )
            )

            fig.update_layout(
                template="plotly_white",
                title="Salinity and Dissolved Oxygen",
                yaxis=dict(
                    title="Salinity"
                ),
                yaxis2=dict(
                    title="Dissolved Oxygen",
                    overlaying="y",
                    side="right"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    if not depth_health.empty:

        st.subheader(
            "Conditions by Depth Zone"
        )

        if "depth_zone" in depth_health.columns:

            numeric_cols = [
                c
                for c in [
                    "average_temperature",
                    "average_salinity",
                    "average_oxygen"
                ]
                if c in depth_health.columns
            ]

            if numeric_cols:

                depth_long = depth_health.melt(
                    id_vars=["depth_zone"],
                    value_vars=numeric_cols,
                    var_name="parameter",
                    value_name="value"
                )

                fig = px.bar(
                    depth_long,
                    x="depth_zone",
                    y="value",
                    color="parameter",
                    barmode="group",
                    title="Ocean Conditions by Depth Zone"
                )

                fig.update_layout(
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# =============================================================================
# PAGE 3 — ANOMALY ANALYSIS
# =============================================================================

elif page == "⚠️ Anomaly Analysis":

    st.title(
        "⚠️ Environmental Anomaly Analysis"
    )

    st.info(
        "An anomaly is an observation whose standardized "
        "environmental variables cross the project's "
        "|z| ≥ 3 threshold. This is an analytical indicator."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Anomalous Observations",
        "74,968"
    )

    c2.metric(
        "Overall Anomaly Rate",
        "1.27%"
    )

    c3.metric(
        "Threshold",
        "|z| ≥ 3"
    )

    if not anomalies_year.empty:

        st.subheader(
            "Anomalies by Year"
        )

        year_col = (
            "year"
            if "year" in anomalies_year.columns
            else anomalies_year.columns[0]
        )

        percentage_col = None

        for col in anomalies_year.columns:

            if "percentage" in col.lower():

                percentage_col = col
                break

        if percentage_col:

            fig = px.line(
                anomalies_year,
                x=year_col,
                y=percentage_col,
                markers=True,
                title="Anomaly Percentage by Year"
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.dataframe(
            anomalies_year,
            use_container_width=True
        )

    if not cluster_anomaly.empty:

        st.subheader(
            "Anomalies Across ML Regimes"
        )

        if "anomaly_percentage" in cluster_anomaly.columns:

            fig = px.bar(
                cluster_anomaly,
                x="cluster",
                y="anomaly_percentage",
                text="anomaly_percentage",
                title="Anomaly Rate by ML Cluster"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.dataframe(
            cluster_anomaly,
            use_container_width=True
        )


# =============================================================================
# PAGE 4 — ML REGIMES
# =============================================================================

elif page == "🤖 ML Oceanographic Regimes":

    st.title(
        "🤖 ML Oceanographic Regimes"
    )

    st.write(
        "PCA and K-Means clustering were used to discover "
        "groups of observations with similar oceanographic "
        "characteristics."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "ML Clusters",
        "5"
    )

    c2.metric(
        "PCA Variance Explained",
        "58.31%"
    )

    c3.metric(
        "Silhouette Score",
        "0.5687"
    )

    if not cluster_profile.empty:

        st.subheader(
            "Cluster Distribution"
        )

        if (
            "cluster" in cluster_profile.columns
            and
            "observations" in cluster_profile.columns
        ):

            fig = px.pie(
                cluster_profile,
                names="cluster",
                values="observations",
                hole=0.35,
                title="Distribution of ML Oceanographic Regimes"
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    if not cluster_interpretation.empty:

        st.subheader(
            "Oceanographic Cluster Profiles"
        )

        display_cols = [
            c
            for c in [
                "cluster",
                "observations",
                "percentage",
                "environmental_pattern",
                "dominant_depth_zone",
                "mean_depth",
                "mean_temperature",
                "mean_salinity",
                "mean_oxygen",
                "mean_environmental_anomaly",
                "total_anomalous",
                "anomaly_percentage",
                "dominant_year"
            ]
            if c in cluster_interpretation.columns
        ]

        if display_cols:

            st.dataframe(
                cluster_interpretation[display_cols],
                use_container_width=True
            )

    if not cluster_profile.empty:

        parameter_cols = [
            c
            for c in [
                "mean_depth",
                "mean_temperature",
                "mean_salinity",
                "mean_oxygen"
            ]
            if c in cluster_profile.columns
        ]

        if parameter_cols:

            st.subheader(
                "Environmental Characteristics"
            )

            cluster_long = cluster_profile.melt(
                id_vars=["cluster"],
                value_vars=parameter_cols,
                var_name="parameter",
                value_name="value"
            )

            fig = px.bar(
                cluster_long,
                x="cluster",
                y="value",
                color="parameter",
                barmode="group",
                title="Environmental Characteristics by ML Cluster"
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =============================================================================
# PAGE 5 — GEOGRAPHIC ANALYSIS
# =============================================================================

elif page == "🗺️ Geographic Analysis":

    st.title(
        "🗺️ Geographic Distribution"
    )

    geo_file = os.path.join(
        VIZ_DIR,
        "10_geographic_ml_clusters.html"
    )

    if os.path.exists(geo_file):

        with open(
            geo_file,
            "r",
            encoding="utf-8"
        ) as f:

            html_content = f.read()

        components.html(
            html_content,
            height=700,
            scrolling=True
        )

    else:

        st.warning(
            "Geographic visualization not found."
        )

    if not spatial_health.empty:

        st.subheader(
            "Regional Ecosystem Health"
        )

        st.dataframe(
            spatial_health,
            use_container_width=True
        )


# =============================================================================
# PAGE 6 — MEHI
# =============================================================================

elif page == "💚 MEHI & Ecosystem Health":

    st.title(
        "💚 Marine Ecosystem Health Index"
    )

    st.write(
        "MEHI summarizes the project's environmental health "
        "indicators across region-year combinations."
    )

    if not mehi.empty:

        st.subheader(
            "MEHI Region-Year Analysis"
        )

        st.dataframe(
            mehi,
            use_container_width=True
        )

        score_col = None
        year_col = None

        for col in mehi.columns:

            if "mehi" in col.lower():

                score_col = col
                break

        for col in mehi.columns:

            if col.lower() == "year":

                year_col = col
                break

        if score_col and year_col:

            yearly_mehi = (
                mehi
                .groupby(year_col)[score_col]
                .mean()
                .reset_index()
            )

            fig = px.line(
                yearly_mehi,
                x=year_col,
                y=score_col,
                markers=True,
                title="Average MEHI by Year"
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "MEHI dataset not found."
        )


# =============================================================================
# PAGE 7 — DATA EXPLORER
# =============================================================================

elif page == "📊 Data Explorer":

    st.title(
        "📊 Data Explorer"
    )

    datasets = {
        "Yearly Conditions": yearly,
        "Depth Health": depth_health,
        "Spatial Health": spatial_health,
        "MEHI": mehi,
        "ML Cluster Profiles": cluster_profile,
        "ML Cluster Anomalies": cluster_anomaly,
        "ML Cluster-Year": cluster_year,
        "ML Cluster-Depth": cluster_depth,
        "Integrated Cluster Summary": cluster_integrated,
        "Integrated Cluster-Year": cluster_year_integrated,
        "Integrated Cluster-Depth": cluster_depth_integrated
    }

    dataset_name = st.selectbox(
        "Select dataset",
        list(datasets.keys())
    )

    selected = datasets[dataset_name]

    if selected.empty:

        st.warning(
            "Dataset unavailable."
        )

    else:

        st.write(
            f"Rows: {len(selected):,} | "
            f"Columns: {len(selected.columns)}"
        )

        st.dataframe(
            selected,
            use_container_width=True,
            height=500
        )

        st.download_button(
            "⬇️ Download CSV",
            selected.to_csv(index=False),
            file_name=(
                dataset_name
                .lower()
                .replace(" ", "_")
                + ".csv"
            ),
            mime="text/csv"
        )


# =============================================================================

# =============================================================================
# PAGE — MARINE ENVIRONMENT SUITABILITY PREDICTOR
# =============================================================================

elif page == "🔮 Suitability Predictor":

    st.title(
        "🔮 Marine Environment Suitability Predictor"
    )

    st.markdown(
        """
        Enter oceanographic conditions below and the trained ML models
        will classify the historical environmental suitability of the
        observation.
        """
    )

    st.info(
        """
        **Important:** This is a project-defined environmental suitability
        prediction based on historical standardized anomaly patterns.
        It is not a biological diagnosis or ecological certification.
        """
    )

    # ---------------------------------------------------------------------
    # Load models
    # ---------------------------------------------------------------------

    try:

        suitability_models, suitability_metadata = (
            load_suitability_models()
        )

    except Exception as e:

        st.error(
            f"Unable to load suitability models: {e}"
        )

        st.stop()

    if len(suitability_models) < 3:

        st.error(
            "Fewer than three suitability models were loaded."
        )

        st.stop()

    feature_columns = suitability_metadata[
        "features"
    ]

    class_mapping = {
        0: "Not Suitable",
        1: "Moderate",
        2: "Suitable"
    }

    best_model_name = suitability_metadata[
        "selected_model"
    ]

    # ---------------------------------------------------------------------
    # Model information
    # ---------------------------------------------------------------------

    st.subheader(
        "🤖 Models Used"
    )

    model_cols = st.columns(3)

    for idx, model_name in enumerate(
        suitability_models.keys()
    ):

        with model_cols[idx]:

            st.success(
                model_name
            )

            if model_name == best_model_name:

                st.caption(
                    "⭐ Selected validation model"
                )

            else:

                st.caption(
                    "Comparison model"
                )

    # ---------------------------------------------------------------------
    # Input section
    # ---------------------------------------------------------------------

    st.subheader(
        "🌊 Enter Oceanographic Conditions"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        latitude_input = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=10.0,
            step=0.1
        )

        longitude_input = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=75.0,
            step=0.1
        )

        depth_input = st.number_input(
            "Depth / Pressure (dbar)",
            min_value=0.0,
            max_value=7000.0,
            value=500.0,
            step=10.0
        )

    with col2:

        temperature_input = st.number_input(
            "Temperature (°C)",
            min_value=-5.0,
            max_value=40.0,
            value=15.0,
            step=0.1
        )

        salinity_input = st.number_input(
            "Salinity (PSU)",
            min_value=0.0,
            max_value=45.0,
            value=35.0,
            step=0.1
        )

        oxygen_input = st.number_input(
            "Dissolved Oxygen",
            min_value=0.0,
            max_value=600.0,
            value=200.0,
            step=1.0
        )

    with col3:

        date_input = st.date_input(
            "Observation Date",
            value=date.today()
        )

        st.write("")

        predict_button = st.button(
            "🔮 Predict Suitability",
            type="primary",
            use_container_width=True
        )

    # ---------------------------------------------------------------------
    # Prepare prediction row
    # ---------------------------------------------------------------------

    if predict_button:

        prediction_row = pd.DataFrame(
            {
                "latitude": [
                    latitude_input
                ],

                "longitude": [
                    longitude_input
                ],

                "depth_m": [
                    depth_input
                ],

                "ctd_temperature": [
                    temperature_input
                ],

                "ctd_salinity": [
                    salinity_input
                ],

                "ctd_oxygen": [
                    oxygen_input
                ],

                "year": [
                    date_input.year
                ],

                "month": [
                    date_input.month
                ],

                "day_of_year": [
                    date_input.timetuple().tm_yday
                ]
            }
        )

        prediction_row = prediction_row[
            feature_columns
        ]

        # -----------------------------------------------------------------
        # Run all three models
        # -----------------------------------------------------------------

        model_results = []

        predictions = {}

        probabilities = {}

        for model_name, model in (
            suitability_models.items()
        ):

            prediction = int(
                model.predict(
                    prediction_row
                )[0]
            )

            probability = model.predict_proba(
                prediction_row
            )[0]

            predictions[
                model_name
            ] = prediction

            probabilities[
                model_name
            ] = probability

            model_results.append(
                {
                    "Model":
                        model_name,

                    "Prediction":
                        class_mapping[prediction],

                    "Confidence (%)":
                        round(
                            float(
                                np.max(
                                    probability
                                )
                            ) * 100,
                            2
                        ),

                    "Suitable (%)":
                        round(
                            float(
                                probability[2]
                            ) * 100,
                            2
                        ),

                    "Moderate (%)":
                        round(
                            float(
                                probability[1]
                            ) * 100,
                            2
                        ),

                    "Not Suitable (%)":
                        round(
                            float(
                                probability[0]
                            ) * 100,
                            2
                        )
                }
            )

        results_df = pd.DataFrame(
            model_results
        )

        # -----------------------------------------------------------------
        # Final prediction from best validation model
        # -----------------------------------------------------------------

        best_prediction = predictions[
            best_model_name
        ]

        best_probability = probabilities[
            best_model_name
        ]

        best_label = class_mapping[
            best_prediction
        ]

        # Probability-weighted score
        #
        # Suitable = 100
        # Moderate = 50
        # Not Suitable = 0
        #
        ml_suitability_score = (
            best_probability[2] * 100
            +
            best_probability[1] * 50
        )

        confidence = (
            np.max(
                best_probability
            ) * 100
        )

        # -----------------------------------------------------------------
        # Final result
        # -----------------------------------------------------------------

        st.divider()

        st.subheader(
            "🎯 Final Prediction"
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            if best_label == "Suitable":

                st.success(
                    f"### 🟢 {best_label}"
                )

            elif best_label == "Moderate":

                st.warning(
                    f"### 🟡 {best_label}"
                )

            else:

                st.error(
                    f"### 🔴 {best_label}"
                )

            st.caption(
                f"Selected model: {best_model_name}"
            )

        with result_col2:

            st.metric(
                "ML Suitability Score",
                f"{ml_suitability_score:.1f}/100"
            )

        with result_col3:

            st.metric(
                "Model Confidence",
                f"{confidence:.1f}%"
            )

        # -----------------------------------------------------------------
        # Three model comparison
        # -----------------------------------------------------------------

        st.subheader(
            "📊 Three-Model Prediction Comparison"
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------------------------------------------
        # Probability chart
        # -----------------------------------------------------------------

        st.subheader(
            "📈 Prediction Probabilities"
        )

        probability_chart = pd.DataFrame(
            {
                "Class": [
                    "Not Suitable",
                    "Moderate",
                    "Suitable"
                ],

                "Probability (%)":
                    best_probability * 100
            }
        )

        fig = px.bar(
            probability_chart,
            x="Class",
            y="Probability (%)",
            text="Probability (%)",
            title=(
                f"{best_model_name} Prediction "
                "Probability"
            )
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_range=[
                0,
                105
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -----------------------------------------------------------------
        # Input parameter summary
        # -----------------------------------------------------------------

        st.subheader(
            "🌡️ Input Conditions"
        )

        input_display = pd.DataFrame(
            {
                "Parameter": [
                    "Latitude",
                    "Longitude",
                    "Depth / Pressure",
                    "Temperature",
                    "Salinity",
                    "Dissolved Oxygen",
                    "Observation Date"
                ],

                "Value": [
                    f"{latitude_input:.3f}°",
                    f"{longitude_input:.3f}°",
                    f"{depth_input:.1f} dbar",
                    f"{temperature_input:.2f} °C",
                    f"{salinity_input:.2f} PSU",
                    f"{oxygen_input:.2f}",
                    str(date_input)
                ]
            }
        )

        st.dataframe(
            input_display,
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------------------------------------------
        # Parameter range explanation
        # -----------------------------------------------------------------

        st.subheader(
            "🔎 Historical Range Check"
        )

        range_data = []

        parameter_values = {
            "latitude":
                latitude_input,

            "longitude":
                longitude_input,

            "depth_m":
                depth_input,

            "ctd_temperature":
                temperature_input,

            "ctd_salinity":
                salinity_input,

            "ctd_oxygen":
                oxygen_input
        }

        parameter_names = {
            "latitude":
                "Latitude",

            "longitude":
                "Longitude",

            "depth_m":
                "Depth / Pressure",

            "ctd_temperature":
                "Temperature",

            "ctd_salinity":
                "Salinity",

            "ctd_oxygen":
                "Dissolved Oxygen"
        }

        # Metadata may not contain ranges in older runs.
        historical_ranges = (
            suitability_metadata.get(
                "historical_ranges",
                {}
            )
        )

        for feature, value in (
            parameter_values.items()
        ):

            if feature in historical_ranges:

                feature_range = historical_ranges[
                    feature
                ]

                minimum = feature_range.get(
                    "min",
                    np.nan
                )

                maximum = feature_range.get(
                    "max",
                    np.nan
                )

                p05 = feature_range.get(
                    "p05",
                    np.nan
                )

                p95 = feature_range.get(
                    "p95",
                    np.nan
                )

                if (
                    value >= p05
                    and value <= p95
                ):

                    status = (
                        "Within common historical range"
                    )

                elif (
                    value >= minimum
                    and value <= maximum
                ):

                    status = (
                        "Within historical range "
                        "but outside common range"
                    )

                else:

                    status = (
                        "Outside historical range"
                    )

                range_data.append(
                    {
                        "Parameter":
                            parameter_names[feature],

                        "Input":
                            round(value, 3),

                        "Historical Min":
                            round(minimum, 3),

                        "Historical Max":
                            round(maximum, 3),

                        "Common 5–95%":
                            f"{p05:.3f} – {p95:.3f}",

                        "Status":
                            status
                    }
                )

        if range_data:

            st.dataframe(
                pd.DataFrame(
                    range_data
                ),
                use_container_width=True,
                hide_index=True
            )

        # -----------------------------------------------------------------
        # Feature importance
        # -----------------------------------------------------------------

        st.subheader(
            "🧠 Model Feature Importance"
        )

        best_model = suitability_models[
            best_model_name
        ]

        if hasattr(
            best_model,
            "feature_importances_"
        ):

            importance_df = pd.DataFrame(
                {
                    "Feature":
                        feature_columns,

                    "Importance":
                        best_model.feature_importances_
                }
            ).sort_values(
                "Importance",
                ascending=False
            )

            importance_df[
                "Feature"
            ] = importance_df[
                "Feature"
            ].replace(
                {
                    "ctd_temperature":
                        "Temperature",

                    "ctd_salinity":
                        "Salinity",

                    "ctd_oxygen":
                        "Dissolved Oxygen",

                    "depth_m":
                        "Depth / Pressure",

                    "latitude":
                        "Latitude",

                    "longitude":
                        "Longitude",

                    "":
                        "Temperature Gradient",

                    "":
                        "Salinity Gradient",

                    "":
                        "Oxygen Gradient",

                    "day_of_year":
                        "Day of Year"
                }
            )

            fig_imp = px.bar(
                importance_df.head(10),
                x="Importance",
                y="Feature",
                orientation="h",
                title=(
                    f"Top Features — "
                    f"{best_model_name}"
                )
            )

            fig_imp.update_layout(
                yaxis={
                    "categoryorder":
                        "total ascending"
                }
            )

            st.plotly_chart(
                fig_imp,
                use_container_width=True
            )

        # -----------------------------------------------------------------
        # Explanation
        # -----------------------------------------------------------------

        st.subheader(
            "💡 Interpretation"
        )

        st.markdown(
            f"""
            The **{best_model_name}** model classified these conditions
            as **{best_label}** with a confidence of
            **{confidence:.1f}%**.

            The probability-based ML suitability score is
            **{ml_suitability_score:.1f}/100**.

            This score is generated from the model's class probabilities
            and is **not the same as the project's MEHI score**.

            The three-model comparison above allows the prediction to be
            checked against Random Forest, XGBoost and LightGBM rather
            than relying on a single algorithm.
            """
        )

    # =========================================================================
    # BATCH CSV PREDICTION
    # =========================================================================

    st.divider()

    st.subheader(
        "📁 Batch CSV Prediction"
    )

    st.markdown(
        """
        Upload a CSV containing oceanographic observations.

        Supported input names include:

        - `latitude`
        - `longitude`
        - `depth_m` or `pressure`
        - `ctd_temperature` or `temperature`
        - `ctd_salinity` or `salinity`
        - `ctd_oxygen` or `oxygen`
        - `year`
        - `month`
        - `day_of_year`

        A `date`, `time` or `timestamp` column can also be used to
        derive the temporal features.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload oceanographic CSV",
        type=["csv"],
        key="suitability_csv"
    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(
            uploaded_file
        )

        st.write(
            f"Uploaded rows: {len(batch_df):,}"
        )

        lower_columns = {
            str(col).lower(): col
            for col in batch_df.columns
        }

        def find_column(
            candidates
        ):

            for candidate in candidates:

                if candidate.lower() in lower_columns:

                    return lower_columns[
                        candidate.lower()
                    ]

            return None

        lat_col = find_column(
            [
                "latitude"
            ]
        )

        lon_col = find_column(
            [
                "longitude"
            ]
        )

        depth_col = find_column(
            [
                "depth_m",
                "pressure",
                "depth"
            ]
        )

        temp_col = find_column(
            [
                "ctd_temperature",
                "temperature",
                "temp"
            ]
        )

        sal_col = find_column(
            [
                "ctd_salinity",
                "salinity",
                "sal"
            ]
        )

        oxy_col = find_column(
            [
                "ctd_oxygen",
                "oxygen",
                "dissolved_oxygen"
            ]
        )

        year_col = find_column(
            [
                "year"
            ]
        )

        month_col = find_column(
            [
                "month"
            ]
        )

        day_col = find_column(
            [
                "day_of_year"
            ]
        )

        date_col = find_column(
            [
                "date",
                "time",
                "timestamp"
            ]
        )

        missing_columns = []

        required_mapping = {
            "latitude":
                lat_col,

            "longitude":
                lon_col,

            "depth_m / pressure":
                depth_col,

            "temperature":
                temp_col,

            "salinity":
                sal_col,

            "oxygen":
                oxy_col
        }

        for name, column in (
            required_mapping.items()
        ):

            if column is None:

                missing_columns.append(
                    name
                )

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(
                    missing_columns
                )
            )

        else:

            # -------------------------------------------------------------
            # Derive temporal features
            # -------------------------------------------------------------

            if date_col is not None:

                parsed_dates = pd.to_datetime(
                    batch_df[date_col],
                    errors="coerce"
                )

                if year_col is None:

                    batch_df["_year"] = (
                        parsed_dates.dt.year
                    )

                    year_col = "_year"

                if month_col is None:

                    batch_df["_month"] = (
                        parsed_dates.dt.month
                    )

                    month_col = "_month"

                if day_col is None:

                    batch_df["_day_of_year"] = (
                        parsed_dates.dt.dayofyear
                    )

                    day_col = "_day_of_year"

            # -------------------------------------------------------------
            # If year/month exist but day_of_year doesn't,
            # use the first day of that month.
            # -------------------------------------------------------------

            if day_col is None:

                if (
                    year_col is not None
                    and month_col is not None
                ):

                    batch_df["_day_of_year"] = (
                        pd.to_datetime(
                            dict(
                                year=pd.to_numeric(
                                    batch_df[
                                        year_col
                                    ],
                                    errors="coerce"
                                ),
                                month=pd.to_numeric(
                                    batch_df[
                                        month_col
                                    ],
                                    errors="coerce"
                                ),
                                day=1
                            ),
                            errors="coerce"
                        ).dt.dayofyear
                    )

                    day_col = "_day_of_year"

            temporal_missing = []

            if year_col is None:

                temporal_missing.append(
                    "year/date"
                )

            if month_col is None:

                temporal_missing.append(
                    "month/date"
                )

            if day_col is None:

                temporal_missing.append(
                    "day_of_year/date"
                )

            if temporal_missing:

                st.error(
                    "Unable to derive temporal "
                    "features: "
                    + ", ".join(
                        temporal_missing
                    )
                )

            else:

                batch_features = pd.DataFrame(
                    {
                        "latitude":
                            pd.to_numeric(
                                batch_df[
                                    lat_col
                                ],
                                errors="coerce"
                            ),

                        "longitude":
                            pd.to_numeric(
                                batch_df[
                                    lon_col
                                ],
                                errors="coerce"
                            ),

                        "depth_m":
                            pd.to_numeric(
                                batch_df[
                                    depth_col
                                ],
                                errors="coerce"
                            ),

                        "ctd_temperature":
                            pd.to_numeric(
                                batch_df[
                                    temp_col
                                ],
                                errors="coerce"
                            ),

                        "ctd_salinity":
                            pd.to_numeric(
                                batch_df[
                                    sal_col
                                ],
                                errors="coerce"
                            ),

                        "ctd_oxygen":
                            pd.to_numeric(
                                batch_df[
                                    oxy_col
                                ],
                                errors="coerce"
                            ),

                        "year":
                            pd.to_numeric(
                                batch_df[
                                    year_col
                                ],
                                errors="coerce"
                            ),

                        "month":
                            pd.to_numeric(
                                batch_df[
                                    month_col
                                ],
                                errors="coerce"
                            ),

                        "day_of_year":
                            pd.to_numeric(
                                batch_df[
                                    day_col
                                ],
                                errors="coerce"
                            )
                    }
                )

                # ---------------------------------------------------------
                # Match training feature order
                # ---------------------------------------------------------

                batch_features = batch_features[
                    feature_columns
                ]

                valid_mask = (
                    batch_features
                    .notna()
                    .all(axis=1)
                )

                valid_batch = batch_features[
                    valid_mask
                ]

                if len(valid_batch) == 0:

                    st.error(
                        "No valid rows were available "
                        "for prediction."
                    )

                else:

                    # -----------------------------------------------------
                    # Best model predictions
                    # -----------------------------------------------------

                    best_model = suitability_models[
                        best_model_name
                    ]

                    batch_predictions = (
                        best_model.predict(
                            valid_batch
                        )
                    )

                    batch_probabilities = (
                        best_model.predict_proba(
                            valid_batch
                        )
                    )

                    output_df = batch_df.loc[
                        valid_mask
                    ].copy()

                    output_df[
                        "predicted_suitability"
                    ] = [
                        class_mapping[
                            int(p)
                        ]
                        for p in batch_predictions
                    ]

                    output_df[
                        "ml_suitability_score"
                    ] = (
                        batch_probabilities[:, 2]
                        * 100
                        +
                        batch_probabilities[:, 1]
                        * 50
                    ).round(2)

                    output_df[
                        "prediction_confidence"
                    ] = (
                        np.max(
                            batch_probabilities,
                            axis=1
                        )
                        * 100
                    ).round(2)

                    output_df[
                        "prob_not_suitable"
                    ] = (
                        batch_probabilities[:, 0]
                        * 100
                    ).round(2)

                    output_df[
                        "prob_moderate"
                    ] = (
                        batch_probabilities[:, 1]
                        * 100
                    ).round(2)

                    output_df[
                        "prob_suitable"
                    ] = (
                        batch_probabilities[:, 2]
                        * 100
                    ).round(2)

                    st.success(
                        f"Predicted {len(output_df):,} rows"
                    )

                    st.dataframe(
                        output_df.head(100),
                        use_container_width=True
                    )

                    csv_output = (
                        output_df
                        .to_csv(
                            index=False
                        )
                        .encode("utf-8")
                    )

                    st.download_button(
                        "⬇️ Download Predictions",
                        data=csv_output,
                        file_name=(
                            "marine_suitability_predictions.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True
                    )



# PAGE 8 — METHODOLOGY
# =============================================================================

elif page == "📚 Methodology":

    st.title(
        "📚 Project Methodology"
    )

    st.header(
        "1. Raw Data"
    )

    st.write(
        "Six yearly CTD datasets from 2020–2025 "
        "were used as raw oceanographic observations."
    )

    st.header(
        "2. Distributed Storage"
    )

    st.write(
        "Hadoop HDFS was used for distributed storage."
    )

    st.header(
        "3. Distributed Processing"
    )

    st.write(
        "Apache Spark and PySpark were used for "
        "data loading, cleaning, feature engineering "
        "and large-scale analysis."
    )

    st.header(
        "4. Feature Engineering"
    )

    st.write(
        "Temporal features, spatial grid identifiers, "
        "depth zones, vertical gradients and standardized "
        "environmental anomalies were generated."
    )

    st.header(
        "5. Anomaly Detection"
    )

    st.write(
        "Standardized temperature, salinity and oxygen "
        "anomalies were evaluated using the project's "
        "|z| ≥ 3 threshold."
    )

    st.header(
        "6. Machine Learning"
    )

    st.write(
        "PCA was used for dimensionality reduction and "
        "K-Means was used to discover five oceanographic "
        "regimes. The first three principal components "
        "explained 58.31% of variance."
    )

    st.header(
        "7. Marine Ecosystem Health"
    )

    st.write(
        "Marine health indicators and MEHI were generated "
        "from processed oceanographic observations."
    )

    st.header(
        "8. Visualization"
    )

    st.write(
        "Plotly and Streamlit provide interactive exploration "
        "of temporal, depth-wise, geographic, anomaly and ML results."
    )

    st.header(
        "Complete Pipeline"
    )

    pipeline = """
Raw CTD Data
      ↓
HDFS
      ↓
PySpark
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Anomaly Detection
      ↓
Marine Health Indicators
      ↓
MEHI
      ↓
PCA
      ↓
K-Means
      ↓
Integrated Analysis
      ↓
Plotly
      ↓
Streamlit Dashboard
"""

    st.code(
        pipeline,
        language="text"
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

st.caption(
    "Distributed Oceanographic Data Analytics for "
    "Marine Ecosystem Health Assessment | "
    "HDFS + PySpark + Spark ML + Plotly + Streamlit"
)




# =============================================================================
# PHASE 16J — ADVANCED SUITABILITY + MEHI DASHBOARD
# =============================================================================

import os
import json
import pickle
import pandas as pd
import numpy as np
import streamlit as st


# -----------------------------------------------------------------------------
# PHASE 16J PATHS
# -----------------------------------------------------------------------------

PHASE16J_PROJECT_DIR = (
    "BASE_DIR"
)

PHASE16J_RESULTS_DIR = os.path.join(
    PHASE16J_PROJECT_DIR,
    "results"
)

PHASE16J_MASTER_DIR = os.path.join(
    PHASE16J_RESULTS_DIR,
    "phase16e_17"
)

PHASE16J_G_DIR = os.path.join(
    PHASE16J_MASTER_DIR,
    "16G_geographic"
)

PHASE16J_H_DIR = os.path.join(
    PHASE16J_MASTER_DIR,
    "16H_batch"
)

PHASE16J_I_DIR = os.path.join(
    PHASE16J_MASTER_DIR,
    "16I_mehi"
)

PHASE16J_MODEL_DIR = os.path.join(
    PHASE16J_RESULTS_DIR,
    "phase16c_models"
)


# -----------------------------------------------------------------------------
# LOAD PHASE 16C MODELS
# -----------------------------------------------------------------------------

@st.cache_resource
def load_phase16j_models():

    models = {}

    model_names = {
        "Random Forest":
            "random_forest_model.pkl",

        "XGBoost":
            "xgboost_model.pkl",

        "LightGBM":
            "lightgbm_model.pkl"
    }

    for name, filename in model_names.items():

        path = os.path.join(
            PHASE16J_MODEL_DIR,
            filename
        )

        if os.path.exists(path):

            with open(
                path,
                "rb"
            ) as f:

                models[name] = pickle.load(f)

    return models


@st.cache_data
def load_phase16j_metadata():

    path = os.path.join(
        PHASE16J_MODEL_DIR,
        "phase16c_model_metadata.json"
    )

    if not os.path.exists(path):

        return {}

    try:

        with open(
            path,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


# -----------------------------------------------------------------------------
# LOAD PHASE 16G–16I DATA
# -----------------------------------------------------------------------------

@st.cache_data
def load_phase16j_predictions():

    path = os.path.join(
        PHASE16J_G_DIR,
        "geographic_suitability_predictions.csv"
    )

    if not os.path.exists(path):

        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_phase16j_year():

    path = os.path.join(
        PHASE16J_H_DIR,
        "suitability_by_year.csv"
    )

    if not os.path.exists(path):

        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_phase16j_depth():

    path = os.path.join(
        PHASE16J_H_DIR,
        "suitability_by_depth_zone.csv"
    )

    if not os.path.exists(path):

        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_phase16j_overall():

    path = os.path.join(
        PHASE16J_H_DIR,
        "overall_suitability_distribution.csv"
    )

    if not os.path.exists(path):

        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_phase16j_mehi():

    path = os.path.join(
        PHASE16J_I_DIR,
        "ml_suitability_mehi_region_year.csv"
    )

    if not os.path.exists(path):

        return pd.DataFrame()

    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# PREDICTOR
# -----------------------------------------------------------------------------

def render_phase16j_predictor():

    st.header(
        "🔮 Marine Environment Suitability Predictor"
    )

    st.write(
        """
        Enter oceanographic conditions to obtain suitability
        predictions from the Phase 16C machine-learning models.
        """
    )

    models = load_phase16j_models()

    if not models:

        st.error(
            "Phase 16C models could not be loaded."
        )

        return

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=10.0,
            step=0.1
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=75.0,
            step=0.1
        )

        depth = st.number_input(
            "Depth / Pressure (dbar)",
            min_value=0.0,
            max_value=7000.0,
            value=500.0,
            step=10.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-5.0,
            max_value=40.0,
            value=10.0,
            step=0.1
        )

    with col2:

        salinity = st.number_input(
            "Salinity",
            min_value=0.0,
            max_value=45.0,
            value=35.0,
            step=0.01
        )

        oxygen = st.number_input(
            "Dissolved Oxygen",
            min_value=0.0,
            max_value=600.0,
            value=200.0,
            step=1.0
        )

        selected_date = st.date_input(
            "Observation Date"
        )

    input_df = pd.DataFrame(
        {
            "latitude": [latitude],
            "longitude": [longitude],
            "depth_m": [depth],
            "ctd_temperature": [temperature],
            "ctd_salinity": [salinity],
            "ctd_oxygen": [oxygen],
            "year": [selected_date.year],
            "month": [selected_date.month],
            "day_of_year": [
                selected_date.timetuple().tm_yday
            ]
        }
    )

    st.subheader(
        "Input Conditions"
    )

    st.dataframe(
        input_df,
        use_container_width=True
    )

    if st.button(
        "🚀 Predict Suitability",
        type="primary"
    ):

        st.subheader(
            "Model Predictions"
        )

        results = []

        class_names = {
            0: "Not Suitable",
            1: "Moderate",
            2: "Suitable"
        }

        for model_name, model in models.items():

            try:

                feature_order = list(
                    getattr(
                        model,
                        "feature_names_in_",
                        input_df.columns
                    )
                )

                prediction_df = input_df[
                    feature_order
                ]

                prediction = int(
                    model.predict(
                        prediction_df
                    )[0]
                )

                probabilities = model.predict_proba(
                    prediction_df
                )[0]

                confidence = float(
                    np.max(probabilities)
                )

                results.append(
                    {
                        "Model":
                            model_name,

                        "Prediction":
                            class_names.get(
                                prediction,
                                str(prediction)
                            ),

                        "Confidence":
                            confidence,

                        "Not Suitable":
                            probabilities[0],

                        "Moderate":
                            probabilities[1],

                        "Suitable":
                            probabilities[2]
                    }
                )

            except Exception as e:

                st.error(
                    f"{model_name}: {e}"
                )

        if results:

            result_df = pd.DataFrame(
                results
            )

            display_df = result_df.copy()

            for column in [
                "Confidence",
                "Not Suitable",
                "Moderate",
                "Suitable"
            ]:

                display_df[column] = (
                    display_df[column] * 100
                ).round(2)

            st.dataframe(
                display_df,
                use_container_width=True
            )

            st.subheader(
                "Model Agreement"
            )

            prediction_counts = (
                result_df["Prediction"]
                .value_counts()
            )

            if len(prediction_counts) == 1:

                st.success(
                    "All three models produced "
                    "the same suitability class."
                )

            else:

                st.warning(
                    "The models produced different "
                    "suitability classes. Review the "
                    "individual probabilities."
                )


# -----------------------------------------------------------------------------
# MODEL COMPARISON
# -----------------------------------------------------------------------------

def render_phase16j_model_comparison():

    st.header(
        "🤖 Suitability Model Comparison"
    )

    comparison = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "XGBoost",
                "LightGBM"
            ],
            "Profile Validation Accuracy": [
                0.998887,
                0.994708,
                0.994504
            ],
            "Profile Validation F1": [
                0.998887,
                0.994708,
                0.994505
            ],
            "ROC-AUC": [
                0.999995,
                0.999932,
                0.999897
            ]
        }
    )

    display_df = comparison.copy()

    for column in [
        "Profile Validation Accuracy",
        "Profile Validation F1",
        "ROC-AUC"
    ]:

        display_df[column] = (
            display_df[column] * 100
        ).round(3)

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.caption(
        """
        These metrics come from profile-based validation.
        Because the suitability target is itself defined from
        environmental anomaly thresholds, these metrics should
        not be interpreted as independent ecological truth
        validation.
        """
    )


# -----------------------------------------------------------------------------
# GEOGRAPHIC MAP
# -----------------------------------------------------------------------------

def render_phase16j_geographic():

    st.header(
        "🌍 Geographic Suitability Analysis"
    )

    map_path = os.path.join(
        PHASE16J_G_DIR,
        "marine_suitability_map.html"
    )

    prediction_df = load_phase16j_predictions()

    if prediction_df.empty:

        st.warning(
            "Geographic prediction data unavailable."
        )

        return

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Observations",
            f"{len(prediction_df):,}"
        )

    with col2:

        st.metric(
            "Mean Confidence",
            f"{prediction_df['confidence'].mean():.3f}"
        )

    with col3:

        st.metric(
            "Mean Suitability Score",
            f"{prediction_df['suitability_score'].mean():.2f}"
        )

    st.subheader(
        "Suitability Distribution"
    )

    distribution = (
        prediction_df[
            "predicted_suitability"
        ]
        .value_counts()
        .rename_axis(
            "Suitability"
        )
        .reset_index(
            name="Observations"
        )
    )

    st.dataframe(
        distribution,
        use_container_width=True
    )

    if os.path.exists(map_path):

        with open(
            map_path,
            "r",
            encoding="utf-8"
        ) as f:

            map_html = f.read()

        st.components.v1.html(
            map_html,
            height=700,
            scrolling=True
        )

    else:

        st.warning(
            "Interactive geographic map not found."
        )


# -----------------------------------------------------------------------------
# BATCH ANALYTICS
# -----------------------------------------------------------------------------

def render_phase16j_batch():

    st.header(
        "📊 Batch Suitability Analytics"
    )

    overall = load_phase16j_overall()
    yearly = load_phase16j_year()
    depth = load_phase16j_depth()

    tab1, tab2, tab3 = st.tabs(
        [
            "Overall",
            "Year",
            "Depth"
        ]
    )

    with tab1:

        if not overall.empty:

            st.dataframe(
                overall,
                use_container_width=True
            )

            import plotly.express as px

            fig = px.bar(
                overall,
                x="suitability",
                y="observations",
                title="Overall Suitability Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with tab2:

        if not yearly.empty:

            st.dataframe(
                yearly,
                use_container_width=True
            )

            fig = px.bar(
                yearly,
                x="year",
                y="percentage",
                color="predicted_suitability",
                barmode="group",
                title="Suitability by Year"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with tab3:

        if not depth.empty:

            st.dataframe(
                depth,
                use_container_width=True
            )

            fig = px.bar(
                depth,
                x="depth_zone",
                y="percentage",
                color="predicted_suitability",
                barmode="group",
                title="Suitability by Depth Zone"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# -----------------------------------------------------------------------------
# MEHI INTEGRATION
# -----------------------------------------------------------------------------

def render_phase16j_mehi():

    st.header(
        "🌊 ML Suitability ↔ MEHI Integration"
    )

    mehi_df = load_phase16j_mehi()

    if mehi_df.empty:

        st.warning(
            "MEHI integration data unavailable."
        )

        return

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Matched Region-Year Records",
            f"{len(mehi_df):,}"
        )

    with col2:

        st.metric(
            "Mean ML Suitability",
            f"{mehi_df['mean_ml_suitability_score'].mean():.2f}"
        )

    with col3:

        st.metric(
            "Mean MEHI",
            f"{mehi_df['MEHI_score'].mean():.2f}"
        )

    correlation = (
        mehi_df[
            [
                "mean_ml_suitability_score",
                "MEHI_score"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    st.metric(
        "Pearson Correlation",
        f"{correlation:.4f}"
    )

    import plotly.express as px

    st.subheader(
        "ML Suitability vs MEHI"
    )

    color_column = (
        "health_category"
        if "health_category" in mehi_df.columns
        else None
    )

    fig = px.scatter(
        mehi_df,
        x="MEHI_score",
        y="mean_ml_suitability_score",
        color=color_column,
        hover_data=[
            "region_id",
            "year",
            "ml_observations"
        ],
        title="ML Suitability Score vs MEHI"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Integrated Region-Year Dataset"
    )

    st.dataframe(
        mehi_df,
        use_container_width=True
    )

    csv_data = mehi_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Download ML + MEHI CSV",
        data=csv_data,
        file_name="ml_suitability_mehi_region_year.csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------------------
# DOWNLOAD CENTER
# -----------------------------------------------------------------------------

def render_phase16j_downloads():

    st.header(
        "📥 Phase 16 Advanced Outputs"
    )

    download_files = {

        "Geographic Predictions":
            os.path.join(
                PHASE16J_G_DIR,
                "geographic_suitability_predictions.csv"
            ),

        "Overall Suitability":
            os.path.join(
                PHASE16J_H_DIR,
                "overall_suitability_distribution.csv"
            ),

        "Suitability by Year":
            os.path.join(
                PHASE16J_H_DIR,
                "suitability_by_year.csv"
            ),

        "Suitability by Depth":
            os.path.join(
                PHASE16J_H_DIR,
                "suitability_by_depth_zone.csv"
            ),

        "ML + MEHI Integration":
            os.path.join(
                PHASE16J_I_DIR,
                "ml_suitability_mehi_region_year.csv"
            )
    }

    for name, path in download_files.items():

        if os.path.exists(path):

            with open(
                path,
                "rb"
            ) as f:

                data = f.read()

            st.download_button(
                f"⬇ {name}",
                data=data,
                file_name=os.path.basename(path),
                mime="text/csv",
                key=f"download_{name}"
            )


# -----------------------------------------------------------------------------
# METHODOLOGY
# -----------------------------------------------------------------------------

def render_phase16j_methodology():

    st.header(
        "📚 Phase 16 Methodology"
    )

    st.markdown(
        """
### Marine Environment Suitability Predictor

The Phase 16 suitability system uses three machine-learning
models:

- Random Forest
- XGBoost
- LightGBM

The predictor uses directly user-enterable environmental
variables:

- Latitude
- Longitude
- Depth / Pressure
- Temperature
- Salinity
- Dissolved Oxygen
- Year
- Month
- Day of Year

### Suitability target

The project-defined target contains three classes:

**Not Suitable**
→ maximum absolute environmental anomaly ≥ 3

**Moderate**
→ maximum absolute environmental anomaly ≥ 1 and < 3

**Suitable**
→ maximum absolute environmental anomaly < 1

Therefore, this is a **project-defined analytical suitability
classification**, not a universally established ecological
standard.

### Validation

Profile-based validation was used so observations belonging
to the same CTD profile were not simultaneously placed in
training and testing groups.

### MEHI integration

ML suitability predictions were aggregated by:

`region_id + year`

and compared with the project's Marine Ecosystem Health Index
(MEHI).

The Phase 16I analysis produced:

- 313 matched region-year groups
- Pearson correlation ≈ 0.7840

This correlation is descriptive and should not be interpreted
as causal evidence or independent ecological validation.

### Data pipeline

Raw CTD data
→ HDFS
→ PySpark cleaning
→ Feature engineering
→ Anomaly detection
→ MEHI
→ ML pattern discovery
→ Suitability prediction
→ Geographic analysis
→ MEHI integration
→ Interactive dashboard
"""
    )


# =============================================================================
# END PHASE 16J MODULE
# =============================================================================




# =============================================================================
# PHASE 16J NAVIGATION
# =============================================================================

if "page" in globals():

    if page == "🔮 Suitability Predictor":

        render_phase16j_predictor()

    elif page == "🤖 Suitability Model Comparison":

        render_phase16j_model_comparison()

    elif page == "🌍 Suitability Geographic Analysis":

        render_phase16j_geographic()

    elif page == "📊 Suitability Batch Analytics":

        render_phase16j_batch()

    elif page == "🌊 ML Suitability + MEHI":

        render_phase16j_mehi()

    elif page == "📥 Advanced Outputs":

        render_phase16j_downloads()

    elif page == "📚 Phase 16 Methodology":

        render_phase16j_methodology()


# =============================================================================
# PHASE 16E–17 — FINAL SUITABILITY DASHBOARD MODULES
# =============================================================================

PHASE16_ROOT = os.path.join(
    RESULTS_DIR,
    "phase16e_17"
)

PHASE16C_ROOT = os.path.join(
    RESULTS_DIR,
    "phase16c_models"
)

PHASE16G_ROOT = os.path.join(
    PHASE16_ROOT,
    "16G_geographic"
)

PHASE16H_ROOT = os.path.join(
    PHASE16_ROOT,
    "16H_batch"
)

PHASE16I_ROOT = os.path.join(
    PHASE16_ROOT,
    "16I_mehi"
)

PHASE16F_ROOT = os.path.join(
    PHASE16_ROOT,
    "16F_explainability"
)


@st.cache_resource
def load_suitability_models():

    result = {}

    files = {
        "Random Forest":
            "random_forest_model.pkl",

        "XGBoost":
            "xgboost_model.pkl",

        "LightGBM":
            "lightgbm_model.pkl"
    }

    for name, filename in files.items():

        path = os.path.join(
            PHASE16C_ROOT,
            filename
        )

        if os.path.exists(path):

            result[name] = joblib.load(path)

    return result


@st.cache_data
def load_suitability_metadata():

    path = os.path.join(
        PHASE16C_ROOT,
        "phase16c_model_metadata.json"
    )

    if os.path.exists(path):

        with open(path, "r") as f:

            return json.load(f)

    return {}


@st.cache_data
def load_phase16_csv(filename, folder):

    path = os.path.join(
        folder,
        filename
    )

    if os.path.exists(path):

        return pd.read_csv(path)

    return pd.DataFrame()


def render_suitability_predictor():

    st.title(
        "🔮 Marine Environment Suitability Predictor"
    )

    st.write(
        """
        Enter environmental conditions to obtain suitability predictions
        from Random Forest, XGBoost and LightGBM.
        """
    )

    models_local = load_suitability_models()

    if not models_local:

        st.error(
            "Suitability models were not found."
        )

        return

    c1, c2, c3 = st.columns(3)

    with c1:

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=10.0
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=75.0
        )

        depth = st.number_input(
            "Depth / Pressure",
            min_value=0.0,
            max_value=7000.0,
            value=500.0
        )

    with c2:

        temperature = st.number_input(
            "Temperature (°C)",
            value=10.0
        )

        salinity = st.number_input(
            "Salinity",
            value=35.0
        )

        oxygen = st.number_input(
            "Dissolved Oxygen",
            value=200.0
        )

    with c3:

        year = st.number_input(
            "Year",
            min_value=2020,
            max_value=2025,
            value=2025
        )

        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=1
        )

        day_of_year = st.number_input(
            "Day of Year",
            min_value=1,
            max_value=366,
            value=1
        )

    input_df = pd.DataFrame(
        [[
            latitude,
            longitude,
            depth,
            temperature,
            salinity,
            oxygen,
            year,
            month,
            day_of_year
        ]],
        columns=FEATURE_COLUMNS
    )

    if st.button(
        "🚀 Predict Suitability",
        use_container_width=True
    ):

        st.subheader(
            "Model Predictions"
        )

        cols = st.columns(
            len(models_local)
        )

        prediction_rows = []

        for i, (name, model) in enumerate(
            models_local.items()
        ):

            prediction = int(
                model.predict(
                    input_df
                )[0]
            )

            probabilities = (
                model.predict_proba(
                    input_df
                )[0]
            )

            confidence = float(
                np.max(probabilities)
            )

            class_name = CLASS_NAMES[
                prediction
            ]

            with cols[i]:

                st.metric(
                    name,
                    class_name
                )

                st.write(
                    f"Confidence: "
                    f"{confidence:.2%}"
                )

            prediction_rows.append({

                "Model": name,

                "Prediction":
                    class_name,

                "Confidence":
                    confidence

            })

        result_df = pd.DataFrame(
            prediction_rows
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            """
            Suitability is a project-defined analytical class derived
            from standardized environmental anomaly thresholds. It is
            not biological ground truth.
            """
        )


def render_model_comparison():

    st.title(
        "🤖 Suitability Model Comparison"
    )

    validation_path = os.path.join(
        PHASE16_ROOT,
        "16E_validation",
        "phase16e_validation_summary.csv"
    )

    df = load_phase16_csv(
        "phase16e_validation_summary.csv",
        os.path.dirname(validation_path)
    )

    if df.empty:

        st.warning(
            "Validation results unavailable."
        )

        return

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    metric_columns = [
        c
        for c in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
        if c in df.columns
    ]

    if metric_columns:

        melted = df.melt(
            id_vars=["model"],
            value_vars=metric_columns,
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            melted,
            x="model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Suitability Model Performance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


def render_explainability():

    st.title(
        "🧠 Suitability Model Explainability"
    )

    path = os.path.join(
        PHASE16F_ROOT,
        "combined_feature_importance.csv"
    )

    if not os.path.exists(path):

        st.warning(
            "Explainability output unavailable."
        )

        return

    df = pd.read_csv(
        path
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    importance_cols = [
        c
        for c in df.columns
        if c != "feature"
    ]

    if importance_cols:

        model_choice = st.selectbox(
            "Select model",
            importance_cols
        )

        plot_df = df[
            ["feature", model_choice]
        ].sort_values(
            model_choice
        )

        fig = px.bar(
            plot_df,
            x=model_choice,
            y="feature",
            orientation="h",
            title=f"{model_choice} Feature Importance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        """
        Feature importance indicates which input variables contribute
        most strongly to the tree-based model's predictions. It does not
        establish causation.
        """
    )


def render_geographic_suitability():

    st.title(
        "🌍 Geographic Suitability"
    )

    path = os.path.join(
        PHASE16G_ROOT,
        "geographic_suitability_predictions.csv"
    )

    if not os.path.exists(path):

        st.warning(
            "Geographic suitability data unavailable."
        )

        return

    df = pd.read_csv(
        path
    )

    st.metric(
        "Geographic Observations",
        f"{len(df):,}"
    )

    if "latitude" in df.columns and \
       "longitude" in df.columns:

        color_col = None

        for candidate in [
            "suitability_score",
            "suitability_class"
        ]:

            if candidate in df.columns:

                color_col = candidate

                break

        if color_col:

            fig = px.scatter_geo(
                df.sample(
                    min(30000, len(df)),
                    random_state=42
                ),
                lat="latitude",
                lon="longitude",
                color=color_col,
                title="Geographic Suitability"
            )

            fig.update_geos(
                showcoastlines=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.dataframe(
        df.head(1000),
        use_container_width=True,
        hide_index=True
    )


def render_year_suitability():

    st.title(
        "📈 Year-wise Suitability"
    )

    path = os.path.join(
        PHASE16H_ROOT,
        "yearly_suitability_summary.csv"
    )

    if not os.path.exists(path):

        path = os.path.join(
            PHASE16H_ROOT,
            "suitability_by_year.csv"
        )

    if not os.path.exists(path):

        st.warning(
            "Year-wise suitability data unavailable."
        )

        return

    df = pd.read_csv(
        path
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    numeric = [
        c
        for c in [
            "mean_suitability_score",
            "suitability_score"
        ]
        if c in df.columns
    ]

    if numeric and "year" in df.columns:

        fig = px.line(
            df,
            x="year",
            y=numeric[0],
            markers=True,
            title="Mean Suitability Score by Year"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


def render_depth_suitability():

    st.title(
        "🌊 Depth-wise Suitability"
    )

    path = os.path.join(
        PHASE16H_ROOT,
        "depth_suitability_summary.csv"
    )

    if not os.path.exists(path):

        path = os.path.join(
            PHASE16H_ROOT,
            "suitability_by_depth_zone.csv"
        )

    if not os.path.exists(path):

        st.warning(
            "Depth-wise suitability data unavailable."
        )

        return

    df = pd.read_csv(
        path
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    numeric = [
        c
        for c in [
            "mean_suitability_score",
            "suitability_score"
        ]
        if c in df.columns
    ]

    if numeric:

        depth_col = None

        for candidate in [
            "depth_zone",
            "zone"
        ]:

            if candidate in df.columns:

                depth_col = candidate

                break

        if depth_col:

            fig = px.bar(
                df,
                x=depth_col,
                y=numeric[0],
                title="Suitability by Depth Zone"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


def render_mehi_comparison():

    st.title(
        "🔗 ML Suitability ↔ MEHI"
    )

    st.write(
        """
        Comparison between the project-defined ML suitability score
        and the project-defined Marine Ecosystem Health Index.
        """
    )

    st.metric(
        "Pearson Correlation",
        "0.7840"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Mean ML Suitability Score",
            "69.95"
        )

    with c2:

        st.metric(
            "Mean MEHI",
            "80.25"
        )

    path = os.path.join(
        PHASE16I_ROOT,
        "ml_suitability_mehi_region_year.csv"
    )

    if os.path.exists(path):

        df = pd.read_csv(
            path
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        x = None
        y = None

        for candidate in [
            "mean_suitability_score",
            "ml_suitability_score",
            "suitability_score"
        ]:

            if candidate in df.columns:

                x = candidate

                break

        for candidate in [
            "mean_mehi",
            "mehi",
            "MEHI"
        ]:

            if candidate in df.columns:

                y = candidate

                break

        if x and y:

            fig = px.scatter(
                df,
                x=x,
                y=y,
                title="ML Suitability vs MEHI"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


def render_download_center():

    st.title(
        "📥 Suitability Download Center"
    )

    folders = {
        "Phase 16E Validation":
            os.path.join(
                PHASE16_ROOT,
                "16E_validation"
            ),

        "Phase 16F Explainability":
            PHASE16F_ROOT,

        "Phase 16G Geographic":
            PHASE16G_ROOT,

        "Phase 16H Batch":
            PHASE16H_ROOT,

        "Phase 16I MEHI":
            PHASE16I_ROOT
    }

    for section, folder in folders.items():

        st.subheader(
            section
        )

        if not os.path.exists(folder):

            st.write(
                "No files available."
            )

            continue

        files = sorted(
            glob.glob(
                os.path.join(
                    folder,
                    "*"
                )
            )
        )

        for file in files:

            if not os.path.isfile(file):
                continue

            with open(
                file,
                "rb"
            ) as f:

                st.download_button(
                    f"⬇️ {os.path.basename(file)}",
                    f.read(),
                    file_name=os.path.basename(file),
                    key=f"download_{section}_{os.path.basename(file)}"
                )


# =============================================================================
# FINAL PAGE ROUTING
# =============================================================================

if page == "🔮 Suitability Predictor":

    render_suitability_predictor()

elif page == "🤖 Suitability Model Comparison":

    render_model_comparison()

elif page == "🧠 Suitability Explainability":

    render_explainability()

elif page == "🌍 Geographic Suitability":

    render_geographic_suitability()

elif page == "📈 Year-wise Suitability":

    render_year_suitability()

elif page == "🌊 Depth-wise Suitability":

    render_depth_suitability()

elif page == "🔗 Suitability ↔ MEHI":

    render_mehi_comparison()

elif page == "📥 Suitability Download Center":

    render_download_center()

