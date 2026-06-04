import io
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import requests

# ============================================================
# App config
# ============================================================
st.set_page_config(
    page_title="University of Oxford Funding Dashboard",
    layout="wide"
)

st.title("University of Oxford Funding and Success Rate Analysis")
st.caption("Interactive dashboard built.")

# ============================================================
# GitHub raw file URLs
# ============================================================
FUNDING_CSV_URL = "https://raw.githubusercontent.com/krubjorn/funding-dashboard/main/projectsearch.csv"
SUCCESS_CSV_URL = "https://raw.githubusercontent.com/krubjorn/funding-dashboard/main/success_rate_data.csv"

# ============================================================
# Column names
# ============================================================
COL_FUNDER = "FundingOrgName"
COL_PROJECT_REF = "ProjectReference"
COL_LEAD_RO = "LeadROName"
COL_DEPARTMENT = "Department"
COL_PROJECT_CATEGORY = "ProjectCategory"
COL_PI_SURNAME = "PISurname"
COL_PI_FIRST = "PIFirstName"
COL_TITLE = "Title"
COL_START = "StartDate"
COL_END = "EndDate"
COL_AWARD = "AwardPounds"
COL_REGION = "Region"
COL_STATUS = "Status"
COL_URL = "GTRProjectUrl"

# ============================================================
# Data loading
# ============================================================
@st.cache_data(show_spinner=True)
def load_csv_from_github(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))

@st.cache_data(show_spinner=True)
def load_data():
    # Funding data
    df_funding_raw = load_csv_from_github(FUNDING_CSV_URL)

    # Filter for University of Oxford
    df_oxford = df_funding_raw[
        df_funding_raw[COL_LEAD_RO].astype(str).str.contains(
            "University of Oxford", case=False, na=False
        )
    ].copy()

    df_oxford[COL_START] = pd.to_datetime(df_oxford[COL_START], errors="coerce", dayfirst=True)
    df_oxford[COL_END] = pd.to_datetime(df_oxford[COL_END], errors="coerce", dayfirst=True)
    df_oxford[COL_AWARD] = pd.to_numeric(df_oxford[COL_AWARD], errors="coerce")
    df_oxford["StartYear"] = df_oxford[COL_START].dt.year

@st.cache_data(show_spinner=True)
def load_data():
    # Funding data
    df_success_raw = load_csv_from_github(SUCCESS_CSV_URL)

    def extract_year(value):
        if pd.isna(value):
            return np.nan
        text = str(value)
        digits = "".join(c for c in text if c.isdigit())
        if len(digits) >= 2:
            return int("20" + digits[:2])
        return np.nan

    df_success_raw["Year"] = df_success_raw["Financial Year"].apply(extract_year)

    status_map = {
        "Approved by funder": "A",
        "Rejected by funder": "U",
        "Submitted to funder": "S"
    }
    df_success_raw["StatusCode"] = df_success_raw["Fund Deci Status Desc"].map(status_map)

    funder_summary = (
        df_success_raw.groupby("Funder", dropna=False)
        .agg(
            Submitted=("Project Ref", "count"),
            Successful=("StatusCode", lambda x: (x == "A").sum())
        )
        .reset_index()
    )
    funder_summary["Success Rate"] = 100 * funder_summary["Successful"] / funder_summary["Submitted"]
    funder_summary = funder_summary.sort_values("Success Rate", ascending=False)

    funder_year = (
        df_success_raw.groupby(["Funder", "Year"], dropna=False)
        .agg(
            Submitted=("Project Ref", "count"),
            Successful=("StatusCode", lambda x: (x == "A").sum())
        )
        .reset_index()
    )
    funder_year["Success Rate"] = 100 * funder_year["Successful"] / funder_year["Submitted"]

    scheme_summary = (
        df_success_raw.groupby(["Funder", "Scheme Name"], dropna=False)
        .agg(
            Submitted=("Project Ref", "count"),
            Successful=("StatusCode", lambda x: (x == "A").sum())
        )
        .reset_index()
    )
    scheme_summary["Success Rate"] = 100 * scheme_summary["Successful"] / scheme_summary["Submitted"]

    return df_oxford, funder_summary, funder_year, scheme_summary

# Load data
try:
    df_oxford_copy, funder_summary, funder_year, scheme_summary = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ============================================================
# Plot functions
# ============================================================
def plot_yearly_awards_summary(df_data):
    yearly_summary = (
        df_data.groupby("StartYear", dropna=False)
        .agg(
            number_of_awards=(COL_PROJECT_REF, "nunique"),
            total_award_pounds=(COL_AWARD, "sum")
        )
        .reset_index()
        .dropna(subset=["StartYear"])
    )

    fig = px.scatter(
        yearly_summary,
        x="StartYear",
        y="number_of_awards",
        size="total_award_pounds",
        hover_name="StartYear",
        hover_data={
            "total_award_pounds": ":,.2f",
            "number_of_awards": True,
            "StartYear": False
        },
        title="Number of Awards and Total Funding by Year (University of Oxford)",
        labels={
            "StartYear": "Year",
            "number_of_awards": "Number of Awards",
            "total_award_pounds": "Total Award Pounds"
        },
        size_max=60
    )
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1), height=600)
    return fig

def plot_success_rate_by_funder(funder_summary_df):
    fig = px.bar(
        funder_summary_df,
        x="Success Rate",
        y="Funder",
        orientation="h",
        color="Success Rate",
        hover_data=["Submitted", "Successful"],
        title="All SoGE Success Rate by Funder"
    )
    fig.update_layout(height=900, yaxis_title="")
    return fig

def plot_success_rate_heatmap(funder_year_df):
    heatmap_data = funder_year_df.pivot(index="Funder", columns="Year", values="Success Rate")
    fig = px.imshow(
        heatmap_data,
        aspect="auto",
        text_auto=".0f",
        color_continuous_scale="RdYlGn",
        title="Success Rate by Funder and Year"
    )
    fig.update_layout(height=900)
    return fig

def plot_scheme_portfolio_performance(scheme_summary_df):
    fig = px.scatter(
        scheme_summary_df,
        x="Submitted",
        y="Success Rate",
        size="Successful",
        color="Funder",
        hover_name="Scheme Name",
        title="Scheme Portfolio Performance"
    )
    fig.update_layout(height=700)
    return fig

# ============================================================
# Sidebar filters
# ============================================================
st.sidebar.header("Filters")

min_year = int(np.nanmin(df_oxford_copy["StartYear"])) if df_oxford_copy["StartYear"].notna().any() else 2000
max_year = int(np.nanmax(df_oxford_copy["StartYear"])) if df_oxford_copy["StartYear"].notna().any() else 2026

year_range = st.sidebar.slider(
    "Oxford award year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1
)

departments = sorted(df_oxford_copy[COL_DEPARTMENT].dropna().astype(str).unique().tolist())
selected_departments = st.sidebar.multiselect(
    "Department",
    options=departments,
    default=departments
)

funders = sorted(funder_summary["Funder"].dropna().astype(str).unique().tolist())
selected_funders = st.sidebar.multiselect(
    "Funder",
    options=funders,
    default=funders[: min(10, len(funders))]
)

# Filter Oxford funding data
filtered_oxford = df_oxford_copy.copy()
filtered_oxford = filtered_oxford[
    (filtered_oxford["StartYear"].fillna(-1) >= year_range[0]) &
    (filtered_oxford["StartYear"].fillna(-1) <= year_range[1])
]
if selected_departments:
    filtered_oxford = filtered_oxford[
        filtered_oxford[COL_DEPARTMENT].astype(str).isin(selected_departments)
    ]

filtered_funder_summary = funder_summary[
    funder_summary["Funder"].astype(str).isin(selected_funders)
].copy()

filtered_funder_year = funder_year[
    funder_year["Funder"].astype(str).isin(selected_funders)
].copy()

filtered_scheme_summary = scheme_summary[
    scheme_summary["Funder"].astype(str).isin(selected_funders)
].copy()

# ============================================================
# Navigation
# ============================================================
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Choose a visualisation:",
    [
        "Yearly Awards Summary",
        "Success Rate by Funder",
        "Success Rate by Funder and Year (Heatmap)",
        "Scheme Portfolio Performance"
    ]
)

# ============================================================
# Main content
# ============================================================
if page == "Yearly Awards Summary":
    st.subheader("Yearly Awards Summary")
    st.write("Oxford awards grouped by year, showing number of awards and total funding.")
    st.plotly_chart(plot_yearly_awards_summary(filtered_oxford), use_container_width=True)
    st.dataframe(
        filtered_oxford[
            [COL_FUNDER, COL_LEAD_RO, COL_DEPARTMENT, COL_TITLE, COL_START, COL_AWARD, COL_URL]
        ],
        use_container_width=True
    )

elif page == "Success Rate by Funder":
    st.subheader("Success Rate by Funder")
    st.write("Ranked success rate by funder for the SoGE success-rate dataset.")
    st.plotly_chart(plot_success_rate_by_funder(filtered_funder_summary), use_container_width=True)
    st.dataframe(filtered_funder_summary, use_container_width=True)

elif page == "Success Rate by Funder and Year (Heatmap)":
    st.subheader("Success Rate by Funder and Year")
    st.write("Heatmap of success rates across funders and financial years.")
    st.plotly_chart(plot_success_rate_heatmap(filtered_funder_year), use_container_width=True)
    st.dataframe(filtered_funder_year, use_container_width=True)

elif page == "Scheme Portfolio Performance":
    st.subheader("Scheme Portfolio Performance")
    st.write("How portfolio volume and success rate vary by scheme and funder.")
    st.plotly_chart(plot_scheme_portfolio_performance(filtered_scheme_summary), use_container_width=True)
    st.dataframe(filtered_scheme_summary, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit and Plotly")
