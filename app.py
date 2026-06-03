import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    # Paths to your data files in Google Drive
    # Ensure these paths are correct and accessible by Streamlit if deployed
    # For Colab context, we use the mounted drive path
    FUNDING_DATA_PATH = "/content/drive/MyDrive/funding_data/projectsearch-1780498470549.csv"
    SUCCESS_RATE_DATA_PATH = "/content/drive/MyDrive/funding_data/Success_rate_data.xlsx"

    # --- Load and preprocess Funding Data ---
    df_funding_raw = pd.read_csv(FUNDING_DATA_PATH)

    # Re-apply necessary cleaning and column preparation for df_oxford_copy
    # This block replicates the logic from various cells in the original notebook

    # Define original column names used in the notebook
    COL_FUNDER = 'FundingOrgName'
    COL_PROJECT_REF = 'ProjectReference'
    COL_LEAD_RO = 'LeadROName'
    COL_DEPARTMENT = 'Department'
    COL_PROJECT_CATEGORY = 'ProjectCategory'
    COL_PI_SURNAME = 'PISurname'
    COL_PI_FIRST = 'PIFirstName'
    COL_TITLE = 'Title'
    COL_START = 'StartDate'
    COL_END = 'EndDate'
    COL_AWARD = 'AwardPounds'
    COL_REGION = 'Region'
    COL_STATUS = 'Status'
    COL_URL = 'GTRProjectUrl'

    # Filter for University of Oxford
    df_oxford = df_funding_raw[df_funding_raw[COL_LEAD_RO].str.contains(r'University of Oxford', case=False, na=False)].copy()

    # Create a copy for further modifications
    df_oxford_copy = df_oxford.copy()

    # Convert date columns and create StartYear
    df_oxford_copy[COL_START] = pd.to_datetime(df_oxford_copy[COL_START], errors='coerce', dayfirst=True)
    df_oxford_copy['StartYear'] = df_oxford_copy[COL_START].dt.year
    df_oxford_copy[COL_AWARD] = pd.to_numeric(df_oxford_copy[COL_AWARD], errors='coerce')

    # --- Load and preprocess Success Rate Data ---
    df_success_raw = pd.read_excel(
        SUCCESS_RATE_DATA_PATH,
        sheet_name="Success rate 2020-21 to present"
    )

    def extract_year(value):
        if pd.isna(value):
            return np.nan
        text = str(value)
        digits = ''.join(c for c in text if c.isdigit())
        if len(digits) >= 2:
            return int("20" + digits[:2])
        return np.nan

    df_success_raw["Year"] = df_success_raw["Financial Year"].apply(extract_year)

    status_map = {
        "Approved by funder": "A",
        "Rejected by funder": "U",
        "Submitted to funder": "S"
    }
    df_success_raw["Fund Deci Status Desc"] = df_success_raw["Fund Deci Status Desc"].map(status_map)

    # Derive summary dataframes for success rate analysis
    funder_summary = (
        df_success_raw.groupby("Funder")
        .agg(
            Submitted=("Project Ref", "count"),
            Successful=("Fund Deci Status Desc", lambda x: (x=="A").sum())
        )
        .reset_index()
    )
    funder_summary["Success Rate"] = (100 * funder_summary["Successful"] / funder_summary["Submitted"])
    funder_summary = funder_summary.sort_values("Success Rate", ascending=False)

    funder_year = (
        df_success_raw.groupby(["Funder", "Year"])
        .agg(
            Submitted=("Project Ref", "count"),
            Successful=("Fund Deci Status Desc", lambda x: (x=="A").sum())
        )
        .reset_index()
    )
    funder_year["Success Rate"] = (100 * funder_year["Successful"] / funder_year["Submitted"])

    scheme_summary = (
        df_success_raw.groupby(["Funder", "Scheme Name"])
        .agg(
            Submitted=("Project Ref", "count"),
            Successful=("Fund Deci Status Desc", lambda x: (x=="A").sum())
        )
        .reset_index()
    )
    scheme_summary["Success Rate"] = (100 * scheme_summary["Successful"] / scheme_summary["Submitted"])


    return df_oxford_copy, funder_summary, funder_year, scheme_summary


df_oxford_copy, funder_summary, funder_year, scheme_summary = load_data()


# --- Visualization Functions ---

def plot_yearly_awards_summary(df_data):
    # This replicates the bubble plot from cell jPzQYO2FLuPc
    yearly_summary = df_data.groupby('StartYear').agg(
        number_of_awards=('ProjectId', 'nunique'),
        total_award_pounds=('AwardPounds', 'sum')
    ).reset_index()
    yearly_summary = yearly_summary.dropna(subset=['StartYear'])

    fig = px.scatter(
        yearly_summary,
        x='StartYear',
        y='number_of_awards',
        size='total_award_pounds',
        hover_name='StartYear',
        hover_data={'total_award_pounds': ':, .2f', 'number_of_awards': True, 'StartYear': False},
        title='Number of Awards and Total Funding by Year (University of Oxford)',
        labels={
            'StartYear': 'Year',
            'number_of_awards': 'Number of Awards',
            'total_award_pounds': 'Total Award Pounds'
        },
        size_max=60
    )
    fig.update_layout(xaxis=dict(tickmode='linear', dtick=1), height=600)
    return fig

def plot_success_rate_by_funder(funder_summary_df):
    # This replicates the bar chart from cell X8dEJx5b7enG
    fig = px.bar(
        funder_summary_df,
        x="Success Rate",
        y="Funder",
        orientation="h",
        color="Success Rate",
        hover_data=[
            "Submitted",
            "Successful"
        ],
        title="All SoGE Success Rate by Funder 2020 - 2025"
    )
    fig.update_layout(height=1000)
    return fig

def plot_success_rate_heatmap(funder_year_df):
    # This replicates the heatmap from cell kBRWWcuP7n8u
    heatmap_data = funder_year_df.pivot(index="Funder", columns="Year", values="Success Rate")
    fig = px.imshow(
        heatmap_data,
        aspect="auto",
        text_auto=".0f",
        color_continuous_scale="RdYlGn",
        title="Success Rate by Funder and Year"
    )
    return fig

def plot_scheme_portfolio_performance(scheme_summary_df):
    # This replicates the scatter plot from cell aq4Gwp6i76Z8
    fig = px.scatter(
        scheme_summary_df,
        x="Submitted",
        y="Success Rate",
        size="Successful",
        color="Funder",
        hover_name="Scheme Name",
        title="Scheme Portfolio Performance"
    )
    return fig

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("University of Oxford Funding and Success Rate Analysis")

# Sidebar for navigation or filters
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a visualization:",
    [
        "Yearly Awards Summary",
        "Success Rate by Funder",
        "Success Rate by Funder and Year (Heatmap)",
        "Scheme Portfolio Performance"
    ]
)

if page == "Yearly Awards Summary":
    st.header("Yearly Awards Summary (Funding Data)")
    st.plotly_chart(plot_yearly_awards_summary(df_oxford_copy), use_container_width=True)

elif page == "Success Rate by Funder":
    st.header("Success Rate by Funder")
    st.plotly_chart(plot_success_rate_by_funder(funder_summary), use_container_width=True)

elif page == "Success Rate by Funder and Year (Heatmap)":
    st.header("Success Rate by Funder and Year")
    st.plotly_chart(plot_success_rate_heatmap(funder_year), use_container_width=True)

elif page == "Scheme Portfolio Performance":
    st.header("Scheme Portfolio Performance")
    st.plotly_chart(plot_scheme_portfolio_performance(scheme_summary), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit and Plotly")
