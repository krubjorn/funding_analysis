import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Oxford Research Funding Dashboard")
st.caption("Interactive summary of funded projects and research themes.")

df = pd.read_csv("cleaned_projects.csv")

funder = st.sidebar.multiselect(
    "Select funder",
    sorted(df["FundingOrgName"].dropna().unique()),
    default=sorted(df["FundingOrgName"].dropna().unique())[:5]
)

filtered = df[df["FundingOrgName"].isin(funder)]

st.subheader("Research themes by funder")
st.write("This chart shows how funding is distributed across the main themes.")

fig = px.treemap(
    filtered,
    path=["FundingOrgName", "Theme"],
    values="AwardPounds",
    color="AwardPounds"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Underlying records")
st.dataframe(filtered, use_container_width=True)
