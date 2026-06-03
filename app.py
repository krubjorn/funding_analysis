import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Oxford Funding Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("University of Oxford Funding Dashboard")
st.caption(
    "Interactive analysis of UKRI Gateway to Research award data. "
    "Use the sidebar to filter institutions and year range."
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
GTR_BASE_URL = "https://gtr.ukri.org/gtr/api"
GTR_ACCEPT = "application/vnd.rcuk.gtr.json-v7"
DEFAULT_PAGE_SIZE = 100

TARGET_INSTITUTIONS = [
    "University of Manchester",
    "University of Edinburgh",
    "University of Glasgow",
    "University College London",
    "University of Reading",
    "University of Birmingham",
    "University of Leeds",
    "Imperial College London",
    "University of Bristol",
    "Queen Mary University of London",
    "University of Cambridge",
    "University of Oxford",
]

INSTITUTION_SHORT = {
    "University of Oxford": "Oxford",
    "University of Cambridge": "Cambridge",
    "University of Manchester": "Manchester",
    "University of Edinburgh": "Edinburgh",
    "University of Glasgow": "Glasgow",
    "University College London": "UCL",
    "University of Reading": "Reading",
    "University of Birmingham": "Birmingham",
    "University of Leeds": "Leeds",
    "Imperial College London": "Imperial",
    "University of Bristol": "Bristol",
    "Queen Mary University of London": "QMUL",
}

ORG_VARIANTS = {
    "University of Manchester": ["University of Manchester", "Manchester"],
    "University of Edinburgh": ["University of Edinburgh", "Edinburgh"],
    "University of Glasgow": ["University of Glasgow", "Glasgow", "Glasgow University"],
    "University College London": ["University College London", "UCL"],
    "University of Reading": ["University of Reading", "Reading"],
    "University of Birmingham": ["University of Birmingham", "Birmingham"],
    "University of Leeds": ["University of Leeds", "Leeds"],
    "Imperial College London": ["Imperial College London", "Imperial"],
    "University of Bristol": ["University of Bristol", "Bristol"],
    "Queen Mary University of London": ["Queen Mary University of London", "QMUL", "Queen Mary"],
    "University of Cambridge": ["University of Cambridge", "Cambridge"],
    "University of Oxford": ["University of Oxford", "Oxford"],
}

HEADERS = {
    "Accept": GTR_ACCEPT,
    "User-Agent": "Oxford-GtR-Streamlit/1.0",
}

COLS = [
    "Funding Organisation name",
    "Institution name",
    "Year of grant",
    "Project Reference",
    "Grant category",
    "PI Surname",
    "PI first name",
    "title of the project",
    "Award amount in Pounds",
    "Region",
    "GTR Project Url",
    "Project abstract",
]

GEOGRAPHY_KEYWORDS = [
    "geography", "climate", "migration", "environment", "urban", "cities", "land use",
    "spatial", "gis", "flood", "water", "sustainability", "transport", "carbon", "adaptation", "net zero",
]

SOCIAL_SCIENCE_KEYWORDS = [
    "society", "economic", "economics", "policy", "inequality", "education", "social",
    "politics", "governance", "public health", "behaviour",
]

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def extract_items(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in [
            "items", "item", "results", "project", "projects", "fund", "funds",
            "organisation", "organisations", "person", "persons", "outcome", "outcomes",
        ]:
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
            if isinstance(val, dict):
                return [val]
        if any(k in payload for k in ["id", "title", "name", "links"]):
            return [payload]
    return []


def extract_total_pages(payload: Any) -> Optional[int]:
    if isinstance(payload, dict):
        for key in ["totalPages", "total_pages", "pages"]:
            if key in payload:
                try:
                    return int(payload[key])
                except Exception:
                    return None
    return None


def coerce_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)) and not pd.isna(x):
        return float(x)
    s = str(x)
    s = re.sub(r"[^0-9.,-]", "", s)
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def norm_text(text: Any) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def project_links(project: Dict[str, Any]) -> List[Dict[str, Any]]:
    links = project.get("links")
    if isinstance(links, dict):
        links = links.get("link", links.get("links", []))
    return _as_list(links)


def infer_year(project: Dict[str, Any]) -> Optional[int]:
    for key in ["start", "startDate", "awardDate", "date", "created", "modified"]:
        raw = project.get(key)
        if raw:
            ts = pd.to_datetime(raw, errors="coerce")
            if pd.notna(ts):
                return int(ts.year)
    return None


def infer_title(project: Dict[str, Any]) -> Optional[str]:
    return project.get("title") or project.get("name") or project.get("projectTitle")


def infer_pi_from_project(project: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    links = project_links(project)
    for link in links:
        rel = str(link.get("rel", "")).upper()
        if "PI_PER" in rel:
            surname = link.get("surname") or link.get("familyName") or link.get("lastName")
            firstname = link.get("forename") or link.get("givenName") or link.get("firstName")
            if surname or firstname:
                return surname, firstname
    surname = project.get("piSurname") or project.get("pi_surname")
    firstname = project.get("piFirstName") or project.get("pi_first_name")
    return surname, firstname


def infer_funding_organisation(project: Dict[str, Any]) -> Optional[str]:
    links = project_links(project)
    for link in links:
        rel = str(link.get("rel", "")).upper()
        if "FUNDER" in rel or rel == "FUND":
            return link.get("title") or link.get("name") or link.get("label")
    return project.get("fundingOrganisation") or project.get("funding_organisation")


def infer_region(project: Dict[str, Any]) -> Optional[str]:
    for key in ["region", "projectRegion", "leadRegion"]:
        if project.get(key):
            return project.get(key)
    return None


def infer_award_amount(project: Dict[str, Any]) -> Optional[float]:
    for key in ["awardAmount", "award_amount", "amount", "value", "totalValue", "fundValue"]:
        num = coerce_number(project.get(key))
        if num is not None:
            return num
    return None


def infer_grant_category(project: Dict[str, Any]) -> Optional[str]:
    text = " ".join(str(project.get(k, "")) for k in ["scheme", "schemeName", "projectType", "title", "abstractText", "abstract", "summary"]).lower()
    rules = {
        "Fellowship": ["fellowship", "fellow "],
        "Programme": ["programme", "program"],
        "Research grant": ["research grant", "project grant", "grant"],
        "Studentship": ["studentship", "phd"],
        "Training": ["training", "doctoral training"],
    }
    for label, tokens in rules.items():
        if any(t in text for t in tokens):
            return label
    return project.get("schemeName") or project.get("projectType")


# ------------------------------------------------------------
# GtR client + API access
# ------------------------------------------------------------
@dataclass
class GTRClient:
    base_url: str = GTR_BASE_URL
    headers: Dict[str, str] = None
    timeout: int = 60

    def __post_init__(self):
        if self.headers is None:
            self.headers = HEADERS.copy()
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def paginate(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        page = 1
        params = dict(params or {})
        params.setdefault("size", page_size)

        while True:
            params["page"] = page
            data = self.get_json(path, params=params)
            items = extract_items(data)
            out.extend(items)

            total_pages = extract_total_pages(data)
            if total_pages is None:
                if not items:
                    break
                if max_pages is not None and page >= max_pages:
                    break
                page += 1
                continue

            if page >= total_pages:
                break
            if max_pages is not None and page >= max_pages:
                break
            page += 1

        return out


@st.cache_resource
def get_client() -> GTRClient:
    return GTRClient()


client = get_client()


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def search_organisations(query: Optional[str] = None, page_size: int = 25, max_pages: int = 2) -> List[Dict[str, Any]]:
    params = {}
    if query:
        params["q"] = query
    return client.paginate("organisations", params=params, page_size=page_size, max_pages=max_pages)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def resolve_one_institution(target_name: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    for query in ORG_VARIANTS.get(target_name, [target_name]):
        try:
            matches = search_organisations(query, page_size=25, max_pages=2)
        except Exception:
            continue

        if not matches:
            continue

        exact = [m for m in matches if norm_text(m.get("name") or m.get("title")) == norm_text(target_name)]
        chosen = exact[0] if exact else matches[0]
        return target_name, {
            "id": chosen.get("id"),
            "name": chosen.get("name") or chosen.get("title") or target_name,
            "query_used": query,
        }

    return target_name, None


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def resolve_selected_institutions(target_names: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in target_names:
        inst, meta = resolve_one_institution(name)
        if meta is not None:
            out[inst] = meta
    return out


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def get_projects_for_org(org_id: Any, page_size: int = DEFAULT_PAGE_SIZE, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    return client.paginate(f"organisations/{org_id}/projects", page_size=page_size, max_pages=max_pages)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def load_projects(
    selected_institutions: Tuple[str, ...],
    year_min: int,
    year_max: int,
    page_size: int,
    max_pages: int,
) -> pd.DataFrame:
    org_lookup = resolve_selected_institutions(selected_institutions)
    rows: List[Dict[str, Any]] = []

    for inst_name, org_meta in org_lookup.items():
        org_id = org_meta.get("id")
        if org_id is None:
            continue
        projects = get_projects_for_org(org_id, page_size=page_size, max_pages=max_pages)

        for project in projects:
            year = infer_year(project)
            if year is None or not (year_min <= year <= year_max):
                continue

            project_ref = project.get("id") or project.get("grantReference") or project.get("reference")
            title = infer_title(project)
            surname, firstname = infer_pi_from_project(project)

            rows.append({
                "Funding Organisation name": infer_funding_organisation(project),
                "Institution name": inst_name,
                "Year of grant": year,
                "Project Reference": project_ref,
                "Grant category": infer_grant_category(project),
                "PI Surname": surname,
                "PI first name": firstname,
                "title of the project": title,
                "Award amount in Pounds": infer_award_amount(project),
                "Region": infer_region(project),
                "GTR Project Url": f"https://gtr.ukri.org/projects?ref={project_ref}" if project_ref else None,
                "Project abstract": project.get("abstractText") or project.get("abstract") or project.get("summary"),
                "project_raw": project,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["Funding Organisation name", "Institution name", "Grant category", "PI Surname", "PI first name", "title of the project", "Region", "GTR Project Url", "Project abstract"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).replace("nan", "")

    df["Award amount in Pounds"] = pd.to_numeric(df["Award amount in Pounds"], errors="coerce")
    df["Subject Area"] = (
        df["title of the project"].fillna("") + " " + df["Project abstract"].fillna("")
    ).apply(classify_subject)

    # Remove duplicates by project reference within institution if available.
    subset_cols = [c for c in ["Institution name", "Project Reference", "title of the project"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols).copy()

    return df


# ------------------------------------------------------------
# Classification helper
# ------------------------------------------------------------
def classify_subject(text: Any) -> str:
    text = str(text).lower()
    geo_score = sum(k in text for k in GEOGRAPHY_KEYWORDS)
    soc_score = sum(k in text for k in SOCIAL_SCIENCE_KEYWORDS)
    if geo_score > 0:
        return "Geography"
    if soc_score > 0:
        return "Social Sciences"
    return "Other"


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("Filters")

selected_institutions = st.sidebar.multiselect(
    "Institutions",
    options=TARGET_INSTITUTIONS,
    default=["University of Oxford", "University of Cambridge", "University College London", "University of Manchester"],
)

if not selected_institutions:
    selected_institutions = ["University of Oxford"]

year_min, year_max = st.sidebar.slider(
    "Year range",
    min_value=2020,
    max_value=2026,
    value=(2020, 2026),
)

page_size = st.sidebar.slider("Page size", min_value=20, max_value=100, value=100, step=10)
max_pages = st.sidebar.slider("Max pages per institution", min_value=1, max_value=20, value=10, step=1)
show_raw = st.sidebar.checkbox("Show raw project JSON in table", value=False)

with st.spinner("Loading projects from GtR..."):
    projects_df = load_projects(tuple(selected_institutions), year_min, year_max, page_size, max_pages)

# ------------------------------------------------------------
# Top-level KPIs
# ------------------------------------------------------------
if projects_df.empty:
    st.warning("No projects were returned for the current selection. Try expanding the institution list or the page limit.")
    st.stop()

projects_df = projects_df.copy()
projects_df["Institution short"] = projects_df["Institution name"].map(INSTITUTION_SHORT).fillna(projects_df["Institution name"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Projects", f"{len(projects_df):,}")
col2.metric("Institutions", f"{projects_df['Institution name'].nunique():,}")
col3.metric("Total award value", f"£{projects_df['Award amount in Pounds'].fillna(0).sum():,.0f}")
col4.metric("Median award", f"£{projects_df['Award amount in Pounds'].median():,.0f}" if projects_df["Award amount in Pounds"].notna().any() else "n/a")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_overview, tab_table, tab_oxford, tab_focus, tab_peer = st.tabs(
    ["Overview", "Data table", "Oxford awards", "Geography / Social Science", "Peer benchmark"]
)

with tab_overview:
    st.subheader("Portfolio overview")
    st.write("Award counts and values across the selected institutions and years.")

    inst_summary = (
        projects_df.groupby("Institution short", dropna=False)
        .agg(
            awards=("Project Reference", "count"),
            total_award_value=("Award amount in Pounds", "sum"),
            median_award_value=("Award amount in Pounds", "median"),
        )
        .reset_index()
        .sort_values("total_award_value", ascending=False)
    )

    fig = px.bar(
        inst_summary.sort_values("total_award_value"),
        x="total_award_value",
        y="Institution short",
        orientation="h",
        title="Total awarded value by institution",
        labels={"total_award_value": "Total awarded value (£)", "Institution short": "Institution"},
    )
    st.plotly_chart(fig, use_container_width=True)

    year_summary = (
        projects_df.groupby("Year of grant", dropna=False)
        .agg(
            awards=("Project Reference", "count"),
            total_award_value=("Award amount in Pounds", "sum"),
        )
        .reset_index()
        .sort_values("Year of grant")
    )

    fig2 = px.bar(
        year_summary,
        x="Year of grant",
        y="total_award_value",
        title="Total award value by year",
        labels={"total_award_value": "Total awarded value (£)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab_table:
    st.subheader("Interactive project table")
    st.write("Use the table below to inspect project-level records.")

    table_cols = [c for c in COLS if c in projects_df.columns]
    table_df = projects_df[table_cols].copy()
    if not show_raw and "project_raw" in projects_df.columns:
        pass

    st.dataframe(
        table_df.sort_values(["Institution name", "Year of grant", "Award amount in Pounds"], ascending=[True, False, False]),
        use_container_width=True,
        hide_index=True,
    )

    csv = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download table as CSV",
        csv,
        file_name="oxford_gtr_projects.csv",
        mime="text/csv",
    )

with tab_oxford:
    st.subheader("University of Oxford awards")
    oxford_awards = projects_df.loc[projects_df["Institution name"] == "University of Oxford"].copy()

    if oxford_awards.empty:
        st.info("No Oxford records were returned for the current filter set.")
    else:
        oxford_summary = (
            oxford_awards.groupby(["Year of grant", "Grant category"], dropna=False)
            .agg(
                awards=("Project Reference", "count"),
                total_award_value=("Award amount in Pounds", "sum"),
                median_award_value=("Award amount in Pounds", "median"),
            )
            .reset_index()
            .sort_values(["Year of grant", "total_award_value"], ascending=[True, False])
        )
        st.dataframe(oxford_summary, use_container_width=True, hide_index=True)

        fig = px.treemap(
            oxford_awards.dropna(subset=["Grant category", "Award amount in Pounds"]),
            path=["Year of grant", "Grant category"],
            values="Award amount in Pounds",
            color="Award amount in Pounds",
            title="Oxford awards by year and grant category",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            oxford_summary,
            x="Year of grant",
            y="total_award_value",
            color="Grant category",
            barmode="stack",
            title="Oxford award value by year and grant category",
            labels={"total_award_value": "Award amount (£)"},
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            oxford_awards.dropna(subset=["Award amount in Pounds"]),
            x="Grant category",
            y="Award amount in Pounds",
            points="all",
            title="Oxford award amount distribution by grant category",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_focus:
    st.subheader("Oxford Geography / Social Science focus")
    oxford_awards = projects_df.loc[projects_df["Institution name"] == "University of Oxford"].copy()

    if oxford_awards.empty:
        st.info("No Oxford records available for subject classification.")
    else:
        oxford_awards["Subject Area"] = (
            oxford_awards["title of the project"].fillna("") + " " + oxford_awards["Project abstract"].fillna("")
        ).apply(classify_subject)

        focus_df = oxford_awards.loc[oxford_awards["Subject Area"].isin(["Geography", "Social Sciences"])].copy()

        if focus_df.empty:
            st.info("No Geography or Social Science records were detected in the Oxford subset.")
        else:
            focus_summary = (
                focus_df.groupby(["Year of grant", "Subject Area", "Grant category"], dropna=False)
                .agg(
                    awards=("Project Reference", "count"),
                    total_award_value=("Award amount in Pounds", "sum"),
                    median_award_value=("Award amount in Pounds", "median"),
                )
                .reset_index()
                .sort_values(["Year of grant", "total_award_value"], ascending=[True, False])
            )
            st.dataframe(focus_summary, use_container_width=True, hide_index=True)

            fig = px.treemap(
                focus_df.dropna(subset=["Award amount in Pounds"]),
                path=["Subject Area", "Year of grant", "Grant category"],
                values="Award amount in Pounds",
                color="Award amount in Pounds",
                title="Oxford Geography / Social Science awards",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(
                focus_df.dropna(subset=["Award amount in Pounds"]),
                x="Year of grant",
                y="Award amount in Pounds",
                color="Subject Area",
                size="Award amount in Pounds",
                hover_data=["title of the project", "PI Surname", "PI first name", "GTR Project Url"],
                title="Oxford Geography / Social Science award portfolio",
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_peer:
    st.subheader("Peer benchmark")
    st.write("Compare Oxford with other selected institutions.")

    peer_awards = projects_df.copy()
    if peer_awards.empty:
        st.info("No peer records available for benchmarking.")
    else:
        sector_summary = (
            peer_awards.groupby("Institution short")
            .agg(
                awards=("Project Reference", "count"),
                total_award_value=("Award amount in Pounds", "sum"),
                median_award_value=("Award amount in Pounds", "median"),
                mean_award_value=("Award amount in Pounds", "mean"),
                share_large_awards=("Award amount in Pounds", lambda s: (pd.to_numeric(s, errors="coerce") >= 1_000_000).mean()),
            )
            .reset_index()
            .sort_values("total_award_value", ascending=False)
        )
        st.dataframe(sector_summary, use_container_width=True, hide_index=True)

        fig = px.bar(
            sector_summary.sort_values("total_award_value"),
            x="total_award_value",
            y="Institution short",
            orientation="h",
            title="Sector benchmark: total awarded value by institution",
            labels={"total_award_value": "Total awarded value (£)", "Institution short": "Institution"},
        )
        st.plotly_chart(fig, use_container_width=True)

        award_mix = (
            peer_awards.groupby(["Institution short", "Grant category"], dropna=False)
            .agg(awards=("Project Reference", "count"), total_value=("Award amount in Pounds", "sum"))
            .reset_index()
        )

        fig = px.bar(
            award_mix,
            x="Institution short",
            y="awards",
            color="Grant category",
            barmode="stack",
            title="Award category mix by institution",
            labels={"awards": "Number of awards", "Grant category": "Grant category"},
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Data source: UKRI Gateway to Research API. The dashboard is interactive but depends on live API access."
)
