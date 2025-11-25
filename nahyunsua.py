import streamlit as st
import pandas as pd
from pathlib import Path
from urllib.parse import quote
import folium
from streamlit_folium import st_folium

# ------------------------------------------------------------
# 1. File names (English column version)
# ------------------------------------------------------------
HIST_NAME = "green_en.csv"           # past data
PRED_NAME = "XGBoostData_en.csv"     # 2050 prediction data

REPO_BASE_URL = "https://raw.githubusercontent.com/suaaus/25aiproject0/main/"


# ------------------------------------------------------------
# 2. Read CSV: try local first, then GitHub RAW
# ------------------------------------------------------------
def read_csv_local_or_github(filename: str) -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    local_path = base_dir / filename

    # 1) local file
    if local_path.exists():
        return pd.read_csv(local_path, encoding="utf-8-sig")

    # 2) GitHub RAW (for deployment)
    url = REPO_BASE_URL + quote(filename)
    return pd.read_csv(url, encoding="utf-8-sig")


@st.cache_data
def load_data():
    hist = read_csv_local_or_github(HIST_NAME)
    pred = read_csv_local_or_github(PRED_NAME)

    # just to be safe, strip spaces in column names
    hist.columns = hist.columns.astype(str).str.strip()
    pred.columns = pred.columns.astype(str).str.strip()

    return hist, pred


# ------------------------------------------------------------
# 3. Streamlit basic UI
# ------------------------------------------------------------
st.set_page_config(page_title="XGBoost 2050 Emissions per Area", layout="wide")
st.title("XGBoost-based 2050 Emissions per Area Dashboard")

st.write("""
This dashboard visualizes:

- **Historical emissions per area** (from `green_en.csv`)
- **XGBoost predictions for 2050** (from `XGBoostData_en.csv`)

You can explore each region on a map, see trends over time, and inspect the raw data.
""")

hist, pred = load_data()

# ------------------------------------------------------------
# 4. Region selection
# ------------------------------------------------------------
regions = sorted(pred["region"].unique())
selected_region = st.sidebar.selectbox("Select region", regions)

tab1, tab2, tab3 = st.tabs(["2050 Map", "Trend by Region", "Data Tables"])


# ------------------------------------------------------------
# 5. 2050 Prediction Map
# ------------------------------------------------------------
with tab1:
    st.subheader("2050 Emissions per Area (Map)")

    center_lat = pred["lat"].mean()
    center_lon = pred["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    max_val = pred["emissions_per_area_2050"].max()

    for _, row in pred.iterrows():
        value = row["emissions_per_area_2050"]
        radius = 5 + 15 * (value / max_val)

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            popup=f"{row['region']} : {value:.2f}",
            color="red",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    st_folium(m, width=900, height=550)


# ------------------------------------------------------------
# 6. Trend by Region (historical + 2050)
# ------------------------------------------------------------
with tab2:
    st.subheader(f"{selected_region} — Historical + 2050 Trend")

    region_hist = (
        hist[hist["region"] == selected_region]
        .groupby("year", as_index=False)["emissions_per_area"]
        .mean()
    )

    row_pred = pred[pred["region"] == selected_region]
    if len(row_pred) > 0:
        val_2050 = row_pred["emissions_per_area_2050"].iloc[0]
        region_hist = pd.concat(
            [
                region_hist,
                pd.DataFrame({"year": [2050], "emissions_per_area": [val_2050]})
            ],
            ignore_index=True,
        )

    region_hist = region_hist.sort_values("year")

    st.line_chart(region_hist.set_index("year"))


# ------------------------------------------------------------
# 7. Data Tables
# ------------------------------------------------------------
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2050 prediction (sorted)")
        st.dataframe(
            pred.sort_values("emissions_per_area_2050", ascending=False),
            use_container_width=True,
        )

    with col2:
        st.subheader("Historical data sample")
        st.dataframe(
            hist[["region", "year", "emissions_per_area"]].head(50),
            use_container_width=True,
        )
