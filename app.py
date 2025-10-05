import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score




# -------------------- Header --------------------
st.markdown(
    """
    <style>
        /* Remove default Streamlit top/bottom padding */
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            z-index: 0;
        }

        /* Header */
        .main-header {
            text-align: center;
            font-size: 38px;
            font-weight: bold;
            margin: 50px 0 10px 0;
            padding: 10px 0;
            width: 100%;
            background: linear-gradient(90deg, #4A90E2/30, #50E3C2/30);
            color: white;
            # border-radius: 8px;
        }


        /* Footer */
        .footer {
            position: relative;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            padding: 6px 0;
            # margin-top: 100px;
            background: linear-gradient(90deg, #50E3C2, #4A90E2);
            color: white;
            font-size: 13px;
            font-weight: 500;
        }
    </style>

    <!-- HEADER -->
    <div class="main-header">🌍 NASA Challenge – Bloom Watch Project 🌸</div>
    """,
    unsafe_allow_html=True
)




# --------------------
# Streamlit Config
# --------------------
st.set_page_config(page_title="NASA Challenge Project", layout="wide")

# --------------------
# Load dataset
# --------------------
df = pd.read_csv("flowerBlooms_Clean.csv")






# 1st viz: Combined Bar and Trend Line Chart with Dual Y-Axis
st.title("🌸 Bloom Area vs Season (Bar + Trend Lines)")
# --------------------
# Define a unified color palette
# --------------------
palette = sns.color_palette("Set2")
sns.set_theme(style="whitegrid", palette=palette)

# # Convert Seaborn palette to HEX for Plotly
shared_colors = [mcolors.to_hex(c) for c in palette]

# --------------------
# Aggregate data
# --------------------
season_summary = df.groupby(["Season", "Type"])["Area"].sum().reset_index()
total_per_season = df.groupby("Season")["Area"].sum().reset_index()

# --------------------
# Create figure with secondary y-axis
# --------------------
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add bar trace for total bloom area per season
fig.add_trace(
    go.Bar(
        x=total_per_season["Season"],
        y=total_per_season["Area"],
        name="Total Bloom Area",
        marker_color="lightblue",
        text=total_per_season["Area"],
        textposition="outside"
    ),
    secondary_y=False
)

# Add line traces for each Type
types = season_summary["Type"].unique()
colors = shared_colors[:len(types)]  # consistent colors

for i, t in enumerate(types):
    type_data = season_summary[season_summary["Type"] == t]
    fig.add_trace(
        go.Scatter(
            x=type_data["Season"],
            y=type_data["Area"],
            mode="lines+markers",   # keep trend lines
            name=f"Trend - {t}",
            line=dict(color=colors[i], width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )

# --------------------
# Layout settings
# --------------------
fig.update_layout(
    xaxis_title="Season",
    yaxis_title="Total Bloom Area",
    yaxis2_title="Bloom Area by Type",
    legend=dict(title="Legend", orientation="h", y=-0.2, x=0.5, xanchor="center"),
    xaxis=dict(showgrid=True),          # keep x-axis grid if desired
    yaxis=dict(showgrid=False),         # remove y-axis grid lines for primary y
    yaxis2=dict(showgrid=False)         # remove y-axis grid lines for secondary y
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)







# 2nd viz: Pie Chart of Bloom Area Proportions by Site
# Aggregate total bloom area per site
site_area = df.groupby("Site")["Area"].sum().reset_index()

fig_pie = px.pie(
    site_area,
    names="Site",
    values="Area",
    title="🌸 Proportion of Bloom Area by Site",
    color="Site",
    color_discrete_sequence=shared_colors,
    width=800,
    height=600
)

fig_pie.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05]*len(site_area))
fig_pie.update_layout(
    title_font=dict(size=40, family="Arial Black"),
    legend_title_text="Sites",
    plot_bgcolor="white"
)

st.plotly_chart(fig_pie, use_container_width=True)







#3rd viz: Donut Chart + Side-by-Side Bar Chart
st.title("🌸 Bloom Area Analysis By Type")

# --- Create two columns for layout ---
col1, col2 = st.columns(2)

# Donut Chart: Total Bloom Area by Type
with col1:
    area_by_type = df.groupby("Type")["Area"].sum().reset_index()
    fig_donut = px.pie(
        area_by_type,
        names="Type",
        values="Area",
        hole=0.5,
        title="Total Bloom Area by Type"
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# Side-by-Side Bar Chart: Bloom Area across Sites grouped by Type
with col2:
    fig_bar = px.bar(
        df,
        x="Site",
        y="Area",
        color="Type",
        barmode="group",
        title="Bloom Area by Type across Sites"
    )
    fig_bar.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig_bar, use_container_width=True)







# 4th viz: Site Selector with Dual Visualizations
st.title("🌸 Bloom By Site per Seasons")

# -------------------------
# Row 1: Dropdown (site selector)
# -------------------------
site = st.selectbox("Select Site:", df["Site"].unique())

# Filter site data
site_data = df[df["Site"] == site]
unique_seasons = site_data["Season"].nunique()

# -------------------------
# Row 2: Two visualizations side by side
# -------------------------
col1, col2 = st.columns(2)

# Left Column → Site-level chart
with col1:
    st.subheader(f"Site-level Bloom at {site}")
    if unique_seasons == 1:
        # Donut Chart if only one season
        fig_site = px.pie(
            site_data,
            names="Season",
            values="Area",
            hole=0.4,
            title=f"Bloom Area Distribution for {site}"
        )
        fig_site.update_traces(textinfo="label+percent")
    else:
        # Bar Chart if multiple seasons
        fig_site = px.bar(
            site_data,
            x="Season",
            y="Area",
            color="Season",
            title=f"Bloom Area for {site}",
        )
        fig_site.update_layout(
            title_font=dict(size=18, family="Arial Black"),
            xaxis=dict(title="Season", showgrid=True),
            yaxis=dict(title="Bloom Area (sq. units)", showgrid=False),
            # plot_bgcolor="white"
        )
    st.plotly_chart(fig_site, use_container_width=True)

# Right Column → Global Donut (all sites)
with col2:
    st.subheader("🌍 Global Bloom Distribution")
    area_by_season = df.groupby("Season")["Area"].sum().reset_index()
    fig_global = px.pie(
        area_by_season,
        names="Season",
        values="Area",
        hole=0.4,
        title="Overall Bloom Area by Season"
    )
    fig_global.update_traces(textinfo="label+percent")
    st.plotly_chart(fig_global, use_container_width=True)








# 5th viz: Radial Plot of Seasonal Bloom Patterns
st.title("🌸 Seasonal Bloom Patterns Across Sites (Radial Plots)")

# Dropdown to pick a site
site_selected = st.selectbox("Select a Site:", df["Site"].unique())
radial_data = df[df["Site"] == site_selected]

fig_radial = px.bar_polar(
    radial_data,
    r="Area",
    theta="Season",
    color="Season",
    title=f"Seasonal Bloom Pattern for {site_selected}",
    color_discrete_sequence=px.colors.sequential.Plasma_r
)

fig_radial.update_layout(
    title_font=dict(size=18, family="Arial Black"),
    polar=dict(radialaxis=dict(showticklabels=True, ticks="")),
    width=500,
    height=500,
)

st.plotly_chart(fig_radial, use_container_width=True)







# 6th Viz: Geospatial Distribution of Bloom Sites
import ast 

# Function to extract first [lon, lat] from Coordinates
def extract_lon_lat(coord_str):
    try:
        coords = ast.literal_eval(coord_str)  # convert string → Python list
        lon, lat = coords[0][0][0]  # first polygon → first ring → first point
        return lon, lat
    except:
        return None, None
    
df["Longitude"], df["Latitude"] = zip(*df["Coordinates"].map(extract_lon_lat))

# Convert to float
df["Latitude"] = df["Latitude"].astype(float)
df["Longitude"] = df["Longitude"].astype(float)

st.title("🗺️ Geospatial Distribution of Bloom Sites")

season_colors = {
    "Spring": "#FF69B4",   # hot pink
    "Summer": "#FFD700",   # golden yellow
    "Fall":   "#FF8C00",   # warm orange
    "Winter": "#1E90FF"    # cool blue
}

fig_map = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    size="Area",
    color="Season",
    hover_name="Site",
    hover_data={"Area": True, "Latitude": False, "Longitude": False},
    zoom=5,
    mapbox_style="carto-positron",
    title="Bloom Area Distribution Across Sites (Geospatial View)",
    color_discrete_map=season_colors   # use custom palette
)

fig_map.update_layout(
    title_font=dict(size=18, family="Arial Black"),
    height=600
)

st.plotly_chart(fig_map, use_container_width=True)








# 7th Viz: Bloom Area by Site (Ascending Order)
st.title("🌸 Bloom Area by Site (Ascending Order)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Bloom Area", f"{df['Area'].sum():,.0f}")
with col2:
    st.metric("Unique Sites", df['Site'].nunique())
with col3:
    st.metric("Flower Types", df['Type'].nunique())

# Aggregate bloom area per site
site_area_sorted = df.groupby("Site")["Area"].sum().reset_index().sort_values(by="Area", ascending=True)

# Create horizontal bar chart
fig_area_sorted = px.bar(
    site_area_sorted,
    x="Area",
    y="Site",
    orientation="h",  # horizontal bars
    color="Area",
    color_continuous_scale="Blues",
)

# Update layout
fig_area_sorted.update_layout(
    xaxis_title="Total Bloom Area",
    yaxis_title="Site",
    height=500
)

# Display
st.plotly_chart(fig_area_sorted, use_container_width=True)







# 8th Viz: Trend Analysis & Prediction of Bloom Area
st.title("🌼 Bloom Area Trend & Prediction Analysis")

# -----------------------------
# Define unified color palette
# -----------------------------
palette = sns.color_palette("Set2")
shared_colors = [mcolors.to_hex(c) for c in palette]
sns.set_theme(style="whitegrid", palette=palette)

# -----------------------------
# Data Preprocessing
# -----------------------------
# Convert 'Season' to categorical and map numbers
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
df['Season'] = pd.Categorical(df['Season'], categories=season_order, ordered=True)
df['Season_num'] = df['Season'].cat.codes  # Safe conversion

# Convert other categorical columns
df['Site_num'] = df['Site'].astype('category').cat.codes
df['Type_num'] = df['Type'].astype('category').cat.codes
df['GeometryType_num'] = df['GeometryType'].astype('category').cat.codes

# Features & Target
X = df[['Season_num', 'Site_num', 'Type_num', 'GeometryType_num']]
y = df['Area']

# -----------------------------
# Train-Test Split & Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -----------------------------
# Visualization (Compact & Themed)
# -----------------------------
st.subheader("🌸 Predicted Bloom Area Trend by Season")

fig, ax = plt.subplots(figsize=(8, 4))

# Scatter for actual data
sns.scatterplot(
    x='Season_num',
    y='Area',
    data=df,
    color=shared_colors[0],
    s=60,
    label='Actual',
    ax=ax
)

# Line for predicted trend
sns.lineplot(
    x=X_test['Season_num'],
    y=y_pred,
    color=shared_colors[2],
    linewidth=2.5,
    label='Predicted',
    ax=ax
)

# ✅ Replace numbers with season names on x-axis
ax.set_xticks(range(len(season_order)))
ax.set_xticklabels(season_order, fontsize=7)

# ✅ Adjust y-axis font size
ax.tick_params(axis='y', labelsize=7)

# Labels and aesthetics
ax.set_title('Predicted Bloom Area Trend by Season', fontsize=10, fontweight='bold')
ax.set_xlabel('Seasons', fontsize=8)
ax.set_ylabel('Bloom Area', fontsize=8)
legend = ax.legend(fontsize=8, loc='upper right', frameon=True)
legend.get_frame().set_facecolor('#f8f9fa')
legend.get_frame().set_edgecolor('#dcdcdc')
legend.get_frame().set_alpha(0.9)
sns.despine()

st.pyplot(fig, use_container_width=False)

# -----------------------------
# Model Performance
# -----------------------------
# st.subheader("📊 Model Performance")
# col1, col2 = st.columns(2)
# col1.metric(label="Mean Squared Error", value=f"{mse:.2f}")
# col2.metric(label="R² Score", value=f"{r2:.2f}")

# -----------------------------
# User Prediction Section
# -----------------------------
st.subheader("🌺 Predict Bloom Area for Custom Input")

col1, col2, col3, col4 = st.columns(4)
season_choice = col1.selectbox("Season", season_order)
site_choice = col2.selectbox("Site", df['Site'].unique())
type_choice = col3.selectbox("Flower Type", df['Type'].unique())
geo_choice = col4.selectbox("Geometry Type", df['GeometryType'].unique())

# Numeric encodings for input
season_val = df[df['Season'] == season_choice]['Season_num'].iloc[0]
site_val = df[df['Site'] == site_choice]['Site_num'].iloc[0]
type_val = df[df['Type'] == type_choice]['Type_num'].iloc[0]
geo_val = df[df['GeometryType'] == geo_choice]['GeometryType_num'].iloc[0]

predicted_area = model.predict([[season_val, site_val, type_val, geo_val]])[0]

st.metric(label=f"Predicted Bloom Area for {season_choice}", value=round(predicted_area, 2))






# -------------------- Footer --------------------
st.markdown(
    """
    <div class="footer">
        🚀 Team <b>Nebula Nexus</b> | NASA Space Apps Challenge 2025 🌌
    </div>
    """,
    unsafe_allow_html=True
)