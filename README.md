# NASA-Challenge_Bloom-Watch-Project_By_NebulaNexus
An Earth Observation Application for Global Flowering Phenology

## Dataset prepartion and cleaning

### Dataset Context
The primary dataset used to this project is refered from the Nasa's official resource **Wildflower Bloom Observations** collected via the [GLOBE Observer App](https://observer.globe.gov/do-globe-observer/do-more/data-requests/wildflower-blooms).
Some outsourced dataset is also included in it and to explore the variety and diversity of data.
The initail dataset was in JSON format it has been converted to CSV to simplify analysis and processing with Python and data visualization tools.

### Concerns for data cleaning
- **Irrevalent Metadata**: Dataset contained columns as properties.id and type which didn't add any relevant analytical values for bloom monitoring.
- **Unnecessary Indexing**: The site column included numbering sequences that were not meaningful for analysis
- **Readability and usability**: As clean and structured data add meaningfull value in analysis process such as visualization, feature extraction, or predictive modeling can be carried out efficiently.

### Data cleaning process
Used Python's (Panda library) the following steps were taken:
- **Loading the Dataset**: Imported the dataset using pandas.read_csv() after conversion from JSON.
Used df.info() and df.head() to explore dataset structure, missing values, and column metadata.
- **Dropping Redundant Columns**: Removed the properties.id column and Dropped the type column.
- **Removing Numbering Series in site Column**: Cleaned up the site field by removing numbering sequences. 
- **Exporting the Clean Dataset**: After prepartion and cleaning, saved the cleaned data as flowerBlooms_Clean.csv, making it ready for further analysis and visualization.
- **Column rename**: Shorten the name of column due to long column naming.

### Futher usage
- Clean data helps in doing visualization properly.



## Data Visualizations Overview
This project features 8 interactive and analytical visualizations built using Streamlit, Plotly, Seaborn, and Matplotlib, focusing on understanding flower bloom patterns, area distribution, and predictive trends.
Each visualization offers a unique analytical perspective on the dataset derived from the NASA Bloom Watch Project.

### 1️⃣ Bloom Area vs Season (Bar + Trend Lines)
📊 Type: Bar Chart + Dual Line Trend (Plotly)
Displays total bloom area per season as bars.
Overlays trend lines by flower type to show comparative seasonal growth.
Dual y-axes separate total vs. type-level bloom area.
Interactive legend and hover tooltips for detailed insight.

### 2️⃣ Proportion of Bloom Area by Site (Pie Chart)
🥧 Type: Pie Chart (Plotly)
Shows percentage contribution of each site to the total bloom area.
Interactive hover labels display site names and their proportional area.
Uses unified shared_colors palette for aesthetic consistency.

### 3️⃣ Bloom Area by Type (Donut + Grouped Bar Chart)
🪷 Type: Donut Chart & Grouped Bar Chart (Plotly)
Donut Chart: Visualizes total bloom area by flower type.
Bar Chart: Compares flower types across different sites.
Layout uses Streamlit’s dual-column design for side-by-side comparison.

### 4️⃣ Bloom by Site per Season (Dynamic Selector)
🏞️ Type: Dropdown-Based Site Comparison (Plotly)
Allows users to select any site from a dropdown.
Displays site-level bloom area per season (bar or donut depending on season count).
A global comparison donut chart summarizes overall seasonal bloom.

### 5️⃣ Seasonal Bloom Patterns Across Sites (Radial Plot)
🌈 Type: Radial Bar Polar Chart (Plotly)
Visualizes bloom patterns per season for a selected site.
Polar coordinates highlight seasonal dominance visually.
Color-coded by season for intuitive comparison.

### 6️⃣ Geospatial Distribution of Bloom Sites
🗺️ Type: Interactive Mapbox Scatter Plot (Plotly)
Maps all bloom sites using their latitude and longitude.
Circle size represents area coverage, while color indicates season.
Fully interactive: users can zoom, pan, and inspect bloom clusters geographically.

### 7️⃣ Bloom Area by Site (Ascending Order)
📈 Type: Horizontal Bar Chart (Plotly)
Ranks sites from smallest to largest bloom area.
Uses a blue color gradient to represent intensity of bloom coverage.
Top metrics displayed:
Total Bloom Area
Number of Unique Sites
Number of Flower Types

### 8️⃣ Bloom Area Trend & Prediction Analysis
**🤖 Type: Machine Learning (Random Forest Regression) + Seaborn Plot**
- Predicts bloom area trends by season using encoded categorical features.
- Visualizes actual vs. predicted bloom areas in a compact scatter-line plot.
- Includes a custom input predictor allowing users to forecast bloom area for chosen parameters.
- Model metrics: Mean Squared Error (MSE) and R² Score.

**🎨 Unified Design Theme**
- All visualizations share a consistent pastel color palette (shared_colors = sns.color_palette("Set2")).
- Smooth gradients, soft contrasts, and rounded card visuals maintain a cohesive aesthetic across dashboards.
- The theme blends analytical clarity with visual appeal, making data insights easy to interpret.

**🧭 Technologies Used**
- Python Libraries: Streamlit, Pandas, Plotly, Seaborn, Matplotlib, Scikit-learn, NumPy
- Frontend Styling: Streamlit Custom CSS
- Color Theme: Seaborn “Set2” palette converted to HEX for Plotly



## Bloom Area Trend & Prediction — Analysis Module
This module focuses on the analysis and prediction of California’s superbloom areas across different seasons.
It explores the cleaned dataset (flowerBlooms_Clean.csv) to identify seasonal bloom trends, evaluate multiple machine learning models, and visualize bloom predictions.

### 🔍 Objectives

Analyze seasonal variations in bloom areas.
Compare performance of different regression models.
Visualize actual vs predicted bloom patterns.
Build an interactive prediction feature for bloom area estimation.

### ⚙️ Technologies Used
Python Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
Machine Learning Models:
Linear Regression
Random Forest Regressor
XGBoost Regressor
Visualization Tool: Streamlit for interactive exploration

### 📈 Process Workflow
Data Preparation
Loaded and cleaned the dataset.
Encoded categorical features (Season, Site, Type, GeometryType).
Model Training & Evaluation
Split data into training and testing sets.
Trained three regression models.
Evaluated performance using Mean Squared Error (MSE) and R² Score.

#### Result Summary
- Model	MSE	R² Score
- Linear Regression	1.65e+10	-0.03
- Random Forest	1.66e+10	-0.03
- XGBoost	1.96e+10	-0.22
- Random Forest provided relatively stable results and was selected for visualization and prediction.

### 🌿 Visualization & Insights

A scatter plot with a trend line shows bloom area patterns across seasons.
Highest bloom area was observed around Fall, while Winter showed a decline.
Interactive dropdowns allow users to predict bloom area by selecting:
- Season
- Site
- Flower Type
- Geometry Type

#### AI Usage:
Utilized chatgpt for extracting demonstration and descriptions related to dataset cleaning, visuals selection and insights extraction.