import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="🚗 Data driven mobility solutions", layout="wide")
st.title("🚗 Data driven mobility solutions")

# Load and cache data
@st.cache_data
def load_data():
    # Load everything as string to avoid PyArrow errors
    df = pd.read_csv("Automobile.csv", dtype=str)
    df.columns = df.columns.str.strip().str.lower()

    # Try converting numeric columns back to numbers
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

# Load dataset
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ 'Automobile.csv' file not found.")
    st.stop()

# Show raw data
if st.checkbox("📂 Show raw data"):
    st.dataframe(df)

# Dataset overview
st.subheader("📊 Dataset Overview")
st.write("Shape:", df.shape)
st.write("Columns:", list(df.columns))
st.dataframe(df.describe(include='all'))

# Handle missing values
st.subheader("🛠️ Missing Value Handling")
missing = df.isnull().sum()
st.write(missing[missing > 0])

action = st.selectbox("Action for missing values", ["Do nothing", "Drop rows", "Fill with mean (numeric only)"])
if action == "Drop rows":
    df.dropna(inplace=True)
    st.success("✅ Dropped rows with missing values.")
elif action == "Fill with mean (numeric only)":
    df.fillna(df.mean(numeric_only=True), inplace=True)
    st.success("✅ Filled missing numeric values with mean.")

# Filter by car make
st.subheader("🔍 Filter by Make")

# Try to detect column name
make_col = None
for candidate in ["make", "company", "brand", "manufacturer"]:
    if candidate in df.columns:
        make_col = candidate
        break

if make_col is None:
    st.warning("⚠️ No 'make' column found. Using all data without filtering by car make.")
    filtered_df = df.copy()
else:
    makes = df[make_col].dropna().unique()
    selected_makes = st.multiselect("Choose car makes", sorted(makes), default=sorted(makes)[:3])
    filtered_df = df[df[make_col].isin(selected_makes)]
    st.write(f"Filtered dataset ({len(filtered_df)} rows):")
    st.dataframe(filtered_df)

# Download filtered data
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Filtered Data", data=csv, file_name='filtered_automobile.csv', mime='text/csv')

# Scatter plot
st.subheader("📈 Scatter Plot")
numeric_cols = df.select_dtypes(include='number').columns
if len(numeric_cols) >= 2:
    x_col = st.selectbox("X-axis", numeric_cols)
    y_col = st.selectbox("Y-axis", numeric_cols, index=1)

    fig1, ax1 = plt.subplots()
    if make_col:
        sns.scatterplot(data=filtered_df, x=x_col, y=y_col, hue=make_col, ax=ax1)
    else:
        sns.scatterplot(data=filtered_df, x=x_col, y=y_col, ax=ax1)
    plt.title(f"{y_col} vs {x_col}")
    st.pyplot(fig1)
else:
    st.warning("⚠️ Not enough numeric columns to plot scatter plot.")

# Correlation heatmap
if st.checkbox("🔁 Show Correlation Heatmap"):
    corr = df[numeric_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

# Histogram
st.subheader("📊 Histogram")
if len(numeric_cols) > 0:
    hist_col = st.selectbox("Select a column to plot histogram", numeric_cols)
    fig3, ax3 = plt.subplots()
    sns.histplot(data=filtered_df, x=hist_col, kde=True, ax=ax3)  # ✅ Fixed
    plt.title(f"Distribution of {hist_col}")
    st.pyplot(fig3)
else:
    st.warning("⚠️ No numeric columns available for histogram.")

# Boxplot
st.subheader("📦 Boxplot by Category")
cat_cols = df.select_dtypes(include='object').columns
if len(cat_cols) > 0 and len(numeric_cols) > 0:
    box_cat = st.selectbox("Select a categorical column", cat_cols)
    box_num = st.selectbox("Select a numeric column", numeric_cols)

    fig4, ax4 = plt.subplots()
    sns.boxplot(data=filtered_df, x=box_cat, y=box_num, ax=ax4)
    plt.xticks(rotation=45)
    plt.title(f"{box_num} by {box_cat}")
    st.pyplot(fig4)
else:
    st.warning("⚠️ Not enough categorical/numeric columns for boxplot.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit -> by Rishi")
