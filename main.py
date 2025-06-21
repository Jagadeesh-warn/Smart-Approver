import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("loan-train.csv")
    return df

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Preprocessing & Data Info", "Model & Accuracy", "Visualizations", "Make Prediction"])

# ---------------------------------------
# Page 1: Preprocessing and Data Info
# ---------------------------------------
if page == "Preprocessing & Data Info":
    st.title("📊 Data Preprocessing & Information")

    st.write("### First 5 Rows of Dataset")
    st.write(df.head())

    st.write("### Data Description")
    st.write(df.describe(include='all'))

    st.write("### Missing Values")
    st.write(df.isnull().sum())

# ---------------------------------------
# Page 2: Model & Accuracy
# ---------------------------------------
elif page == "Model & Accuracy":
    st.title("🧠 Model Training & Accuracy")

    # Drop rows with missing values for simplicity
    df_clean = df.dropna()

    # Sidebar: Select Target and Features
    target_col = st.sidebar.selectbox("Select Target Column", df_clean.columns)
    features = st.sidebar.multiselect("Select Feature Columns", df_clean.columns.drop(target_col))

    if features:
        X = df_clean[features]
        y = df_clean[target_col]

        # Convert categorical to numeric
        X = pd.get_dummies(X)
        y = pd.factorize(y)[0]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"### ✅ Accuracy: {acc:.2f}")
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))
    else:
        st.warning("Please select at least one feature column.")

# ---------------------------------------
# Page 3: Display Visualizations
# ---------------------------------------

elif page == "Visualizations":
    st.title("📈 Interactive Visualizations")

    # Interactive categorical count plots
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        st.write("### 🟦 Categorical Feature Distributions")
        for col in categorical_cols:
            fig = px.histogram(df, x=col, color=col,
                               title=f"Distribution of {col}",
                               color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig)

    # Interactive correlation heatmap
    st.write("### 🔥 Correlation Heatmap (Numerical Columns Only)")
    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()

    heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Turbo',
        hoverongaps=False,
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation")
    ))
    heatmap.update_layout(title="Correlation Matrix Heatmap",
                          margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(heatmap)

    # Optional: Scatter matrix
    st.write("### ✨ Scatter Matrix (Numerical Columns)")
    fig_matrix = px.scatter_matrix(df, dimensions=numeric_df.columns,
                                   color=df[categorical_cols[0]] if categorical_cols else None,
                                   title="Scatter Matrix of Numerical Features")
    st.plotly_chart(fig_matrix)


# ---------------------------------------
# Page 4: Make Prediction
# ---------------------------------------
elif page == "Make Prediction":
    st.title("🎯 Predict on New Data")

    df_clean = df.dropna()
    target_col = st.sidebar.selectbox("Select Target Column", df_clean.columns, key="predict_target")
    features = st.sidebar.multiselect("Select Feature Columns", df_clean.columns.drop(target_col), key="predict_features")

    if features:
        input_data = {}
        for col in features:
            if df[col].dtype == 'object':
                input_data[col] = st.selectbox(f"Select {col}", df[col].dropna().unique())
            else:
                input_data[col] = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()))

        X = df_clean[features]
        y = df_clean[target_col]

        X = pd.get_dummies(X)
        y_encoded, uniques = pd.factorize(y)

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        model = RandomForestClassifier()
        model.fit(X, y_encoded)

        prediction = model.predict(input_df)[0]
        st.success(f"🎉 Prediction: {uniques[prediction]}")
    else:
        st.warning("Please select at least one feature column.")
