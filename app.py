
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="House Price EDA Dashboard",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 House Price Analysis Dashboard")
st.write(
    "Exploratory Data Analysis of the Ames Housing Dataset"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset Overview",
        "Price Analysis",
        "Correlation Analysis",
        "Neighborhood Analysis"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Upload your dataset below")

uploaded_file = st.sidebar.file_uploader(
    "Upload train.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if page == "Home":

        st.header("Project Overview")

        st.write(
            "This project explores house prices "
            "using Python and Exploratory Data Analysis."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    elif page == "Dataset Overview":

        st.header("Dataset Overview")

        st.write("Dataset Shape:", df.shape)

        st.subheader("Column Information")
        st.dataframe(
            pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isnull().sum().values
            })
        )

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

    elif page == "Price Analysis":

        st.header("House Price Distribution")

        if "SalePrice" in df.columns:

            fig, ax = plt.subplots()
            sns.histplot(
                df["SalePrice"].dropna(),
                kde=True,
                ax=ax
            )
            ax.set_title("Sale Price Distribution")
            ax.set_xlabel("Sale Price")
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            sns.boxplot(x=df["SalePrice"], ax=ax)
            ax.set_title("Sale Price Boxplot")
            st.pyplot(fig)
            plt.close(fig)

            st.write(
                "Mean Sale Price:",
                df["SalePrice"].mean()
            )
            st.write(
                "Median Sale Price:",
                df["SalePrice"].median()
            )

        else:
            st.error("SalePrice column not found.")

    elif page == "Correlation Analysis":

        st.header("Correlation Analysis")

        numeric_df = df.select_dtypes(
            include=np.number
        )

        if "SalePrice" in numeric_df.columns:

            correlations = (
                numeric_df.corr()["SalePrice"]
                .drop("SalePrice")
                .abs()
                .sort_values(ascending=False)
            )

            st.subheader("Top Correlated Features")
            st.dataframe(correlations.head(10))

            top_features = (
                correlations.head(10).index.tolist()
                + ["SalePrice"]
            )

            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(
                numeric_df[top_features].corr(),
                annot=True,
                cmap="coolwarm",
                center=0,
                ax=ax
            )
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
            plt.close(fig)

    elif page == "Neighborhood Analysis":

        st.header("Neighborhood Price Comparison")

        if {"Neighborhood", "SalePrice"}.issubset(df.columns):

            neighborhood_prices = (
                df.groupby("Neighborhood")["SalePrice"]
                .median()
                .sort_values(ascending=False)
            )

            st.dataframe(neighborhood_prices)

            fig, ax = plt.subplots(figsize=(12, 6))
            neighborhood_prices.plot(
                kind="bar",
                ax=ax
            )
            ax.set_title("Median House Price by Neighborhood")
            ax.set_xlabel("Neighborhood")
            ax.set_ylabel("Median Sale Price")
            plt.xticks(rotation=90)
            st.pyplot(fig)
            plt.close(fig)

else:

    st.info(
        "Please upload the train.csv dataset "
        "using the sidebar to explore the dashboard."
    )

    st.subheader("What this dashboard offers")

    st.write("- Dataset exploration")
    st.write("- House price distribution")
    st.write("- Correlation analysis")
    st.write("- Neighborhood comparison")

    st.caption("Built using Python, Pandas, Matplotlib and Seaborn.")
