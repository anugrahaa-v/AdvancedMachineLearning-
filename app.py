import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="California Housing ML",
    layout="wide"
)

st.title("California Housing Price Prediction")
st.write("Linear Regression, Lasso and Ridge Regression")


# =========================
# LOAD DATASET
# =========================

@st.cache_data
def load_dataset():

    df = pd.read_csv("housing.csv")

    # Remove unwanted spaces from column names
    df.columns = df.columns.str.strip()

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))

    return df


df = load_dataset()


# =========================
# TARGET COLUMN
# =========================

target = "median_house_value"


if target not in df.columns:

    st.error(
        "Target column 'median_house_value' was not found."
    )

    st.write("Columns found in your dataset:")
    st.write(df.columns.tolist())

    st.stop()


# =========================
# FEATURE ENGINEERING
# =========================

# Rooms per household
if "total_rooms" in df.columns and "households" in df.columns:

    df["rooms_per_household"] = (
        df["total_rooms"] /
        (df["households"] + 1)
    )


# Bedrooms per room
if "total_bedrooms" in df.columns and "total_rooms" in df.columns:

    df["bedrooms_per_room"] = (
        df["total_bedrooms"] /
        (df["total_rooms"] + 1)
    )


# Population per household
if "population" in df.columns and "households" in df.columns:

    df["population_per_household"] = (
        df["population"] /
        (df["households"] + 1)
    )


# Income per room
if "median_income" in df.columns and "total_rooms" in df.columns:

    df["income_per_room"] = (
        df["median_income"] /
        (df["total_rooms"] + 1)
    )


# =========================
# FEATURES AND TARGET
# =========================

X = df.drop(
    target,
    axis=1
)

y = df[target]


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# LINEAR REGRESSION
# =========================

linear_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LinearRegression()
    )
])

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)


# =========================
# LASSO REGRESSION
# =========================

lasso_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        Lasso(
            max_iter=10000
        )
    )
])

lasso_grid = GridSearchCV(
    lasso_model,

    {
        "model__alpha":
        np.logspace(
            -4,
            1,
            20
        )
    },

    cv=5,

    scoring="neg_mean_squared_error",

    n_jobs=-1
)

lasso_grid.fit(
    X_train,
    y_train
)

lasso_pred = lasso_grid.predict(
    X_test
)


# =========================
# RIDGE REGRESSION
# =========================

ridge_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        Ridge()
    )
])

ridge_grid = GridSearchCV(
    ridge_model,

    {
        "model__alpha":
        np.logspace(
            -4,
            4,
            20
        )
    },

    cv=5,

    scoring="neg_mean_squared_error",

    n_jobs=-1
)

ridge_grid.fit(
    X_train,
    y_train
)

ridge_pred = ridge_grid.predict(
    X_test
)


# =========================
# METRICS
# =========================

def get_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, mse, rmse, r2


linear_metrics = get_metrics(
    y_test,
    linear_pred
)

lasso_metrics = get_metrics(
    y_test,
    lasso_pred
)

ridge_metrics = get_metrics(
    y_test,
    ridge_pred
)


# =========================
# SIDEBAR
# =========================

st.sidebar.title("Menu")

option = st.sidebar.selectbox(
    "Choose Section",
    [
        "Dataset",
        "Model Performance",
        "Prediction",
        "Coefficients",
        "Alpha Analysis"
    ]
)


# =========================
# DATASET
# =========================

if option == "Dataset":

    st.header("Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Features",
        X.shape[1]
    )

    col3.metric(
        "Target",
        "median_house_value"
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader(
        "Dataset Dimensions"
    )

    st.write(
        f"Rows: {df.shape[0]}"
    )

    st.write(
        f"Columns: {df.shape[1]}"
    )

    st.subheader(
        "Missing Values"
    )

    st.dataframe(
        df.isnull().sum(),
        use_container_width=True
    )

    st.write(
        "Duplicate Records:",
        df.duplicated().sum()
    )

    st.subheader(
        "Descriptive Statistics"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# =========================
# MODEL PERFORMANCE
# =========================

elif option == "Model Performance":

    st.header(
        "Model Performance"
    )

    results = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Lasso Regression",
            "Ridge Regression"
        ],

        "MAE": [
            linear_metrics[0],
            lasso_metrics[0],
            ridge_metrics[0]
        ],

        "MSE": [
            linear_metrics[1],
            lasso_metrics[1],
            ridge_metrics[1]
        ],

        "RMSE": [
            linear_metrics[2],
            lasso_metrics[2],
            ridge_metrics[2]
        ],

        "R2": [
            linear_metrics[3],
            lasso_metrics[3],
            ridge_metrics[3]
        ]
    })

    st.dataframe(
        results.round(4),
        use_container_width=True
    )

    # Error comparison

    st.subheader(
        "Error Comparison"
    )

    fig, ax = plt.subplots()

    results.set_index(
        "Model"
    )[[
        "MAE",
        "RMSE"
    ]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel(
        "Error"
    )

    ax.set_title(
        "MAE and RMSE Comparison"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(fig)


    # R2

    st.subheader(
        "R² Score Comparison"
    )

    fig2, ax2 = plt.subplots()

    results.set_index(
        "Model"
    )["R2"].plot(
        kind="bar",
        ax=ax2
    )

    ax2.set_ylabel(
        "R² Score"
    )

    ax2.set_title(
        "R² Score"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(fig2)


    # Alpha

    col1, col2 = st.columns(2)

    col1.metric(
        "Best Lasso Alpha",
        f"{lasso_grid.best_params_['model__alpha']:.6f}"
    )

    col2.metric(
        "Best Ridge Alpha",
        f"{ridge_grid.best_params_['model__alpha']:.6f}"
    )


# =========================
# PREDICTION
# =========================

elif option == "Prediction":

    st.header(
        "House Value Prediction"
    )

    st.write(
        "Enter the required feature values."
    )

    values = {}

    for feature in X.columns:

        values[feature] = st.number_input(
            feature,
            value=float(
                X[feature].median()
            )
        )

    input_data = pd.DataFrame(
        [values]
    )

    model_name = st.selectbox(
        "Select Model",
        [
            "Linear Regression",
            "Lasso Regression",
            "Ridge Regression"
        ]
    )

    if st.button(
        "Predict House Value"
    ):

        if model_name == "Linear Regression":

            prediction = linear_model.predict(
                input_data
            )[0]

        elif model_name == "Lasso Regression":

            prediction = lasso_grid.predict(
                input_data
            )[0]

        else:

            prediction = ridge_grid.predict(
                input_data
            )[0]

        st.success(
            f"Predicted House Value: "
            f"${prediction:,.2f}"
        )


# =========================
# COEFFICIENTS
# =========================

elif option == "Coefficients":

    st.header(
        "Feature Coefficients"
    )

    linear_coef = (
        linear_model
        .named_steps["model"]
        .coef_
    )

    lasso_coef = (
        lasso_grid
        .best_estimator_
        .named_steps["model"]
        .coef_
    )

    ridge_coef = (
        ridge_grid
        .best_estimator_
        .named_steps["model"]
        .coef_
    )

    coef_df = pd.DataFrame({

        "Feature": X.columns,

        "Linear": linear_coef,

        "Lasso": lasso_coef,

        "Ridge": ridge_coef
    })

    st.dataframe(
        coef_df.round(4),
        use_container_width=True
    )

    st.subheader(
        "Coefficient Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    coef_df.set_index(
        "Feature"
    ).plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel(
        "Coefficient"
    )

    ax.set_title(
        "Linear vs Lasso vs Ridge Coefficients"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)


    # Zero coefficients

    zero_features = coef_df[
        np.isclose(
            coef_df["Lasso"],
            0
        )
    ]["Feature"].tolist()

    st.subheader(
        "Lasso Zero Coefficients"
    )

    if len(zero_features) > 0:

        for feature in zero_features:

            st.write(
                "•",
                feature
            )

    else:

        st.write(
            "No zero coefficients."
        )


# =========================
# ALPHA ANALYSIS
# =========================

elif option == "Alpha Analysis":

    st.header(
        "Grid Search Analysis"
    )


    # LASSO

    lasso_results = pd.DataFrame(
        lasso_grid.cv_results_
    )

    lasso_alpha = (
        lasso_results[
            "param_model__alpha"
        ]
        .astype(float)
    )

    lasso_error = -lasso_results[
        "mean_test_score"
    ]

    st.subheader(
        "Lasso Alpha vs CV Error"
    )

    fig, ax = plt.subplots()

    ax.semilogx(
        lasso_alpha,
        lasso_error,
        marker="o"
    )

    ax.set_xlabel(
        "Alpha"
    )

    ax.set_ylabel(
        "CV MSE"
    )

    ax.set_title(
        "Lasso Grid Search"
    )

    plt.tight_layout()

    st.pyplot(fig)


    # RIDGE

    ridge_results = pd.DataFrame(
        ridge_grid.cv_results_
    )

    ridge_alpha = (
        ridge_results[
            "param_model__alpha"
        ]
        .astype(float)
    )

    ridge_error = -ridge_results[
        "mean_test_score"
    ]

    st.subheader(
        "Ridge Alpha vs CV Error"
    )

    fig2, ax2 = plt.subplots()

    ax2.semilogx(
        ridge_alpha,
        ridge_error,
        marker="o"
    )

    ax2.set_xlabel(
        "Alpha"
    )

    ax2.set_ylabel(
        "CV MSE"
    )

    ax2.set_title(
        "Ridge Grid Search"
    )

    plt.tight_layout()

    st.pyplot(fig2)


    st.info(
        "Best Lasso Alpha: "
        + str(
            lasso_grid.best_params_[
                "model__alpha"
            ]
        )
    )

    st.info(
        "Best Ridge Alpha: "
        + str(
            ridge_grid.best_params_[
                "model__alpha"
            ]
        )
    )
