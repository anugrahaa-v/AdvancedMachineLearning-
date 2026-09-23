import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


st.set_page_config(
    page_title="California Housing ML",
    layout="wide"
)

st.title("California Housing Price Prediction")
st.write("Linear Regression, Lasso and Ridge Regression")


# Load Dataset
@st.cache_data
def load_dataset():

    data = fetch_california_housing(as_frame=True)

    df = data.frame.copy()

    df["RoomsPerPerson"] = (
        df["AveRooms"] / (df["Population"] + 1)
    )

    df["BedroomsPerRoom"] = (
        df["AveBedrms"] / (df["AveRooms"] + 1)
    )

    df["PopulationPerHousehold"] = (
        df["Population"] / (df["AveOccup"] + 1)
    )

    df["RoomsPerHousehold"] = (
        df["AveRooms"] / (df["AveOccup"] + 1)
    )

    df["IncomePerRoom"] = (
        df["MedInc"] / (df["AveRooms"] + 1)
    )

    return df


df = load_dataset()

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Linear Regression
@st.cache_resource
def train_linear():

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    model.fit(X_train, y_train)

    return model


linear_model = train_linear()

linear_pred = linear_model.predict(X_test)


# Lasso Regression
@st.cache_resource
def train_lasso():

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(max_iter=10000))
    ])

    params = {
        "model__alpha": np.logspace(-4, 1, 20)
    }

    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid


lasso_grid = train_lasso()

lasso_pred = lasso_grid.predict(X_test)


# Ridge Regression
@st.cache_resource
def train_ridge():

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])

    params = {
        "model__alpha": np.logspace(-4, 4, 20)
    }

    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid


ridge_grid = train_ridge()

ridge_pred = ridge_grid.predict(X_test)


# Metrics
def get_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = np.sqrt(mse)

    r2 = r2_score(actual, predicted)

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


# Sidebar
st.sidebar.title("Menu")

option = st.sidebar.selectbox(
    "Choose Section",
    [
        "Dataset",
        "Model Performance",
        "Prediction",
        "Coefficients",
        "Actual vs Predicted",
        "Alpha Analysis"
    ]
)


# Dataset
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
        "MedHouseVal"
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Missing Values")

    st.dataframe(
        df.isnull().sum(),
        use_container_width=True
    )

    st.write(
        "Duplicate Records:",
        df.duplicated().sum()
    )


# Model Performance
elif option == "Model Performance":

    st.header("Model Performance")

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

    st.subheader("MAE and RMSE Comparison")

    error_chart = results.set_index("Model")[
        ["MAE", "RMSE"]
    ]

    st.bar_chart(error_chart)

    st.subheader("R2 Score Comparison")

    r2_chart = results.set_index("Model")[
        ["R2"]
    ]

    st.bar_chart(r2_chart)

    st.subheader("Best Alpha Values")

    col1, col2 = st.columns(2)

    col1.metric(
        "Best Lasso Alpha",
        f"{lasso_grid.best_params_['model__alpha']:.6f}"
    )

    col2.metric(
        "Best Ridge Alpha",
        f"{ridge_grid.best_params_['model__alpha']:.6f}"
    )


# Prediction
elif option == "Prediction":

    st.header("House Value Prediction")

    st.write(
        "Enter the feature values below."
    )

    values = {}

    for feature in X.columns:

        values[feature] = st.number_input(
            feature,
            value=float(X[feature].median())
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

    if st.button("Predict House Value"):

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
            f"${prediction * 100000:,.2f}"
        )


# Coefficients
elif option == "Coefficients":

    st.header("Feature Coefficients")

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

    st.subheader("Coefficient Comparison")

    coefficient_chart = coef_df.set_index(
        "Feature"
    )[[
        "Linear",
        "Lasso",
        "Ridge"
    ]]

    st.bar_chart(coefficient_chart)

    zero_features = coef_df[
        np.isclose(coef_df["Lasso"], 0)
    ]["Feature"].tolist()

    st.subheader(
        "Features with Zero Lasso Coefficient"
    )

    if zero_features:

        for feature in zero_features:
            st.write("•", feature)

    else:

        st.write(
            "No coefficients became zero."
        )


# Actual vs Predicted
elif option == "Actual vs Predicted":

    st.header("Actual vs Predicted Values")

    prediction_df = pd.DataFrame({
        "Actual": y_test.values,
        "Lasso Predicted": lasso_pred,
        "Ridge Predicted": ridge_pred
    })

    st.subheader("Lasso Regression")

    st.line_chart(
        prediction_df[
            ["Actual", "Lasso Predicted"]
        ].head(100)
    )

    st.subheader("Ridge Regression")

    st.line_chart(
        prediction_df[
            ["Actual", "Ridge Predicted"]
        ].head(100)
    )

    st.dataframe(
        prediction_df.head(20).round(4),
        use_container_width=True
    )


# Alpha Analysis
elif option == "Alpha Analysis":

    st.header("Grid Search Alpha Analysis")

    # Lasso
    lasso_results = pd.DataFrame(
        lasso_grid.cv_results_
    )

    lasso_alpha = lasso_results[
        "param_model__alpha"
    ].astype(float)

    lasso_error = -lasso_results[
        "mean_test_score"
    ]

    lasso_chart = pd.DataFrame({
        "Alpha": lasso_alpha.values,
        "CV MSE": lasso_error.values
    })

    st.subheader(
        "Lasso Alpha vs Cross-Validation Error"
    )

    st.line_chart(
        lasso_chart.set_index("Alpha")
    )

    # Ridge
    ridge_results = pd.DataFrame(
        ridge_grid.cv_results_
    )

    ridge_alpha = ridge_results[
        "param_model__alpha"
    ].astype(float)

    ridge_error = -ridge_results[
        "mean_test_score"
    ]

    ridge_chart = pd.DataFrame({
        "Alpha": ridge_alpha.values,
        "CV MSE": ridge_error.values
    })

    st.subheader(
        "Ridge Alpha vs Cross-Validation Error"
    )

    st.line_chart(
        ridge_chart.set_index("Alpha")
    )

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
