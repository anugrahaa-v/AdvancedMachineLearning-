import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="California Housing ML",
    layout="wide"
)

st.title("California Housing Price Prediction")
st.write(
    "Linear Regression, Lasso and Ridge Regression "
    "with Cross-Validation and Grid Search"
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv("housing.csv")

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Replace infinity values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


df = load_dataset()


# =========================================================
# TARGET
# =========================================================

target = "median_house_value"


if target not in df.columns:

    st.error(
        "The target column 'median_house_value' was not found."
    )

    st.write("Columns found in housing.csv:")

    st.write(
        df.columns.tolist()
    )

    st.stop()


# =========================================================
# REMOVE DUPLICATES
# =========================================================

duplicate_count = df.duplicated().sum()


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


categorical_columns = df.select_dtypes(
    exclude=np.number
).columns

for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

# Rooms per household
if (
    "total_rooms" in df.columns
    and "households" in df.columns
):

    df["rooms_per_household"] = (
        df["total_rooms"]
        / (df["households"] + 1)
    )


# Bedrooms per room
if (
    "total_bedrooms" in df.columns
    and "total_rooms" in df.columns
):

    df["bedrooms_per_room"] = (
        df["total_bedrooms"]
        / (df["total_rooms"] + 1)
    )


# Population per household
if (
    "population" in df.columns
    and "households" in df.columns
):

    df["population_per_household"] = (
        df["population"]
        / (df["households"] + 1)
    )


# Income per room
if (
    "median_income" in df.columns
    and "total_rooms" in df.columns
):

    df["income_per_room"] = (
        df["median_income"]
        / (df["total_rooms"] + 1)
    )


# Replace infinity created by feature engineering
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill newly created missing values
for column in df.select_dtypes(
    include=np.number
).columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# =========================================================
# SEPARATE FEATURES AND TARGET
# =========================================================

y = pd.to_numeric(
    df[target],
    errors="coerce"
)

X = df.drop(
    target,
    axis=1
)


# =========================================================
# ENCODE CATEGORICAL VARIABLES
# =========================================================

X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=float
)


# Convert everything to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# Handle any remaining missing values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)


# Remove rows where target is missing
valid_rows = y.notna()

X = X.loc[
    valid_rows
]

y = y.loc[
    valid_rows
]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# LINEAR REGRESSION
# =========================================================

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


# =========================================================
# LASSO REGRESSION + GRID SEARCH
# =========================================================

lasso_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        Lasso(
            max_iter=20000
        )
    )
])


lasso_grid = GridSearchCV(
    estimator=lasso_model,

    param_grid={
        "model__alpha": np.logspace(
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


# =========================================================
# RIDGE REGRESSION + GRID SEARCH
# =========================================================

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
    estimator=ridge_model,

    param_grid={
        "model__alpha": np.logspace(
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


# =========================================================
# METRICS FUNCTION
# =========================================================

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

    rmse = np.sqrt(
        mse
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, mse, rmse, r2


# =========================================================
# CALCULATE METRICS
# =========================================================

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


# =========================================================
# SIDEBAR
# =========================================================

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


# =========================================================
# DATASET
# =========================================================

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
        "Rows:",
        df.shape[0]
    )

    st.write(
        "Columns:",
        df.shape[1]
    )


    st.subheader(
        "Missing Values"
    )

    missing_values = df.isnull().sum()

    st.dataframe(
        missing_values,
        use_container_width=True
    )


    st.subheader(
        "Duplicate Records"
    )

    st.write(
        duplicate_count
    )


    st.subheader(
        "Descriptive Statistics"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


    st.subheader(
        "Median Income vs House Value"
    )

    if (
        "median_income" in df.columns
        and target in df.columns
    ):

        fig, ax = plt.subplots()

        ax.scatter(
            df["median_income"],
            df[target],
            alpha=0.3
        )

        ax.set_xlabel(
            "Median Income"
        )

        ax.set_ylabel(
            "Median House Value"
        )

        ax.set_title(
            "Median Income vs House Value"
        )

        st.pyplot(fig)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

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
        "MAE and RMSE Comparison"
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
        "Model Error Comparison"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(fig)


    # R2 comparison

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
        "R² Score Comparison"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(fig2)


    # Alpha values

    col1, col2 = st.columns(2)

    col1.metric(
        "Best Lasso Alpha",
        f"{lasso_grid.best_params_['model__alpha']:.6f}"
    )

    col2.metric(
        "Best Ridge Alpha",
        f"{ridge_grid.best_params_['model__alpha']:.6f}"
    )


# =========================================================
# PREDICTION
# =========================================================

elif option == "Prediction":

    st.header(
        "House Value Prediction"
    )

    st.write(
        "Enter the feature values."
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


# =========================================================
# COEFFICIENTS
# =========================================================

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


    # Lasso zero coefficients

    zero_features = coef_df[
        np.isclose(
            coef_df["Lasso"],
            0
        )
    ]["Feature"].tolist()


    st.subheader(
        "Features with Zero Lasso Coefficients"
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


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

elif option == "Actual vs Predicted":

    st.header(
        "Actual vs Predicted Values"
    )


    # Lasso

    st.subheader(
        "Lasso Regression"
    )

    fig, ax = plt.subplots()

    ax.scatter(
        y_test,
        lasso_pred,
        alpha=0.4
    )

    minimum = min(
        y_test.min(),
        lasso_pred.min()
    )

    maximum = max(
        y_test.max(),
        lasso_pred.max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum]
    )

    ax.set_xlabel(
        "Actual Values"
    )

    ax.set_ylabel(
        "Predicted Values"
    )

    ax.set_title(
        "Lasso: Actual vs Predicted"
    )

    st.pyplot(fig)


    # Ridge

    st.subheader(
        "Ridge Regression"
    )

    fig2, ax2 = plt.subplots()

    ax2.scatter(
        y_test,
        ridge_pred,
        alpha=0.4
    )

    minimum2 = min(
        y_test.min(),
        ridge_pred.min()
    )

    maximum2 = max(
        y_test.max(),
        ridge_pred.max()
    )

    ax2.plot(
        [minimum2, maximum2],
        [minimum2, maximum2]
    )

    ax2.set_xlabel(
        "Actual Values"
    )

    ax2.set_ylabel(
        "Predicted Values"
    )

    ax2.set_title(
        "Ridge: Actual vs Predicted"
    )

    st.pyplot(fig2)


# =========================================================
# ALPHA ANALYSIS
# =========================================================

elif option == "Alpha Analysis":

    st.header(
        "Grid Search Analysis"
    )


    # -------------------------
    # LASSO
    # -------------------------

    st.subheader(
        "Lasso Alpha vs Cross-Validation Error"
    )


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


    # -------------------------
    # RIDGE
    # -------------------------

    st.subheader(
        "Ridge Alpha vs Cross-Validation Error"
    )


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


    # Best alpha

    st.success(
        "Best Lasso Alpha: "
        + str(
            lasso_grid.best_params_[
                "model__alpha"
            ]
        )
    )


    st.success(
        "Best Ridge Alpha: "
        + str(
            ridge_grid.best_params_[
                "model__alpha"
            ]
        )
    )
