import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Household Power Consumption Analysis",
    page_icon="⚡",
    layout="wide"
)

# ============================================================
# MATPLOTLIB / SEABORN STYLE
# ============================================================

sns.set_theme(style="whitegrid")

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "#f8f9fa"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 11


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(filepath):

    try:

        # Household Power Consumption dataset is commonly
        # stored with semicolon separator.
        df = pd.read_csv(
            filepath,
            sep=None,
            engine="python"
        )

        # Remove unnecessary spaces from column names
        df.columns = df.columns.str.strip()

        # Numerical columns
        numerical_cols = [
            "Global_active_power",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3"
        ]

        # Convert numerical columns
        for col in numerical_cols:

            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        # Create DateTime
        if "Date" in df.columns and "Time" in df.columns:

            df["DateTime"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                format="%d/%m/%Y %H:%M:%S",
                errors="coerce"
            )

            # Sort data
            df = df.sort_values("DateTime")

            # Time-based features
            df["Year"] = df["DateTime"].dt.year
            df["Month"] = df["DateTime"].dt.month
            df["Day"] = df["DateTime"].dt.day
            df["Hour"] = df["DateTime"].dt.hour
            df["DayOfWeek"] = df["DateTime"].dt.day_name()
            df["Date_only"] = df["DateTime"].dt.date

        return df.reset_index(drop=True)

    except Exception as e:

        st.error(f"Error loading dataset: {e}")
        return None


# ============================================================
# COLUMN INFORMATION
# ============================================================

def get_column_types(df):

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Remove time features
    numeric_cols = [
        col for col in numeric_cols
        if col not in ["Year", "Month", "Day", "Hour"]
    ]

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns.tolist()

    categorical_cols = [
        col for col in categorical_cols
        if col not in ["Date", "Time"]
    ]

    return numeric_cols, categorical_cols


# ============================================================
# INTRODUCTION
# ============================================================

def show_introduction(df):

    st.header("📌 Introduction")

    st.subheader(
        "⚡ Household Electric Power Consumption Analysis"
    )

    st.write(
        """
        This dashboard presents a comprehensive exploratory
        data analysis of individual household electric power
        consumption.

        The dataset contains minute-level measurements of
        electricity consumption, voltage, current intensity,
        reactive power and three different sub-metering systems.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------

    st.subheader("📊 Dataset Overview")

    numeric_cols, categorical_cols = get_column_types(df)

    total_rows = len(df)
    total_columns = df.shape[1]
    total_missing = df.isnull().sum().sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📈 Total Records",
            f"{total_rows:,}"
        )

    with col2:
        st.metric(
            "📋 Total Columns",
            total_columns
        )

    with col3:
        st.metric(
            "🔢 Numerical Features",
            len(numeric_cols)
        )

    with col4:
        st.metric(
            "⚠️ Missing Values",
            f"{total_missing:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # DATE RANGE
    # --------------------------------------------------------

    if "DateTime" in df.columns:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "📅 Start Date",
                df["DateTime"].min().strftime(
                    "%d-%m-%Y %H:%M"
                )
            )

        with col2:
            st.metric(
                "📅 End Date",
                df["DateTime"].max().strftime(
                    "%d-%m-%Y %H:%M"
                )
            )

    st.divider()

    # --------------------------------------------------------
    # FEATURE INFORMATION
    # --------------------------------------------------------

    st.subheader("🔍 Feature Information")

    feature_info = []

    for col in df.columns:

        feature_info.append({
            "Feature": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Values": df[col].notna().sum(),
            "Missing Values": df[col].isna().sum(),
            "Unique Values": df[col].nunique()
        })

    feature_df = pd.DataFrame(feature_info)

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # FEATURE DESCRIPTION
    # --------------------------------------------------------

    st.subheader("📝 Feature Descriptions")

    descriptions = {

        "Global_active_power":
            "Household global minute-averaged active power (kW).",

        "Global_reactive_power":
            "Household global minute-averaged reactive power (kW).",

        "Voltage":
            "Minute-averaged voltage (V).",

        "Global_intensity":
            "Household global minute-averaged current intensity (A).",

        "Sub_metering_1":
            "Energy sub-metering 1, mainly kitchen appliances (Wh).",

        "Sub_metering_2":
            "Energy sub-metering 2, mainly laundry appliances (Wh).",

        "Sub_metering_3":
            "Energy sub-metering 3, mainly water-heater / AC (Wh)."
    }

    for feature, description in descriptions.items():

        if feature in df.columns:

            st.write(
                f"**{feature}:** {description}"
            )


# ============================================================
# DATA QUALITY ANALYSIS
# ============================================================

def show_data_quality(df):

    st.subheader(
        "🧹 Data Quality Analysis"
    )

    numeric_cols, _ = get_column_types(df)

    # --------------------------------------------------------
    # MISSING VALUES TABLE
    # --------------------------------------------------------

    missing_df = pd.DataFrame({

        "Feature": numeric_cols,

        "Missing Count": [
            df[col].isnull().sum()
            for col in numeric_cols
        ],

        "Missing Percentage": [
            round(
                df[col].isnull().mean() * 100,
                2
            )
            for col in numeric_cols
        ]
    })

    st.write("### Missing Values")

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MISSING VALUE CHART
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    missing_percentage = [
        df[col].isnull().mean() * 100
        for col in numeric_cols
    ]

    ax.barh(
        numeric_cols,
        missing_percentage
    )

    ax.set_xlabel(
        "Missing Values (%)"
    )

    ax.set_title(
        "Missing Values Distribution"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close()

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Duplicate Rows",
            f"{duplicate_count:,}"
        )

    with col2:

        st.metric(
            "Total Missing Values",
            f"{df.isnull().sum().sum():,}"
        )


# ============================================================
# UNIVARIATE ANALYSIS
# ============================================================

def show_univariate(df):

    st.subheader(
        "📊 Univariate Analysis"
    )

    numeric_cols, _ = get_column_types(df)

    # --------------------------------------------------------
    # HISTOGRAMS
    # --------------------------------------------------------

    st.write(
        "### Distribution of Numerical Features"
    )

    selected_cols = numeric_cols[:6]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 10)
    )

    axes = axes.flatten()

    for i, col in enumerate(selected_cols):

        data = df[col].dropna()

        axes[i].hist(
            data,
            bins=50,
            edgecolor="black",
            alpha=0.75
        )

        axes[i].set_title(
            col
        )

        axes[i].set_xlabel(
            "Value"
        )

        axes[i].set_ylabel(
            "Frequency"
        )

        mean_value = data.mean()

        median_value = data.median()

        axes[i].axvline(
            mean_value,
            linestyle="--",
            linewidth=2,
            label=f"Mean = {mean_value:.2f}"
        )

        axes[i].axvline(
            median_value,
            linestyle="--",
            linewidth=2,
            label=f"Median = {median_value:.2f}"
        )

        axes[i].legend()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close()

    # --------------------------------------------------------
    # BOX PLOTS
    # --------------------------------------------------------

    st.write(
        "### Box Plot / Outlier Detection"
    )

    box_cols = [
        col for col in [
            "Global_active_power",
            "Voltage",
            "Global_intensity"
        ]
        if col in df.columns
    ]

    fig, axes = plt.subplots(
        1,
        len(box_cols),
        figsize=(14, 5)
    )

    if len(box_cols) == 1:
        axes = [axes]

    for i, col in enumerate(box_cols):

        axes[i].boxplot(
            df[col].dropna()
        )

        axes[i].set_title(
            col
        )

        axes[i].set_ylabel(
            "Value"
        )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close()

    # --------------------------------------------------------
    # STATISTICAL SUMMARY
    # --------------------------------------------------------

    st.write(
        "### Statistical Summary"
    )

    stats = df[numeric_cols].describe().T

    stats["Skewness"] = [
        df[col].skew()
        for col in numeric_cols
    ]

    stats["Kurtosis"] = [
        df[col].kurt()
        for col in numeric_cols
    ]

    st.dataframe(
        stats,
        use_container_width=True
    )


# ============================================================
# BIVARIATE ANALYSIS
# ============================================================

def show_bivariate(df):

    st.subheader(
        "🔗 Bivariate Analysis"
    )

    numeric_cols, _ = get_column_types(df)

    # --------------------------------------------------------
    # CORRELATION MATRIX
    # --------------------------------------------------------

    st.write(
        "### Correlation Matrix"
    )

    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        ax=ax
    )

    ax.set_title(
        "Correlation Matrix"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close()

    # --------------------------------------------------------
    # STRONGEST CORRELATIONS
    # --------------------------------------------------------

    st.write(
        "### Strongest Correlations"
    )

    pairs = []

    for i in range(
        len(corr_matrix.columns)
    ):

        for j in range(
            i + 1,
            len(corr_matrix.columns)
        ):

            pairs.append({

                "Feature 1":
                    corr_matrix.columns[i],

                "Feature 2":
                    corr_matrix.columns[j],

                "Correlation":
                    corr_matrix.iloc[i, j]
            })

    pairs_df = pd.DataFrame(
        pairs
    )

    pairs_df["Absolute Correlation"] = (
        pairs_df["Correlation"].abs()
    )

    pairs_df = (
        pairs_df
        .sort_values(
            "Absolute Correlation",
            ascending=False
        )
        .head(10)
        .drop(
            columns="Absolute Correlation"
        )
    )

    st.dataframe(
        pairs_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # SCATTER PLOT
    # --------------------------------------------------------

    st.write(
        "### Global Active Power vs Global Intensity"
    )

    if (
        "Global_active_power" in df.columns
        and
        "Global_intensity" in df.columns
    ):

        plot_data = df[
            [
                "Global_intensity",
                "Global_active_power"
            ]
        ].dropna()

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.scatterplot(
            data=plot_data,
            x="Global_intensity",
            y="Global_active_power",
            alpha=0.3,
            s=15,
            ax=ax
        )

        ax.set_title(
            "Global Active Power vs Global Intensity"
        )

        ax.set_xlabel(
            "Global Intensity (A)"
        )

        ax.set_ylabel(
            "Global Active Power (kW)"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()


# ============================================================
# MULTIVARIATE ANALYSIS
# ============================================================

def show_multivariate(df):

    st.subheader(
        "📈 Multivariate Analysis"
    )

    # --------------------------------------------------------
    # HOURLY ANALYSIS
    # --------------------------------------------------------

    st.write(
        "### Average Power Consumption by Hour"
    )

    required_cols = [
        "Hour",
        "Global_active_power",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    if all(
        col in df.columns
        for col in required_cols
    ):

        hourly = df.groupby(
            "Hour"
        )[
            [
                "Global_active_power",
                "Sub_metering_1",
                "Sub_metering_2",
                "Sub_metering_3"
            ]
        ].mean()

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.plot(
            hourly.index,
            hourly["Global_active_power"],
            marker="o",
            label="Global Active Power"
        )

        ax.plot(
            hourly.index,
            hourly["Sub_metering_1"] / 100,
            marker="s",
            label="Sub-metering 1"
        )

        ax.plot(
            hourly.index,
            hourly["Sub_metering_2"] / 100,
            marker="^",
            label="Sub-metering 2"
        )

        ax.plot(
            hourly.index,
            hourly["Sub_metering_3"] / 100,
            marker="d",
            label="Sub-metering 3"
        )

        ax.set_xlabel(
            "Hour of Day"
        )

        ax.set_ylabel(
            "Average Consumption"
        )

        ax.set_title(
            "Average Power Consumption by Hour"
        )

        ax.set_xticks(
            range(24)
        )

        ax.legend()

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()

    # --------------------------------------------------------
    # MONTHLY ANALYSIS
    # --------------------------------------------------------

    st.write(
        "### Monthly Power Consumption"
    )

    if "Month" in df.columns:

        monthly = df.groupby(
            "Month"
        )[
            [
                "Global_active_power",
                "Voltage",
                "Global_intensity"
            ]
        ].mean()

        month_names = [
            "Jan", "Feb", "Mar",
            "Apr", "May", "Jun",
            "Jul", "Aug", "Sep",
            "Oct", "Nov", "Dec"
        ]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16, 5)
        )

        axes[0].bar(
            monthly.index,
            monthly["Global_active_power"]
        )

        axes[0].set_title(
            "Average Active Power"
        )

        axes[0].set_xlabel(
            "Month"
        )

        axes[0].set_ylabel(
            "Power (kW)"
        )

        axes[1].bar(
            monthly.index,
            monthly["Voltage"]
        )

        axes[1].set_title(
            "Average Voltage"
        )

        axes[1].set_xlabel(
            "Month"
        )

        axes[1].set_ylabel(
            "Voltage (V)"
        )

        axes[2].bar(
            monthly.index,
            monthly["Global_intensity"]
        )

        axes[2].set_title(
            "Average Global Intensity"
        )

        axes[2].set_xlabel(
            "Month"
        )

        axes[2].set_ylabel(
            "Intensity (A)"
        )

        for ax in axes:

            ax.set_xticks(
                range(1, 13)
            )

            ax.set_xticklabels(
                month_names,
                rotation=45
            )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()

    # --------------------------------------------------------
    # DAY OF WEEK
    # --------------------------------------------------------

    st.write(
        "### Power Consumption by Day of Week"
    )

    if "DayOfWeek" in df.columns:

        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        weekly = df.groupby(
            "DayOfWeek"
        )[
            [
                "Global_active_power",
                "Global_intensity"
            ]
        ].mean()

        weekly = weekly.reindex(
            day_order
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        x = np.arange(
            len(day_order)
        )

        width = 0.35

        ax.bar(
            x - width / 2,
            weekly["Global_active_power"],
            width,
            label="Global Active Power"
        )

        ax.bar(
            x + width / 2,
            weekly["Global_intensity"] / 10,
            width,
            label="Global Intensity / 10"
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            day_order,
            rotation=45
        )

        ax.set_xlabel(
            "Day of Week"
        )

        ax.set_ylabel(
            "Average Consumption"
        )

        ax.set_title(
            "Power Consumption by Day of Week"
        )

        ax.legend()

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

def show_outliers(df):

    st.subheader(
        "⚠️ Outlier Analysis"
    )

    numeric_cols, _ = get_column_types(df)

    outlier_data = []

    for col in numeric_cols:

        data = df[col].dropna()

        Q1 = data.quantile(0.25)

        Q3 = data.quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR

        upper = Q3 + 1.5 * IQR

        outliers = data[
            (data < lower) |
            (data > upper)
        ]

        outlier_data.append({

            "Feature": col,

            "Outlier Count":
                len(outliers),

            "Outlier Percentage":
                round(
                    len(outliers) /
                    len(data) * 100,
                    2
                ),

            "Lower Bound":
                round(lower, 2),

            "Upper Bound":
                round(upper, 2)
        })

    outlier_df = pd.DataFrame(
        outlier_data
    )

    st.dataframe(
        outlier_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # BOXPLOTS
    # --------------------------------------------------------

    st.write(
        "### Box Plots"
    )

    selected_cols = numeric_cols[:6]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8)
    )

    axes = axes.flatten()

    for i, col in enumerate(selected_cols):

        data = df[col].dropna()

        axes[i].boxplot(
            data
        )

        axes[i].set_title(
            col
        )

        axes[i].set_ylabel(
            "Value"
        )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close()


# ============================================================
# EDA SECTION
# ============================================================

def show_eda(df):

    st.header(
        "📊 Exploratory Data Analysis"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🧹 Data Quality",
            "📊 Univariate",
            "🔗 Bivariate",
            "📈 Multivariate",
            "⚠️ Outliers"
        ]
    )

    with tab1:
        show_data_quality(df)

    with tab2:
        show_univariate(df)

    with tab3:
        show_bivariate(df)

    with tab4:
        show_multivariate(df)

    with tab5:
        show_outliers(df)


# ============================================================
# CONCLUSION
# ============================================================

def show_conclusion(df):

    st.header(
        "✅ Conclusion"
    )

    # --------------------------------------------------------
    # POWER STATISTICS
    # --------------------------------------------------------

    avg_power = df[
        "Global_active_power"
    ].mean()

    max_power = df[
        "Global_active_power"
    ].max()

    min_power = df[
        "Global_active_power"
    ].min()

    st.subheader(
        "🔍 Major Findings"
    )

    st.write(
        f"""
        **1. Household Energy Usage**

        The average global active power consumption is
        **{avg_power:.2f} kW**, while the maximum recorded
        consumption is **{max_power:.2f} kW** and the minimum
        is **{min_power:.2f} kW**.
        """
    )

    # --------------------------------------------------------
    # SUB-METERING
    # --------------------------------------------------------

    sub1 = df[
        "Sub_metering_1"
    ].mean()

    sub2 = df[
        "Sub_metering_2"
    ].mean()

    sub3 = df[
        "Sub_metering_3"
    ].mean()

    total = sub1 + sub2 + sub3

    if total > 0:

        p1 = sub1 / total * 100
        p2 = sub2 / total * 100
        p3 = sub3 / total * 100

        st.write(
            f"""
            **2. Circuit-Level Consumption**

            Sub-metering 1 contributes approximately
            **{p1:.1f}%**, Sub-metering 2 contributes
            **{p2:.1f}%**, and Sub-metering 3 contributes
            **{p3:.1f}%** of the average metered consumption.
            """
        )

    # --------------------------------------------------------
    # VOLTAGE
    # --------------------------------------------------------

    avg_voltage = df[
        "Voltage"
    ].mean()

    voltage_std = df[
        "Voltage"
    ].std()

    st.write(
        f"""
        **3. Voltage Stability**

        The average voltage is approximately
        **{avg_voltage:.2f} V**, with a standard deviation
        of **{voltage_std:.2f} V**.
        """
    )

    # --------------------------------------------------------
    # HOURLY PATTERN
    # --------------------------------------------------------

    hourly = df.groupby(
        "Hour"
    )["Global_active_power"].mean()

    peak_hour = hourly.idxmax()

    lowest_hour = hourly.idxmin()

    st.write(
        f"""
        **4. Daily Consumption Pattern**

        The highest average power consumption occurs around
        **{peak_hour}:00**, while the lowest average
        consumption occurs around **{lowest_hour}:00**.
        """
    )

    # --------------------------------------------------------
    # MONTHLY PATTERN
    # --------------------------------------------------------

    monthly = df.groupby(
        "Month"
    )["Global_active_power"].mean()

    highest_month = monthly.idxmax()

    lowest_month = monthly.idxmin()

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    st.write(
        f"""
        **5. Seasonal Pattern**

        The highest average monthly consumption occurs in
        **{month_names[highest_month - 1]}**, while the lowest
        occurs in **{month_names[lowest_month - 1]}**.
        """
    )

    # --------------------------------------------------------
    # MACHINE LEARNING INSIGHTS
    # --------------------------------------------------------

    st.subheader(
        "💡 Data Science Insights"
    )

    st.write(
        """
        **Feature Engineering**

        Hour, day, month and day-of-week features can be
        useful for predicting future electricity consumption.
        """
    )

    st.write(
        """
        **Predictive Modeling**

        This dataset can be used for machine learning and
        time-series forecasting models such as Linear
        Regression, Random Forest, XGBoost, ARIMA and LSTM.
        """
    )

    st.write(
        """
        **Anomaly Detection**

        Outliers can be investigated to identify unusual
        electricity consumption or possible equipment issues.
        """
    )

    st.write(
        """
        **Energy Optimization**

        Hourly and circuit-level patterns can help identify
        periods of high electricity usage and potential
        energy-saving opportunities.
        """
    )

    # --------------------------------------------------------
    # FINAL CONCLUSION
    # --------------------------------------------------------

    st.subheader(
        "🎯 Final Conclusion"
    )

    st.info(
        """
        The Household Electric Power Consumption dataset
        provides valuable information about residential
        electricity usage.

        The analysis reveals clear daily, weekly and
        monthly consumption patterns. The strong relationship
        between active power and current intensity demonstrates
        the usefulness of electrical measurements for
        understanding household energy behavior.

        The dataset is suitable for advanced data science
        applications including:

        • Energy consumption forecasting

        • Anomaly detection

        • Load prediction

        • Energy optimization

        • Smart-home applications

        • Demand-side management

        • Machine learning based prediction
        """,
        icon="⚡"
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # CHANGE THIS FILE NAME IF REQUIRED
    # --------------------------------------------------------

    file_path = "household_power_consumption.csv"

    # Load dataset
    df = load_data(file_path)

    if df is None or df.empty:

        st.error(
            "Dataset could not be loaded."
        )

        st.info(
            "Make sure household_power_consumption.csv "
            "is in the same folder as app.py."
        )

        return

    # --------------------------------------------------------
    # MAIN TITLE
    # --------------------------------------------------------

    st.title(
        "⚡ Household Electric Power Consumption Analysis"
    )

    st.write(
        "Comprehensive Data Analysis Dashboard using "
        "Python, Pandas, NumPy, Matplotlib, Seaborn and Streamlit."
    )

    st.divider()

    # --------------------------------------------------------
    # THREE MAIN SECTIONS
    # --------------------------------------------------------

    introduction = st.expander(
        "📌 1. Introduction",
        expanded=True
    )

    with introduction:
        show_introduction(df)

    st.divider()

    eda = st.expander(
        "📊 2. Exploratory Data Analysis",
        expanded=False
    )

    with eda:
        show_eda(df)

    st.divider()

    conclusion = st.expander(
        "✅ 3. Conclusion",
        expanded=False
    )

    with conclusion:
        show_conclusion(df)

    st.divider()

    st.caption(
        "Household Power Consumption Analysis | "
        "Python + Streamlit + Pandas + NumPy + "
        "Matplotlib + Seaborn"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()