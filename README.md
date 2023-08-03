# Satellite Telemetry Analyzer

The Satellite Telemetry Analyzer is a Python application designed to process, visualize, analyze, and model satellite telemetry data. It provides a comprehensive set of functionalities, including data preprocessing, visualization, outlier detection, correlation analysis, modeling, and more.

## Table of Contents

1. [Requirements](#requirements)
2. [Usage](#usage)
3. [Functionality](#functionality)
   - [Data Loading](#data-loading)
   - [Data Preprocessing](#data-preprocessing)
   - [Data Visualization](#data-visualization)
   - [Missing Value Analysis](#missing-value-analysis)
   - [Outlier Analysis](#outlier-analysis)
   - [Correlation Analysis](#correlation-analysis)
   - [Model Fitting](#model-fitting)
   - [Result Export](#result-export)
   - [Variance Inflation Factor Analysis](#variance-inflation-factor-analysis)
   - [Spinning Classification](#spinning-classification)
   - [ROC Curve Plotting](#roc-curve-plotting)
   - [Scaled Data Visualization](#scaled-data-visualization)
4. [Background](#background)
   - [Important definitions](#important-definitions)
      - [Telemetry](#telemetry)
      - [Bus](#bus)
      - [Reaction Wheel](#reaction-wheel)
   - [Implications](#implications)
5. [Hypothesis](#hypothesis)
   - [Null Hypothesis](#null-hypothesis)
   - [Alternate Hypothesis](#alternate-hypothesis)
6. [Data Cleaning](#data-cleaning)
7. [Visualizations](#visualizations)
8. [Conclusion](#conclusion)
   - [What this means for telemetry analysis](#what-this-means-for-telemetry-analysis)
   


## Requirements

The following Python libraries are required to run the code:

- pandas
- seaborn
- matplotlib
- numpy
- scikit-learn
- scipy
- statsmodels
- plotly
- argparse

You can install them using the following command:

```bash
pip install pandas seaborn matplotlib numpy scikit-learn scipy statsmodels plotly argparse
```

## Usage

To run the code, you can utilize the command line arguments to specify which functions you want to execute. Here's an example of running the complete analysis:

```bash
python yourscriptname.py --run_all
```

You can also select specific functionalities:

```bash
python yourscriptname.py --load_data --preprocess_data --visualize_data
```

## Functionality

### Data Loading

The `load_data` function loads telemetry data from specific CSV files, including battery temperature, bus voltage, total bus current, wheel RPM, and wheel temperature.

### Data Preprocessing

The `preprocess_data` function converts timestamps to the datetime format and merges dataframes based on the timestamp.

### Data Visualization

The `visualize_data` function provides pairplot, boxplot, and time series plot to display different aspects of the data.

### Missing Value Analysis

The `analyze_missing_values` function analyzes and handles missing values by filling them with the mean value of the corresponding column.

### Outlier Analysis

The `analyze_outliers` function identifies and removes outliers using Z-scores.

### Correlation Analysis

The `analyze_correlation` function visualizes the correlation matrix using a heatmap.

### Model Fitting

The `fit_model` function fits a Ridge regression model to the data and calculates the mean squared error of the predictions.

### Result Export

The `export_results` function exports the merged data to a CSV file.

### Variance Inflation Factor Analysis

The `analyze_vif` function computes the Variance Inflation Factor (VIF) for each feature.

### Spinning Classification

The `classify_spinning` function transforms RPM into a binary outcome and trains a logistic regression model to classify whether the satellite is spinning or not. It also evaluates the accuracy.

### ROC Curve Plotting

The `plot_roc_curve` function calculates and plots the Receiver Operating Characteristic (ROC) curve.

### Scaled Data Visualization

The `visualize_scaled_data` function scales numeric columns and plots a time series, along with shading regions where RPM is nonzero.

Please refer to the comments in the code for more specific details on the functioning and usage of each method.