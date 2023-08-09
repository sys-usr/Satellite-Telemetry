import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import plotly.express as px
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class SatelliteTelemetryAnalyzer:
    def __init__(self):
        # Initialize dataframes
        self.battery_temp = None
        self.bus_voltage = None
        self.total_bus_current = None
        self.wheel_rpm = None
        self.wheel_temp = None
        self.merged_df = None

    def load_data(self):
        # Load data
        self.battery_temp = pd.read_csv('Satellite-Telemetry\\Data\\BatteryTemperature.csv', header=None, names=['timestamp', 'temperature'])
        self.bus_voltage = pd.read_csv('Satellite-Telemetry\\Data\\BusVoltage.csv', header=None, names=['timestamp', 'voltage'])
        self.total_bus_current = pd.read_csv('Satellite-Telemetry\\Data\\TotalBusCurrent.csv', header=None, names=['timestamp', 'current'])
        self.wheel_rpm = pd.read_csv('Satellite-Telemetry\\Data\\WheelRPM.csv', header=None, names=['timestamp', 'rpm'])
        self.wheel_temp = pd.read_csv('Satellite-Telemetry\\Data\\WheelTemperature.csv', header=None, names=['timestamp', 'temperature'])

    def preprocess_data(self):
        # Convert 'timestamp' column to datetime and merge dataframes
        dataframes = [self.battery_temp, self.bus_voltage, self.total_bus_current, self.wheel_rpm, self.wheel_temp]
        for df in dataframes:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        self._merge_dataframes()

    def visualize_data(self):
        # Pairplot, boxplot, and time series plot
        sns.pairplot(self.merged_df, vars=['temperature', 'voltage', 'current', 'rpm'], corner=True)
        sns.boxplot(data=self.merged_df)
        plt.show()

    def analyze_missing_values(self):
        # Analyze and handle missing values
        missing_values = self.merged_df.isnull().sum()
        print(missing_values)
        self.merged_df.fillna(self.merged_df.mean(), inplace=True)

    def analyze_outliers(self):
        # IDs outliers using a Z-sacore and removes them
        self.merged_df = self.merged_df[(self.merged_df.select_dtypes(include=['float64', 'int64']) >= 0).all(axis=1)]
        numeric_cols = self.merged_df.select_dtypes(include=['float64', 'int64'])
        z_scores = stats.zscore(numeric_cols)
        abs_z_scores = np.abs(z_scores)
        outliers = (abs_z_scores > 2).any(axis=1)
        self.merged_df = self.merged_df[~outliers]

    def analyze_correlation(self):
        # Correlation matrix and heatmap
        correlation_matrix = self.merged_df.corr()
        plt.figure(figsize=(12, 9))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()

    def fit_model(self):
        # Define features and target, split the data, and fit a Ridge regressor
        X = self.merged_df[['voltage', 'current', 'rpm', 'temperature_wheel_temp', 'current_total_bus_current']]
        y = self.merged_df['temperature']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        ridge_regressor = Ridge(alpha=1.0)
        ridge_regressor.fit(X_train, y_train)
        y_pred_ridge = ridge_regressor.predict(X_test)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        print("Ridge Regression Mean Squared Error:", mse_ridge)

    def export_results(self):
        # Export merged data to CSV
        file_path = 'Satellite-Telemetry/Data/MergedData.csv'
        self.merged_df.to_csv(file_path, index=False)

    def _merge_dataframes(self):
        # Merge the dataframes on 'timestamp'
        self.merged_df = self.battery_temp
        self.merged_df = pd.merge(self.merged_df, self.bus_voltage, on='timestamp', suffixes=('', '_bus_voltage'))
        self.merged_df = pd.merge(self.merged_df, self.total_bus_current, on='timestamp', suffixes=('', '_total_bus_current'))
        self.merged_df = pd.merge(self.merged_df, self.wheel_rpm, on='timestamp', suffixes=('', '_wheel_rpm'))
        self.merged_df = pd.merge(self.merged_df, self.wheel_temp, on='timestamp', suffixes=('', '_wheel_temp'))

    def analyze_vif(self):
        # Compute VIF for each feature
        X = add_constant(self.merged_df[['temperature', 'voltage', 'current', 'rpm', 'temperature_wheel_temp', 'current_total_bus_current']])
        vif = pd.DataFrame()
        vif['Feature'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(vif)

    def classify_spinning(self):
        # Transform rpm into binary outcome, train logistic regression model, and evaluate accuracy
        self.merged_df['is_spinning'] = (self.merged_df['rpm'] > 0).astype(int)
        X = self.merged_df[['temperature', 'voltage', 'current', 'temperature_wheel_temp', 'current_total_bus_current']]
        y = self.merged_df['is_spinning']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logistic_regressor = LogisticRegression()
        logistic_regressor.fit(X_train, y_train)
        y_pred_logistic = logistic_regressor.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_logistic)
        print("Logistic Regression Accuracy:", accuracy)

    def plot_roc_curve(self):
        # Define features and target for the ROC curve calculation
        X = self.merged_df[['voltage', 'current', 'rpm', 'temperature_wheel_temp', 'current_total_bus_current']]
        y = self.merged_df['is_spinning']

        # Calculate and plot the ROC curve
        model = LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()

    def t_test_battery_temp_spinning_vs_not(self):
        spinning_data = self.merged_df[self.merged_df['is_spinning'] == 1]['temperature']
        not_spinning_data = self.merged_df[self.merged_df['is_spinning'] == 0]['temperature']
    
        t_stat, p_value = stats.ttest_ind(spinning_data, not_spinning_data)
    
        print("T-statistic:", t_stat)
        print("P-value:", p_value)
    
        if p_value < 0.05:
            print("There is a significant difference in battery temperatures between spinning and non-spinning conditions.")
        else:
            print("There is no significant difference in battery temperatures between spinning and non-spinning conditions.")

    def visualize_scaled_data(self):
        # Scale numeric columns and plot time series
        numeric_cols = self.merged_df.select_dtypes(include=['float64', 'int64'])
        cols_to_drop = ['timestamp', 'rpm']
        numeric_cols = numeric_cols.drop(columns=[col for col in cols_to_drop if col in numeric_cols.columns])
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(numeric_cols)
        scaled_df = pd.DataFrame(scaled_values, columns=numeric_cols.columns)
        scaled_df['timestamp'] = self.merged_df['timestamp']
        fig = px.line()
        for col in scaled_df.columns[:-1]:
            fig.add_scatter(x=scaled_df['timestamp'], y=scaled_df[col], name=col)
        rpm_column = self.merged_df['rpm']
        where_rpm_non_zero = rpm_column != 0
        fig.add_scatter(x=self.merged_df['timestamp'], y=where_rpm_non_zero * 3 - 3, fill='toself', fillcolor='grey', opacity=0.1, name='rpm_non_zero')
        fig.update_traces(line=dict(width=2))
        fig.update_layout(
            title='Scaled Numeric Columns with Spinning Regions Shaded',
            xaxis_title='Time',
            yaxis_title='Standardized Value',
        )
        fig.show()

if __name__ == '__main__':
    # create parser
    parser = argparse.ArgumentParser(description="Satellite Telemetry Analyzer")

    # funct args
    parser.add_argument('--load_data', action='store_true', help='Load telemetry data')
    parser.add_argument('--preprocess_data', action='store_true', help='Preprocess loaded data')
    parser.add_argument('--visualize_data', action='store_true', help='Visualize the telemetry data')
    parser.add_argument('--analyze_missing', action='store_true', help='Analyze and handle missing values')
    parser.add_argument('--analyze_outliers', action='store_true', help='Identify and remove outliers')
    parser.add_argument('--analyze_correlation', action='store_true', help='Analyze correlation between variables')
    parser.add_argument('--fit_model', action='store_true', help='Fit a Ridge regression model')
    parser.add_argument('--export_results', action='store_true', help='Export merged data to CSV')
    parser.add_argument('--analyze_vif', action='store_true', help='Analyze Variance Inflation Factor (VIF)')
    parser.add_argument('--classify_spinning', action='store_true', help='Classify spinning based on RPM')
    parser.add_argument('--plot_roc_curve', action='store_true', help='Plot the ROC curve')
    parser.add_argument('--visualize_scaled_data', action='store_true', help='Visualize scaled numeric columns')
    parser.add_argument('--run_all', action='store_true', help='Run all functions')

    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of SatelliteTelemetryAnalyzer
    analyzer = SatelliteTelemetryAnalyzer()

    # Helper function to run all functions
    def run_all_functions():
        analyzer.load_data()
        analyzer.preprocess_data()
        analyzer.visualize_data()
        analyzer.analyze_missing_values()
        analyzer.analyze_outliers()
        analyzer.analyze_correlation()
        analyzer.fit_model()
        analyzer.export_results()
        analyzer.analyze_vif()
        analyzer.classify_spinning()
        analyzer.plot_roc_curve()
        analyzer.visualize_scaled_data()

    if args.run_all:
        run_all_functions()
    else:
        # Run selected functions based on provided arguments
        if args.load_data:
            analyzer.load_data()
        if args.preprocess_data:
            analyzer.preprocess_data()
        if args.visualize_data:
            analyzer.visualize_data()
        if args.analyze_missing:
            analyzer.analyze_missing_values()
        if args.analyze_outliers:
            analyzer.analyze_outliers()
        if args.analyze_correlation:
            analyzer.analyze_correlation()
        if args.fit_model:
            analyzer.fit_model()
        if args.export_results:
            analyzer.export_results()
        if args.analyze_vif:
            analyzer.analyze_vif()
        if args.classify_spinning:
            analyzer.classify_spinning()
        if args.plot_roc_curve:
            analyzer.plot_roc_curve()
        if args.visualize_scaled_data:
            analyzer.visualize_scaled_data()