# Standard modules
import datetime
import progressbar
import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas import datetime
from pandas import read_csv

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def parser(x):
    new_time = ''.join(x.split('.')[0])  # remove microseconds from bus voltage data
    return datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')  # for bus voltage data


def detect_anomalies_flat_mean_and_std(ts):
    """Detect outliers in the bus voltage data by comparing points against two standard deviations from the mean.

       Inputs:
           ts [pd Series]: A pandas Series with a DatetimeIndex and a column for voltage.

       Optional Inputs:
           None

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, and columns for Bus Voltage (real values) and Outlier (True or False).
           outliers [pd Series]: The detected outliers, as a pandas Series with a DatetimeIndex and a column for the outlier value.

       Optional Outputs:
           None

       Example:
           bus_voltage_with_outliers, outliers = detect_anomalies_flat_mean_and_std(time_series)
       """

    # Gather statistics in preparation for outlier detection
    mean = float(ts.values.mean())
    std = float(ts.values.std(ddof=0))
    X = ts.values
    outliers = pd.Series()
    time_series_with_outliers = pd.DataFrame({'Bus Voltage': ts})
    time_series_with_outliers['Outlier'] = 'False'

    # Start a progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
    progress_bar_sliding_window = progressbar.ProgressBar(
        widgets=[progressbar.FormatLabel('Bus Voltage Outliers ')] + widgets,
        max_value=int(len(X))).start()

    # Label outliers using standard deviation
    for t in range(len(X)):
        obs = X[t]
        if abs(mean-obs) > std*2:
            time_series_with_outliers.at[ts.index[t], 'Outlier'] = 'True'
            outlier = pd.Series(obs, index=[ts.index[t]])
            outliers = outliers.append(outlier)
        progress_bar_sliding_window.update(t)  # advance progress bar

    return time_series_with_outliers, outliers


def standard_deviation_anomalies_bus_voltage(dataset_path='Data/BusVoltage.csv', plots_save_path=None,
                                             verbose=False):
    """Detect outliers in the bus voltage data by comparing points against two standard deviations from the mean.

       Inputs:
           dataset_path [str]: A string path to the bus voltage data. Data is read as a pandas Series with a DatetimeIndex and a column for voltage.

       Optional Inputs:
           plots_save_path [str]: Set to a path in order to save the plot of the data with outliers to disk.
                                  Default is None, meaning no plots will be saved to disk.
           verbose [bool]:        Set to display extra dataset information (plot of the data, its head, and statistics).
                                  Default is False.

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, and columns for Bus Voltage (real values) and Outlier (True or False).

       Optional Outputs:
           None

       Example:
           bus_voltage_with_outliers = standard_deviation_anomalies_bus_voltage(verbose=True, plots_save_path='Plots/')
       """

    # Load the dataset
    print('Reading the dataset: ' + dataset_path)
    time_series = read_csv(dataset_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

    # Preprocess
    # time_series.interpolate(inplace=True)  # remove NaN's with linear interpolation
    # time_series = time_series.resample('24H').mean()
    # time_series.to_csv('BusVoltageAveraged3Hourly.csv')

    if verbose:
        # describe the loaded dataset
        print(time_series.head())
        print(time_series.describe())
        time_series.plot(title=dataset_path + ' Dataset')  # plots the data
        pyplot.show()

    time_series_with_outliers, outliers = detect_anomalies_flat_mean_and_std(time_series)

    # Plot the outliers
    time_series.plot(color='blue', title=dataset_path.split('/')[-1] + ' Dataset with Outliers')
    pyplot.xlabel('Time')
    pyplot.ylabel('Bus Voltage')
    if len(outliers) > 0:
        print('\nDetected Outliers: ' + str(len(outliers)) + "\n")
        outliers.plot(color='red', style='.')
    if plots_save_path:
        current_time = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        filename = plots_save_path + dataset_path.split('/')[-1] + ' with Outliers (' + current_time + ').png'
        pyplot.savefig(filename, dpi=500)
    pyplot.show()

    return time_series_with_outliers


if __name__ == "__main__":
    print('Standard_Deviation_Anomalies_Bus_Voltage.py is being run directly')
    bus_voltage_with_outliers = standard_deviation_anomalies_bus_voltage(verbose=True)

else:
    print('Standard_Deviation_Anomalies_Bus_Voltage.py is being imported into another module')