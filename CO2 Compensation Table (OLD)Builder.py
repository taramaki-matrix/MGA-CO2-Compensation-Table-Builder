import numpy as np
from scipy.interpolate import RegularGridInterpolator, Rbf
import csv
import math
import matplotlib.pyplot as plt
import json


# INPUTS START      ##########################
###

# How to structure data for the LUT
low_pressure_for_LUT = 17.5
high_pressure_for_LUT = 22
high_concentration_for_LUT = 510000
num_pressures_for_LUT = 3
num_concentrations_for_LUT = 56

# Data file is a csv structured with columns as so:
# p1_int (measured by uncompensated MGA),p1_pressure (pressure during measurement), p1_ext (measurement at output) ... repeat for more pressures
data_filename = 'data.csv'

###
# INPUTS END        #########################
def cubic_rbf_interpolator(data_filename):
    """
    Performs cubic radial basis function interpolation on calibration data 
    and generates a calibration lookup table (LUT) in CSV and JSON formats. 
    Also produces a 3D visualization of the interpolation surface.

    Args:
        data_filename (str): Path to the input CSV file containing measured data.
    """
    points_z, points_x, points_y = [], [], []

    # Read CSV and extract data for interpolation
    with open(data_filename, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        num_pressures = len(header) // 3  # Columns are grouped in threes
        for row in datareader:
            for i in range(num_pressures):
                idx = i * 3
                points_z.append(float(row[idx]))
                points_x.append(float(row[idx + 1]))
                points_y.append(float(row[idx + 2]))

    # Create RBF interpolator using cubic function
    rbfi = Rbf(points_x, points_y, points_z, function='cubic') 

    # Generate interpolation grid (visual)
    xi_vals = np.linspace(low_pressure_for_LUT, high_pressure_for_LUT, 10)
    yi_vals = np.linspace(0, high_concentration_for_LUT, 80)
    xi, yi = np.meshgrid(xi_vals, yi_vals)
    zi = rbfi(xi, yi)

    # Generate table of interpolated results
    pressures = np.linspace(low_pressure_for_LUT, high_pressure_for_LUT, num_pressures_for_LUT)
    external_ppms = np.linspace(0, high_concentration_for_LUT, num_concentrations_for_LUT)
    ints_results = []
    for p in pressures:
        temp = []
        for e in external_ppms:
            temp.append(round(float(rbfi(p, e)), 1))
        ints_results.append(temp)

    # Write calibration table to CSV
    csv_file = "cal_table_output.csv"
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ["ext"]
        for p in range(num_pressures_for_LUT):
            header.append(str("p" + str(p+1) + "_pressure"))
            header.append(str("p" + str(p+1) + "_int"))
        csv_writer.writerow(header)
        for e in range(len(external_ppms)):
            row = [int(external_ppms[e])]
            for p in range(num_pressures_for_LUT):
                row.append(round(pressures[p], 2))
                row.append(ints_results[p][e])
            csv_writer.writerow(row)

    # Write JSON LUT
    with open('co2_compensation.json', 'w') as json_file:
        json.dump({
            "compensation": ints_results,
            "low_ppm": 0.0,
            "high_ppm": high_concentration_for_LUT,
            "low_pressure": low_pressure_for_LUT,
            "high_pressure": high_pressure_for_LUT
        }, json_file, indent=4)

    # Plot 3D surface of interpolated fit
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xi, yi, zi, alpha=.5, cmap='plasma', label='Fitted Surface')  # Fitted surface
    ax.scatter(points_x, points_y, points_z, label='Data Points', s=25, marker='o')  # Raw data points

    # Overlay LUT points for visualization
    l = 'Points for LUT'
    for i in range(len(pressures)):
        pressures_for_plotting = [pressures[i]] * len(external_ppms)
        ax.scatter(pressures_for_plotting, external_ppms, ints_results[i],
                   label=l, s=50, marker='o', color='red')
        l = ''  # Avoid duplicate legend entries

    # Label axes and show plot
    ax.set_xlabel('Pressure (psi)')
    ax.set_ylabel('External ppm (real)')
    ax.set_zlabel('Internal ppm (elevated)')
    ax.set_title('Pressure/ExtCO2/IntCO2 Fit')
    ax.legend()
    plt.show()

def LUT_accuracy_check(LUT_filename, data_filename):
    """
    Evaluates the accuracy of a calibration lookup table (LUT) against the original data.

    Args:
        LUT_filename (str): Path to the CSV LUT file to evaluate.
        data_filename (str): Path to the original CSV file used to generate the LUT.
    """
    pressure, external, internal = [], [], []

    # Load LUT and extract pressure and internal CO2 values
    pressure_recorded = False
    with open(LUT_filename, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        num_pressures = len(header) // 2 # columns grouped in twos
        for row in datareader:
            internals_sub = np.array([])
            external = np.append(external, float(row[0]))
            for i in range(num_pressures):
                internals_sub = np.append(internals_sub, float(row[i*2+2]))  # Internal CO2
                if not pressure_recorded:
                    pressure.append(float(row[i*2+1]))  # Pressure
            internal.append(internals_sub)
            pressure_recorded = True

    # Load original data points used for comparison
    points_to_interpolate = []
    ints_to_check = []
    with open(data_filename, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        num_pressures = len(header) // 3
        for row in datareader:
            for i in range(num_pressures):
                sub_list = []
                ints_to_check.append(float(row[i*3]))  # Actual internal CO2
                sub_list.append(float(row[i*3+2]))     # External CO2
                sub_list.append(float(row[i*3+1]))     # Pressure
                points_to_interpolate.append(sub_list)

    # Interpolate values using LUT
    interpolator = RegularGridInterpolator((external, pressure), internal, method='linear')
    interpolated_values = interpolator(points_to_interpolate)

    # Calculate error metrics
    errors = []
    sse = 0
    for i in range(len(interpolated_values)):
        error = round(float(interpolated_values[i] - ints_to_check[i]), 1)
        errors.append(error)
        sse += error * error

    # Output error statistics
    print("errors of LUT: ")
    print(errors)
    print("average error (rms): ")
    print(round(float(math.sqrt(sse / len(interpolated_values))), 4))

cubic_rbf_interpolator(data_filename)

LUT_accuracy_check('cal_table_output.csv',data_filename)
