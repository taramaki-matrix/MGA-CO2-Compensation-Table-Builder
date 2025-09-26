import numpy as np
import csv
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Gets Inputs for Performing Interpolation
def get_inputs():
    low_pressure = float(input("Enter the Lowest Pressure for LUT: "))
    high_pressure = float(input("Enter the Highest Pressure for LUT: "))
    num_pressures = int(input("Enter the Number of Pressure Steps for LUT: "))
    high_CO2 = int(input("Enter the Highest CO2 Concentration for LUT: "))
    num_concentrations = int(input("Enter the Number of CO2 Steps for LUT: "))
    data_file = input("Enter the Data File Directory (relative or absolute): ")
    out_path = input("Enter the Output Path Directory: ")

    inputs = {
        'low_pressure': low_pressure,
        'high_pressure': high_pressure,
        'num_pressure': num_pressures,
        'high_co2': high_CO2,
        'num_concentrations': num_concentrations,
        'data_file': data_file,
        'out_path': out_path
    }

    print(inputs)
    inputs_confirmed = input("Are these inputs correct? (y/n): ")
    if (inputs_confirmed == "n" or inputs_confirmed == "N"):
        print("Please provide new inputs.")
        inputs = get_inputs()

    return inputs

#Load Data from Inputs Data File
def load_data(inputs):
    skipped = 0 # Keep track of skipped rows
    internal_CO2, pressure, external_CO2 = [], [], []
    data_file = inputs['data_file']

    with open(data_file, newline='') as csvFile:
        dataReader = csv.reader(csvFile, delimiter=',')
        header = next(dataReader)
        num_pressures = len(header) // 3 #Each Calibration Has 3 Columns
        for row in dataReader:
            for i in range(num_pressures):
                idx = i * 3
                try:
                    ci = float(row[idx])
                    p = float(row[idx + 1])
                    ce = float(row[idx + 2])

                    if ci < inputs["high_co2"]: # If it should be based on external, change ci to ce
                        internal_CO2.append(ci)
                        pressure.append(p)
                        external_CO2.append(ce)
                    else:
                        skipped += 1
                except (ValueError, IndexError):
                    continue
    
    print(f"Filtered out {skipped} data points with internal CO₂ > {inputs["high_co2"]}")
    return [internal_CO2, pressure, external_CO2];


# Uses TensorFlow Nueral Network for Fitting Data (Overkill, but extrapolates very well)
def interpolate_data_nn(inputs, internal_CO2, pressure, external_CO2):
    # Convert inputs to numpy arrays
    internal_CO2 = np.array(internal_CO2)
    pressure = np.array(pressure)
    external_CO2 = np.array(external_CO2)
    
    # Normalize input and output (min-max scaling)
    internal_min, internal_max = internal_CO2.min(), internal_CO2.max()
    pressure_min, pressure_max = pressure.min(), pressure.max()
    external_min, external_max = external_CO2.min(), external_CO2.max()

    internal_norm = (internal_CO2 - internal_min) / (internal_max - internal_min)
    pressure_norm = (pressure - pressure_min) / (pressure_max - pressure_min)
    external_norm = (external_CO2 - external_min) / (external_max - external_min)

    # Prepare features (X) and target (y)
    X_train = np.column_stack((internal_norm, pressure_norm))
    y_train = external_norm

    # Build neural network model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),  # 2 inputs: internal_CO2_norm, pressure_norm
        Dense(64, activation='relu'),
        Dense(1)  # Single output: external_CO2_norm
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)

    # Create mesh grid for prediction in normalized scale
    xi_linspace = np.linspace(0, 1, 80)  # internal_CO2 normalized range
    yi_linspace = np.linspace(0, 1, 80)  # pressure normalized range
    xi_norm_grid, yi_norm_grid = np.meshgrid(xi_linspace, yi_linspace)

    # Flatten grid for prediction
    grid_points = np.column_stack((xi_norm_grid.ravel(), yi_norm_grid.ravel()))
    
    # Predict normalized external_CO2
    zi_norm = model.predict(grid_points).reshape(xi_norm_grid.shape)
    
    # Denormalize predictions
    zi = zi_norm * (external_max - external_min) + external_min

    # Generate lookup table similarly
    mapped_internals = np.linspace(0, 1, inputs["num_concentrations"])  # normalized
    mapped_pressures = np.linspace(0, 1, inputs["num_pressure"])        # normalized

    # Map Output for Export and Plotting
    results = []
    for p_norm in mapped_pressures:
        temp = []
        for e_norm in mapped_internals:
            val_norm = model.predict(np.array([[e_norm, p_norm]]))[0,0]
            val = val_norm * (external_max - external_min) + external_min
            temp.append(round(float(val), 1))
        results.append(temp)

    # Optional: compute RMSE on training data for evaluation
    y_pred_train = model.predict(X_train).flatten()
    rmse = np.sqrt(np.mean((y_pred_train - y_train) ** 2))
    print(f"NN RMSE (normalized scale): {rmse}")

    # Return denormalized meshgrid and predictions
    return [xi_norm_grid * (internal_max - internal_min) + internal_min,
            yi_norm_grid * (pressure_max - pressure_min) + pressure_min,
            zi], [mapped_internals * (internal_max - internal_min) + internal_min,
                  mapped_pressures * (pressure_max - pressure_min) + pressure_min,
                  results]

#Write LUT into .json File for MGA Device
def write_data(inputs, results):
    output_path = inputs["out_path"]

    json_output = os.path.join(output_path, "co2_compensation.json")
    # Write JSON LUT
    with open(json_output, 'w') as json_file:
        json.dump({
            "compensation": results,
            "low_ppm": 0.0,
            "high_ppm": inputs["high_co2"],
            "low_pressure": inputs["low_pressure"],
            "high_pressure": inputs["high_pressure"]
        }, json_file, indent=4)

    return

def plot_data(input_data, fitted_surface, output):
    #Unpack Data
    raw_x, raw_y, raw_z = input_data
    xi, yi, zi = fitted_surface
    mapped_internals, mapped_pressures, results = output

    # Convert results to 2D array for plotting
    results_array = np.array(results)
    
    # Plot
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot fitted surface
    ax.plot_surface(xi, yi, zi, alpha=0.5, cmap='plasma')

    # Raw data points
    ax.scatter(raw_x, raw_y, raw_z, color='blue', s=25, label='Data Points')

    results_array = np.array(results)  # (num_concentrations, num_pressures)

    # Create meshgrid of internal and pressure for plotting points:
    internals_grid, pressures_grid = np.meshgrid(mapped_internals, mapped_pressures, indexing='xy')

    # Flatten all to 1D arrays:
    xs = internals_grid.flatten()
    ys = pressures_grid.flatten()
    zs = results_array.flatten()

    # Single scatter call:
    ax.scatter(xs, ys, zs, color='red', s=50, label='Points for LUT')

    ax.set_xlabel('Internal ppm (elevated)')
    ax.set_ylabel('Pressure (psi)')
    ax.set_zlabel('External ppm (real)')
    ax.set_title('Pressure/ExtCO2/IntCO2 Fit')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    inputs = get_inputs()
    data = load_data(inputs)
    print("Data Loaded!")
    fitted_surface, output = interpolate_data_nn(inputs, *data)
    print("Data Interpolated!")
    plot_data(data, fitted_surface, output)
    write_data(inputs, output[2])
    print("Output Saved!")



