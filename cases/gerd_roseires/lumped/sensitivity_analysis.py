# py -m cases.gerd_roseires.lumped.sensitivity_analysis

"""
This script performs a sensitivity analysis on the `main.run` function.

It systematically perturbs each input parameter, runs the simulation, and
calculates the sensitivity of each output metric to that change.
The sensitivity is quantified as the "percent change in output per percent
change in input".

The results are saved in two formats:
1.  `sensitivity_analysis_results.md`: A human-readable Markdown table.
2.  `sensitivity_analysis_results.csv`: A CSV file for easy data import and processing.
"""
from . import main
import csv

def run_sensitivity_analysis():
    """
    Main function to orchestrate the sensitivity analysis.
    """
    # 1. Define the baseline values for the input parameters.
    baseline_params = {
        'rep_length_fraction': 0.5,
        'wet_n': 0.027,
        'dry_n': 0.030,
        'n_steepness': 0.15,
        'Cc': 0.5,
        'K_q': 0.1
    }

    # Define the percentage of change to apply to each input parameter.
    # A 1% perturbation is a common choice.
    perturbation = 0.01

    # 2. Run the simulation with baseline parameters to get baseline outputs.
    base_outputs_tuple = main.run(**baseline_params)
    baseline_outputs = {
        'attenuation': base_outputs_tuple[0],
        'us_peak_amp': base_outputs_tuple[1],
        'ds_peak_amp': base_outputs_tuple[2]
    }

    # Handle cases where a baseline output is zero to prevent division errors.
    for key, value in baseline_outputs.items():
        if value == 0:
            print(f"Warning: Baseline output '{key}' is zero. Sensitivity will be infinite or NaN.")
            # Set to a very small number to avoid division by zero.
            baseline_outputs[key] = 1e-9

    # 3. Initialize a dictionary to store the sensitivity results.
    sensitivity_results = {}

    # 4. Iterate through each parameter to analyze its sensitivity.
    for param_name in baseline_params:
        perturbed_params = baseline_params.copy()
        base_param_value = baseline_params[param_name]

        # Apply the perturbation to the current parameter.
        perturbed_params[param_name] = base_param_value * (1 + perturbation)

        # 5. Run the simulation with the single perturbed parameter.
        new_outputs_tuple = main.run(**perturbed_params)
        new_outputs = {
            'attenuation': new_outputs_tuple[0],
            'us_peak_amp': new_outputs_tuple[1],
            'ds_peak_amp': new_outputs_tuple[2]
        }

        # 6. Calculate sensitivity for each output.
        # Formula: Sensitivity = (% change in output) / (% change in input)
        param_sensitivities = {}
        for out_name, base_out_value in baseline_outputs.items():
            new_out_value = new_outputs[out_name]
            percent_change_output = (new_out_value - base_out_value) / base_out_value
            sensitivity = percent_change_output / perturbation
            param_sensitivities[out_name] = sensitivity

        sensitivity_results[param_name] = param_sensitivities

    # 7. Save the calculated results to files.
    output_results_to_files(sensitivity_results)
    print("Sensitivity analysis complete.")
    print("Results saved to 'sensitivity_analysis_results.md' and 'sensitivity_analysis_results.csv'.")


def output_results_to_files(results):
    """
    Writes the sensitivity analysis results to Markdown and CSV files.

    Args:
        results (dict): A dictionary containing the sensitivity results.
    """
    # Get the names of the output variables from the first result entry.
    output_names = list(next(iter(results.values())).keys())

    # --- Write to Markdown file for easy viewing ---
    md_filepath = 'cases\\gerd_roseires\\lumped\\sensitivity_analysis.py\\sensitivity_analysis_results.md'
    with open(md_filepath, 'w') as f:
        f.write("# Sensitivity Analysis Results\n\n")
        f.write("This table shows the sensitivity of each output variable to a 1% change in each input parameter.\n")
        f.write("A sensitivity value of 1.5, for example, means that a 1% increase in the input parameter\n")
        f.write("leads to a 1.5% increase in the output variable.\n\n")
        f.write("Calculation: `Sensitivity = (% change in output) / (% change in input)`\n\n")

        header = "| Input Parameter         | " + " | ".join([f"Sensitivity of {name}" for name in output_names]) + " |\n"
        f.write(header)
        f.write("|-------------------------|-" + "-|-".join(["-" * (len(name) + 17) for name in output_names]) + "-|\n")

        for param_name, sensitivities in results.items():
            row_values = [f"{sensitivities[out_name]:.4f}" for out_name in output_names]
            row = f"| {param_name:<23} | " + " | ".join(row_values) + " |\n"
            f.write(row)

    # --- Write to CSV file for data processing ---
    csv_filepath = 'cases\\gerd_roseires\\lumped\\sensitivity_analysis.py\\sensitivity_analysis_results.csv'
    with open(csv_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Input Parameter'] + [f'Sensitivity of {name}' for name in output_names]
        writer.writerow(header)

        for param_name, sensitivities in results.items():
            row = [param_name] + [sensitivities[out_name] for out_name in output_names]
            writer.writerow(row)

run_sensitivity_analysis()
