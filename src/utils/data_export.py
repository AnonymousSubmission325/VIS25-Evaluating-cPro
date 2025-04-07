# src/utils/data_export.py
import os
import pandas as pd

def export_evaluation_results(results, max_time, output_filename_prefix="evaluation_results"):
    """
    Exports the evaluation results to a CSV file in the 'src/results/records' directory,
    including the max_time parameter in the filename.

    Parameters:
    -----------
    results : list of dict
        A list of dictionaries containing evaluation metrics and other information.
    max_time : int
        The maximum allowed time (in seconds) for each projection method.
    output_filename_prefix : str, optional
        The prefix of the CSV file name. Defaults to 'evaluation_results'.
    """
    # Ensure the directory for saving records exists
    output_dir = "src/results/records"
    os.makedirs(output_dir, exist_ok=True)

    # Construct the filename with max_time included
    output_filename = f"{output_filename_prefix}_max_time_{max_time}s.csv"
    results_path = os.path.join(output_dir, output_filename)

    # Convert the results to a DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"Results exported to {results_path}")
