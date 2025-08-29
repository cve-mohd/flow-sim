import os
import shutil
import pandas as pd

# Root directory containing the "infXXXX_stgXXX" folders
root_dir = "results"

# List of csv files to merge
csv_files = [
    "analytical_wave_celerity.csv",
    "flow_area.csv",
    "flow_depth.csv",
    "flow_rate.csv",
    "flow_velocity.csv",
    "Peak_amplitude_profile.csv",
    "water_surface_level.csv",
]

# Loop through first-level folders
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    # Process each of the 7 files
    for csv_name in csv_files:
        merged_df = None

        for reach in range(5):
            reach_path = os.path.join(folder_path, f"reach_{reach}", csv_name)
            df = pd.read_csv(reach_path)

            if reach == 0:
                # First reach: keep time column, drop last column (duplicate if not final)
                if reach != 4:
                    df = df.iloc[:, :-1]
                merged_df = df
            else:
                # From reach_1 onwards: drop time column
                df = df.iloc[:, 1:]

                # Drop last column if not the last reach
                if reach != 4:
                    df = df.iloc[:, :-1]

                merged_df = pd.concat([merged_df, df], axis=1)

        # Save merged result back to the top-level folder
        output_path = os.path.join(folder_path, csv_name)
        merged_df.to_csv(output_path, index=False)

    # After processing all csv files, remove reach_* subfolders
    for reach in range(5):
        reach_dir = os.path.join(folder_path, f"reach_{reach}")
        if os.path.isdir(reach_dir):
            shutil.rmtree(reach_dir)

print('Done.')
