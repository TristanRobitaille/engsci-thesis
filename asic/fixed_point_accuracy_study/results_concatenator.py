import os
import csv
import glob
import shutil
import datetime

#----- CONSTANTS -----#
RESULTS_TEMPLATE_FP = "./results/results_template_w_python.csv"
START_ROW_OFFSET = 2
START_COLUMN_OFFSET = 3 - 8

def main():
    # Make a copy of the template file
    new_fp = f"./results/results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    shutil.copy(RESULTS_TEMPLATE_FP, new_fp)

    # Read the template file
    with open(new_fp, 'r') as f:
        reader = csv.reader(f)
        template_data = list(reader)

    # Go through each partial results file
    files = glob.glob(os.path.join(os.path.dirname(RESULTS_TEMPLATE_FP), "results_template_w_python_*"))
    for file in files:
        # Separate the file name into its components
        filename = file.replace(".csv", "")
        file_name_parts = filename.split('_')        
        num_bits = int(file_name_parts[-3])
        start_index = int(file_name_parts[-2])
        end_index = int(file_name_parts[-1])
        with open(file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)

        # Paste the copied cells into the template file
        for row in range(start_index+START_ROW_OFFSET, end_index+START_ROW_OFFSET+1):
            template_data[row][num_bits+START_COLUMN_OFFSET] = data[row][num_bits+START_COLUMN_OFFSET]

    # Save the complete file
    with open(new_fp, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(template_data)

    # Delete files
    for file in files:
        os.remove(file)

if __name__ == "__main__":
    main()
