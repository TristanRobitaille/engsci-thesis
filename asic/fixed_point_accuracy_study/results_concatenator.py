import os
import csv
import glob
import argparse
import datetime

#----- CONSTANTS -----#
RESULTS_TEMPLATE_FP = "./results/results_template_w_python.csv"
START_ROW_OFFSET = 2
START_COLUMN_OFFSET = 3 - 2

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['partial', 'complete'], help="Defines how to save concatenated results file. 'partial' will maintain metadata in title for subsequent processing. 'complete' removes metadata and appends time'")
    parser.add_argument("--target_bit", type=int, help="Defines the target bit whose results files to concatenate. Only relevant for --type=partial")
    parser.add_argument("--bit_type", choices=['int_res', 'params'], help="Defines whether to use the int_res bits or params bits in concatenation logic.")
    args = parser.parse_args()

    # Make a copy of the template file
    new_fp = f"./results/results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    # Read the template file
    with open(RESULTS_TEMPLATE_FP, 'r') as f:
        reader = csv.reader(f)
        template_data = list(reader)

    # Go through each partial results file
    start_index_total = 1000000
    end_index_total = 0
    if (args.type == "complete"): files = glob.glob(os.path.join(os.path.dirname(RESULTS_TEMPLATE_FP), f"results_template_w_python_*.csv"))
    elif (args.bit_type == "int_res"): files = glob.glob(os.path.join(os.path.dirname(RESULTS_TEMPLATE_FP), f"results_template_w_python_*_{args.target_bit}_*"))
    elif (args.bit_type == "params"): files = glob.glob(os.path.join(os.path.dirname(RESULTS_TEMPLATE_FP), f"results_template_w_python_{args.target_bit}_*"))
    if (files == []):
        print("No files found!")
        return
    
    for file in files:
        # Separate the file name into its components
        filename = file.replace(".csv", "")
        file_name_parts = filename.split("_")
        if args.bit_type == "int_res":
            num_bits = int(file_name_parts[-3])
            num_bits_fixed = int(file_name_parts[-4])
        elif args.bit_type == "params":
            num_bits = int(file_name_parts[-4])
            num_bits_fixed = int(file_name_parts[-3])

        if (args.type == "partial" and num_bits != args.target_bit): continue
        start_index = int(file_name_parts[-2])
        end_index = int(file_name_parts[-1])
       
        if (start_index < start_index_total): start_index_total = start_index
        if (end_index > end_index_total): end_index_total = end_index
        
        with open(file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)

        # Paste the copied cells into the template file
        for row in range(start_index+START_ROW_OFFSET, end_index+START_ROW_OFFSET+1):
            template_data[row][num_bits+START_COLUMN_OFFSET] = data[row][num_bits+START_COLUMN_OFFSET]

    # Save the complete file
    if args.type == "partial":
        fp_parts = files[0].split('_')
        fp_parts[-1] = str(end_index_total)
        fp_parts[-2] = str(start_index_total)
        new_fp = ""
        for i in range(len(fp_parts)):
            if (i == 0): new_fp += fp_parts[i]
            else: new_fp += f"_{fp_parts[i]}"
        
        new_fp += ".csv"
    elif args.type == "complete":
        if args.bit_type == "int_res":  new_fp = f"./results/results_params-{num_bits_fixed}_int-res-variable_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        elif args.bit_type == "params": new_fp = f"./results/results_params-variable_int-res-{num_bits_fixed}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    # Compute accuracies if concatenating the final files
    if args.type == "complete":
        accuracies = len(template_data[-1])*["0"]
        accuracies[0] = "Accuracy"
        accuracies[1] = "."
        total_clips_ran = 0
        for row in template_data:
            at_least_one_n_sto_inferred = False
            if row[0].isdigit(): # If this row is an actual clip
                ground_truth = int(row[1])
                for i in range(3,len(row)): # Only Python and N_STOs
                    if row[i] == ".": continue
                    if row[1] == "0": break # If ground truth is undefined
                    at_least_one_n_sto_inferred = True # If clip is inference from functional simulation
                    if int(row[i]) == ground_truth:
                        accuracies[i] = str(int(accuracies[i])+1)
            
                if at_least_one_n_sto_inferred: # Can now consider Tensorflow reference
                    total_clips_ran += 1
                    if (int(row[2])) == ground_truth: accuracies[2] = str(int(accuracies[2])+1)

        for i in range(2, len(accuracies)):
            accuracies[i] = str(round(int(accuracies[i])/total_clips_ran, 4))
        template_data.append(accuracies)

    with open(new_fp, 'w', newline='') as f:
        print(f"new_fp: {new_fp}")
        writer = csv.writer(f)
        writer.writerows(template_data)

    # Delete files
    for file in files:
        os.remove(file)
        
if __name__ == "__main__":
    main()
