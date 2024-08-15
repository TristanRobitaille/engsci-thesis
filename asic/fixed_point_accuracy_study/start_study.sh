#!/bin/bash

# Spawn processes of the program to run the fixed point accuracy study
num_total_clips=962
num_process=481
starting_num_bits_int_res=6
num_bit_studies_int_res=10
starting_num_bits_params=5
num_bit_studies_params=10

for int_res_bits in $(seq $starting_num_bits_int_res $((starting_num_bits_int_res + num_bit_studies_int_res - 1))) # Loop through int_res bits
do
    cd asic/func_sim
    export N_STO_INT_RES=${int_res_bits}
    for param_bits in $(seq $starting_num_bits_params $((starting_num_bits_params + num_bit_studies_params - 1))) # Loop through param bits
    do
        # Generate new binary
        export N_STO_PARAMS=${param_bits}
        make clean
        cmake CMakeLists.txt
        make -j24
        mv build/Func_Sim build/Func_Sim_${param_bits}b
        
        for process_num in $(seq 0 $((num_process-1)))
        do
            ./build/Func_Sim_${param_bits}b --start_index $((process_num*(num_total_clips/num_process))) --end_index $(((process_num+1)*(num_total_clips/num_process)-1)) --results_csv_fp "../fixed_point_accuracy_study/results/results_template_w_python.csv" --study_bit_type "PARAMS" &
        done
        wait # Wait for all background jobs to finish

        rm build/Func_Sim_${param_bits}b # Delete the binary

        cd ../fixed_point_accuracy_study
        python3 results_concatenator.py --type="partial" --target_bit=${param_bits} --bit_type="params"
        cd ../func_sim
    done

    cd ../fixed_point_accuracy_study # Go back
    python3 results_concatenator.py --type="complete" --bit_type="params"
    cd ../../ # Go back to root directory of engsci-thesis
done

# Reset
export N_STO_INT_RES=20
export N_STO_PARAMS=20
