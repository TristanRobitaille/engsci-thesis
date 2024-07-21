# Spawn processes of the program to run the fixed point accuracy study
num_total_clips=960
num_process=192
starting_num_bits=7
num_bit_studies=14

cd ../func_sim
export N_STO_PARAMS=20

for i in $(seq $starting_num_bits $((starting_num_bits + num_bit_studies - 1)))
do
    # Generate new binary
    export N_STO_INT_RES=${i}
    echo $N_STO_INT_RES
    make clean
    make configureNoBoost
    make build
    mv build/Func_Sim build/Func_Sim_${i}b
    
    for j in $(seq 0 $((num_process-1)))
    do
        ./build/Func_Sim_${i}b --start_index $((j*(num_total_clips/num_process))) --end_index $(((j+1)*(num_total_clips/num_process)-1)) --results_csv_fp "../fixed_point_accuracy_study/results/results_template_w_python.csv" &
    done
    wait # Wait for all background jobs to finish

    rm build/Func_Sim_${i}b # Delete the binary

    cd ../fixed_point_accuracy_study
    python3 results_concatenator.py --type="partial" --target_bit=${i} --bit_type="int_res"
    cd ../func_sim
done

cd ../fixed_point_accuracy_study # Go back
export N_STO_INT_RES=20 # Reset back to default
export N_STO_PARAMS=20 # Reset back to default

python3 results_concatenator.py --type="complete" --bit_type="int_res"
