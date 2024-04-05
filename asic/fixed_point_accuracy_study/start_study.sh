# Spawn runs of the program to run the fixed point accuracy study
num_runs=48
for i in {0..19}
do
    ../func_sim/func_sim --start_index $((i*num_runs)) --end_index $(((i+1)*num_runs-1)) --results_csv_fp "accuracy_study_results.csv" &
    sleep 1
done

wait # Wait for all background jobs to finish