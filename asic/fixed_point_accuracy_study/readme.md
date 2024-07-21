## Fixed Point Accuracy Study
### How to run
1. Make a copy of `results_template.csv`. Rename it.
2. Run `run_accuracy_study()` in `python_prototype/utilities.py` (from `python_prototype`), making sure to use the correct arguments. This adds the ground truth sleep stages and inferred sleep stages from the Python model to results CSV.
3. Update `DATA_BASE_DIR` in `asic/func_sim/include/Misc.hpp` to `"./"`.
4. From `asic/fixed_point_accuracy_study`, run `source start_study.sh`. You can change the total number of clips in the dataset and the number of concurrent processes to spawn. Please make sure the total number of clips is a multiple of the number of processes. Also, change the filepath of the results CSV. The script will compile a functional simulation binary, spawn concurrent processes, each running the functional simulation on a subset of the clips in the dataset. These will write their inferred sleep stage in the results CSV once all their clips have ran. Once all processes for a given `N_STO_INT_RES` have terminated, the script will compile another binary with `N_STO_INT_RES + 1` and re-run all the processes.\
Note: For Compute Canada, run `run_accuracy_study_compute_canada.sh` from `asic/fixed_point_accuracy_study` and change `num_process` in `start_study` to `160`. It should take under 10h to run.

### Important note
For the fixed-point accuracy study, there is no averaging filter on the softmax of the model as this would complicate parallelization of the study. We simply compare the argmax of the current clip's softmax vector.