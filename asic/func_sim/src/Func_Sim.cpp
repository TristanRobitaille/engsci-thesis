#include <Func_Sim.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- GLOBAL -----*/
uint32_t epoch_cnt;
map<int, FcnPtr> event_schedule;
struct ext_signals ext_sigs;
vector<uint32_t> softmax_max_indices;

#if DISTRIBUTED_ARCH
Bus bus;
vector<CiM> cims;
Master_Ctrl ctrl(string(DATA_BASE_DIR)+"eeg.h5", string(DATA_BASE_DIR)+"model_weights.h5");
#elif CENTRALIZED_ARCH
CiM_Centralized cim(string(DATA_BASE_DIR)+"model_weights.h5");
#endif

/*----- DEFINITION -----*/
int init(){
    cout << "N_STO_INT_RES: " << N_STO_INT_RES << endl;
    cout << "N_STO_PARAMS: " << N_STO_PARAMS << endl;
    cout << "NUM. TERMS TAYLOR EXPANSION EXP APPROX: " << NUM_TERMS_EXP_TAYLOR_APPROX << endl;
    ext_sigs.master_nrst = false;
    ext_sigs.start_param_load = false;
    ext_sigs.new_sleep_epoch = false;

    // Define schedule for external events (triggered by the RISC-V processor)
    event_schedule[0] = master_nrst;
    event_schedule[1] = master_nrst_reset;
    event_schedule[4] = param_load;
    event_schedule[5] = param_load_reset;
    event_schedule[40000] = epoch_start;
    event_schedule[40001] = epoch_start_reset;

#if DISTRIBUTED_ARCH
    // Construct CiMs
    for (int16_t i = 0; i < NUM_CIM; ++i) {
        cims.emplace_back(i);
    }
#endif //DISTRIBUTED_ARCH
    return 0;
}

void copy_file(const char *src, const char *dst) {
    ifstream src_file(src, ios::binary);
    if (!src_file) {
        cerr << "Error opening source file: " << strerror(errno) << endl;
        return;
    }
    ofstream dst_file(dst, ios::binary);
    if (!dst_file) {
        cerr << "Error opening destination file: " << strerror(errno) << endl;
        return;
    }
    dst_file << src_file.rdbuf();
    src_file.close();
    dst_file.close();
}

void update_results_csv(uint32_t starting_clip_num, uint32_t num_clips, string template_csv_fp, string study_bit_type) {
    /*
    Exports a copy of the given template CSV file with the softmax argmaxes for the given clips.
    Filename: <template_csv_fp>_<N_STO_PARAMS>_<N_STO_INT_RES>_<starting_clip_index>_<end_clip_index>.csv
    */
    
    // Create a copy of template
    string results_csv_fp = template_csv_fp.substr(0, template_csv_fp.size()-4); // Remove .csv from fp
    results_csv_fp = results_csv_fp + "_" + to_string(N_STO_PARAMS) + "_" + to_string(N_STO_INT_RES) + "_" + to_string(starting_clip_num) + "_" + to_string(starting_clip_num+num_clips-1) + ".csv";
    copy_file(template_csv_fp.c_str(), results_csv_fp.c_str());

    // Update the CSV
    try {
        rapidcsv::Document results_csv(results_csv_fp, rapidcsv::LabelParams(-1, -1));
	int column = 0;
	if (study_bit_type == "INT_RES") {
	    column = N_STO_INT_RES - FIXED_POINT_ACCURACY_STUDY_START_N_STO + 3;
	} else if (study_bit_type == "PARAMS") {
	    column = N_STO_PARAMS - FIXED_POINT_ACCURACY_STUDY_START_N_STO + 3;
	}

        int starting_row = starting_clip_num + 2;
        for (int i=0; i<num_clips; i++) { results_csv.SetCell<int>(column, starting_row+i, softmax_max_indices[i]); }
        results_csv.Save(results_csv_fp);
    } catch (const exception &e) {
        cerr << "Error updating CSV: " << e.what() << endl;
        return;
    }
}

void run_sim(uint32_t clip_num, string results_csv_fp) {
    cout << ">----- STARTING SIMULATION -----<" << endl;
    uint64_t epoch_cnt = 0;
    while (1) {
#if DISTRIBUTED_ARCH
        if ((epoch_cnt == 7) && (ctrl.get_are_params_loaded() == true)) { epoch_cnt = 12500; } // Fast forward if params don't need to be loaded
        if (event_schedule.count(epoch_cnt) > 0) { event_schedule[epoch_cnt](&ext_sigs); } // Update external signals if needed
        for (auto& cim: cims) { cim.run(&ext_sigs, &bus); } // Run CiMs
        if (ctrl.run(&ext_sigs, &bus, cims, clip_num) == EVERYTHING_FINISHED) { break; }; // Run Master Controller
        bus.run(); // Run bus
        epoch_cnt++;
#elif CENTRALIZED_ARCH
        if (event_schedule.count(epoch_cnt) > 0) { event_schedule[epoch_cnt](&ext_sigs); } // Update external signals if needed
        if (cim.run(&ext_sigs, string(DATA_BASE_DIR)+"dummy_softmax_", string(DATA_BASE_DIR)+"eeg.h5", clip_num) == EVERYTHING_FINISHED) { break; }; // Run CiM
        epoch_cnt++;
#else 
        throw invalid_argument("Please define either DISTRIBUTED_ARCH or CENTRALIZED_ARCH!");
#endif
    }

    print_intermediate_value_stats();
    
    cout << "Total number of epochs: " << epoch_cnt << endl;
    cout << ">----- SIMULATION FINISHED -----<" << endl;
#if DISTRIBUTED_ARCH //FIXME: Temporary fix to avoid fault while centralized architecture is brought up
    softmax_max_indices.emplace_back(ctrl.get_softmax_max_index());
#endif //DISTRIBUTED_ARCH
    reset_stats();
}

int main(int argc, char *argv[]){
    uint32_t start_index;
    uint32_t end_index;
    string results_csv_fp;
    string study_bit_type;

    // Parse arguments
    if (argc == 1) { // No arguments provided, run on first clip and don't write to CSV
        start_index = 0;
        end_index = 0;
        results_csv_fp = "";
    } else if ((argc == 1) && (string(argv[1]) == "--help")) {
        cout << "Usage: " << argv[0] << " [--start_index] [--end_index] [--results_csv_fp]" << endl;
        cout << "--start_index: The index of the clip to use for the first run" << endl;
        cout << "--end_index: The index of the clip to use for the last run" << endl;
        cout << "--results_csv_fp: The file path to the results CSV file" << endl;
        cout << "--study_bit_type: Type of bits used to index in fixed-point accuracy study results CSV column. Either 'INT_RES' or 'PARAMS'." << endl;
	return 0;
    } else if (argc == 9) { // All arguments provided
        start_index = stoi(argv[2]);
        end_index = stoi(argv[4]);
        results_csv_fp = string(argv[6]);
    	study_bit_type = string(argv[8]);
    } else {
        throw invalid_argument("Please provide all or none of the arguments! Use --help for more information.");
    }
    
    // Check N_STO_x
    const char* env_vars[2] = {"N_STO_PARAMS", "N_STO_INT_RES"};
    bool missing_env_var = false;
    for (int i=0; i<2; i++){
    	if (getenv(env_vars[i]) == NULL) {
 	    cout << "Could not find environment variable " << env_vars[i] << "! Did you forget to define it?" << endl;
            missing_env_var = true;
        }
    }
    if (missing_env_var) { return 1; }

    // Run functional simulation
    init();
    if (start_index > end_index) { throw invalid_argument("Start index must be less than or equal to end index!"); }
    for (uint32_t i = 0; i < ((end_index + 1) - start_index); i++) {
        cout << "Run #" << i+1 << " out of " << (end_index-start_index+1) << " (clip index: " << (start_index+i) << ")" << endl;
        run_sim(start_index+i, results_csv_fp);
    }
    cout << "Finished running all simulation runs." << endl;
    if (results_csv_fp != "") { update_results_csv(start_index, (end_index-start_index+1), results_csv_fp, study_bit_type); }

    return 0;
}
