#include <Func_Sim.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- GLOBAL -----*/
uint32_t epoch_cnt;
map<int, FcnPtr> event_schedule;
struct ext_signals ext_sigs;
vector<uint32_t> softmax_max_indices;

Bus bus;
vector<CiM> cims;
Master_Ctrl ctrl((string(DATA_BASE_DIR)+"eeg.h5"), (string(DATA_BASE_DIR)+"model_weights.h5"));

int init(){
    cout << "N_STO: " << N_STO << endl;
    ext_sigs.master_nrst = false;
    ext_sigs.new_sleep_epoch = false;

    // Define schedule for external events (triggered by he RISC-V processor)
    event_schedule[0] = master_nrst;
    event_schedule[2] = master_nrst_reset;
    event_schedule[4] = master_param_load;
    event_schedule[6] = master_param_load_reset;
    event_schedule[40000] = epoch_start;
    event_schedule[40002] = epoch_start_reset;

    // Construct CiMs
    for (int16_t i = 0; i < NUM_CIM; ++i) {
        cims.emplace_back(i);
    }
    return 0;
}

/*----- FUNCTIONS -----*/
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

void update_results_csv(uint32_t starting_clip_num, uint32_t num_clips, string template_csv_fp) {
    /*
    Exports a copy of the given template CSV file with the softmax argmaxes for the given clips.
    Filename: <template_csv_fp>_<N_STO>_<starting_clip_index>_<end_clip_index>.csv
    */
    
    // Create a copy of template
    string results_csv_fp = template_csv_fp.substr(0, template_csv_fp.size()-4); // Remove .csv from fp
    results_csv_fp = results_csv_fp + "_" + to_string(N_STO) + "_" + to_string(starting_clip_num) + "_" + to_string(starting_clip_num+num_clips-1) + ".csv";
    copy_file(template_csv_fp.c_str(), results_csv_fp.c_str());

    // Update the CSV
    try {
        rapidcsv::Document results_csv(results_csv_fp, rapidcsv::LabelParams(-1, -1));
        int column = N_STO - FIXED_POINT_ACCURACY_STUDY_START_N_STO + 3;
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
        if ((epoch_cnt == 7) && (ctrl.get_are_params_loaded() == true)) { epoch_cnt = 12500; } // Fast forward if params don't need to be loaded
        if (event_schedule.count(epoch_cnt) > 0) { event_schedule[epoch_cnt](&ext_sigs); } // Update external signals if needed
        for (auto& cim: cims) { cim.run(&ext_sigs, &bus); } // Run CiMs
        if (ctrl.run(&ext_sigs, &bus, cims, clip_num) == EVERYTHING_FINISHED) { break; }; // Run Master Controller
        bus.run(); // Run bus
        epoch_cnt++;
    }

    print_intermediate_value_stats();
    cout << "Total number of epochs: " << epoch_cnt << endl;
    cout << ">----- SIMULATION FINISHED -----<" << endl;
    softmax_max_indices.emplace_back(ctrl.get_softmax_max_index());
    reset_stats();
}

int main(int argc, char *argv[]){
    uint32_t start_index;
    uint32_t end_index;
    string results_csv_fp;

    if (argc == 1) { // No arguments provided, run on first clip and don't write to CSV
        start_index = 0;
        end_index = 0;
        results_csv_fp = "";
    } else if ((argc == 1) && (string(argv[1]) == "--help")) {
        cout << "Usage: " << argv[0] << " [--start_index] [--end_index] [--results_csv_fp]" << endl;
        cout << "--start_index: The index of the clip to use for the first run" << endl;
        cout << "--end_index: The index of the clip to use for the last run" << endl;
        cout << "--results_csv_fp: The file path to the results CSV file" << endl;
        return 0;
    } else if (argc == 7) { // All arguments provided
        start_index = stoi(argv[2]);
        end_index = stoi(argv[4]);
        results_csv_fp = string(argv[6]);
    } else {
        throw invalid_argument("Please provide all or none of the arguments! Use --help for more information.");
    }

    init();
    if (start_index > end_index) { throw invalid_argument("Start index must be less than or equal to end index!"); }
    for (uint32_t i = 0; i < ((end_index + 1) - start_index); i++) {
        cout << "Run #" << i+1 << " out of " << (end_index-start_index+1) << " (clip index: " << (start_index+i) << ")" << endl;
        run_sim(start_index+i, results_csv_fp);
    }
    cout << "Finished running all simulation runs." << endl;
    if (results_csv_fp != "") { update_results_csv(start_index, (end_index-start_index+1), results_csv_fp); }

    return 0;
}