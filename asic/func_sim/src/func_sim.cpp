#include <func_sim.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- GLOBAL -----*/
uint32_t epoch_cnt;
map<int, FcnPtr> event_schedule;
struct ext_signals ext_sigs;

Bus bus;
std::vector<CiM> cims;
Master_Ctrl ctrl((std::string(DATA_BASE_DIR)+"eeg.h5"), (std::string(DATA_BASE_DIR)+"model_weights.h5"));

int init(){
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
void update_results_csv(uint32_t inferred_sleep_stage, uint32_t clip_num, string results_csv_fp) {
    rapidcsv::Document results_csv(results_csv_fp);
    results_csv.SetCell<int>((NUM_FRAC_BITS-1+3), (clip_num+1), inferred_sleep_stage); // +3 and +1 are to account for headers
    results_csv.Save();
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
    if (results_csv_fp != "") { update_results_csv(ctrl.get_inferred_sleep_stage(), clip_num, results_csv_fp); }
    reset_stats();
}

int main(int argc, char *argv[]){
    if (argc == 1) { throw std::invalid_argument("No arguments provided!"); }
    if (std::string(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0] << " [--start_index] [--end_index] [--results_csv_fp]" << std::endl;
        std::cout << "--start_index: The index of the clip to use for the first run" << std::endl;
        std::cout << "--end_index: The index of the clip to use for the last run" << std::endl;
        std::cout << "--results_csv_fp: The file path to the results CSV file" << std::endl;
        return 0;
    }

    init();
    uint32_t start_index = std::stoi(argv[2]);
    uint32_t end_index = std::stoi(argv[4]);
    if (start_index > end_index) { throw std::invalid_argument("Start index must be less than or equal to end index!"); }
    for (uint32_t i = 0; i < ((end_index + 1) - start_index); i++) {
        cout << "Run #" << i << " out of " << (end_index-start_index+1) << " (clip index: " << (start_index+i) << ")" << std::endl;
        run_sim(start_index+i, argv[6]);
    }
    cout << "Finished running all simulation runs." << endl;

    return 0;
}
