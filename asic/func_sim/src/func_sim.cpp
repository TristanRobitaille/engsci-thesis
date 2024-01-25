#include <func_sim.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- GLOBAL -----*/
uint32_t epoch_cnt;
map<int, FcnPtr> event_schedule;
struct ext_signals ext_sigs;

Bus bus;
CiM cims[NUM_CIM];
Master_ctrl ctrl("data/eeg.h5", "data/params.h5");

int init(){
    ext_sigs.master_nrst = false;
    ext_sigs.new_sleep_epoch = false;

    // Define schedule for external events
    event_schedule[0] = master_nrst;
    event_schedule[2] = master_nrst_reset;
    event_schedule[4] = epoch_start;
    event_schedule[5] = epoch_start_reset;

    // Construct CiMs
    for (int16_t i = 0; i < NUM_CIM; ++i) { cims[i] = CiM(i); }
    return 0;
}

/*----- MAIN -----*/
int main(){
    cout << ">----- STARTING SIMULATION -----<" << endl;

    init();

    while (1) {
        if (event_schedule.count(epoch_cnt) > 0) { event_schedule[epoch_cnt](&ext_sigs); } // Update external signals if needed
        for (auto& cim: cims) { cim.run(&ext_sigs, &bus); } // Run CiMs
        if (ctrl.run(&ext_sigs, &bus) == DONE) { break; }; // Run Master Controller
        bus.run(); // Run bus
        epoch_cnt++;
    }

    cout << ">----- SIMULATION DONE -----<" << endl;
    return 0;
}
