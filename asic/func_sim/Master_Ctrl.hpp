#ifndef MASTER_CTRL_H
#define MASTER_CTRL_H

#include <iostream>
#include "csv-parser/single_include/csv.hpp"
#include "Misc.hpp"

/*----- DEFINE -----*/
#define CENTRALIZED_STORAGE_WEIGHTS_KB 2048

/*----- CLASS -----*/
class Counter;

class Master_ctrl {
    private:
        enum state {
            IDLE,
            SIGNAL_LOAD,
            INFERENCE_RUNNING,
            RESET,
            INVALID = -1
        };

        float storage[CENTRALIZED_STORAGE_WEIGHTS_KB];
        Counter gen_cnt_8b;
        Counter gen_cnt_10b;
        csv::CSVReader eeg;
        state state;

        int broadcast_inst(struct instruction);

    public:
        Master_ctrl(const std::string eeg_fp);
        int reset();
        system_state run(struct ext_signals* ext_sigs, Bus* bus);
        int start_signal_load();
        int start_weights_load();
};

#endif //MASTER_CTRL_H