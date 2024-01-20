#include "Master_Ctrl.hpp"
#include "csv-parser/single_include/csv.hpp"

/*----- DECLARATIONS -----*/
Master_ctrl::Master_ctrl(string eeg_fp){
    csv::CSVReader eeg(eeg_fp);
}
int Master_ctrl::reset(){return 0;}
int Master_ctrl::load_signal(){
    /* Sequentially parses the input EEG file and forwards it to the correct CiM */
    return 0;
}
int Master_ctrl::load_weights(){return 0;}
int Master_ctrl::broadcast_inst(){return 0;}
