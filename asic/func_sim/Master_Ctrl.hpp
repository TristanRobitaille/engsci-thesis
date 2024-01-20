#include <iostream>
#include "csv-parser/single_include/csv.hpp"

/*----- NAMESPACE -----*/
using namespace std;

/*----- CLASS -----*/
class Master_ctrl {
    private:
        int broadcast_inst();
        
    public:
        Master_ctrl(string eeg_fp);
        int reset();
        int load_signal();
        int load_weights();
};