`ifndef _soc_interface_vh_
`define _soc_interface_vh_

import Defines::*;

interface SoCInterface;

    // Signals
    logic rst_n, new_sleep_epoch;
    logic start_eeg_load, new_eeg_data;
    logic inference_complete;
    AdcData_t eeg;
    modport data_out (
        input rst_n, new_sleep_epoch,
        input start_eeg_load, new_eeg_data, eeg,
        output inference_complete
    );

endinterface
`endif
