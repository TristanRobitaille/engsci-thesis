`ifndef _soc_interface_vh_
`define _soc_interface_vh_

import Defines::*;

interface SoCInterface;

    // Signals
    logic rst_n, new_sleep_epoch;
    logic start_eeg_load, new_eeg_data;
    logic inference_complete;
    SleepStage_t inferred_sleep_stage;
    AdcData_t eeg;

    modport cim (
        input rst_n, new_sleep_epoch,
        input start_eeg_load, new_eeg_data, eeg,
        output inference_complete,
        output inferred_sleep_stage
    );

    modport soc (
        output rst_n, new_sleep_epoch,
        output start_eeg_load, new_eeg_data, eeg,
        input inference_complete,
        input inferred_sleep_stage
    );

endinterface
`endif
