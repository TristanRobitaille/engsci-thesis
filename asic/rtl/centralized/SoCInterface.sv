import Defines::*;

interface SoCInterface;

    // Signals
    logic rst_n, new_sleep_epoch;

    modport data_out (
        input rst_n, new_sleep_epoch
    );

endinterface
