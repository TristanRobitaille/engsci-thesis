`ifndef _soc_interface_vh_
`define _soc_interface_vh_

import Defines::*;

interface SoCInterface;

    // Signals
    logic rst_n, new_sleep_epoch;
    logic inference_complete;
    modport data_out (
        input rst_n, new_sleep_epoch,
        output inference_complete
    );

endinterface
`endif
