import Defines::*;

interface CounterInterface # (
    parameter int WIDTH = 8
);

    // Signals
    logic inc, rst_n;
    logic [WIDTH-1:0] cnt;

    modport data_in (
        input inc, rst_n,
        output cnt
    );

    modport data_out (
        output inc, rst_n,
        input cnt
    );

endinterface
