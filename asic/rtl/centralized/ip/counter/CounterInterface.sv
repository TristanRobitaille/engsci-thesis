import Defines::*;

interface CounterInterface # (
    parameter int WIDTH
);

    // Signals
    logic inc, rst_n;
    logic [WIDTH-1:0] cnt;

    modport data_out (
        input inc, rst_n,
        output cnt
    );

endinterface
