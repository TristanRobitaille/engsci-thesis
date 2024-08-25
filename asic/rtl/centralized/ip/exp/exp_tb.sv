module exp_tb (
    input wire clk,
    input wire rst_n,

    input wire start,
    input CompFx_t input_q,

    output wire busy, done, overflow,
    output CompFx_t output_q
);

    ComputeIPInterface adder_io ();
    ComputeIPInterface mult_io ();
    ComputeIPInterface io ();

    // Internal wires
    assign io.start = start;
    assign io.in_1 = input_q;
    assign io.in_2 = 'd0;
    assign busy = io.busy;
    assign done = io.done;
    assign output_q = io.out;
    assign overflow = io.overflow;

    adder adder (.clk, .rst_n, .io(adder_io));
    multiplier mult (.clk, .rst_n, .io(mult_io));
    exp exp (.clk, .rst_n, .adder_io, .mult_io, .io);

endmodule // exp_wrapper_tb
