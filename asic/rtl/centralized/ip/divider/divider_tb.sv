`ifndef _divider_tb_sv_
`define _divider_tb_sv_

module divider_tb (
    input wire clk,
    input wire rst_n,

    input start,
    input CompFx_t dividend, divisor,
    output busy, done,
    output CompFx_t output_q,
    output wire overflow
);

    ComputeIPInterface io ();
  
    assign io.start = start;
    assign io.in_1 = dividend;
    assign io.in_2 = divisor;

    assign busy = io.busy;
    assign done = io.done;
    assign output_q = io.out;
    assign overflow = io.overflow;

    divider divider (.clk, .rst_n, .io);
endmodule
`endif
