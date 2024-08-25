`ifndef _multiplier_tb_sv_
`define _multiplier_tb_sv_

module multiplier_tb (
    input wire clk,
    input wire rst_n,

    input start,
    input CompFx_t input_q_1, input_q_2,
    output CompFx_t output_q,
    output wire overflow
);

    ComputeIPInterface io ();
  
    assign io.start = start;
    assign io.in_1 = input_q_1;
    assign io.in_2 = input_q_2;
    assign output_q = io.out;
    assign overflow = io.overflow;

    multiplier multiplier (.clk, .rst_n, .io);
endmodule
`endif
