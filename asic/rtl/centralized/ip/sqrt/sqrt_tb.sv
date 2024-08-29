`ifndef _sqrt_tb_sv_
`define _sqrt_tb_sv_

module sqrt_tb (
    input wire clk,
    input wire rst_n,

    input start,
    input CompFx_t rad_q,
    output wire done, overflow,
    output CompFx_t root_q
);

    ComputeIPInterface io ();
  
    assign io.start = start;
    assign io.in_1 = rad_q;
    assign io.in_2 = 'd0;

    assign done = io.done;
    assign root_q = io.out;
    assign overflow = io.overflow;

    sqrt sqrt (.clk, .rst_n, .io);
endmodule
`endif
