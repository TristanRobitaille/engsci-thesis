`ifndef _adder_top_sv_
`define _adder_top_sv_

`include "types.svh"

module adder_top (
    input wire clk,
    input wire rst_n,

    input refresh_0, refresh_1,

    // Data in 2's complement format
    input COMP_WORD_T input_q_1_0,
    input COMP_WORD_T input_q_2_0,
    output COMP_WORD_T output_q_0,
    output wire overflow_0,

    // Data in 2's complement format
    input COMP_WORD_T input_q_1_1,
    input COMP_WORD_T input_q_2_1,
    output COMP_WORD_T output_q_1,
    output wire overflow_1
);

    adder adder_0 (.clk(clk), .rst_n(rst_n), .refresh(refresh_0), .input_q_1(input_q_1_0), .input_q_2(input_q_2_0), .output_q(output_q_0), .overflow(overflow_0));
    adder adder_1 (.clk(clk), .rst_n(rst_n), .refresh(refresh_1), .input_q_1(input_q_1_1), .input_q_2(input_q_2_1), .output_q(output_q_1), .overflow(overflow_1));

endmodule
`endif
