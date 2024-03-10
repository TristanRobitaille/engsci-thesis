`include "ip/adder/adder.sv"
`include "ip/counter/counter.sv"
`include "ip/divider/divider.sv"
`include "ip/multiplier/multiplier.sv"
`include "ip/exp/exp.sv"
`include "ip/sqrt/sqrt.sv"

module top # (
    parameter N = 22, // 22b total
    parameter Q = 10  // 10b fractional
) (
    input wire clk, rst_n,

    // Add
    input wire refresh_add,
    input wire signed [N-1:0] input_q_1, input_q_2,
    output logic overflow,
    output logic signed [N-1:0] output_q,

    // Div
    input wire start_div,
    input wire signed [N-1:0] dividend, divisor,
    output logic busy_div, done_div, dbz_div, overflow_div,
    output logic signed [N-1:0] div_out_q,

    // Multiplier
    input wire refresh_mult,
    input wire signed [N-1:0] mult_input_q_1, mult_input_q_2,
    output logic overflow_mult,
    output logic [N-1:0] mult_out_q,

    // Sqrt
    input wire start_sqrt,
    input wire signed [N-1:0] rad_q,
    output logic busy, done, neg_rad,
    output logic [N-1:0] root_q,

    // Counters
    input wire inc_0, inc_1, inc_2,
    output logic [9:0] cnt_0, cnt_1, cnt_2
);

    adder #(N=N) add_0_inst (.clk(clk), .rst_n(rst_n), .refresh(refresh_add), .overflow(overflow), .input_q_1(input_q_1), .input_q_2(input_q_2), .output_q(output_q));
    divider #(N=N, Q=Q) divider_0_inst (.clk(clk), .rst_n(rst_n), .start(start_div), .busy(busy_div), .done(done_div), .dbz(dbz_div), .overflow(overflow_div), .dividend(dividend), .divisor(divisor), .output_q(div_out_q));
    multiplier #(N=N, Q=Q) multiplier_0 (.clk(clk), .rst_n(rst_n), .refresh(refresh_mult), .input_q_1(mult_input_q_1), .input_q_2(mult_input_q_2), .output_q(mult_out_q), .overflow(overflow_mult));
    sqrt #(N=N, Q=Q) sqrt_0(.clk(clk), .rst_n(rst_n), .start(start_sqrt), .rad_q(rad_q), .busy(busy), .done(done), .neg_rad(neg_rad), .root_q(root_q));
    
    // exp exp_0_inst (.clk(clk), .rst_n(rst_n), .start(), .);

    counter #(WIDTH=10, MODE=0) cnt_0_inst (.clk(clk), .rst_n(rst_n), .inc(inc_0), .cnt(cnt_0));
    counter #(WIDTH=10, MODE=0) cnt_1_inst (.clk(clk), .rst_n(rst_n), .inc(inc_1), .cnt(cnt_1));
    counter #(WIDTH=10, MODE=0) cnt_2_inst (.clk(clk), .rst_n(rst_n), .inc(inc_2), .cnt(cnt_2));

endmodule
