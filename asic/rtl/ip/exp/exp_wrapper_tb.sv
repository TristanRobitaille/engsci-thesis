/* Note:
- Wrapper for exp module testbench as we need to connect it to add, mult and global temp registers
*/
`include "../adder/adder.sv"
`include "../multiplier/multiplier.sv"

module exp_wrapper_tb # (
    parameter N = 22, // 22b total
    parameter Q = 10  // 10b fractional
)(
    input wire clk,
    input wire rst_n,
    input wire start,

    input wire signed [N-1:0] input_q,
    output wire busy, done,
    output logic signed [N-1:0] output_q
);

    // Internal wires
    logic adder_refresh, mult_refresh;
    logic signed [N-1:0] adder_output, mult_output, mult_add_input_1, mult_add_input_2;
    logic signed [N-1:0] adder_input_1_wire, adder_input_2_wire, mult_input_1_wire, mult_input_2_wire;

    exp #(N,Q) exp (.clk(clk), .rst_n(rst_n), .start(start), 
                    .adder_output(adder_output), .adder_refresh(adder_refresh), .adder_input_1(adder_input_1_wire), .adder_input_2(adder_input_2_wire), 
                    .mult_output(mult_output), .mult_refresh(mult_refresh), .mult_input_1(mult_input_1_wire), .mult_input_2(mult_input_2_wire),
                    .input_q(input_q), .busy(busy), .done(done), .output_q(output_q));
    multiplier #(N,Q) multiplier (.clk(clk), .rst_n(rst_n), .refresh(mult_refresh), .input_q_1(mult_input_1_wire), .input_q_2(mult_input_2_wire), .output_q(mult_output), /* verilator lint_off PINCONNECTEMPTY */.overflow()/* verilator lint_off PINCONNECTEMPTY */);
    adder #(N) adder (.clk(clk), .rst_n(rst_n), .refresh(adder_refresh), .input_q_1(adder_input_1_wire), .input_q_2(adder_input_2_wire), .output_q(adder_output), /* verilator lint_off PINCONNECTEMPTY */.overflow()/* verilator lint_off PINCONNECTEMPTY */);

endmodule // exp_wrapper_tb
