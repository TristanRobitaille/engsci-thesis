/* Note:
- Single-cycle fixed-point adder
- Output is updated when refresh is set
- Provides notice of overflow
*/
module adder #(
    parameter N = 22    // 22b total
)(
    input logic clk,
    input logic rst_n,
    input refresh, // Update output when this is set

    // Data in 2's complement format
    input wire signed [N-1:0] input_q_1,
    input wire signed [N-1:0] input_q_2,
    output logic signed [N-1:0] output_q,
    output wire overflow
);

    always_ff @(posedge clk or negedge rst_n) begin : adder_logic
        if (!rst_n) begin
            output_q <= '0;
        end else if (refresh) begin
            output_q <= input_q_1 + input_q_2;
        end
    end

    assign overflow = (input_q_1[N-1] == input_q_2[N-1]) && (input_q_1[N-1] != output_q[N-1]);

endmodule // adder
