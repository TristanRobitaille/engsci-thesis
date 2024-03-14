/* Note:
- Single-cycle fixed-point multiplier
- Output is updated when refresh is set
- Provides notice of overflow
*/
module multiplier #(
    parameter N = 22, // 22b total
    parameter Q = 10 // 10b fractional
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

    logic signed [2*N-1:0] temp_result;
    always_ff @ (posedge clk) begin : multiplier_logic
        if (!rst_n) begin
            temp_result <= '0;
        end else if (refresh) begin
            temp_result <= signed'(input_q_1[N-2:0]) * signed'(input_q_2[N-2:0]);
        end
    end
    assign output_q = temp_result[N-1+Q:Q];
    assign overflow = (input_q_1[N-1] ^ input_q_2[N-1]) ? (~output_q[N-1]) : (output_q[N-1] != input_q_1[N-1]); // Overflow

endmodule
