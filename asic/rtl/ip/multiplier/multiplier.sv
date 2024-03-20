/* Note:
- Single-cycle fixed-point multiplier
- Output is updated when refresh is set
- Provides notice of overflow
*/
// `include "../types.svh"
module multiplier (
    input wire clk,
    input wire rst_n,
    input refresh, // Update output when this is set

    // Data in 2's complement format
    input wire signed [N_COMP-1:0] input_q_1,
    input wire signed [N_COMP-1:0] input_q_2,
    output logic signed [N_COMP-1:0] output_q
);

    logic signed [2*N_COMP-1:0] temp_result;
    always_ff @ (posedge clk) begin : multiplier_logic
        if (!rst_n) begin
            temp_result <= '0;
        end else if (refresh) begin
            temp_result <= signed'(input_q_1[N_COMP-2:0]) * signed'(input_q_2[N_COMP-2:0]);
        end
    end
    assign output_q = temp_result[N_COMP-1+Q:Q];
endmodule
