`ifndef _multiplier_sv_
`define _multiplier_sv_

/* Note:
- Single-cycle fixed-point multiplier
- Output is updated when refresh is set
- Provides notice of overflow (note: only valid in the cycle after the posedge on refresh, when the data is ready)
- To avoid introducing bias, the output is rounded to the nearest integer (Gaussian rounding)
*/

module multiplier (
    input wire clk,
    input wire rst_n,
    input refresh, // Update output when this is set

    // Data in 2's complement format
    input COMP_WORD_T input_q_1,
    input COMP_WORD_T input_q_2,
    output wire overflow,
    output COMP_WORD_T output_q
);

    localparam MSB = 2*N_COMP - (N_COMP - Q) - 1;
    localparam LSB = N_COMP - (N_COMP - Q);
    localparam HALF = {1'b1, {Q-1{1'b0}}};

    wire [N_COMP-1:0] unrounded_out_q;
    logic refresh_delayed;
    logic signed [2*N_COMP-1:0] temp_result;
    
    always_ff @ (posedge clk) begin : multiplier_logic
        if (!rst_n) begin
            temp_result <= 'd0;
            refresh_delayed <= 1'b0;
        end else begin
            temp_result <= (refresh) ? (input_q_1 * input_q_2) : temp_result;
            refresh_delayed <= refresh;
        end
    end

    // Warning: This amount of combinational logic might slow down Fmax significantly...might need to pipeline it
    assign unrounded_out_q = temp_result[MSB:LSB];
    assign output_q = (temp_result[Q-1+:1] && !(~temp_result[Q+:1] && temp_result[Q-1:0] == HALF)) ? unrounded_out_q + 1 : unrounded_out_q; // Gaussian rounding

    // Overflow detection
    // Note: Overflow will falsely trigger if one of the input is 0 and the other is negative. The amount of logic required to fix this is not worth it.
    assign overflow = (input_q_1[N_COMP-1] ^ input_q_2[N_COMP-1]) ? ~output_q[N_COMP-1] & refresh_delayed : output_q[N_COMP-1] & refresh_delayed; // Overflow
    always_ff @ (posedge clk) begin : overflow_detection
        if (overflow) begin
            $display("Overflow detected in multiplier at time %t", $time);
        end
    end
    
endmodule

`endif
