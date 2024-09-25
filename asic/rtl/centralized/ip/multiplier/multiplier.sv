`ifndef _multiplier_sv_
`define _multiplier_sv_

/* Note:
- Single-cycle fixed-point multiplier
- Output is updated when refresh is set
- Provides notice of overflow (note: only valid in the cycle after the posedge on refresh, when the data is ready)
- To avoid introducing bias, the output's fractional part is rounded as Gaussian rounding ("convergent rounding")
*/

module multiplier (
    input wire clk,
    input wire rst_n,
    input ComputeIPInterface.basic_in io
);

    localparam LSB = Q_COMP; // LSB of fractional part that we retain in output
    localparam HALF_FRACT = 'b1 << (Q_COMP - 1); // 0.5 in Q format for the bottom half of the fractional part of the temporary result

    CompFx_t in_1_q, in_2_q;
    logic signed [2*N_COMP-1:0] temp_result, rounded_temp;
    
    always_ff @ (posedge clk) begin : multiplier_logic
        if (!rst_n) begin
            temp_result <= 'd0;
            io.done <= 1'b0;
            in_1_q <= 'd0;
            in_2_q <= 'd0;
        end else begin
            temp_result <= (io.start) ? (io.in_1 * io.in_2) : temp_result;
            io.done <= io.start;
            in_1_q <= io.in_1;
            in_2_q <= io.in_2;
        end
    end

    always_comb begin : multiplier_rounding
        if (temp_result[Q_COMP-1:0] == HALF_FRACT) begin // Need convergent rounding
            if (temp_result[LSB]) begin // LSb of fractional part is 1 (odd)
                rounded_temp = {temp_result[2*N_COMP-1:LSB] + 1, {LSB{1'b0}}};
            end else begin // First integer bit is 0, so integer part is even (just truncate)
                rounded_temp = {temp_result[2*N_COMP-1:LSB], {LSB{1'b0}}};
            end
        end else if (temp_result[Q_COMP-1:0] > HALF_FRACT) begin
            rounded_temp = {temp_result[2*N_COMP-1:LSB]+1, {LSB{1'b0}}};
        end else begin
            rounded_temp = {temp_result[2*N_COMP-1:LSB], {LSB{1'b0}}};
        end
    end

    always_comb begin : multiplier_overflow
        if (io.overflow) begin
            if (rounded_temp[N_COMP+Q_COMP]) 
                io.out = {1'b0, {N_COMP-1{1'b1}}}; // Positive overflow --> Saturate to max. positive value
            else 
                io.out = {1'b1, {N_COMP-1{1'b0}}}; // Negative overflow --> Saturate to min. negative value
        end else io.out = rounded_temp[N_COMP+Q_COMP-1:Q_COMP]; // Essentially AP_TRN (truncation towards minus infinity)
    end

    // Overflow detection
    // Note that if the result is too small to be represented (and thus "0"), the overflow may trigger if one of the input was negative
    always_comb begin : multipler_overflow_detection
        logic one_input_is_zero = (in_1_q == 'd0) || (in_2_q == 'd0);
        logic out_is_neg = rounded_temp[N_COMP+Q_COMP];
        logic out_is_zero = (rounded_temp[N_COMP+Q_COMP-2:Q_COMP] == 'd0); // Ignore sign bit

        if (io.done) begin
            if (one_input_is_zero & out_is_neg) begin
                assign io.overflow = 1'b1;
            end else if ((in_1_q[N_COMP-1] == in_2_q[N_COMP-1]) & out_is_neg) begin
                assign io.overflow = 1'b1;
            end else if ((in_1_q[N_COMP-1] != in_2_q[N_COMP-1] & ~one_input_is_zero) & ~out_is_neg & ~out_is_zero) begin
                assign io.overflow = 1'b1;
            end else begin
                assign io.overflow = 1'b0;
            end
        end else begin
            assign io.overflow = 1'b0;
        end
    end

`ifdef OVERFLOW_WARNING
        always_ff @ (posedge clk) begin : overflow_detection
            if (io.overflow) begin
                $display("Overflow detected in multiplier on previous posedge of time %t (in_1_q: %h, in_2_q: %h, out: %h)", $time, in_1_q, in_2_q, io.out);
            end
        end
`endif
endmodule

`endif
