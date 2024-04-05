`ifndef _divider_sv_
`define _divider_sv_

/* Note:
    - The divider is implemented using a non-restoring division algorithm.
    - A pulse on start will initiate the division process.
    - Signal done is asserted when the division is complete, and persists until the next start pulse.
    - Signal overflow is asserted when the division result has overflowed, and persists until the next start pulse.
    - Signal busy is asserted during the division process.
    - Inputs must remain stable until done is asserted.
    - Computation time is N_COMP+Q+3 cycles, where N_COMP is the total number of bits, and Q is the number of fractional bits.
*/

`include "../../parameters.svh"
`include "../../types.svh"

module divider (
    input wire clk,
    input wire rst_n,
    input wire start,

    // Data in 2's complement format
    output logic busy, done, dbz, overflow,
    input COMP_WORD_T dividend,
    input COMP_WORD_T divisor,
    output COMP_WORD_T output_q
);

    logic [$clog2(N_COMP):0] count;
    logic [N_COMP-2:0] quo, quo_next, dividend_abs, divisor_abs;
    logic [N_COMP-1:0] acc, acc_next;

    always_comb begin : divider_comb
        if (acc >= {1'b0, divisor_abs}) begin
            acc_next = acc - divisor_abs;
            {acc_next, quo_next} = {acc_next[N_COMP-2:0], quo, 1'b1};
        end else begin
            {acc_next, quo_next} = {acc, quo} << 1;
        end
    end

    always_comb begin : divider_abs_comb
        dividend_abs = (dividend[N_COMP-1]) ? -dividend[N_COMP-2:0] : dividend[N_COMP-2:0]; // Take abs value 
        divisor_abs = (divisor[N_COMP-1]) ? -divisor[N_COMP-2:0] : divisor[N_COMP-2:0]; // Take abs value
    end

    enum [1:0] {IDLE, CALC, SIGN, ROUND} state;
    always_ff @ (posedge clk) begin : divider_FSM
        if (!rst_n) begin
            state       <= IDLE;
            output_q    <= '0;
            busy        <= 0;
            overflow    <= 0;
            dbz         <= 0;
            done        <= 0;
        end else begin
            unique case (state)
                IDLE: begin
                    count <= 'd0;
                    done <= 0;
                    overflow <= start ? 0 : overflow;
                    busy <= start;
                    state <= start ? CALC : IDLE;
                    {acc, quo} <= {{N_COMP-1{1'b0}}, dividend_abs, 1'b0};
                    dbz <= (divisor == '0);
                end
                CALC: begin
                    if (count == N_COMP+Q-1) begin
                        state <= ROUND;
                    end else begin
                        count <= count + 1;
                        acc <= acc_next;
                        quo <= quo_next;
                        state <= CALC;
                        overflow <= (count == N_COMP-1) && (quo_next[N_COMP-2:N_COMP-1-Q] != '0);
                    end
                end
                ROUND: begin // Gaussian rounding
                    state <= SIGN;
                    if (quo_next[0] == 1'b1) begin // Next digit is 1, so consider rounding
                        if (quo[0] == 1'b1 || acc_next[N_COMP-1:1] != 0) quo <= quo + 1; // Round up if quotient is odd or remainder is non-zero
                    end
                end
                SIGN: begin
                    if (quo != 0) output_q <= (divisor[N_COMP-1] ^ dividend[N_COMP-1]) ? {1'b1, -quo} : {1'b0, quo}; // If sign of inputs is different, then quotient is negative
                    done <= 1;
                    busy <= 0;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

    always_ff @ (posedge clk) begin : div_assertions
        if (start && ($countones(divisor[N_COMP-2:0]) == 1)) begin
            $display("Warning from divide module: Divisor is a power of 2, consider using a bit-shift instead of division.");
        end
    end
endmodule

`endif
