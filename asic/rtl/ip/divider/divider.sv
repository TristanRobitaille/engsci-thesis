/* Note:
    - The divider is implemented using a non-restoring division algorithm.
    - A pulse on start will initiate the division process.
    - Signal done is asserted when the division is complete, and persists until the next start pulse.
    - Signal overflow is asserted when the division result has overflowed, and persists until the next start pulse.
    - Signal busy is asserted during the division process.
    - Inputs must remain stable until done is asserted.
    - Computation time is N+Q+2 cycles, where N is the total number of bits, and Q is the number of fractional bits.
*/

module divider #(
    parameter N = 22, // 22b total
    parameter Q = 10 // 10b fractional
)(
    input logic clk,
    input logic rst_n,
    input logic start,

    // Data in 2's complement format
    output logic busy, done, dbz, overflow,
    input logic signed [N-1:0] dividend,
    input logic signed [N-1:0] divisor,
    output logic signed [N-1:0] output_q
);

    logic [$clog2(N):0] count;
    logic [N-2:0] quo, quo_next, dividend_abs, divisor_abs;
    logic [N-1:0] acc, acc_next;

    always_comb begin : divider_comb
        if (acc >= {1'b0, divisor_abs}) begin
            acc_next = acc - divisor_abs;
            {acc_next, quo_next} = {acc_next[N-2:0], quo, 1'b1};
        end else begin
            {acc_next, quo_next} = {acc, quo} << 1;
        end
    end

    always_comb begin : divider_abs_comb
        dividend_abs = (dividend[N-1]) ? -dividend[N-2:0] : dividend[N-2:0]; // Take abs value 
        divisor_abs = (divisor[N-1]) ? -divisor[N-2:0] : divisor[N-2:0]; // Take abs value
    end

    enum [1:0] {IDLE, CALC, SIGN} state;
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
                    count <= 'b0;
                    done <= start ? 0 : done;
                    overflow <= start ? 0 : overflow;
                    busy <= start;
                    state <= start ? CALC : IDLE;
                    {acc, quo} <= {{N-1{1'b0}}, dividend_abs, 1'b0};
                    dbz <= (divisor == '0);
                end
                CALC: begin
                    if (count == N+Q-1) begin
                        state <= SIGN;
                    end else begin
                        count <= count + 1;
                        acc <= acc_next;
                        quo <= quo_next;
                        state <= CALC;
                        overflow <= (count == N-1) && (quo_next[N-2:N-1-Q] != '0);
                    end
                end
                SIGN: begin
                    if (quo != 0) output_q <= (divisor[N-1] ^ dividend[N-1]) ? {1'b1, -quo} : {1'b0, quo}; // If sign of inputs is different, then quotient is negative
                    done <= 1;
                    busy <= 0;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
