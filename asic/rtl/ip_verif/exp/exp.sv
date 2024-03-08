/* Note:
    - Approxmiate the exponential function using the e^x = 2^(x/ln(2)) identity and Taylor series expansion.
    - A pulse on start will initiate the computation process.
    - Signal done is asserted when the computation is complete, and persists until the next start pulse.
    - Signal busy is asserted during the computation process.
    - Inputs must remain stable until done is asserted.
    - Computation time is 24 cycles.
*/
module exp # (
    parameter N = 22, // 22b total
    parameter Q = 10  // 10b fractional
)(
    input wire clk,
    input wire rst_n,
    input wire start,

    // Adder module
    input wire signed [N-1:0] adder_output,
    output logic adder_refresh,
    output logic signed [N-1:0] adder_input_1, adder_input_2,

    // Multiplier module
    input wire signed [N-1:0] mult_output,
    output logic mult_refresh,
    output logic signed [N-1:0] mult_input_1, mult_input_2,

    input wire signed [N-1:0] input_q,
    output logic busy, done,
    output logic signed [N-1:0] output_q
);

// Signals
logic signed [N-1:0] input_mapped, taylor_sum;

// Constants
wire signed [N-1:0] ln2_inv, Taylor_mult_1, Taylor_mult_2, Taylor_mult_3;
assign ln2_inv          = 22'b1_0111000101; // 1/ln(2) = 1.44269504089
assign Taylor_mult_1    = 22'b0_1011000110; // ln(2) = 0.69314718056
assign Taylor_mult_2    = 22'b0_0011110110; // ln(2)^2/2! = 0.46209812037
assign Taylor_mult_3    = 22'b0_0000111001; // ln(2)^3/3! = 0.24022650696

enum [2:0] {IDLE, UPDATE_SUM, MULT, TAYLOR_TERM_1, TAYLOR_TERM_2, TAYLOR_TERM_3, FINISH} state, next_state;
enum [1:0] {ITERATION_1_SUBSTATE, ITERATION_2_SUBSTATE, ITERATION_3_SUBSTATE, ITERATION_4_SUBSTATE} sub_state, sub_state_sum;

always_ff @ (posedge clk or negedge rst_n) begin : exp_FSM
    if (!rst_n) begin
        busy <= 0;
        done <= 0;
        output_q <= '0;
        mult_refresh <= 0;
        adder_refresh <= 0;
        state <= IDLE;
        next_state <= IDLE;
        sub_state <= ITERATION_1_SUBSTATE;
        adder_input_1 <= '0;
        adder_input_2 <= '0;
        mult_input_1 <= '0;
        mult_input_2 <= '0;
        taylor_sum <= '0;
    end else begin
        case (state)
            IDLE: begin
                busy <= start;
                done <= (start) ? 0 : done;
                state <= (start) ? MULT : IDLE;
                next_state <= TAYLOR_TERM_1;
                sub_state <= ITERATION_1_SUBSTATE;
                mult_refresh <= start; // Multiply is single-cycle
                mult_input_1 <= input_q;
                mult_input_2 <= ln2_inv;
                taylor_sum <= 22'b1_0000000000; // 1st term of Taylor series is 1
            end

            UPDATE_SUM: begin
                if (sub_state_sum == ITERATION_1_SUBSTATE) begin
                    adder_input_1 <= taylor_sum;
                    adder_input_2 <= mult_output;
                    adder_refresh <= 1;
                    mult_refresh <= 0;
                    sub_state_sum <= ITERATION_2_SUBSTATE;
                end else begin
                    state <= next_state;
                    sub_state_sum <= ITERATION_1_SUBSTATE;
                end
            end

            MULT: begin
                state <= next_state;
            end

            TAYLOR_TERM_1: begin // x_fractional * ln(2)
                if (sub_state == ITERATION_1_SUBSTATE) begin // Square fractional part of input
                    input_mapped <= mult_output;
                    mult_input_1 <= {{(N-Q){input_q[N-1]}}, mult_output[Q-1:0]}; // Grab fractional part
                    mult_input_2 <= Taylor_mult_1;
                    next_state <= TAYLOR_TERM_1;
                    state <= MULT;
                    sub_state <= ITERATION_2_SUBSTATE;
                end else begin
                    state <= UPDATE_SUM;
                    next_state <= TAYLOR_TERM_2;
                    sub_state <= ITERATION_1_SUBSTATE;
                end
            end

            TAYLOR_TERM_2: begin
                if (sub_state == ITERATION_1_SUBSTATE) begin // Square fractional part of input
                    taylor_sum <= adder_output;
                    mult_input_1 <= {{(N-Q){input_q[N-1]}}, input_mapped[Q-1:0]}; // Grab fractional part (while keeping the sign)
                    mult_input_2 <= {{(N-Q){input_q[N-1]}}, input_mapped[Q-1:0]}; // Grab fractional part (while keeping the sign)
                    mult_refresh <= 1;
                    adder_refresh <= 0;
                    state <= MULT;
                    next_state <= TAYLOR_TERM_2;
                    sub_state <= ITERATION_2_SUBSTATE;
                end else if (sub_state == ITERATION_2_SUBSTATE) begin
                    mult_input_1 <= mult_output;
                    mult_input_2 <= Taylor_mult_2;
                    state <= MULT;
                    next_state <= TAYLOR_TERM_2;
                    sub_state <= ITERATION_3_SUBSTATE;
                end else if (sub_state == ITERATION_3_SUBSTATE) begin
                    state <= UPDATE_SUM;
                    next_state <= TAYLOR_TERM_3;
                    sub_state <= ITERATION_1_SUBSTATE;
                end
            end

            TAYLOR_TERM_3: begin
                if (sub_state == ITERATION_1_SUBSTATE) begin // Square fractional part of input
                    taylor_sum <= adder_output;
                    mult_refresh <= 1;
                    mult_input_1 <= {{(N-Q){input_q[N-1]}}, input_mapped[Q-1:0]}; // Grab fractional part (while keeping the sign)
                    mult_input_2 <= {{(N-Q){input_q[N-1]}}, input_mapped[Q-1:0]}; // Grab fractional part (while keeping the sign)
                    state <= MULT;
                    next_state <= TAYLOR_TERM_3;
                    sub_state <= ITERATION_2_SUBSTATE;
                end else if (sub_state == ITERATION_2_SUBSTATE) begin // Cube fractional part of input
                    mult_input_2 <= mult_output;
                    sub_state <= ITERATION_3_SUBSTATE;
                    state <= MULT;
                end else if (sub_state == ITERATION_3_SUBSTATE) begin
                    mult_input_1 <= mult_output; // Grab fractional part (while keeping the sign)
                    mult_input_2 <= Taylor_mult_3;
                    sub_state <= ITERATION_4_SUBSTATE;
                    state <= MULT;
                end else if (sub_state == ITERATION_4_SUBSTATE) begin
                    state <= UPDATE_SUM;
                    next_state <= FINISH;
                end
            end

            FINISH: begin
                output_q <= (input_q[N-1]) ? (adder_output >> ~input_mapped[N-1:Q]) : (adder_output << input_mapped[N-1:Q]); // Multiply fractional part by integer part (2^floor(x/ln(2)))
                done <= 1;
                state <= IDLE;
            end

            default: state <= IDLE;
        endcase
    end
end

wire [N-1:Q] input_mapped_complement;
assign input_mapped_complement = ~input_mapped[N-1:Q];

endmodule
