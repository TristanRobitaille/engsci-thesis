`ifndef _exp_sv_
`define _exp_sv_

/* Note:
    - Approxmiate the exponential function using the e^x = 2^(x/ln(2)) identity and Taylor series expansion.
    - A pulse on start will initiate the computation process.
    - Signal done (pulse) is asserted when the computation is complete.
    - Signal busy is asserted during the computation process.
    - Computation time is 24 cycles.
*/

module exp (
    input wire clk,
    input wire rst_n,

    input ComputeIPInterface.basic_in io,
    output ComputeIPInterface.basic_out adder_io,
    output ComputeIPInterface.basic_out mult_io
);

function automatic CompFx_t real_to_fixed(input real decimal_float);
    real scale_factor;
    CompFx_t fixed_value;
    begin
        // Convert the real number to fixed-point by scaling and rounding
        fixed_value = CompFx_t'($rtoi(decimal_float * (1 << Q_COMP)));
        return fixed_value;
    end
endfunction

// Constants
localparam CompFx_t ln2_inv = real_to_fixed(1.44269504089); // 1/ln(2)
localparam CompFx_t Taylor_mult[0:3] = {
    real_to_fixed(1.0),             // 1.0
    real_to_fixed(0.69314718056),   // ln(2)
    real_to_fixed(0.24022650700),   // ln(2)^2/2!
    real_to_fixed(0.05550410866)    // ln(2)^3/3!
};

enum [2:0] {IDLE, UPDATE_SUM, MULT, TAYLOR_TERM_1, TAYLOR_TERM_2, TAYLOR_TERM_3, FINISH} state, next_state;
enum [1:0] {ITERATION_1_SUBSTATE, ITERATION_2_SUBSTATE, ITERATION_3_SUBSTATE, ITERATION_4_SUBSTATE} sub_state, sub_state_sum;

// Signals
CompFx_t input_capture, input_mapped, taylor_sum;
wire sign_bit_input;
wire [N_COMP-1:Q_COMP] input_mapped_complement;

assign input_mapped_complement = ~input_mapped[N_COMP-1:Q_COMP];
assign sign_bit_input = input_capture[N_COMP-1];
assign io.overflow = io.done && io.out[N_COMP-1]; // Overflow occurs when the output is negative since exp(x) is always positive

always_ff @ (posedge clk) begin : exp_FSM
    if (!rst_n) begin
        io.busy <= 0;
        io.done <= 0;
        io.out <= '0;
        adder_io.start <= 0;
        mult_io.start <= 0;
        adder_io.in_1 <= '0;
        adder_io.in_2 <= '0;
        mult_io.in_1 <= '0;
        mult_io.in_2 <= '0;
        state <= IDLE;
        next_state <= IDLE;
        sub_state <= ITERATION_1_SUBSTATE;
        taylor_sum <= '0;
    end else begin
        case (state)
            IDLE: begin
                io.busy <= io.start;
                io.done <= 0;
                mult_io.start <= io.start;
                adder_io.start <= 0;
                mult_io.in_1 <= io.in_1;
                mult_io.in_2 <= ln2_inv;
                state <= (io.start) ? MULT : IDLE;
                input_capture <= (io.start) ? io.in_1 : input_capture;
                next_state <= TAYLOR_TERM_1;
                sub_state <= ITERATION_1_SUBSTATE;
                taylor_sum <= Taylor_mult[0]; // 1st term of Taylor series is 1
            end

            UPDATE_SUM: begin
                if (sub_state_sum == ITERATION_1_SUBSTATE) begin
                    adder_io.in_1 <= taylor_sum;
                    adder_io.in_2 <= mult_io.out;
                    adder_io.start <= 1;
                    mult_io.start <= 0;
                    sub_state_sum <= ITERATION_2_SUBSTATE;
                end else begin
                    adder_io.start <= 0;
                    state <= next_state;
                    sub_state_sum <= ITERATION_1_SUBSTATE;
                end
            end

            MULT: begin
                state <= next_state;
            end

            TAYLOR_TERM_1: begin // x_fractional * ln(2)
                if (sub_state == ITERATION_1_SUBSTATE) begin
                    input_mapped <= mult_io.out;
                    mult_io.in_1 <= {{(N_COMP-Q_COMP){sign_bit_input}}, mult_io.out[Q_COMP-1:0]}; // Grab fractional part
                    mult_io.in_2 <= Taylor_mult[1];
                    next_state <= TAYLOR_TERM_1;
                    state <= MULT;
                    sub_state <= ITERATION_2_SUBSTATE;
                end else begin
                    state <= UPDATE_SUM;
                    mult_io.start <= 0;
                    next_state <= TAYLOR_TERM_2;
                    sub_state <= ITERATION_1_SUBSTATE;
                end
            end

            TAYLOR_TERM_2: begin
                if (sub_state == ITERATION_1_SUBSTATE) begin // Square fractional part of mapped input
                    taylor_sum <= adder_io.out;
                    mult_io.in_1 <= {{(N_COMP-Q_COMP){sign_bit_input}}, input_mapped[Q_COMP-1:0]}; // Grab fractional part (while keeping the sign)
                    mult_io.in_2 <= {{(N_COMP-Q_COMP){sign_bit_input}}, input_mapped[Q_COMP-1:0]}; // Grab fractional part (while keeping the sign)
                    mult_io.start <= 1;
                    adder_io.start <= 0;
                    state <= MULT;
                    next_state <= TAYLOR_TERM_2;
                    sub_state <= ITERATION_2_SUBSTATE;
                end else if (sub_state == ITERATION_2_SUBSTATE) begin // x_fract^2 * ln(2)^2/2!
                    mult_io.in_1 <= mult_io.out;
                    mult_io.in_2 <= Taylor_mult[2];
                    state <= MULT;
                    next_state <= TAYLOR_TERM_2;
                    sub_state <= ITERATION_3_SUBSTATE;
                end else if (sub_state == ITERATION_3_SUBSTATE) begin // Update sum
                    state <= UPDATE_SUM;
                    mult_io.start <= 0;
                    next_state <= TAYLOR_TERM_3;
                    sub_state <= ITERATION_1_SUBSTATE;
                end
            end

            TAYLOR_TERM_3: begin
                if (sub_state == ITERATION_1_SUBSTATE) begin // Square fractional part of mapped input
                    taylor_sum <= adder_io.out;
                    mult_io.start <= 1;
                    mult_io.in_1 <= {{(N_COMP-Q_COMP){sign_bit_input}}, input_mapped[Q_COMP-1:0]}; // Grab fractional part (while keeping the sign)
                    mult_io.in_2 <= {{(N_COMP-Q_COMP){sign_bit_input}}, input_mapped[Q_COMP-1:0]}; // Grab fractional part (while keeping the sign)
                    state <= MULT;
                    next_state <= TAYLOR_TERM_3;
                    sub_state <= ITERATION_2_SUBSTATE;
                end else if (sub_state == ITERATION_2_SUBSTATE) begin // Cube fractional part of mapped input
                    mult_io.in_2 <= mult_io.out;
                    sub_state <= ITERATION_3_SUBSTATE;
                    state <= MULT;
                end else if (sub_state == ITERATION_3_SUBSTATE) begin
                    mult_io.in_1 <= mult_io.out; // Grab fractional part (while keeping the sign)
                    mult_io.in_2 <= Taylor_mult[3];
                    sub_state <= ITERATION_4_SUBSTATE;
                    state <= MULT;
                end else if (sub_state == ITERATION_4_SUBSTATE) begin
                    state <= UPDATE_SUM;
                    next_state <= FINISH;
                    mult_io.start <= 0;
                end
            end

            FINISH: begin
                io.out <= (sign_bit_input) ? (adder_io.out >> ~input_mapped[N_COMP-1:Q_COMP]) : (adder_io.out << input_mapped[N_COMP-1:Q_COMP]); // Multiply fractional part by integer part (2^floor(x/ln(2)))
                io.done <= 1;
                state <= IDLE;
            end

            default: state <= IDLE;
        endcase
    end
end

endmodule

`endif
