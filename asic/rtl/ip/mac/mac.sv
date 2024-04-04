`ifndef _mac_sv_
`define _mac_sv_

/* Note:
- Fixed-point MAC
- Done signal is a single-cycle pulse
- Uses external adder, multiplier, divider and exp modules to be shared with other modules in the CiM.
- Implements three different activations: No activation (no bias), linear (adds bias to MAC result) and SWISH (adds bias to MAC result, then applies SWISH activation function)
*/
`include "../../types.svh"

module mac
(
    input wire clk,
    input wire rst_n,

    // Control signals
    input wire start, param_type,
    input wire [1:0] activation,
    input wire [$clog2(MAC_MAX_LEN+1)-1:0] len,
    input PARAMS_ADDR_T bias_addr,
    input TEMP_RES_ADDR_T start_addr1,
    input TEMP_RES_ADDR_T start_addr2, // Either intermediate res or params
    output logic busy, done,

    // Memory access signals
    MemAccessSignals int_res_access_signals,
    MemAccessSignals params_access_signals,
    input STORAGE_WORD_T param_data,
    input STORAGE_WORD_T int_res_data,

    // Computation signals
    output COMP_WORD_T computation_result,

    // Adder signals
    input COMP_WORD_T add_output_q, add_out_flipped,
    output COMP_WORD_T add_input_q_1, add_input_q_2,
    output logic add_refresh,

    // Multiplier signals
    input COMP_WORD_T mult_output_q,
    output COMP_WORD_T mult_input_q_1, mult_input_q_2,
    output logic mult_refresh,

    // Divide signals
    input wire div_busy, div_done,
    input COMP_WORD_T div_output_q,
    output COMP_WORD_T div_dividend, div_divisor,
    output logic div_start,

    // Exponential signals
    input wire exp_busy, exp_done,
    input COMP_WORD_T exp_output_q,
    output COMP_WORD_T exp_input,
    output logic exp_start
);

   /*----- LOGIC -----*/
    logic [1:0] delay_signal;
    logic [$clog2(MAC_MAX_LEN+1)-1:0] index;
    COMP_WORD_T compute_temp, compute_temp_2; // TODO: Consider if it should be shared with other modules in CiM

    enum logic [3:0] {IDLE, COMPUTE_MUL_IN1, COMPUTE_MUL_IN2, COMPUTE_MUL_OUT, COMPUTE_ADD, BASIC_MAC_DONE, COMPUTE_LINEAR_ACTIVATION, COMPUTE_SWISH_ACTIVATION, SWISH_ACTIVATION_FINAL_ADD} state;
    always_ff @ (posedge clk) begin : mac_fsm
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            unique case (state)
                IDLE: begin
                    if (start) begin
                        state <= COMPUTE_MUL_IN1;
                        busy <= 1'b1;
                        int_res_access_signals.addr_table[MAC] <= start_addr1;
                        int_res_access_signals.read_req_src[MAC] <= 1'b1;
                        compute_temp <= 'd0;
                    end else begin
                        done <= 1'b0;
                        index <= 'd0;
                        mult_refresh <= 1'b0;
                        add_refresh <= 1'b0;
                        int_res_access_signals.read_req_src[MAC] <= 1'b0;
                        delay_signal <= 2'b0;
                        busy <= 1'b0;
                    end
                end 
                COMPUTE_MUL_IN1: begin
                    index <= index + 1;
                    int_res_access_signals.addr_table[MAC] <= (param_type == INTERMEDIATE_RES) ? (start_addr2 + {3'd0, index}) : (int_res_access_signals.addr_table[MAC]);
                    params_access_signals.addr_table[MAC] <= (param_type == MODEL_PARAM) ? (start_addr2 + {3'd0, index}) : (params_access_signals.addr_table[MAC]);
                    int_res_access_signals.read_req_src[MAC] <= (param_type == INTERMEDIATE_RES);
                    params_access_signals.read_req_src[MAC] <= (param_type == MODEL_PARAM);
                    state <= COMPUTE_MUL_IN2;
                end
                COMPUTE_MUL_IN2: begin
                    compute_temp <= (index > 1) ? add_output_q : compute_temp; // Grab data from previous iteration (unless it's the first iteration)
                    mult_input_q_1 <= {{(N_COMP-N_STORAGE){int_res_data[N_STORAGE-1]}}, int_res_data}; // Sign extend
                    int_res_access_signals.read_req_src[MAC] <= 1'b0;
                    params_access_signals.read_req_src[MAC] <= 1'b0;
                    state <= COMPUTE_MUL_OUT;
                    add_refresh <= 1'b0;
                end
                COMPUTE_MUL_OUT: begin
                    mult_input_q_2 <= (param_type == INTERMEDIATE_RES) ? {{(N_COMP-N_STORAGE){int_res_data[N_STORAGE-1]}}, int_res_data} : {{(N_COMP-N_STORAGE){param_data[N_STORAGE-1]}}, param_data};
                    mult_refresh <= 1'b1;
                    state <= (mult_refresh) ? COMPUTE_ADD : COMPUTE_MUL_OUT;
                end
                COMPUTE_ADD: begin
                    add_input_q_1 <= mult_output_q;
                    add_input_q_2 <= compute_temp;
                    mult_refresh <= 1'b0;
                    add_refresh <= 1'b1;
                    if (add_refresh) begin
                        state <= (index == len) ? BASIC_MAC_DONE : COMPUTE_MUL_IN1;
                    end
                    // In case we need to go back to COMPUTE_MUL_IN1
                    int_res_access_signals.addr_table[MAC] <= start_addr1 + {3'd0, index};
                    int_res_access_signals.read_req_src[MAC] <= (index != len);
                end
                BASIC_MAC_DONE: begin
                    add_refresh <= 1'b0;
                    if (activation == NO_ACTIVATION) begin
                        computation_result <= add_output_q;
                        done <= 1'b1;
                        state <= IDLE;
                    end else if (activation == LINEAR_ACTIVATION || activation == SWISH_ACTIVATION) begin
                        params_access_signals.addr_table[MAC] <= bias_addr;
                        params_access_signals.read_req_src[MAC] <= 1'b1;
                        if (delay_signal[1]) begin
                            delay_signal <= (activation == LINEAR_ACTIVATION) ? delay_signal : 'd0; // Reset signal
                            state <= (activation == LINEAR_ACTIVATION) ? COMPUTE_LINEAR_ACTIVATION : COMPUTE_SWISH_ACTIVATION;
                        end else begin
                            delay_signal <= {delay_signal[0], 1'b1}; // Need to delay signal by 1 cycle while we wait for bias
                        end
                    end
                end

                COMPUTE_LINEAR_ACTIVATION: begin
                    add_input_q_1 <= add_output_q;
                    add_input_q_2 <= {{(N_COMP-N_STORAGE){param_data[N_STORAGE-1]}}, param_data}; // Sign extend;
                    params_access_signals.read_req_src[MAC] <= 1'b0;
                    add_refresh <= 1'b1;
                    delay_signal <= {delay_signal[0], 1'b0}; // Need to delay signal by 1 cycle while we wait for add

                    if (add_refresh & ~delay_signal[1]) begin
                        computation_result <= add_output_q;
                        done <= 1'b1;
                        state <= IDLE;
                    end
                end

                COMPUTE_SWISH_ACTIVATION: begin
                    // Start add
                    compute_temp <= (params_access_signals.read_req_src[MAC]) ? add_output_q : compute_temp; // Capture result of MAC for last step
                    add_input_q_1 <= (params_access_signals.read_req_src[MAC]) ? add_output_q : compute_temp; // For first add, use previous add_output_q and for the 2nd add, use div_output_q
                    add_input_q_2 <= (params_access_signals.read_req_src[MAC]) ? {{(N_COMP-N_STORAGE){param_data[N_STORAGE-1]}}, param_data} : div_output_q; // Sign extend;
                    params_access_signals.read_req_src[MAC] <= 1'b0;
                    add_refresh <= params_access_signals.read_req_src[MAC] || div_done; // read_req_src and div_done are convenient pulses to start add

                    // Save add result and start exp
                    delay_signal <= {delay_signal[0], add_refresh}; // Need to delay signal by 1 cycle while we wait for add
                    compute_temp_2 <= (delay_signal[1]) ? add_output_q : compute_temp_2;
                    exp_start <= delay_signal[1];
                    exp_input <= add_out_flipped;

                    // Start div
                    div_divisor <= exp_output_q + /* +1 */('d1 << Q);
                    div_dividend <= compute_temp_2;
                    div_start <= exp_done;

                    // Done
                    state <= (div_done) ? SWISH_ACTIVATION_FINAL_ADD : COMPUTE_SWISH_ACTIVATION;
                end

                SWISH_ACTIVATION_FINAL_ADD: begin
                    computation_result <= add_output_q;
                    delay_signal <= {delay_signal[0], add_refresh};
                    add_refresh <= 1'b0;
                    done <= (delay_signal[1]);
                    state <= (delay_signal[1]) ? IDLE : SWISH_ACTIVATION_FINAL_ADD;
                end
                default: begin
                    $fatal("Invalid state in MAC FSM");
                end
            endcase
        end
    end

endmodule

`endif
