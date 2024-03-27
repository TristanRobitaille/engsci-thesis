/* Note:
- Fixed-point MAC
- Done signal is a single-cycle pulse
- Uses external multiplier and adder modules to be shared with other modules in the CiM.
- Implements three different activations: No activation (no bias), linear (adds bias to MAC result) and TODO: SWISH
*/
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
    input MemAccessSignals int_res_access_signals,
    input MemAccessSignals params_access_signals,
    input STORAGE_WORD_T param_data,
    input STORAGE_WORD_T int_res_data,

    // Computation signals
    output COMP_WORD_T computation_result,

    // Adder signals
    input COMP_WORD_T add_output_q,
    output COMP_WORD_T add_input_q_1, add_input_q_2,
    output logic add_refresh,

    // Multiplier signals
    input COMP_WORD_T mult_output_q,
    output COMP_WORD_T mult_input_q_1, mult_input_q_2,
    output logic mult_refresh
);

   /*----- LOGIC -----*/
    logic [1:0] delay_signal;
    logic [$clog2(MAC_MAX_LEN+1)-1:0] index;
    COMP_WORD_T compute_temp, compute_temp_2; // TODO: Consider if it should be shared with other modules in CiM

    enum logic [2:0] {IDLE, COMPUTE_MUL_IN1, COMPUTE_MUL_IN2, COMPUTE_MUL_OUT, COMPUTE_ADD, BASIC_MAC_DONE, COMPUTE_LINEAR_ACTIVATION} state;
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
                    end else if (activation == LINEAR_ACTIVATION) begin
                        params_access_signals.addr_table[MAC] <= (activation == LINEAR_ACTIVATION) ? bias_addr : params_access_signals.addr_table[MAC];
                        params_access_signals.read_req_src[MAC] <= (activation == LINEAR_ACTIVATION);
                        delay_signal <= {delay_signal[0], 1'b1}; // Need to delay signal by 1 cycle while we wait for bias
                        state <= (delay_signal[1]) ? COMPUTE_LINEAR_ACTIVATION : BASIC_MAC_DONE;
                    end else if (activation == SWISH_ACTIVATION) begin
                        // TODO
                        $fatal("SWISH activation not implementedin MAC unit");
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
                default: begin
                    $fatal("Invalid state in MAC FSM");
                end
            endcase
        end
    end

endmodule
