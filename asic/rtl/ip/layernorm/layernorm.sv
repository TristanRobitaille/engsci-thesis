module layernorm (
    input wire clk,
    input wire rst_n,

    // Control signals
    input wire start, half_select,
    input TEMP_RES_ADDR_T start_addr,
    input PARAMS_ADDR_T beta_addr,
    input PARAMS_ADDR_T gamma_addr,
    output logic busy, done,

    // Memory access signals
    input MemAccessSignals int_res_access_signals,
    input MemAccessSignals params_access_signals,
    input STORAGE_WORD_T param_data,
    input STORAGE_WORD_T int_res_data,

    // Adder signals
    input COMP_WORD_T add_output_q, add_output_flipped,
    output COMP_WORD_T add_input_q_1, add_input_q_2,
    output logic add_refresh,

    // Multiplier signals
    input COMP_WORD_T mult_output_q, mult_output_flipped,
    output COMP_WORD_T mult_input_q_1, mult_input_q_2,
    output logic mult_refresh,

    // Divider signals
    input wire div_done, div_busy,
    input COMP_WORD_T div_output_q,
    output logic div_start,
    output COMP_WORD_T div_dividend, div_divisor,

    // Sqrt signals
    input wire sqrt_done, sqrt_busy,
    input COMP_WORD_T sqrt_root_q,
    output logic sqrt_start,
    output COMP_WORD_T sqrt_rad_q
);

    /*----- LOGIC -----*/
    localparam  LEN_FIRST_HALF = 64, // LEN_FIRST_HALF must be a power of 2 since we divide by bit shifting
                LEN_SECOND_HALF = 61;
    logic [2:0] gen_reg_3b;
    logic [$clog2(EMB_DEPTH+1)-1:0] index;
    COMP_WORD_T compute_temp, compute_temp_2, compute_temp_3; // TODO: Consider if it should be shared with other modules in CiM

    enum logic [1:0] {MEAN_SUM, VARIANCE, PARTIAL_NORM_FIRST_HALF} loop_sum_type;
    enum logic [3:0] {IDLE, LOOP_SUM, VARIANCE_LOOP, VARIANCE_SQRT, NORM_FIRST_HALF, BETA_LOAD, GAMMA_LOAD, SECOND_HALF_LOOP, DONE} state;

    always_ff @ (posedge clk) begin : layernorm_fsm
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            unique case (state)
                IDLE : begin
                    if (start) begin
                        if (half_select == FIRST_HALF) begin
                            state <= LOOP_SUM;
                        end else if (half_select == SECOND_HALF) begin
                            params_access_signals.addr_table[LAYERNORM] <= gamma_addr;
                            params_access_signals.read_req_src[LAYERNORM] <= 1'b1;
                            state <= GAMMA_LOAD;
                        end
                        busy <= 1'b1;
                    end else begin
                        done <= 1'b0;
                        loop_sum_type <= MEAN_SUM;
                        index <= 'd0;
                        gen_reg_3b <= 'd0;
                        compute_temp <= 'd0;
                        compute_temp_2 <= 'd0;
                        compute_temp_3 <= 'd0;
                        add_refresh <= 1'b0;
                        mult_refresh <= 1'b0;
                        div_start <= 1'b0;
                        sqrt_start <= 1'b0;
                        busy <= 1'b0;
                    end
                end
                LOOP_SUM : begin // Can reuse this state whenever we need to sum over LEN_FIRST_HALF addresses
                    if (index < LEN_FIRST_HALF) begin
                        if (gen_reg_3b == 'd0) begin // Set int_res addr and read request
                            int_res_access_signals.addr_table[LAYERNORM] <= start_addr + {3'd0, index};
                            int_res_access_signals.read_req_src[LAYERNORM] <= 1'b1;
                            if (int_res_access_signals.read_req_src[LAYERNORM] == 1'b1) begin 
                                gen_reg_3b <= 'd1;
                                int_res_access_signals.read_req_src[LAYERNORM] <= 1'b0;
                            end
                        end else if (gen_reg_3b == 'd1) begin
                            add_input_q_1 <= {{(N_COMP-N_STORAGE){int_res_data[N_STORAGE-1]}}, int_res_data};
                            add_input_q_2 <= ((loop_sum_type == VARIANCE) || (loop_sum_type == PARTIAL_NORM_FIRST_HALF)) ? (~compute_temp + 1'd1) : compute_temp; // Can subtract by adding the flipped version (data is 2's complement)
                            add_refresh <= 1'b1;
                            if (add_refresh) begin
                                gen_reg_3b <= 'd2;
                            end
                        end else if (gen_reg_3b == 'd2) begin
                            if (loop_sum_type == MEAN_SUM) begin // Move to next
                                compute_temp <= add_output_q;
                                gen_reg_3b <= 'd0;
                                index <= index + 'd1;
                            end else if (loop_sum_type == VARIANCE) begin
                                // To continue variance computation, need to square and sum the result
                                add_refresh <= 1'b0;
                                mult_refresh <= 1'b1;
                                mult_input_q_1 <= add_output_q;
                                mult_input_q_2 <= add_output_q;
                                if (mult_refresh) begin
                                    gen_reg_3b <= 'd3;
                                end
                            end else if (loop_sum_type == PARTIAL_NORM_FIRST_HALF) begin
                                mult_input_q_1 <= compute_temp_3;
                                mult_input_q_2 <= (mult_refresh) ? mult_input_q_2 : add_output_q; // Only output when mult_refresh is 0 to avoid false positive transient overflows
                                mult_refresh <= 1'b1;
                                gen_reg_3b <= (mult_refresh) ? 'd4 : 'd2;
                            end
                            add_refresh <= 1'b0;
                        end else if (gen_reg_3b == 'd3) begin
                            mult_refresh <= 1'b0;
                            add_refresh <= 1'b1;
                            add_input_q_1 <= compute_temp_2;
                            add_input_q_2 <= mult_output_q;
                            gen_reg_3b <= 'd4;
                        end else if (gen_reg_3b == 'd4) begin
                            add_refresh <= 1'b0;
                            mult_refresh <= 1'b0;
                            if (~add_refresh & ~mult_refresh) begin
                                gen_reg_3b <= 'd0;
                                index <= index + 'd1;
                                compute_temp_2 <= add_output_q;
                            end
                            int_res_access_signals.write_req_src[LAYERNORM] <= mult_refresh; // Just because add_refresh mult_refresh is 1 when arriving here so it's a convenient signal to use
                            int_res_access_signals.write_data[LAYERNORM] <= (mult_output_q[N_COMP-1]) ? (~mult_output_flipped[N_STORAGE-1:0] + {{(N_STORAGE-1){1'b0}}, 1'd1}) : mult_output_q[N_STORAGE-1:0];
                        end
                    end else begin
                        index <= 'd0;
                        if (loop_sum_type == MEAN_SUM) begin
                            state <= VARIANCE_LOOP;
                            compute_temp <= (compute_temp >>> $clog2(LEN_FIRST_HALF)); // Divide by LEN_FIRST_HALF (shift right by log2(LEN_FIRST_HALF) bits.
                        end else if (loop_sum_type == VARIANCE) begin
                            compute_temp_2 <= (compute_temp_2 >>> $clog2(LEN_FIRST_HALF)); // Divide by LEN_FIRST_HALF (shift right by log2(LEN_FIRST_HALF) bits
                            state <= VARIANCE_SQRT;
                            mult_input_q_1 <= compute_temp_3;
                            mult_input_q_2 <= compute_temp_3;
                            mult_refresh <= 1'b1;
                        end else if (loop_sum_type == PARTIAL_NORM_FIRST_HALF) begin
                            state <= DONE;
                        end
                    end
                end
                VARIANCE_LOOP : begin
                    loop_sum_type <= VARIANCE;
                    state <= LOOP_SUM;
                end
                VARIANCE_SQRT : begin
                    sqrt_rad_q <= compute_temp_2;
                    sqrt_start <= ~sqrt_busy & ~sqrt_done;
                    state <= (sqrt_done) ? NORM_FIRST_HALF : VARIANCE_SQRT;
                end
                NORM_FIRST_HALF : begin
                    // Compute 1.0/variance such that we can multiply by that instead of dividing
                    compute_temp_3 <= div_output_q;
                    state <= (div_done) ? LOOP_SUM : NORM_FIRST_HALF;
                    div_dividend <= 1 << Q; // 1.0 in Q format
                    div_divisor <= sqrt_root_q;
                    div_start <= ~div_done & ~div_busy;
                    mult_refresh <= 1'b0;
                    loop_sum_type <= PARTIAL_NORM_FIRST_HALF;
                end
                GAMMA_LOAD: begin // Following states are for second-half
                    gen_reg_3b <= gen_reg_3b + 'd1;
                    if (gen_reg_3b == 'd1) begin
                        compute_temp_2 <= {{(N_COMP-N_STORAGE){param_data[N_STORAGE-1]}}, param_data}; // gamma (w/ sign extension)
                        params_access_signals.addr_table[LAYERNORM] <= beta_addr;
                        state <= BETA_LOAD;
                    end
                end
                BETA_LOAD: begin
                    params_access_signals.read_req_src[LAYERNORM] <= 1'b0;
                    gen_reg_3b <= 'd0;
                    if (gen_reg_3b == 'd0) begin
                        compute_temp_3 <= {{(N_COMP-N_STORAGE){param_data[N_STORAGE-1]}}, param_data}; // beta (w/ sign extension)
                        state <= SECOND_HALF_LOOP;
                    end
                end
                SECOND_HALF_LOOP: begin
                    if (index < LEN_SECOND_HALF) begin
                        if (gen_reg_3b == 'd0) begin // Set int_res addr and read request
                            int_res_access_signals.addr_table[LAYERNORM] <= start_addr + {3'd0, index};
                            int_res_access_signals.read_req_src[LAYERNORM] <= 1'b1;
                            if (int_res_access_signals.read_req_src[LAYERNORM] == 1'b1) gen_reg_3b <= 'd1;
                        end else if (gen_reg_3b == 'd1) begin
                            int_res_access_signals.read_req_src[LAYERNORM] <= 1'b0;
                            mult_input_q_1 <= compute_temp_2; // gamma
                            mult_input_q_2 <= {{(N_COMP-N_STORAGE){int_res_data[N_STORAGE-1]}}, int_res_data};
                            mult_refresh <= 1'b1;
                            if (mult_refresh) begin
                                gen_reg_3b <= 'd2;
                            end
                        end else if (gen_reg_3b == 'd2) begin
                            mult_refresh <= 1'b0;
                            add_input_q_1 <= mult_output_q;
                            add_input_q_2 <= compute_temp_3; // beta
                            add_refresh <= 1'b1;
                            if (add_refresh) begin
                                gen_reg_3b <= 'd3;
                            end
                        end else if (gen_reg_3b == 'd3) begin
                            add_refresh <= 1'b0;
                            int_res_access_signals.write_req_src[LAYERNORM] <= 1'b1;
                            int_res_access_signals.write_data[LAYERNORM] <= (add_output_q[N_COMP-1]) ? (~add_output_flipped[N_STORAGE-1:0] + {{(N_STORAGE-1){1'b0}}, 1'd1}) : add_output_q[N_STORAGE-1:0];;
                            gen_reg_3b <= 'd4;
                        end else begin
                            int_res_access_signals.write_req_src[LAYERNORM] <= 1'b0;
                            gen_reg_3b <= 'd0;
                            index <= index + 'd1;
                        end
                    end else begin
                        state <= DONE;
                    end
                end
                DONE: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
