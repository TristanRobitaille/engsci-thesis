// Includes
`include "cim.svh"

/*TODO
    - Consider using a function to start MAC computation for more compact code
*/

module cim # (
    parameter ID = 0,
    parameter STANDALONE_TB = 1
)(
    input wire clk,
    input wire rst_n,

    // Bus
    inout wire [BUS_OP_WIDTH-1:0] bus_op,
    inout wire signed [2:0][N_STORAGE-1:0] bus_data,
    inout wire [$clog2(NUM_CIMS)-1:0] bus_target_or_sender,

    output logic is_ready
);

    // Initialize arrays
    `include "top_init.sv"
    `include "cim_init.sv"

    // Memory
    MemAccessSignals params_access_signals();
    MemAccessSignals int_res_access_signals();
    wire [N_STORAGE-1:0] int_res_read_data, params_read_data;
    cim_mem mem (   .clk(clk), .params_access_signals(params_access_signals), .int_res_access_signals(int_res_access_signals),
                    .int_res_read_data(int_res_read_data), .params_read_data(params_read_data));

    // Bus
    logic bus_drive, bus_drive_delayed;
    logic [BUS_OP_WIDTH-1:0] bus_op_write;
    logic signed [2:0][N_STORAGE-1:0] bus_data_write;
    logic [$clog2(NUM_CIMS)-1:0] bus_target_or_sender_write;
    assign bus_op = (bus_drive) ? bus_op_write : 'Z;
    assign bus_data = (bus_drive) ? bus_data_write : 'Z;
    assign bus_target_or_sender = (bus_drive) ? bus_target_or_sender_write : 'Z;

    wire [BUS_OP_WIDTH-1:0] bus_op_read;
    wire signed [2:0][N_STORAGE-1:0] bus_data_read;
    wire [$clog2(NUM_CIMS)-1:0] bus_target_or_sender_read;
    if (STANDALONE_TB == 0) begin : bus_read_assignment // Cannot drive inout's in CocoTB testbench, so we drive bus_x_read directly and do not assign to them
        assign bus_op_read = bus_op;
        assign bus_data_read = bus_data;
        assign bus_target_or_sender_read = bus_target_or_sender;
    end

    // Internal signals
    wire MAC_compute_in_progress;
    logic [1:0] gen_reg_2b;
    logic [$clog2(NUM_CIMS+1)-1:0] sender_id, data_len;
    TEMP_RES_ADDR_T tx_addr, rx_addr;
    logic [N_COMP-1:0] compute_temp, compute_temp_2, compute_temp_3, computation_result;

    // Counters
    logic gen_cnt_7b_rst_n, gen_cnt_7b_2_rst_n, word_rec_cnt_rst_n, word_snt_cnt_rst_n;
    logic [6:0] gen_cnt_7b_inc, gen_cnt_7b_2_inc, word_rec_cnt_inc, word_snt_cnt_inc;
    wire [6:0] gen_cnt_7b_cnt, gen_cnt_7b_2_cnt, word_rec_cnt, word_snt_cnt;
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b_inst      (.clk(clk), .rst_n(gen_cnt_7b_rst_n),   .inc(gen_cnt_7b_inc),   .cnt(gen_cnt_7b_cnt));
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b_2_inst    (.clk(clk), .rst_n(gen_cnt_7b_2_rst_n), .inc(gen_cnt_7b_2_inc), .cnt(gen_cnt_7b_2_cnt));
    counter #(.WIDTH(7), .MODE(0)) word_rec_cnt_inst    (.clk(clk), .rst_n(word_rec_cnt_rst_n), .inc(word_rec_cnt_inc), .cnt(word_rec_cnt));
    counter #(.WIDTH(7), .MODE(0)) word_snt_cnt_inst    (.clk(clk), .rst_n(word_snt_cnt_rst_n), .inc(word_snt_cnt_inc), .cnt(word_snt_cnt));

    // Adder module
    wire add_overflow;
    logic add_refresh, cim_add_refresh;
    logic [N_COMP-1:0] add_input_q_1, add_input_q_2, cim_add_input_q_1, cim_add_input_q_2, add_output_q, add_out_flipped;
    always_latch begin : adder_input_MUX
        if (cim_add_refresh) begin
            add_input_q_1 = cim_add_input_q_1;
            add_input_q_2 = cim_add_input_q_2;
        end else if (mac_add_refresh) begin
            add_input_q_1 = mac_add_input_q_1;
            add_input_q_2 = mac_add_input_q_2;
        end else if (layernorm_add_refresh) begin
            add_input_q_1 = layernorm_add_input_q_1;
            add_input_q_2 = layernorm_add_input_q_2;
        end else if (exp_add_refresh) begin
            add_input_q_1 = exp_add_input_q_1;
            add_input_q_2 = exp_add_input_q_2;
        end
        add_refresh = (cim_add_refresh || mac_add_refresh || layernorm_add_refresh || exp_add_refresh);
    end

    always_ff @ (posedge clk) begin : adder_assertions
        assert ($countones({cim_add_refresh, mac_add_refresh, layernorm_add_refresh, exp_add_refresh}) <= 1) else $fatal("Multiple add_refresh signals are asserted simultaneously!");
    end
    adder add_inst (.clk(clk), .rst_n(rst_n), .refresh(add_refresh), .overflow(add_overflow), .input_q_1(add_input_q_1), .input_q_2(add_input_q_2), .output_q(add_output_q));

    // Multiplier module
    logic mult_refresh, mul_overflow;
    logic [N_COMP-1:0] mult_input_q_1, mult_input_q_2, mult_output_q, mult_out_flipped;
    always_latch begin : mult_input_MUX
        if (layernorm_mult_refresh) begin
            mult_input_q_1 = layernorm_mult_input_q_1;
            mult_input_q_2 = layernorm_mult_input_q_2;
        end else if (mac_mult_refresh) begin
            mult_input_q_1 = mac_mult_input_q_1;
            mult_input_q_2 = mac_mult_input_q_2;
        end else if (exp_mult_refresh) begin
            mult_input_q_1 = exp_mult_input_1;
            mult_input_q_2 = exp_mult_input_2;
        end
        mult_refresh = (layernorm_mult_refresh || mac_mult_refresh || exp_mult_refresh);
    end
    always_ff @ (posedge clk) begin : mult_assertions
        assert ($countones({layernorm_mult_refresh, mac_mult_refresh, exp_mult_refresh}) <= 1) else $fatal("Multiple mult_refresh signals are asserted simultaneously!");
    end
    multiplier mult_inst (.clk(clk), .rst_n(rst_n), .refresh(mult_refresh), .overflow(mul_overflow), .input_q_1(mult_input_q_1), .input_q_2(mult_input_q_2), .output_q(mult_output_q));

    // Divider module
    logic div_dbz, div_overflow, div_done, div_busy, div_start, logic_fsm_div_start;
    logic [N_COMP-1:0] div_output_q, div_out_flipped, div_dividend, div_divisor, logic_fsm_div_dividend, logic_fsm_div_divisor;
    always_latch begin : div_input_MUX
        if (logic_fsm_div_start) begin
            div_dividend = logic_fsm_div_dividend;
            div_divisor = logic_fsm_div_divisor;
        end else if (layernorm_div_start) begin
            div_dividend = layernorm_div_dividend;
            div_divisor = layernorm_div_divisor;
        end else if (softmax_div_start) begin
            div_dividend = softmax_div_dividend;
            div_divisor = softmax_div_divisor;
        end
        div_start = (logic_fsm_div_start || layernorm_div_start || softmax_div_start);
    end
    always_ff @ (posedge clk) begin : div_assertions
        assert ($countones({logic_fsm_div_start, layernorm_div_start, softmax_div_start}) <= 1) else $fatal("Multiple div_start signals are asserted simultaneously!");
    end
    divider div_inst (.clk(clk), .rst_n(rst_n), .start(div_start), .dividend(div_dividend), .divisor(div_divisor), .done(div_done), .busy(div_busy), .output_q(div_output_q), .dbz(div_dbz), .overflow(div_overflow));

    // Sqrt module
    logic sqrt_start, sqrt_done, sqrt_busy, sqrt_neg_rad;
    logic [N_COMP-1:0] sqrt_rad_q, sqrt_root_q;
    sqrt sqrt_inst (.clk(clk), .rst_n(rst_n), .start(sqrt_start), .done(sqrt_done), .busy(sqrt_busy), .rad_q(sqrt_rad_q), .root_q(sqrt_root_q), .neg_rad(sqrt_neg_rad));

    // MAC module
    logic mac_start, mac_done, mac_busy, mac_add_refresh, mac_mult_refresh;
    logic [$clog2(MAC_MAX_LEN+1)-1:0] mac_len;
    TEMP_RES_ADDR_T mac_start_addr1, mac_start_addr2;
    PARAMS_ADDR_T mac_params_addr, mac_bias_addr;
    logic [N_COMP-1:0] mac_out, mac_out_flipped, mac_add_input_q_1, mac_add_input_q_2;
    logic [N_COMP-1:0] mac_mult_input_q_1, mac_mult_input_q_2;
    PARAM_TYPE_T mac_param_type;
    ACTIVATION_TYPE_T mac_activation;
    mac mac_inst (.clk(clk), .rst_n(rst_n), .start(mac_start), .done(mac_done), .busy(mac_busy), .param_type(mac_param_type), .len(mac_len), .activation(mac_activation),
                  .start_addr1(mac_start_addr1), .start_addr2(mac_start_addr2), .bias_addr(mac_bias_addr),
                  .params_access_signals(params_access_signals), .int_res_access_signals(int_res_access_signals),
                  .param_data(params_read_data), .int_res_data(int_res_read_data), .computation_result(mac_out),
                  .add_input_q_1(mac_add_input_q_1), .add_input_q_2(mac_add_input_q_2), .add_output_q(add_output_q), .add_refresh(mac_add_refresh),
                  .mult_input_q_1(mac_mult_input_q_1), .mult_input_q_2(mac_mult_input_q_2), .mult_output_q(mult_output_q), .mult_refresh(mac_mult_refresh));

    // LayerNorm module
    logic layernorm_start, layernorm_half_select, layernorm_done, layernorm_busy, layernorm_add_refresh, layernorm_mult_refresh, layernorm_div_start;
    PARAMS_ADDR_T beta_addr, gamma_addr;
    TEMP_RES_ADDR_T layernorm_start_addr;
    logic [N_COMP-1:0] layernorm_add_input_q_1, layernorm_add_input_q_2, layernorm_mult_input_q_1, layernorm_mult_input_q_2, layernorm_div_dividend, layernorm_div_divisor;
    layernorm layernorm_inst (.clk(clk), .rst_n(rst_n), .start(layernorm_start), .half_select(layernorm_half_select), .busy(layernorm_busy), .done(layernorm_done),
                              .start_addr(layernorm_start_addr), .beta_addr(beta_addr), .gamma_addr(gamma_addr),
                              .int_res_access_signals(int_res_access_signals), .params_access_signals(params_access_signals),
                              .param_data(params_read_data), .int_res_data(int_res_read_data),
                              .add_input_q_1(layernorm_add_input_q_1), .add_input_q_2(layernorm_add_input_q_2), .add_refresh(layernorm_add_refresh), .add_output_q(add_output_q), .add_output_flipped(add_out_flipped),
                              .mult_input_q_1(layernorm_mult_input_q_1), .mult_input_q_2(layernorm_mult_input_q_2), .mult_refresh(layernorm_mult_refresh), .mult_output_q(mult_output_q), .mult_output_flipped(mult_out_flipped),
                              .div_done(div_done), .div_busy(div_busy), .div_start(layernorm_div_start),
                              .div_output_q(div_output_q), .div_dividend(layernorm_div_dividend), .div_divisor(layernorm_div_divisor),
                              .sqrt_done(sqrt_done), .sqrt_busy(sqrt_busy), .sqrt_start(sqrt_start), .sqrt_rad_q(sqrt_rad_q), .sqrt_root_q(sqrt_root_q));

    // Softmax module
    logic softmax_start, softmax_busy, softmax_done, softmax_add_refresh, softmax_mul_refresh, softmax_div_start, softmax_exp_start;
    logic [$clog2(MAC_MAX_LEN+1)-1:0] softmax_len;
    TEMP_RES_ADDR_T softmax_start_addr;
    COMP_WORD_T softmax_add_input_1, softmax_add_input_2, softmax_mul_input_1, softmax_mul_input_2, softmax_div_dividend, softmax_div_divisor, softmax_exp_input;
    softmax softmax_inst (.clk(clk), .rst_n(rst_n), .start(softmax_start), .len(softmax_len), .start_addr(softmax_start_addr), .busy(softmax_busy), .done(softmax_done),
                          .int_res_access_signals(int_res_access_signals), .int_res_data(int_res_read_data),
                          .add_output_q(add_output_q), .add_out_flipped(add_out_flipped), .add_input_q_1(softmax_add_input_1), .add_input_q_2(softmax_add_input_2), .add_refresh(softmax_add_refresh),
                          .mult_output_q(mult_output_q), .mult_out_flipped(mult_out_flipped), .mult_input_q_1(softmax_mul_input_1), .mult_input_q_2(softmax_mul_input_2), .mult_refresh(softmax_mul_refresh),
                          .div_busy(div_busy), .div_done(div_done), .div_output_q(div_output_q), .div_dividend(softmax_div_dividend), .div_divisor(softmax_div_divisor), .div_start(softmax_div_start),
                          .exp_busy(exp_busy), .exp_done(exp_done), .exp_output_q(exp_output), .exp_out_flipped(exp_output_flipped), .exp_input_q(softmax_exp_input), .exp_start(softmax_exp_start));

    // Exponential module
    logic exp_add_refresh, exp_mult_refresh, exp_busy, exp_done, exp_start;
    COMP_WORD_T exp_input, exp_output, exp_output_flipped, exp_add_input_q_1, exp_add_input_q_2, exp_mult_input_1, exp_mult_input_2;
    always_latch begin : exp_MUX
        if (softmax_exp_start) begin
            exp_input = softmax_exp_input;
        end
        exp_start = softmax_exp_start;
    end
    exp exp_inst (  .clk(clk), .rst_n(rst_n), .start(exp_start), .busy(exp_busy), .done(exp_done), .input_q(exp_input), .output_q(exp_output),
                    .adder_output(add_output_q), .adder_refresh(exp_add_refresh), .adder_input_1(exp_add_input_q_1), .adder_input_2(exp_add_input_q_2),
                    .mult_output(mult_output_q), .mult_refresh(exp_mult_refresh), .mult_input_1(exp_mult_input_1), .mult_input_2(exp_mult_input_2));

    // Comms FSM
    logic signed [N_STORAGE-1:0] bus_data_copy_1, bus_data_copy_2;
    always_ff @ (posedge clk) begin : cim_comms_fsm
        if (!rst_n) begin
            word_rec_cnt_rst_n <= RST;
            word_snt_cnt_rst_n <= RST;
            sender_id <= 'd0;
            data_len <= 'd0;
            tx_addr <= 'd0;
            rx_addr <= 'd0;
        end else begin
            unique case (bus_op_read)
                PATCH_LOAD_BROADCAST_START_OP: begin
                    word_rec_cnt_rst_n <= RST;
                    gen_cnt_7b_2_rst_n <= RST;
                end
                PATCH_LOAD_BROADCAST_OP: begin
                    int_res_access_signals.addr_table[BUS_FSM] <= {3'd0, word_rec_cnt};
                    int_res_access_signals.write_data[BUS_FSM] <= bus_data_read[0];
                    int_res_access_signals.write_req_src[BUS_FSM] <= 1'b1;
                    word_rec_cnt_inc <= 'd1;
                end
                DENSE_BROADCAST_START_OP,
                TRANS_BROADCAST_START_OP: begin
                    if (bus_target_or_sender_read == ID) begin // Start broadcasting data
                        tx_addr <= bus_data_read[0][$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0];
                        data_fill_start_addr <= bus_data_read[0][$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0];
                        bus_op_write <= (bus_op_read == DENSE_BROADCAST_START_OP) ? DENSE_BROADCAST_DATA_OP : TRANS_BROADCAST_DATA_OP;
                        start_inst_fill <= 1'b1;
                        num_words_to_fill <= (bus_data_read[1] == 'd1) ? 'd1 : 'd3;
                    end
                    if (current_inf_step == ENC_MHSA_MULT_V_STEP) begin
                        gen_cnt_7b_2_inc <= {6'd0, (bus_target_or_sender_read == 'd0)}; // Count matrix
                        word_rec_cnt_rst_n <= RST;
                    end

                    data_len <= bus_data_read[1][$clog2(NUM_CIMS+1)-1:0];
                    rx_addr <= bus_data_read[2][$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0];
                    sender_id <= {1'd0, bus_target_or_sender_read};
                    word_snt_cnt_rst_n <= RST;
                    is_ready_internal <= 1'b0;
                end

                DENSE_BROADCAST_DATA_OP,
                TRANS_BROADCAST_DATA_OP: begin
                    word_snt_cnt_rst_n <= RUN;
                    word_snt_cnt_inc <= 'd3;

                    // Grab appropriate data
                    if (bus_op_read == TRANS_BROADCAST_DATA_OP) begin
                        word_rec_cnt_inc <= {6'd0, has_my_data(word_snt_cnt, ID) || (current_inf_step == MLP_HEAD_DENSE_2_STEP) || (bus_op_read == DENSE_BROADCAST_DATA_OP)};
                        if (has_my_data(word_snt_cnt, ID)) begin
                            int_res_access_signals.write_req_src[BUS_FSM] <= 1'b1;
                            int_res_access_signals.addr_table[BUS_FSM] <= rx_addr + {4'd0, bus_target_or_sender_read};
                            if ((word_snt_cnt+3) <= data_len) begin // More than 3 word left to receive
                                int_res_access_signals.write_data[BUS_FSM] <= bus_data_read[ID - word_snt_cnt];
                            end else if (word_snt_cnt == ID) begin
                                int_res_access_signals.write_data[BUS_FSM] <= bus_data_read[2];
                            end
                        end
                        // For a transpose broadcast, there is only one word to save so we can start sending immediately
                        start_inst_fill <= (bus_target_or_sender_read == ID) && (word_snt_cnt+3 < data_len);
                    end else if (bus_op_read == DENSE_BROADCAST_DATA_OP) begin
                        logic [6:0] num_words_left = data_len - word_snt_cnt;
                        int_res_access_signals.write_req_src[BUS_FSM] <= 1'b1;
                        int_res_access_signals.write_data[BUS_FSM] <= bus_data_read[2];
                        save_dense_broadcast_start <= (word_snt_cnt < data_len);
                        need_to_send_dense <= (bus_target_or_sender_read == ID) && (word_snt_cnt+3 < data_len);
                        word_rec_cnt_inc <= 'd3; // Capture all three words in the bus instruction
                        if ((num_words_left) >= 3) begin
                            int_res_access_signals.addr_table[BUS_FSM] <= rx_addr + {3'd0, word_snt_cnt} + 'd2;
                            bus_data_copy_1 <= bus_data_read[1];
                            bus_data_copy_2 <= bus_data_read[0];
                            save_dense_broadcast_num_words <= 3;
                        end else if (num_words_left == 2) begin
                            int_res_access_signals.addr_table[BUS_FSM] <= rx_addr + {3'd0, word_snt_cnt} + 'd1;
                            bus_data_copy_1 <= bus_data_read[1];
                            save_dense_broadcast_num_words <= 2;
                        end else if (num_words_left == 1) begin
                            int_res_access_signals.addr_table[BUS_FSM] <= rx_addr + {3'd0, word_snt_cnt};
                            save_dense_broadcast_num_words <= 1;
                        end
                    end

                    if ((bus_target_or_sender_read == ID) && (word_snt_cnt+3 < data_len)) begin // Need to send
                        data_fill_start_addr <= tx_addr + {3'd0, word_snt_cnt} + 'd3;
                        if ((word_snt_cnt+6) <= data_len) begin
                            num_words_to_fill <= 'd3;
                        end else begin
                            num_words_to_fill <= data_len - word_snt_cnt - 3;
                        end
                    end
                end

                PARAM_STREAM_START_OP: begin
                    rx_addr <= bus_data_read[0][$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0];
                    word_rec_cnt_rst_n <= RST;
                end
                PARAM_STREAM_OP: begin
                    if (bus_target_or_sender_read == ID) begin // The master will progressively fill bus_data[0..1..2] as it receives data from external memory
                        unique case (gen_reg_2b)
                            'd0: begin
                                params_access_signals.addr_table[BUS_FSM] <= rx_addr + {3'd0, word_rec_cnt};
                                params_access_signals.write_data[BUS_FSM] <= bus_data_read[0];
                                gen_reg_2b <= 'd1;
                                is_ready_internal <= 1'b0;
                            end
                            'd1: begin
                                params_access_signals.addr_table[BUS_FSM] <= rx_addr + {3'd0, word_rec_cnt};
                                params_access_signals.write_data[BUS_FSM] <= bus_data_read[1];
                                gen_reg_2b <= 'd2;
                            end
                            'd2: begin
                                params_access_signals.addr_table[BUS_FSM] <= rx_addr + {3'd0, word_rec_cnt};
                                params_access_signals.write_data[BUS_FSM] <= bus_data_read[2];
                                gen_reg_2b <= 'd0;
                                is_ready_internal <= 1'b1;
                            end
                        endcase
                    end
                    params_access_signals.write_req_src[BUS_FSM] <= (bus_target_or_sender_read == ID);
                    word_rec_cnt_inc <= {6'd0, (bus_target_or_sender_read == ID)};
                    word_rec_cnt_rst_n <= (bus_target_or_sender_read == ID) ? RUN : RST; // Hold under reset if CiM isn't recipient to save power
                end
                NOP: begin
                    int_res_access_signals.write_req_src[BUS_FSM] <= 1'b0;
                    params_access_signals.write_req_src[BUS_FSM] <= 1'b0;
                    word_rec_cnt_inc <= 'd0;
                    word_snt_cnt_inc <= 'd0;
                    start_inst_fill <= 1'b0;
                    save_dense_broadcast_start <= 1'b0;
                end
                PISTOL_START_OP: begin
                    word_snt_cnt_rst_n <= RST;
                end
                INFERENCE_RESULT_OP: begin
                end
                default: begin
                end
            endcase
        end
    end

    // Compute control FSM
    CIM_STATE_T cim_state;
    INFERENCE_STEP_T current_inf_step;

    always_ff @ (posedge clk) begin : cim_compute_control_fsm
        if (!rst_n) begin
            cim_state <= CIM_IDLE;
            current_inf_step <= CLASS_TOKEN_CONCAT_STEP;
            gen_cnt_7b_rst_n <= RST;
            gen_cnt_7b_2_rst_n <= RST;
            compute_temp <= 'd0;
            compute_temp_2 <= 'd0;
            compute_temp_3 <= 'd0;
            computation_result <= 'd0;
            // TODO: Reset intermediate_res
        end else begin
            unique case (cim_state)
                CIM_IDLE: begin
                    if (bus_op_read == PATCH_LOAD_BROADCAST_START_OP) begin
                        cim_state <= CIM_PATCH_LOAD;
                    end
                end
                CIM_PATCH_LOAD: begin
                    unique case (gen_reg_2b)
                        'd0: begin
                            if (~MAC_compute_in_progress && (word_rec_cnt == PATCH_LEN)) begin
                                gen_reg_2b <= 'd1;
                                gen_cnt_7b_2_inc <= 'd0;
                                mac_start <= 1'b1;
                                mac_start_addr1 <= 'd0;
                                mac_start_addr2 <= 'd0;
                                mac_len <= PATCH_LEN;
                                mac_param_type <= MODEL_PARAM;
                                mac_activation <= LINEAR_ACTIVATION;
                                mac_bias_addr <= param_addr_map[SINGLE_PARAMS].addr + PATCH_PROJ_BIAS_OFF;
                            end
                        end
                        'd1: begin
                            mac_start <= 1'b0;
                            if (mac_done) begin
                                int_res_access_signals.addr_table[LOGIC_FSM] <= {3'd0, gen_cnt_7b_2_cnt} + mem_map[PATCH_MEM];
                                int_res_access_signals.write_data[LOGIC_FSM] <= (mac_out[N_COMP-1]) ? (~mac_out_flipped[N_STORAGE-1:0]+1'd1) : mac_out[N_STORAGE-1:0]; // Selecting the bottom N_STORAGE bits requires converting to positive two's complement if the number is negative, selecting, and converting back
                                gen_reg_2b <= 'd0;
                                gen_cnt_7b_2_inc <= 'd1;
                            end
                        end
                        default: begin
                            $fatal("Invalid gen_reg_2b value in CIM_PATCH_LOAD state");
                        end
                    endcase
                    word_rec_cnt_rst_n <= (~MAC_compute_in_progress && (word_rec_cnt == PATCH_LEN)) ? RST : RUN;
                    int_res_access_signals.write_req_src[LOGIC_FSM] <= (mac_done);
                    is_ready_internal <= (gen_cnt_7b_2_cnt == NUM_PATCHES);
                    cim_state <= (gen_cnt_7b_2_cnt == NUM_PATCHES) ? CIM_INFERENCE_RUNNING : cim_state;
                    gen_cnt_7b_2_rst_n <= (gen_cnt_7b_2_cnt == NUM_PATCHES) ? RST : RUN;
                end
                CIM_INFERENCE_RUNNING: begin
                    unique case (current_inf_step)
                        CLASS_TOKEN_CONCAT_STEP : begin // Move classification token from parameters memory to intermediate storage
                            gen_reg_2b <= gen_reg_2b + 'd1;
                            if (gen_reg_2b == 0) begin // Read from model parameters memory
                                params_access_signals.addr_table[LOGIC_FSM] <= param_addr_map[SINGLE_PARAMS].addr + CLASS_TOKEN_OFF;
                                params_access_signals.read_req_src[LOGIC_FSM] <= 1'b1;
                            end else if (gen_reg_2b == 2) begin
                                params_access_signals.read_req_src[LOGIC_FSM] <= 1'b0;
                                int_res_access_signals.write_req_src[LOGIC_FSM] <= 1'b1;
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[CLASS_TOKEN_MEM];
                                int_res_access_signals.write_data[LOGIC_FSM] <= params_read_data;
                            end else if (gen_reg_2b == 3) begin
                                int_res_access_signals.write_req_src[LOGIC_FSM] <= 1'b0;
                                gen_reg_2b <= 'd0;
                                current_inf_step <= POS_EMB_STEP;
                            end
                        end

                        POS_EMB_STEP : begin
                            if (gen_cnt_7b_cnt < (NUM_PATCHES+1)) begin
                                if (gen_reg_2b == 'd0) begin // Read from intermediate result and model parameter memories
                                    params_access_signals.read_req_src[LOGIC_FSM] <= 1'b1;
                                    params_access_signals.addr_table[LOGIC_FSM] <= param_addr_map[POS_EMB_PARAMS].addr + {3'd0, gen_cnt_7b_cnt};
                                    int_res_access_signals.read_req_src[LOGIC_FSM] <= 1'b1;
                                    int_res_access_signals.write_req_src[LOGIC_FSM] <= 1'b0;
                                    int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[POS_EMB_MEM] + {3'd0, gen_cnt_7b_cnt};
                                    gen_reg_2b <= 'd1;
                                end else if (gen_reg_2b == 'd1) begin // Start addition
                                    cim_add_input_q_1 <= {{(N_COMP-N_STORAGE){'0}}, params_read_data};
                                    cim_add_input_q_2 <= {{(N_COMP-N_STORAGE){'0}}, int_res_read_data};
                                    params_access_signals.read_req_src[LOGIC_FSM] <= 1'b0;
                                    int_res_access_signals.read_req_src[LOGIC_FSM] <= 1'b0;
                                    gen_reg_2b <= 'd2;
                                end else if (gen_reg_2b == 'd2) begin // Save results
                                    int_res_access_signals.write_req_src[LOGIC_FSM] <= 1'b1;
                                    int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[POS_EMB_MEM] + {3'd0, gen_cnt_7b_cnt};
                                    int_res_access_signals.write_data[LOGIC_FSM] <= (add_output_q[N_COMP-1]) ? (~add_out_flipped[N_STORAGE-1:0]+1'd1) : add_output_q[N_STORAGE-1:0];;
                                    gen_reg_2b <= 'd0;
                                end
                                gen_cnt_7b_inc <= {6'd0, (gen_reg_2b == 'd2)};
                            end else begin
                                gen_cnt_7b_inc <= 'd0;
                                gen_reg_2b <= 'd0;
                                int_res_access_signals.write_req_src[LOGIC_FSM] <= 1'b0;
                                current_inf_step <= (gen_cnt_7b_cnt == (NUM_PATCHES+1)) ? ENC_LAYERNORM_1_1ST_HALF_STEP : POS_EMB_STEP;
                                int_res_access_signals.read_req_src[LOGIC_FSM] <= 1'b0;
                                params_access_signals.read_req_src[LOGIC_FSM] <= 1'b0;
                                $display("Done with POS_EMB_STEP");
                                is_ready_internal <= 1'b1;
                            end
                            gen_cnt_7b_rst_n <= (gen_cnt_7b_cnt == (NUM_PATCHES+1)) ? RST : RUN;
                            cim_add_refresh <= (gen_cnt_7b_cnt != (NUM_PATCHES+1));
                        end

                        ENC_LAYERNORM_1_1ST_HALF_STEP,
                        ENC_LAYERNORM_2_1ST_HALF_STEP,
                        ENC_LAYERNORM_3_1ST_HALF_STEP: begin
                            layernorm_start <= (word_rec_cnt == EMB_DEPTH);
                            word_rec_cnt_rst_n <= (word_rec_cnt == EMB_DEPTH) ? RST : RUN;
                            is_ready_internal <= layernorm_done || ((bus_target_or_sender_read == ID) & (word_snt_cnt >= NUM_PATCHES+1)); // If I'm the one sending and I've sent everything (//TODO: Check if this makes sense...what if I've CiM #63)

                            if (current_inf_step == ENC_LAYERNORM_1_1ST_HALF_STEP)      layernorm_start_addr <= mem_map[ENC_LN1_1ST_HALF_MEM];
                            else if (current_inf_step == ENC_LAYERNORM_2_1ST_HALF_STEP) layernorm_start_addr <= mem_map[ENC_LN2_1ST_HALF_MEM];
                            else if (current_inf_step == ENC_LAYERNORM_3_1ST_HALF_STEP) layernorm_start_addr <= mem_map[MLP_HEAD_LN_1ST_HALF_MEM];

                            if (bus_op_read == PISTOL_START_OP) begin
                                current_inf_step <= INFERENCE_STEP_T'(current_inf_step + 6'd1);
                                $display("Finished LayerNorm (1st half) step at time: %d", $time);
                            end
                        end

                        ENC_LAYERNORM_1_2ND_HALF_STEP,
                        ENC_LAYERNORM_2_2ND_HALF_STEP,
                        ENC_LAYERNORM_3_2ND_HALF_STEP: begin
                            logic all_data_received = (word_rec_cnt == NUM_PATCHES+1) || ((word_rec_cnt == 1) && (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP));
                            layernorm_start <= all_data_received;
                            is_ready_internal <= (layernorm_done | is_ready_internal);
                            layernorm_half_select <= SECOND_HALF;
                            word_rec_cnt_rst_n <= all_data_received ? RST : RUN;
                            if (current_inf_step == ENC_LAYERNORM_1_2ND_HALF_STEP) begin
                                layernorm_start_addr <= mem_map[ENC_LN1_2ND_HALF_MEM];
                                beta_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_1_BETA_OFF;
                                gamma_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_1_GAMMA_OFF;
                            end else if (current_inf_step == ENC_LAYERNORM_2_2ND_HALF_STEP) begin
                                layernorm_start_addr <= mem_map[ENC_LN2_2ND_HALF_MEM];
                                beta_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_2_BETA_OFF;
                                gamma_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_2_GAMMA_OFF;
                            end else if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP) begin
                                layernorm_start_addr <= mem_map[MLP_HEAD_LN_2ND_HALF_MEM];
                                beta_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_3_BETA_OFF;
                                gamma_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_3_GAMMA_OFF;
                            end

                            if (bus_op_read == PISTOL_START_OP) begin
                                if      (current_inf_step == ENC_LAYERNORM_1_2ND_HALF_STEP) current_inf_step <= POST_LAYERNORM_TRANSPOSE_STEP;
                                else if (current_inf_step == ENC_LAYERNORM_2_2ND_HALF_STEP) current_inf_step <= ENC_PRE_MLP_TRANSPOSE_STEP;
                                else if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP) current_inf_step <= MLP_HEAD_PRE_DENSE_1_TRANSPOSE_STEP;
                                gen_reg_2b <= 'd0;
                                $display("Finished LayerNorm (2nd half) step at time: %d", $time);
                            end
                        end

                        POST_LAYERNORM_TRANSPOSE_STEP,
                        ENC_MHSA_Q_TRANSPOSE_STEP,
                        ENC_MHSA_K_TRANSPOSE_STEP,
                        ENC_POST_MHSA_TRANSPOSE_STEP: begin
                            is_ready_internal <= (word_snt_cnt >= data_len);
                            if (bus_op_read == PISTOL_START_OP) begin
                                word_rec_cnt_rst_n <= RST;
                                gen_cnt_7b_rst_n <= RST;
                                gen_cnt_7b_2_rst_n <= RST;
                                current_inf_step <= INFERENCE_STEP_T'(current_inf_step + 6'd1);
                            end
                        end

                        ENC_MHSA_DENSE_STEP: begin
                            word_rec_cnt_rst_n <= (word_rec_cnt >= EMB_DEPTH) ? RST : RUN;
                            gen_cnt_7b_rst_n <= RUN;
                            gen_cnt_7b_2_inc <= {6'd0, (mac_done && (gen_reg_2b == 'd2))};
                            gen_cnt_7b_2_rst_n <= (bus_op_read == PISTOL_START_OP) ? RST : RUN;

                            int_res_access_signals.write_req_src[LOGIC_FSM] <= mac_done;
                            int_res_access_signals.write_data[LOGIC_FSM] <= (mac_out[N_COMP-1]) ? (~mac_out_flipped[N_STORAGE-1:0]+1'd1) : mac_out[N_STORAGE-1:0];;

                            if (current_inf_step == ENC_MHSA_DENSE_STEP) begin
                                if (gen_reg_2b == 0) begin
                                    mac_start_addr1 <= mem_map[ENC_QVK_IN_MEM];
                                    mac_start_addr2 <= param_addr_map[ENC_Q_DENSE_KERNEL_PARAMS].addr;
                                    mac_len <= EMB_DEPTH;
                                    mac_bias_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_Q_DENSE_BIAS_0FF;
                                    mac_param_type <= MODEL_PARAM;
                                    mac_activation <= LINEAR_ACTIVATION;
                                    gen_reg_2b <= (mac_done) ? 'd1 : 'd0;
                                    mac_start <= (word_rec_cnt >= EMB_DEPTH) & (word_rec_cnt_rst_n == RUN);// & (gen_cnt_7b_2_cnt < EMB_DEPTH);
                                end else if (gen_reg_2b == 1) begin
                                    int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[ENC_Q_MEM] + {3'd0, sender_id};
                                    mac_start_addr2 <= param_addr_map[ENC_K_DENSE_KERNEL_PARAMS].addr;
                                    mac_bias_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_K_DENSE_BIAS_0FF;
                                    gen_reg_2b <= (mac_done) ? 'd2 : 'd1;
                                    mac_start <= ~mac_busy & ~mac_done;
                                end else if (gen_reg_2b == 2) begin
                                    int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[ENC_K_MEM] + {3'd0, sender_id};
                                    mac_start_addr2 <= param_addr_map[ENC_V_DENSE_KERNEL_PARAMS].addr;
                                    mac_bias_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_K_DENSE_BIAS_0FF;
                                    gen_reg_2b <= (mac_done) ? 'd3 : 'd2;
                                    mac_start <= ~mac_busy & ~mac_done;
                                end else if (gen_reg_2b == 3) begin
                                    int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[ENC_V_MEM] + {3'd0, sender_id};
                                    int_res_access_signals.write_data[LOGIC_FSM] <= (mac_out[N_COMP-1]) ? (~mac_out_flipped[N_STORAGE-1:0]+1'd1) : mac_out[N_STORAGE-1:0];;
                                    is_ready_internal <= 1'b1;
                                    gen_reg_2b <= 'd0;
                                end
                            end

                            if (bus_op_read == PISTOL_START_OP) begin
                                if (current_inf_step == ENC_MHSA_DENSE_STEP) begin
                                    current_inf_step <= ENC_MHSA_Q_TRANSPOSE_STEP;
                                    $display("Finished MHSA Dense step at time: %d", $time);
                                end
                            end
                        end

                        ENC_MHSA_QK_T_STEP: begin
                            // Perform a MAC, then divide by sqrt(NUM_HEADS), then save to intermediate results memory and inputs to modules is correct
                            // TODO: Change this to multiply by 1/sqrt(num_heads) instead of dividing
                            TEMP_RES_ADDR_T MAC_storage_addr = mem_map[ENC_QK_T_MEM] + gen_cnt_7b_2_cnt*(NUM_PATCHES+1) + {3'd0, sender_id};
                            gen_cnt_7b_inc <= {6'd0, div_done};
                            gen_cnt_7b_2_inc <= {6'd0, (gen_cnt_7b_cnt == (NUM_PATCHES+1))};
                            gen_cnt_7b_rst_n <= (gen_cnt_7b_cnt == (NUM_PATCHES+1)) ? RST : RUN;
                            word_rec_cnt_rst_n <= (word_rec_cnt >= NUM_HEADS) ? RST : RUN;

                            mac_start <= (word_rec_cnt >= NUM_HEADS) & (word_rec_cnt_rst_n == RUN); // Start a MAC
                            logic_fsm_div_start <= mac_done; // Start a division once MAC is done
                            is_ready_internal <= div_done; // Ready to move on once division is done

                            int_res_access_signals.write_req_src[LOGIC_FSM] <= div_done; // Write to memory once division is done
                            int_res_access_signals.addr_table[LOGIC_FSM] <= MAC_storage_addr;
                            int_res_access_signals.write_data[LOGIC_FSM] <= (div_output_q[N_COMP-1]) ? (~div_out_flipped[N_STORAGE-1:0]+1'd1) : div_output_q[N_STORAGE-1:0];
                            params_access_signals.read_req_src[LOGIC_FSM] <= (word_rec_cnt >= NUM_HEADS) & (word_rec_cnt_rst_n == RUN); // Start a read of the parameter memory since it will be needed by the division module
                            params_access_signals.addr_table[LOGIC_FSM] <= param_addr_map[SINGLE_PARAMS].addr + ENC_SQRT_NUM_HEADS_OFF;

                            mac_start_addr1 <= mem_map[ENC_QK_T_IN_MEM];
                            mac_start_addr2 <= mem_map[ENC_K_T_MEM] + gen_cnt_7b_2_cnt*NUM_HEADS;
                            mac_activation <= NO_ACTIVATION;
                            mac_len <= NUM_HEADS;
                            mac_param_type <= INTERMEDIATE_RES;

                            logic_fsm_div_dividend <= mac_out;
                            logic_fsm_div_divisor <= {{(N_COMP-N_STORAGE){params_read_data[N_STORAGE-1]}}, params_read_data}; // Sign extend
    
                            if (bus_op_read == PISTOL_START_OP) begin
                                $display("CiM: Finished encoder's MHSA QK_T");
                                gen_cnt_7b_2_rst_n <= RST;
                                current_inf_step <= ENC_MHSA_PRE_SOFTMAX_TRANSPOSE_STEP;
                            end
                        end

                        ENC_MHSA_PRE_SOFTMAX_TRANSPOSE_STEP: begin
                            gen_cnt_7b_2_inc <= {6'd0, (word_rec_cnt == (NUM_PATCHES+1))};
                            word_rec_cnt_rst_n <= (word_rec_cnt == (NUM_PATCHES+1)) ? RST : RUN;
                            is_ready_internal <= ((bus_target_or_sender_read == ID) && (word_snt_cnt >= (NUM_PATCHES+1))) | (bus_target_or_sender_read != ID); // If I'm the one sending and I've sent everything (//TODO: Check if this makes sense...what if I've CiM #63)
                            gen_cnt_7b_2_rst_n <= (gen_cnt_7b_2_cnt == NUM_HEADS || ID == 'd63) ? RST : RUN;
                            current_inf_step <= (gen_cnt_7b_2_cnt == NUM_HEADS || ID == 'd63) ? ENC_MHSA_SOFTMAX_STEP : ENC_MHSA_PRE_SOFTMAX_TRANSPOSE_STEP;
                        end

                        ENC_MHSA_SOFTMAX_STEP: begin
                            gen_cnt_7b_2_inc <= {6'd0, softmax_start}; // Increment after starting so it's ready for the next start
                            softmax_start <= (gen_cnt_7b_2_cnt < NUM_HEADS) && (ID < (NUM_PATCHES+1)) && (~softmax_busy);
                            softmax_len <= NUM_PATCHES + 'd1;
                            softmax_start_addr <= mem_map[ENC_PRE_SOFTMAX_MEM] + gen_cnt_7b_2_cnt*(NUM_PATCHES+1);
                            is_ready_internal <= ((~softmax_busy) && (gen_cnt_7b_2_cnt == NUM_HEADS)) || (ID >= NUM_PATCHES+1);
                            gen_cnt_7b_2_rst_n <= (bus_op_read == PISTOL_START_OP) ? RST : RUN;

                            if (bus_op_read == PISTOL_START_OP) begin
                                gen_cnt_7b_rst_n <= RST;
                                current_inf_step <= ENC_MHSA_MULT_V_STEP;
                                $display("Done with ENC_MHSA_SOFTMAX_STEP");
                            end
                        end

                        ENC_MHSA_MULT_V_STEP: begin
                            logic is_my_matrix = (ID >= gen_cnt_7b_2_cnt*NUM_HEADS) && (ID < (gen_cnt_7b_2_cnt+1)*NUM_HEADS);

                            // Start MAC
                            mac_start <= ~mac_busy && is_my_matrix && (word_rec_cnt >= (NUM_PATCHES+1));
                            mac_start_addr1 <= mem_map[ENC_V_MULT_IN_MEM];
                            mac_start_addr2 <= mem_map[ENC_V_MEM];
                            mac_bias_addr <= 'd0;
                            mac_len <= NUM_PATCHES+1;
                            mac_activation <= NO_ACTIVATION;
                            mac_param_type <= INTERMEDIATE_RES;

                            // Save results
                            int_res_access_signals.write_req_src[LOGIC_FSM] <= mac_done;
                            int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[ENC_V_MULT_MEM] + {3'd0, gen_cnt_7b_cnt};
                            int_res_access_signals.write_data[LOGIC_FSM] <= (mac_out[N_COMP-1]) ? (~mac_out_flipped[N_STORAGE-1:0]+1'd1) : mac_out[N_STORAGE-1:0];
                            gen_cnt_7b_inc <= {6'd0, mac_done};
                            gen_cnt_7b_rst_n <= (gen_cnt_7b_cnt == (NUM_PATCHES+'d1)) ? RST : RUN;

                            // Pistol start
                            if (bus_op_read == PISTOL_START_OP) begin
                                gen_cnt_7b_2_rst_n <= RST;
                                current_inf_step <= ENC_POST_MHSA_TRANSPOSE_STEP;
                                $display("Finished MHSA MULT V step at time: %d", $time);
                                word_rec_cnt_rst_n <= RST;
                                is_ready_internal <= 1'b1;
                            end else begin
                                word_rec_cnt_rst_n <= (word_rec_cnt >= (NUM_PATCHES+1)) ? RST : RUN;
                                is_ready_internal <= ~(ID >= (NUM_CIMS - NUM_HEADS) && (sender_id == NUM_PATCHES)) && mac_done;
                            end
                        end

                        default: begin
                            $fatal("Invalid current_inf_step value in CIM_INFERENCE_RUNNING state");
                        end
                    endcase
                end
                CIM_INVALID: begin
                    $fatal("Invalid state in CIM!");
                end
                default: begin
                    cim_state <= CIM_IDLE;
                end
            endcase
        end
    end

    // Data fill and send FSM (small FSM to fill a register with data and send it to the bus)
    logic start_inst_fill, inst_fill_substate, need_to_send_dense;
    logic [1:0] addr_offset_counter, addr_offset_counter_delayed;
    logic [6:0] num_words_to_fill;
    TEMP_RES_ADDR_T data_fill_start_addr;

    enum logic {FILL_INST_IDLE, FILL_INST} data_fill_state;
    always_ff @ (posedge clk) begin : data_inst_fill_and_send_fsm
        if (!rst_n) begin
            data_fill_state <= FILL_INST_IDLE;
        end else begin
            unique case (data_fill_state)
                FILL_INST_IDLE: begin
                    if (start_inst_fill || (done_dense_save_broadcast && need_to_send_dense)) begin
                        int_res_access_signals.addr_table[DATA_FILL_FSM] <= data_fill_start_addr;
                        int_res_access_signals.read_req_src[DATA_FILL_FSM] <= 1'b1;
                        data_fill_state <= FILL_INST;
                    end
                    addr_offset_counter <= 'd0;
                    addr_offset_counter_delayed <= 'd0;
                    bus_drive <= 1'b0;
                    bus_data_write <= 'd0;
                end

                FILL_INST: begin
                    if (inst_fill_substate == 'd0) begin
                        inst_fill_substate <= 'd1;
                    end else if (inst_fill_substate == 'd1) begin // Data has come back from intermediate results memory
                        /* verilator lint_off WIDTHEXPAND */
                        if (num_words_to_fill[1:0] == 'd1) begin
                            bus_data_write['d2] <= int_res_read_data;
                        end else if (num_words_to_fill[1:0] == 'd2) begin
                            bus_data_write[2'd1 + addr_offset_counter] <= int_res_read_data;
                        end else begin
                            bus_data_write[addr_offset_counter] <= int_res_read_data;
                        end
                        /* verilator lint_on WIDTHEXPAND */
                        inst_fill_substate <= 'd0;
                        addr_offset_counter <= addr_offset_counter + 'd1;
                        int_res_access_signals.addr_table[DATA_FILL_FSM] <= data_fill_start_addr + {8'd0, addr_offset_counter + 2'd1};
                        int_res_access_signals.read_req_src[DATA_FILL_FSM] <= ~(addr_offset_counter_delayed == (num_words_to_fill[1:0]-2'd1));
                        data_fill_state <= (addr_offset_counter_delayed == (num_words_to_fill[1:0]-2'd1)) ? FILL_INST_IDLE : FILL_INST;
                        bus_drive <= (addr_offset_counter_delayed == (num_words_to_fill[1:0]-2'd1));
                    end
                    addr_offset_counter_delayed <= addr_offset_counter;
                end
                default:
                    $fatal("Invalid data_fill_state value in data_inst_fill_and_send_fsm");
            endcase
        end
    end

    // FSM to save words from dense broadcast instruction
    logic save_dense_broadcast_start, done_dense_save_broadcast;
    logic [1:0] save_dense_broadcast_num_words;
    enum logic {SAVE_WORD_DENSE_BROADCAST_IDLE, SAVE_WORD} dense_broadcast_save_state;
    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            dense_broadcast_save_state <= SAVE_WORD_DENSE_BROADCAST_IDLE;
        end else begin
            unique case (dense_broadcast_save_state)
                SAVE_WORD_DENSE_BROADCAST_IDLE: begin
                    if (save_dense_broadcast_start) begin
                        if (save_dense_broadcast_num_words > 'd1) begin // When we get here, the first word has already been saved, so we can start with the second word
                            int_res_access_signals.addr_table[DENSE_BROADCAST_SAVE_FSM] <= int_res_access_signals.addr_table[BUS_FSM] - 'd1;
                            int_res_access_signals.write_req_src[DENSE_BROADCAST_SAVE_FSM] <= 1'b1;
                            int_res_access_signals.write_data[DENSE_BROADCAST_SAVE_FSM] <= bus_data_copy_1;
                        end
                        dense_broadcast_save_state <= (save_dense_broadcast_num_words == 'd3) ? SAVE_WORD : SAVE_WORD_DENSE_BROADCAST_IDLE; // Only go to next state if there's a third word to save
                        done_dense_save_broadcast <= (save_dense_broadcast_num_words != 'd3); // If finished saving, start sending data to the bus
                    end else begin
                        int_res_access_signals.write_req_src[DENSE_BROADCAST_SAVE_FSM] <= 1'b0;
                        done_dense_save_broadcast <= 1'b0;
                    end
                end
                SAVE_WORD: begin
                    if (save_dense_broadcast_num_words == 'd3) begin
                        int_res_access_signals.addr_table[DENSE_BROADCAST_SAVE_FSM] <= int_res_access_signals.addr_table[DENSE_BROADCAST_SAVE_FSM] - 'd1;
                        int_res_access_signals.write_data[DENSE_BROADCAST_SAVE_FSM] <= bus_data_copy_2;
                        int_res_access_signals.write_req_src[DENSE_BROADCAST_SAVE_FSM] <= 1'b1;
                    end
                    dense_broadcast_save_state <= SAVE_WORD_DENSE_BROADCAST_IDLE;
                    done_dense_save_broadcast <= 1'b1;
                end
                default: begin
                    $fatal("Invalid dense_broadcast_save_state value in dense broadcast save data FSM");
                end
            endcase
        end
    end

    // Miscellanous combinational logic
    always_comb begin : computation_twos_comp_flip
        mult_out_flipped = ~mult_output_q + 'd1;
        mac_out_flipped = ~mac_out + 'd1;
        add_out_flipped = ~add_output_q + 'd1;
        div_out_flipped = ~div_output_q + 'd1;
        exp_output_flipped = ~exp_output + 'd1;
    end

    logic is_ready_internal;
    always_comb begin : is_ready_comb
        is_ready = is_ready_internal & ~(layernorm_busy | div_busy | sqrt_busy | mac_busy | exp_busy | softmax_busy);
    end

endmodule
