`ifndef _cim_sv_
`define _cim_sv_

// Includes
`include "cim/cim.svh"
`include "cim/cim_fcn.sv"

module cim # (
    parameter logic [5:0] ID = 0,
    parameter int STANDALONE_TB = 1,
    parameter int SRAM = 0
)(
    input wire clk,
    input wire rst_n,
    output logic is_ready,
    // Bus
    inout wire [BUS_OP_WIDTH-1:0] bus_op,
    inout wire signed [2:0][N_STORAGE-1:0] bus_data,
    inout wire [$clog2(NUM_CIMS)-1:0] bus_target_or_sender
);

    // Initialize arrays
    `include "top_init.sv"
    `include "cim/cim_init.sv"

    // Memory
    // TEMP_RES_ADDR_T mem_map [PREV_SOFTMAX_OUTPUT_MEM+'d1];
    MemAccessSignals params_access_signals();
    MemAccessSignals int_res_access_signals();
    STORAGE_WORD_T int_res_read_data, params_read_data;
    cim_mem #(.SRAM(SRAM)) mem  (   .clk(clk), .params_access_signals(params_access_signals), .int_res_access_signals(int_res_access_signals),
                                    .int_res_read_data(int_res_read_data), .params_read_data(params_read_data));

    // Bus
    logic bus_drive;
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
    logic [1:0] gen_reg_2b, gen_reg_2b_comms;
    logic [$clog2(NUM_CIMS+1)-1:0] sender_id, data_len;
    SLEEP_STAGE_T inferred_sleep_stage;
    TEMP_RES_ADDR_T tx_addr, rx_addr;
    COMP_WORD_T compute_temp;
    INFERENCE_STEP_T current_inf_step;
    logic save_dense_broadcast_start, done_dense_save_broadcast;
    logic [1:0] save_dense_broadcast_num_words;

    // Counters
    logic gen_cnt_7b_rst_n, gen_cnt_7b_2_rst_n, word_rec_cnt_rst_n_comms, word_rec_cnt_rst_n_logic, word_rec_cnt_rst_n, word_snt_cnt_rst_n;
    logic [6:0] gen_cnt_7b_inc, gen_cnt_7b_2_inc, word_rec_cnt_inc, word_snt_cnt_inc;
    wire [6:0] gen_cnt_7b_cnt, gen_cnt_7b_2_cnt, word_rec_cnt, word_snt_cnt;
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b_inst      (.clk(clk), .rst_n(gen_cnt_7b_rst_n),   .inc(gen_cnt_7b_inc),   .cnt(gen_cnt_7b_cnt));
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b_2_inst    (.clk(clk), .rst_n(gen_cnt_7b_2_rst_n), .inc(gen_cnt_7b_2_inc), .cnt(gen_cnt_7b_2_cnt));
    counter #(.WIDTH(7), .MODE(0)) word_rec_cnt_inst    (.clk(clk), .rst_n(word_rec_cnt_rst_n), .inc(word_rec_cnt_inc), .cnt(word_rec_cnt));
    counter #(.WIDTH(7), .MODE(0)) word_snt_cnt_inst    (.clk(clk), .rst_n(word_snt_cnt_rst_n), .inc(word_snt_cnt_inc), .cnt(word_snt_cnt));

    // Adder module
    wire add_overflow;
    logic add_refresh, cim_add_refresh, mac_add_refresh, layernorm_add_refresh, exp_add_refresh, softmax_add_refresh, cim_add_refresh_delayed;
    COMP_WORD_T add_input_q_1, add_input_q_2, cim_add_input_q_1, cim_add_input_q_2, mac_add_input_q_1, mac_add_input_q_2, layernorm_add_input_q_1, layernorm_add_input_q_2, exp_add_input_q_1, exp_add_input_q_2, softmax_add_input_1, softmax_add_input_2;
    COMP_WORD_T add_output_q, add_out_flipped;
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
        end else if (softmax_add_refresh) begin
            add_input_q_1 = softmax_add_input_1;
            add_input_q_2 = softmax_add_input_2;
        end
        add_refresh = (cim_add_refresh || mac_add_refresh || layernorm_add_refresh || exp_add_refresh || softmax_add_refresh);
    end

    //synopsys translate_off
    always_ff @ (posedge clk) begin : adder_assertions
        assert ($countones({cim_add_refresh, mac_add_refresh, layernorm_add_refresh, exp_add_refresh, softmax_add_refresh}) <= 1) else $fatal("Multiple add_refresh signals are asserted simultaneously!");
    end
    //synopsys translate_on
    adder add_inst (.clk(clk), .rst_n(rst_n), .refresh(add_refresh), .overflow(add_overflow), .input_q_1(add_input_q_1), .input_q_2(add_input_q_2), .output_q(add_output_q));

    // Multiplier module
    logic mult_refresh, layernorm_mult_refresh, mac_mult_refresh, exp_mult_refresh, logic_fsm_mul_refresh, softmax_mult_refresh, mul_overflow;
    COMP_WORD_T mult_input_q_1, mult_input_q_2, logic_fsm_mul_input_1, logic_fsm_mul_input_2, softmax_mult_input_1, softmax_mult_input_2;
    COMP_WORD_T exp_mult_input_1, exp_mult_input_2, layernorm_mult_input_q_1, layernorm_mult_input_q_2, mac_mult_input_q_1, mac_mult_input_q_2;
    COMP_WORD_T mult_output_q, mult_out_flipped;
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
        end else if (logic_fsm_mul_refresh) begin
            mult_input_q_1 = logic_fsm_mul_input_1;
            mult_input_q_2 = logic_fsm_mul_input_2;
        end else if (softmax_mult_refresh) begin
            mult_input_q_1 = softmax_mult_input_1;
            mult_input_q_2 = softmax_mult_input_2;
        end
        mult_refresh = (layernorm_mult_refresh || mac_mult_refresh || exp_mult_refresh || logic_fsm_mul_refresh);
    end
    //synopsys translate_off
    always_ff @ (posedge clk) begin : mult_assertions
        assert ($countones({layernorm_mult_refresh, mac_mult_refresh, exp_mult_refresh, logic_fsm_mul_refresh}) <= 1) else $fatal("Multiple mult_refresh signals are asserted simultaneously!");
    end
    //synopsys translate_on
    multiplier mult_inst (.clk(clk), .rst_n(rst_n), .refresh(mult_refresh), .overflow(mul_overflow), .input_q_1(mult_input_q_1), .input_q_2(mult_input_q_2), .output_q(mult_output_q));

    // Exp module signals
    COMP_WORD_T exp_output, exp_output_flipped;

    // Divider module
    logic div_dbz, div_overflow, div_done, div_busy, div_start, logic_fsm_div_start, layernorm_div_start, softmax_div_start, mac_div_start;
    COMP_WORD_T div_output_q, div_out_flipped, div_dividend, div_divisor, logic_fsm_div_dividend, logic_fsm_div_divisor;
    COMP_WORD_T softmax_div_dividend, softmax_div_divisor, layernorm_div_dividend, layernorm_div_divisor, mac_div_dividend, mac_div_divisor;
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
        end else if (mac_div_start) begin
            div_dividend = mac_div_dividend;
            div_divisor = mac_div_divisor;
        end
        div_start = (logic_fsm_div_start || layernorm_div_start || softmax_div_start || mac_div_start);
    end
    //synopsys translate_off
    always_ff @ (posedge clk) begin : div_assertions
        assert ($countones({logic_fsm_div_start, layernorm_div_start, softmax_div_start, mac_div_start}) <= 1) else $fatal("Multiple div_start signals are asserted simultaneously!");
    end
    //synopsys translate_on
    divider div_inst (.clk(clk), .rst_n(rst_n), .start(div_start), .dividend(div_dividend), .divisor(div_divisor), .done(div_done), .busy(div_busy), .output_q(div_output_q), .dbz(div_dbz), .overflow(div_overflow));

    // Sqrt module
    logic sqrt_start, sqrt_done, sqrt_busy, sqrt_neg_rad;
    COMP_WORD_T sqrt_rad_q, sqrt_root_q;
    sqrt sqrt_inst (.clk(clk), .rst_n(rst_n), .start(sqrt_start), .done(sqrt_done), .busy(sqrt_busy), .rad_q(sqrt_rad_q), .root_q(sqrt_root_q), .neg_rad(sqrt_neg_rad));

    // MAC module
    logic mac_start, mac_done, mac_busy, mac_exp_start, exp_busy, exp_done;
    logic [$clog2(MAC_MAX_LEN+1)-1:0] mac_len;
    TEMP_RES_ADDR_T mac_start_addr1, mac_start_addr2;
    PARAMS_ADDR_T mac_params_addr, mac_bias_addr;
    COMP_WORD_T mac_out, mac_out_flipped;
    COMP_WORD_T mac_exp_input;
    PARAM_TYPE_T mac_param_type;
    ACTIVATION_TYPE_T mac_activation;
    mac mac_inst (.clk(clk), .rst_n(rst_n), .start(mac_start), .done(mac_done), .busy(mac_busy), .param_type(mac_param_type), .len(mac_len), .activation(mac_activation),
                  .start_addr1(mac_start_addr1), .start_addr2(mac_start_addr2), .bias_addr(mac_bias_addr),
                  .params_access_signals(params_access_signals), .int_res_access_signals(int_res_access_signals),
                  .param_data(params_read_data), .int_res_data(int_res_read_data), .computation_result(mac_out),
                  .add_input_q_1(mac_add_input_q_1), .add_input_q_2(mac_add_input_q_2), .add_output_q(add_output_q), .add_out_flipped(add_out_flipped), .add_refresh(mac_add_refresh),
                  .mult_input_q_1(mac_mult_input_q_1), .mult_input_q_2(mac_mult_input_q_2), .mult_output_q(mult_output_q), .mult_refresh(mac_mult_refresh),
                  .div_busy(div_busy), .div_done(div_done), .div_start(mac_div_start), .div_output_q(div_output_q), .div_dividend(mac_div_dividend), .div_divisor(mac_div_divisor),
                  .exp_busy(exp_busy), .exp_done(exp_done), .exp_start(mac_exp_start), .exp_output_q(exp_output), .exp_input(mac_exp_input));

    // LayerNorm module
    logic layernorm_start, layernorm_half_select, layernorm_done, layernorm_busy;
    PARAMS_ADDR_T beta_addr, gamma_addr;
    TEMP_RES_ADDR_T layernorm_start_addr;
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
    logic softmax_start, softmax_busy, softmax_done, softmax_exp_start;
    logic [$clog2(MAC_MAX_LEN+1)-1:0] softmax_len;
    TEMP_RES_ADDR_T softmax_start_addr;
    COMP_WORD_T softmax_exp_input;
    softmax softmax_inst (.clk(clk), .rst_n(rst_n), .start(softmax_start), .len(softmax_len), .start_addr(softmax_start_addr), .busy(softmax_busy), .done(softmax_done),
                          .int_res_access_signals(int_res_access_signals), .int_res_data(int_res_read_data),
                          .add_output_q(add_output_q), .add_input_q_1(softmax_add_input_1), .add_input_q_2(softmax_add_input_2), .add_refresh(softmax_add_refresh),
                          .mult_output_q(mult_output_q), .mult_out_flipped(mult_out_flipped), .mult_input_q_1(softmax_mult_input_1), .mult_input_q_2(softmax_mult_input_2), .mult_refresh(softmax_mult_refresh),
                          .div_done(div_done), .div_output_q(div_output_q), .div_dividend(softmax_div_dividend), .div_divisor(softmax_div_divisor), .div_start(softmax_div_start),
                          .exp_done(exp_done), .exp_output_q(exp_output), .exp_out_flipped(exp_output_flipped), .exp_input_q(softmax_exp_input), .exp_start(softmax_exp_start));

    // Exponential module
    logic exp_start;
    COMP_WORD_T exp_input;
    always_latch begin : exp_MUX
        if (softmax_exp_start) begin
            exp_input = softmax_exp_input;
        end else if (mac_exp_start) begin
            exp_input = mac_exp_input;
        end
        exp_start = (softmax_exp_start || mac_exp_start);
    end
    exp exp_inst (  .clk(clk), .rst_n(rst_n), .start(exp_start), .busy(exp_busy), .done(exp_done), .input_q(exp_input), .output_q(exp_output),
                    .adder_output(add_output_q), .adder_refresh(exp_add_refresh), .adder_input_1(exp_add_input_q_1), .adder_input_2(exp_add_input_q_2),
                    .mult_output(mult_output_q), .mult_refresh(exp_mult_refresh), .mult_input_1(exp_mult_input_1), .mult_input_2(exp_mult_input_2));

    // Comms FSM
    logic start_inst_fill, inst_fill_substate, need_to_send_dense, is_ready_internal, is_ready_internal_comms;
    logic [6:0] num_words_to_fill;
    STORAGE_WORD_T bus_data_copy_1, bus_data_copy_2;
    TEMP_RES_ADDR_T data_fill_start_addr;
    always_ff @ (posedge clk) begin : cim_comms_fsm
        if (!rst_n) begin
            word_snt_cnt_rst_n <= RST;
            sender_id <= 'd0;
            data_len <= 'd0;
            tx_addr <= 'd0;
            rx_addr <= 'd0;
            is_ready_internal_comms <= 1'b1;
        end else begin
            unique case (bus_op_read)
                PATCH_LOAD_BROADCAST_START_OP: begin
                    // Nothing to do
                    word_snt_cnt_rst_n <= RUN;
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
                        start_inst_fill <= 1'b1;
                        num_words_to_fill <= (bus_data_read[1] == 'd1) ? 'd1 : 'd3;
                    end
                    data_len <= bus_data_read[1][$clog2(NUM_CIMS+1)-1:0];
                    rx_addr <= bus_data_read[2][$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0];
                    sender_id <= {1'd0, bus_target_or_sender_read};
                    word_snt_cnt_rst_n <= RST;
                    // is_ready_internal <= 1'b0;
                    // is_ready_internal_comms <= 1'b0;
                end

                DENSE_BROADCAST_DATA_OP,
                TRANS_BROADCAST_DATA_OP: begin
                    word_snt_cnt_rst_n <= RUN;
                    word_snt_cnt_inc <= 'd3;
                    // is_ready_internal_comms <= 1'b1;

                    // Grab appropriate data
                    if (bus_op_read == TRANS_BROADCAST_DATA_OP) begin
                        word_rec_cnt_inc <= {6'd0, has_my_data(word_snt_cnt, {1'b0,ID}) || (current_inf_step == MLP_HEAD_DENSE_2_STEP_CIM) || (bus_op_read == DENSE_BROADCAST_DATA_OP)};
                        if (has_my_data(word_snt_cnt, {1'b0,ID})) begin
                            int_res_access_signals.write_req_src[BUS_FSM] <= 1'b1;
                            int_res_access_signals.addr_table[BUS_FSM] <= rx_addr + {4'd0, bus_target_or_sender_read};
                            if ((word_snt_cnt+3) <= data_len) begin // More than 3 word left to receive
                                int_res_access_signals.write_data[BUS_FSM] <= bus_data_read[ID - word_snt_cnt];
                            end else if (word_snt_cnt == {1'b0,ID}) begin
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
                    word_rec_cnt_rst_n_comms <= RST;
                end
                PARAM_STREAM_OP: begin
                    word_rec_cnt_rst_n_comms <= RUN;
                    if (bus_target_or_sender_read == ID) begin // The master will progressively fill bus_data[0..1..2] as it receives data from external memory
                        unique case (gen_reg_2b_comms)
                            'd0: begin
                                params_access_signals.addr_table[BUS_FSM] <= rx_addr + {3'd0, word_rec_cnt};
                                params_access_signals.write_data[BUS_FSM] <= bus_data_read[0];
                                gen_reg_2b_comms <= 'd1;
                                is_ready_internal_comms <= 1'b0;
                            end
                            'd1: begin
                                params_access_signals.addr_table[BUS_FSM] <= rx_addr + {3'd0, word_rec_cnt};
                                params_access_signals.write_data[BUS_FSM] <= bus_data_read[1];
                                gen_reg_2b_comms <= 'd2;
                            end
                            'd2: begin
                                params_access_signals.addr_table[BUS_FSM] <= rx_addr + {3'd0, word_rec_cnt};
                                params_access_signals.write_data[BUS_FSM] <= bus_data_read[2];
                                gen_reg_2b_comms <= 'd0;
                                is_ready_internal_comms <= 1'b1;
                            end
                        endcase
                    end
                    params_access_signals.write_req_src[BUS_FSM] <= (bus_target_or_sender_read == ID);
                    word_rec_cnt_inc <= {6'd0, (bus_target_or_sender_read == ID)};
                    // word_rec_cnt_rst_n <= (bus_target_or_sender_read == ID) ? RUN : RST; // Hold under reset if CiM isn't recipient to save power
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
                    word_snt_cnt_rst_n <= RST;
                end
                default: begin
                end
            endcase
        end
    end

    // Compute control FSM
    logic send_inference_res_start;
    CIM_STATE_T cim_state;
    always_ff @ (posedge clk) begin : cim_compute_control_fsm
        if (!rst_n) begin
            cim_state <= CIM_IDLE;
            current_inf_step <= CLASS_TOKEN_CONCAT_STEP_CIM;
            gen_cnt_7b_rst_n <= RST;
            gen_cnt_7b_2_rst_n <= RST;
            word_rec_cnt_rst_n_logic <= RST;
            compute_temp <= 'd0;
            // TODO: Reset intermediate_res
        end else begin
            unique case (cim_state)
                CIM_IDLE: begin
                    if (bus_op_read == PATCH_LOAD_BROADCAST_START_OP) begin
                        cim_state <= CIM_PATCH_LOAD;
                    end
                    current_inf_step <= CLASS_TOKEN_CONCAT_STEP_CIM;
                    gen_cnt_7b_rst_n <= RUN;
                    gen_cnt_7b_2_rst_n <= RUN;
                    word_rec_cnt_rst_n_logic <= RUN;
                    is_ready_internal <= 1'b1;
                end
                
                CIM_PATCH_LOAD: begin
                    unique case (gen_reg_2b)
                        'd0: begin
                            if (~mac_busy && (word_rec_cnt == PATCH_LEN)) begin
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
                            //synopsys translate_off
                            $fatal("Invalid gen_reg_2b value in CIM_PATCH_LOAD state");
                            //synopsys translate_on
                        end
                    endcase
                    word_rec_cnt_rst_n_logic <= (~mac_busy && (word_rec_cnt == PATCH_LEN)) ? RST : RUN;
                    int_res_access_signals.write_req_src[LOGIC_FSM] <= (mac_done);
                    is_ready_internal <= (gen_cnt_7b_2_cnt == NUM_PATCHES);
                    cim_state <= (gen_cnt_7b_2_cnt == NUM_PATCHES) ? CIM_INFERENCE_RUNNING : cim_state;
                    gen_cnt_7b_2_rst_n <= (gen_cnt_7b_2_cnt == NUM_PATCHES) ? RST : RUN;
                end

                CIM_INFERENCE_RUNNING: begin
                    unique case (current_inf_step)
                        CLASS_TOKEN_CONCAT_STEP_CIM : begin // Move classification token from parameters memory to intermediate storage
                            gen_reg_2b <= gen_reg_2b + 'd1;
                            if (gen_reg_2b == 0) begin // Read from model parameters memory
                                is_ready_internal <= 1'b0;
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
                                current_inf_step <= POS_EMB_STEP_CIM;
                            end
                        end

                        POS_EMB_STEP_CIM : begin
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
                                    int_res_access_signals.write_data[LOGIC_FSM] <= (add_output_q[N_COMP-1]) ? (~add_out_flipped[N_STORAGE-1:0]+1'd1) : add_output_q[N_STORAGE-1:0];
                                    gen_reg_2b <= 'd0;
                                end
                                gen_cnt_7b_inc <= {6'd0, (gen_reg_2b == 'd2)};
                            end else begin
                                gen_cnt_7b_inc <= 'd0;
                                gen_reg_2b <= 'd0;
                                int_res_access_signals.write_req_src[LOGIC_FSM] <= 1'b0;
                                current_inf_step <= (gen_cnt_7b_cnt == (NUM_PATCHES+1)) ? ENC_LAYERNORM_1_1ST_HALF_STEP_CIM : POS_EMB_STEP_CIM;
                                int_res_access_signals.read_req_src[LOGIC_FSM] <= 1'b0;
                                params_access_signals.read_req_src[LOGIC_FSM] <= 1'b0;
                                $display("Done with POS_EMB_STEP_CIM");
                                is_ready_internal <= 1'b1;
                            end
                            gen_cnt_7b_rst_n <= (gen_cnt_7b_cnt == (NUM_PATCHES+1)) ? RST : RUN;
                            cim_add_refresh <= (gen_cnt_7b_cnt != (NUM_PATCHES+1));
                        end

                        ENC_LAYERNORM_1_1ST_HALF_STEP_CIM,
                        ENC_LAYERNORM_2_1ST_HALF_STEP_CIM,
                        ENC_LAYERNORM_3_1ST_HALF_STEP_CIM: begin
                            layernorm_start <= (word_rec_cnt == EMB_DEPTH);
                            word_rec_cnt_rst_n_logic <= (word_rec_cnt == EMB_DEPTH) ? RST : RUN;
                            if ((bus_op_read == PATCH_LOAD_BROADCAST_START_OP) || (bus_op_read == PATCH_LOAD_BROADCAST_OP)) begin
                                is_ready_internal <= 'd0;
                            end else begin
                                if (current_inf_step == ENC_LAYERNORM_3_1ST_HALF_STEP_CIM) begin
                                    is_ready_internal <= layernorm_done || ((bus_target_or_sender_read == ID) & (word_snt_cnt >= 1)); // If I'm the one sending and I've sent everything (//TODO: Check if this makes sense...what if I've CiM #63)
                                end else begin
                                    is_ready_internal <= layernorm_done || ((bus_target_or_sender_read == ID) & (word_snt_cnt >= NUM_PATCHES+1)); // If I'm the one sending and I've sent everything (//TODO: Check if this makes sense...what if I've CiM #63)
                                end
                            end
                            if (current_inf_step == ENC_LAYERNORM_1_1ST_HALF_STEP_CIM)      layernorm_start_addr <= mem_map[ENC_LN1_1ST_HALF_MEM];
                            else if (current_inf_step == ENC_LAYERNORM_2_1ST_HALF_STEP_CIM) layernorm_start_addr <= mem_map[ENC_LN2_1ST_HALF_MEM];
                            else if (current_inf_step == ENC_LAYERNORM_3_1ST_HALF_STEP_CIM) layernorm_start_addr <= mem_map[MLP_HEAD_LN_1ST_HALF_MEM];

                            if (bus_op_read == PISTOL_START_OP) begin
                                current_inf_step <= INFERENCE_STEP_T'(current_inf_step + 6'd1);
                                $display("Finished LayerNorm (1st half) step at time: %d", $time);
                            end
                        end

                        ENC_LAYERNORM_1_2ND_HALF_STEP_CIM,
                        ENC_LAYERNORM_2_2ND_HALF_STEP_CIM,
                        ENC_LAYERNORM_3_2ND_HALF_STEP_CIM: begin
                            logic all_data_received;
                            if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP_CIM) begin
                                all_data_received = (word_rec_cnt == 1) && (word_snt_cnt >= NUM_PATCHES+1);
                            end else begin
                                all_data_received = (word_rec_cnt == NUM_PATCHES+1);
                            end

                            layernorm_start <= all_data_received;
                            if ((bus_op_read == PATCH_LOAD_BROADCAST_START_OP) || (bus_op_read == PATCH_LOAD_BROADCAST_OP)) begin
                                is_ready_internal <= 'd0;
                            end else begin
                                is_ready_internal <= (layernorm_done | is_ready_internal) & ~((bus_op_read == TRANS_BROADCAST_START_OP) || ((bus_op_read == DENSE_BROADCAST_START_OP)));
                            end
                            layernorm_half_select <= SECOND_HALF;
                            word_rec_cnt_rst_n_logic <= all_data_received ? RST : RUN;
                            if (current_inf_step == ENC_LAYERNORM_1_2ND_HALF_STEP_CIM) begin
                                layernorm_start_addr <= mem_map[ENC_LN1_2ND_HALF_MEM];
                                beta_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_1_BETA_OFF;
                                gamma_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_1_GAMMA_OFF;
                            end else if (current_inf_step == ENC_LAYERNORM_2_2ND_HALF_STEP_CIM) begin
                                layernorm_start_addr <= mem_map[ENC_LN2_2ND_HALF_MEM];
                                beta_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_2_BETA_OFF;
                                gamma_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_2_GAMMA_OFF;
                            end else if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP_CIM) begin
                                layernorm_start_addr <= mem_map[MLP_HEAD_LN_2ND_HALF_MEM];
                                beta_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_3_BETA_OFF;
                                gamma_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_LAYERNORM_3_GAMMA_OFF;
                            end

                            if (bus_op_read == PISTOL_START_OP) begin
                                if      (current_inf_step == ENC_LAYERNORM_1_2ND_HALF_STEP_CIM) current_inf_step <= POST_LAYERNORM_TRANSPOSE_STEP_CIM;
                                else if (current_inf_step == ENC_LAYERNORM_2_2ND_HALF_STEP_CIM) current_inf_step <= ENC_PRE_MLP_TRANSPOSE_STEP_CIM;
                                else if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP_CIM) current_inf_step <= MLP_HEAD_PRE_DENSE_1_TRANSPOSE_STEP_CIM;
                                gen_reg_2b <= 'd0;
                                $display("Finished LayerNorm (2nd half) step at time: %d", $time);
                            end
                        end

                        POST_LAYERNORM_TRANSPOSE_STEP_CIM,
                        ENC_MHSA_Q_TRANSPOSE_STEP_CIM,
                        ENC_MHSA_K_TRANSPOSE_STEP_CIM,
                        ENC_POST_MHSA_TRANSPOSE_STEP_CIM,
                        ENC_PRE_MLP_TRANSPOSE_STEP_CIM,
                        ENC_POST_DENSE_1_TRANSPOSE_STEP_CIM,
                        MLP_HEAD_PRE_DENSE_1_TRANSPOSE_STEP_CIM,
                        MLP_HEAD_PRE_DENSE_2_TRANSPOSE_STEP_CIM: begin
                            if ((bus_op_read == PATCH_LOAD_BROADCAST_START_OP) || (bus_op_read == PATCH_LOAD_BROADCAST_OP)) begin
                                is_ready_internal <= 'd0;
                            end else begin
                                is_ready_internal <= (word_snt_cnt >= data_len);
                            end
                            if (bus_op_read == PISTOL_START_OP) begin
                                word_rec_cnt_rst_n_logic <= RST;
                                gen_cnt_7b_rst_n <= RST;
                                gen_cnt_7b_2_rst_n <= RST;
                                current_inf_step <= INFERENCE_STEP_T'(current_inf_step + 6'd1);
                            end
                        end

                        ENC_MHSA_DENSE_STEP_CIM,
                        MLP_DENSE_1_STEP_CIM,
                        MLP_DENSE_2_AND_SUM_STEP_CIM,
                        MLP_HEAD_DENSE_1_STEP_CIM,
                        MLP_HEAD_DENSE_2_STEP_CIM: begin
                            gen_cnt_7b_rst_n <= RUN;
                            gen_cnt_7b_2_inc <= {6'd0, (mac_done && (gen_reg_2b == 'd2))};
                            gen_cnt_7b_2_rst_n <= (bus_op_read == PISTOL_START_OP) ? RST : RUN;

                            int_res_access_signals.write_req_src[LOGIC_FSM] <= mac_done;
                            int_res_access_signals.write_data[LOGIC_FSM] <= (mac_out[N_COMP-1]) ? (~mac_out_flipped[N_STORAGE-1:0]+1'd1) : mac_out[N_STORAGE-1:0];

                            if (current_inf_step == ENC_MHSA_DENSE_STEP_CIM) begin
                                word_rec_cnt_rst_n_logic <= (word_rec_cnt >= EMB_DEPTH) ? RST : RUN;
                                if (gen_reg_2b == 0) begin
                                    mac_start_addr1 <= mem_map[ENC_QVK_IN_MEM];
                                    mac_start_addr2 <= param_addr_map[ENC_Q_DENSE_KERNEL_PARAMS].addr;
                                    mac_len <= EMB_DEPTH;
                                    mac_bias_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_Q_DENSE_BIAS_0FF;
                                    mac_param_type <= MODEL_PARAM;
                                    mac_activation <= LINEAR_ACTIVATION;
                                    gen_reg_2b <= (mac_done) ? 'd1 : 'd0;
                                    mac_start <= (word_rec_cnt >= EMB_DEPTH) & (word_rec_cnt_rst_n_logic == RUN);
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
                                    int_res_access_signals.write_data[LOGIC_FSM] <= (mac_out[N_COMP-1]) ? (~mac_out_flipped[N_STORAGE-1:0]+1'd1) : mac_out[N_STORAGE-1:0];
                                    gen_reg_2b <= 'd0;
                                end
                                if (bus_op_read == TRANS_BROADCAST_START_OP || bus_op_read == DENSE_BROADCAST_START_OP) begin
                                    is_ready_internal <= 1'b0;
                                end else if (gen_reg_2b == 3) begin
                                    is_ready_internal <= 1'b1;
                                end
                            end else if (current_inf_step == MLP_DENSE_1_STEP_CIM) begin
                                word_rec_cnt_rst_n_logic <= (word_rec_cnt >= EMB_DEPTH) ? RST : RUN;
                                // Start MAC
                                mac_start_addr1 <= mem_map[ENC_MLP_IN_MEM];
                                mac_start_addr2 <= param_addr_map[ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_KERNEL_PARAMS].addr;
                                mac_bias_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_MLP_DENSE_1_MLP_HEAD_DENSE_1_BIAS_OFF;
                                mac_param_type <= MODEL_PARAM;
                                mac_activation <= SWISH_ACTIVATION;
                                mac_len <= EMB_DEPTH;
                                mac_start <= (word_rec_cnt >= EMB_DEPTH) & (word_rec_cnt_rst_n_logic == RUN) & (ID < MLP_DIM);

                                // Save MAC results
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[ENC_MLP_DENSE1_MEM] + {3'd0, sender_id};
                                int_res_access_signals.write_data[LOGIC_FSM] <= (mac_out[N_COMP-1]) ? (~mac_out_flipped[N_STORAGE-1:0]+1'd1) : mac_out[N_STORAGE-1:0];
                                int_res_access_signals.write_req_src[LOGIC_FSM] <= mac_done;

                                if (bus_op_read == TRANS_BROADCAST_START_OP || bus_op_read == DENSE_BROADCAST_START_OP) begin
                                    is_ready_internal <= 1'b0;
                                end else begin
                                    is_ready_internal <= mac_done;
                                end

                            end else if (current_inf_step == MLP_DENSE_2_AND_SUM_STEP_CIM) begin
                                word_rec_cnt_rst_n_logic <= (word_rec_cnt >= MLP_DIM) ? RST : RUN;

                                // Start MAC
                                mac_start_addr1 <= mem_map[ENC_MLP_DENSE2_IN_MEM];
                                mac_start_addr2 <= param_addr_map[ENC_MLP_DENSE_2_KERNEL_PARAMS].addr;
                                mac_bias_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_MLP_DENSE_2_BIAS_OFF;
                                mac_param_type <= MODEL_PARAM;
                                mac_activation <= LINEAR_ACTIVATION;
                                mac_len <= MLP_DIM;
                                mac_start <= (word_rec_cnt >= MLP_DIM) & (word_rec_cnt_rst_n_logic == RUN);

                                // Start add
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[ENC_MLP_OUT_MEM] + {3'd0, sender_id}; // Residual connection with encoder input
                                int_res_access_signals.read_req_src[LOGIC_FSM] <= mac_done;
                                gen_reg_2b <= (mac_done || cim_add_refresh || is_ready_internal) ? (gen_reg_2b + 'd1) : gen_reg_2b;
                                cim_add_input_q_1 <= mac_out;
                                cim_add_input_q_2 <= {{(N_COMP-N_STORAGE){int_res_read_data[N_STORAGE-1]}}, int_res_read_data}; // Sign extend
                                cim_add_refresh <= (gen_reg_2b == 'd1);

                                // Save MAC results
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[ENC_MLP_OUT_MEM] + {3'd0, sender_id};
                                int_res_access_signals.write_data[LOGIC_FSM] <= (add_output_q[N_COMP-1]) ? (~add_out_flipped[N_STORAGE-1:0]+1'd1) : add_output_q[N_STORAGE-1:0];
                                int_res_access_signals.write_req_src[LOGIC_FSM] <= (gen_reg_2b == 'd2);

                                if (bus_op_read == TRANS_BROADCAST_START_OP || bus_op_read == DENSE_BROADCAST_START_OP) begin
                                    is_ready_internal <= 1'b0;
                                end else begin
                                    is_ready_internal <= (gen_reg_2b == 'd2);
                                end
                            end else if (current_inf_step == MLP_HEAD_DENSE_1_STEP_CIM) begin
                                word_rec_cnt_rst_n_logic <= (word_rec_cnt >= EMB_DEPTH) ? RST : RUN;

                                // Start MAC
                                mac_start_addr1 <= mem_map[MLP_HEAD_DENSE_1_IN_MEM];
                                mac_start_addr2 <= param_addr_map[ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_KERNEL_PARAMS].addr;
                                mac_bias_addr <= param_addr_map[SINGLE_PARAMS].addr + ENC_MLP_DENSE_1_MLP_HEAD_DENSE_1_BIAS_OFF;
                                mac_param_type <= MODEL_PARAM;
                                mac_activation <= SWISH_ACTIVATION;
                                mac_len <= EMB_DEPTH;
                                mac_start <= (word_rec_cnt >= EMB_DEPTH) & (word_rec_cnt_rst_n_logic == RUN) & (ID < MLP_DIM); // TODO: Here, we need ID >= MLP_DIM but for testing we use ID < MLP_DIM

                                // Save MAC results
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[MLP_HEAD_DENSE_1_OUT_MEM] + {3'd0, sender_id};
                                int_res_access_signals.write_data[LOGIC_FSM] <= (mac_out[N_COMP-1]) ? (~mac_out_flipped[N_STORAGE-1:0]+1'd1) : mac_out[N_STORAGE-1:0];
                                int_res_access_signals.write_req_src[LOGIC_FSM] <= mac_done;

                                if (bus_op_read == TRANS_BROADCAST_START_OP || bus_op_read == DENSE_BROADCAST_START_OP) begin
                                    is_ready_internal <= 1'b0;
                                end else begin
                                    is_ready_internal <= mac_done;
                                end
                            end else if (current_inf_step == MLP_HEAD_DENSE_2_STEP_CIM) begin
                                word_rec_cnt_rst_n_logic <= (word_rec_cnt >= MLP_DIM) ? RST : RUN;

                                // Start MAC
                                mac_start_addr1 <= mem_map[MLP_HEAD_DENSE_2_IN_MEM];
                                mac_start_addr2 <= param_addr_map[ENC_MLP_DENSE_2_KERNEL_PARAMS].addr;
                                mac_bias_addr <= param_addr_map[SINGLE_PARAMS].addr + MLP_HEAD_DENSE_2_BIAS_OFF;
                                mac_param_type <= MODEL_PARAM;
                                mac_activation <= LINEAR_ACTIVATION;
                                mac_len <= MLP_DIM;
                                mac_start <= (word_rec_cnt >= MLP_DIM) & (word_rec_cnt_rst_n_logic == RUN);

                                // Save MAC results
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[MLP_HEAD_DENSE_2_OUT_MEM];
                                int_res_access_signals.write_data[LOGIC_FSM] <= (mac_out[N_COMP-1]) ? (~mac_out_flipped[N_STORAGE-1:0]+1'd1) : mac_out[N_STORAGE-1:0];
                                int_res_access_signals.write_req_src[LOGIC_FSM] <= mac_done;

                                if (bus_op_read == TRANS_BROADCAST_START_OP || bus_op_read == DENSE_BROADCAST_START_OP) begin
                                    is_ready_internal <= 1'b0;
                                end else begin
                                    is_ready_internal <= mac_done;
                                end
                            end

                            if (bus_op_read == PISTOL_START_OP) begin
                                if (current_inf_step == ENC_MHSA_DENSE_STEP_CIM) begin
                                    current_inf_step <= ENC_MHSA_Q_TRANSPOSE_STEP_CIM;
                                    $display("Finished MHSA Dense step at time: %d", $time);
                                end else if (current_inf_step == MLP_DENSE_1_STEP_CIM) begin
                                    current_inf_step <= ENC_POST_DENSE_1_TRANSPOSE_STEP_CIM;
                                    $display("Finished MLP Dense 1 step at time: %d", $time);
                                end else if (current_inf_step == MLP_DENSE_2_AND_SUM_STEP_CIM) begin
                                    current_inf_step <= ENC_LAYERNORM_3_1ST_HALF_STEP_CIM;
                                    $display("Finished MLP Dense 2 and sum step at time: %d", $time);
                                end else if (current_inf_step == MLP_HEAD_DENSE_1_STEP_CIM) begin
                                    current_inf_step <= MLP_HEAD_PRE_DENSE_2_TRANSPOSE_STEP_CIM;
                                    $display("Finished MLP Head Dense 1 step at time: %d", $time);
                                end else if (current_inf_step == MLP_HEAD_DENSE_2_STEP_CIM) begin
                                    current_inf_step <= MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP_CIM;
                                    $display("Finished MLP Head Dense 2 step at time: %d", $time);
                                end
                            end
                        end

                        ENC_MHSA_QK_T_STEP_CIM: begin
                            // Perform a MAC, then divide by sqrt(NUM_HEADS), then save to intermediate results memory and inputs to modules is correct
                            TEMP_RES_ADDR_T MAC_storage_addr = mem_map[ENC_QK_T_MEM] + gen_cnt_7b_2_cnt*(NUM_PATCHES+1) + {3'd0, sender_id};
                            gen_cnt_7b_inc <= {6'd0, logic_fsm_mul_refresh};
                            gen_cnt_7b_2_inc <= {6'd0, (gen_cnt_7b_cnt == (NUM_PATCHES+1))};
                            gen_cnt_7b_rst_n <= (gen_cnt_7b_cnt == (NUM_PATCHES+1)) ? RST : RUN;
                            word_rec_cnt_rst_n_logic <= (word_rec_cnt >= NUM_HEADS) ? RST : RUN;

                            if ((bus_op_read == PATCH_LOAD_BROADCAST_START_OP) || (bus_op_read == PATCH_LOAD_BROADCAST_OP)) begin
                                is_ready_internal <= 'd0;
                            end else begin
                                is_ready_internal <= logic_fsm_mul_refresh;
                            end

                            // Start a MAC
                            mac_start <= (word_rec_cnt >= NUM_HEADS) & (word_rec_cnt_rst_n_logic == RUN);
                            mac_start_addr1 <= mem_map[ENC_QK_T_IN_MEM];
                            mac_start_addr2 <= mem_map[ENC_K_T_MEM] + gen_cnt_7b_2_cnt*NUM_HEADS;
                            mac_activation <= NO_ACTIVATION;
                            mac_len <= NUM_HEADS;
                            mac_param_type <= INTERMEDIATE_RES;

                            // Divide by sqrt(NUM_HEADS)
                            logic_fsm_mul_refresh <= mac_done; // Start a multiplication once MAC is done
                            logic_fsm_mul_input_1 <= mac_out;
                            logic_fsm_mul_input_2 <= {{(N_COMP-N_STORAGE){params_read_data[N_STORAGE-1]}}, params_read_data}; // Sign extend

                            // Write to memory once division is done
                            int_res_access_signals.write_req_src[LOGIC_FSM] <= logic_fsm_mul_refresh;
                            int_res_access_signals.addr_table[LOGIC_FSM] <= MAC_storage_addr;
                            int_res_access_signals.write_data[LOGIC_FSM] <= (mult_output_q[N_COMP-1]) ? (~mult_out_flipped[N_STORAGE-1:0]+1'd1) : mult_output_q[N_STORAGE-1:0];
                            params_access_signals.read_req_src[LOGIC_FSM] <= (word_rec_cnt >= NUM_HEADS) & (word_rec_cnt_rst_n_logic == RUN); // Start a read of the parameter memory since it will be needed by the division module
                            params_access_signals.addr_table[LOGIC_FSM] <= param_addr_map[SINGLE_PARAMS].addr + ENC_INV_SQRT_NUM_HEADS_OFF;

                            if (bus_op_read == PISTOL_START_OP) begin
                                $display("CiM: Finished encoder's MHSA QK_T");
                                gen_cnt_7b_2_rst_n <= RST;
                                current_inf_step <= ENC_MHSA_PRE_SOFTMAX_TRANSPOSE_STEP_CIM;
                            end
                        end

                        ENC_MHSA_PRE_SOFTMAX_TRANSPOSE_STEP_CIM: begin
                            gen_cnt_7b_2_inc <= {6'd0, (word_rec_cnt == (NUM_PATCHES+1))};
                            word_rec_cnt_rst_n_logic <= (word_rec_cnt == (NUM_PATCHES+1)) ? RST : RUN;
                            gen_cnt_7b_2_rst_n <= (gen_cnt_7b_2_cnt == NUM_HEADS || ID == 'd63) ? RST : RUN;
                            current_inf_step <= (gen_cnt_7b_2_cnt == NUM_HEADS || ID == 'd63) ? ENC_MHSA_SOFTMAX_STEP_CIM : ENC_MHSA_PRE_SOFTMAX_TRANSPOSE_STEP_CIM;

                            if ((bus_op_read == PATCH_LOAD_BROADCAST_START_OP) || (bus_op_read == PATCH_LOAD_BROADCAST_OP)) begin
                                is_ready_internal <= 'd0;
                            end else begin
                                is_ready_internal <= ((bus_target_or_sender_read == ID) && (word_snt_cnt >= (NUM_PATCHES+1))) | (bus_target_or_sender_read != ID); // If I'm the one sending and I've sent everything (//TODO: Check if this makes sense...what if I've CiM #63)
                            end
                        end

                        ENC_MHSA_SOFTMAX_STEP_CIM: begin
                            gen_cnt_7b_2_inc <= {6'd0, softmax_start}; // Increment after starting so it's ready for the next start
                            softmax_start <= (gen_cnt_7b_2_cnt < NUM_HEADS) && (ID < (NUM_PATCHES+1)) && (~softmax_busy);
                            softmax_len <= NUM_PATCHES + 'd1;
                            softmax_start_addr <= mem_map[ENC_PRE_SOFTMAX_MEM] + gen_cnt_7b_2_cnt*(NUM_PATCHES+1);
                            gen_cnt_7b_2_rst_n <= (bus_op_read == PISTOL_START_OP) ? RST : RUN;

                            if ((bus_op_read == PATCH_LOAD_BROADCAST_START_OP) || (bus_op_read == PATCH_LOAD_BROADCAST_OP)) begin
                                is_ready_internal <= 'd0;
                            end else begin
                                is_ready_internal <= ((~softmax_busy) && (gen_cnt_7b_2_cnt == NUM_HEADS)) || (ID >= NUM_PATCHES+1);
                            end
    
                            if (bus_op_read == PISTOL_START_OP) begin
                                gen_cnt_7b_rst_n <= RST;
                                current_inf_step <= ENC_MHSA_MULT_V_STEP_CIM;
                                $display("Done with ENC_MHSA_SOFTMAX_STEP");
                            end
                        end

                        ENC_MHSA_MULT_V_STEP_CIM: begin
                            logic is_my_matrix = ({1'b0,ID} >= gen_cnt_7b_2_cnt*NUM_HEADS) && ({1'b0,ID} < (gen_cnt_7b_2_cnt+1)*NUM_HEADS);
                            if ((bus_op_read == DENSE_BROADCAST_START_OP) || (bus_op_read == TRANS_BROADCAST_START_OP)) begin
                                gen_cnt_7b_2_inc <= {6'd0, (bus_target_or_sender_read == 'd0)}; // Count matrix
                                word_rec_cnt_rst_n_logic <= RST;
                            end

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
                                current_inf_step <= ENC_POST_MHSA_TRANSPOSE_STEP_CIM;
                                $display("Finished MHSA MULT V step at time: %d", $time);
                                word_rec_cnt_rst_n_logic <= RST;
                                is_ready_internal <= 1'b1;
                            end else begin
                                word_rec_cnt_rst_n_logic <= (word_rec_cnt >= (NUM_PATCHES+1)) ? RST : RUN;
                                if ((bus_op_read == PATCH_LOAD_BROADCAST_START_OP) || (bus_op_read == PATCH_LOAD_BROADCAST_OP)) begin
                                    is_ready_internal <= 'd0;
                                end else begin
                                    is_ready_internal <= ~({1'b0,ID} >= (NUM_CIMS - NUM_HEADS) && (sender_id == NUM_PATCHES)) && mac_done;
                                end
                            end
                        end

                        ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP_CIM: begin
                            // Start MAC
                            mac_start <= ~mac_busy && (word_rec_cnt >= EMB_DEPTH);
                            mac_start_addr1 <= mem_map[ENC_DENSE_IN_MEM];
                            mac_start_addr2 <= param_addr_map[ENC_COMB_HEAD_KERNEL_PARAMS].addr;
                            mac_bias_addr <= param_addr_map[SINGLE_PARAMS].addr+ENC_COMB_HEAD_BIAS_OFF;
                            mac_len <= EMB_DEPTH;
                            mac_activation <= LINEAR_ACTIVATION;
                            mac_param_type <= MODEL_PARAM;
                            word_rec_cnt_rst_n_logic <= (word_rec_cnt >= EMB_DEPTH) ? RST : RUN;

                            // Read encoder's input for residual connection
                            int_res_access_signals.read_req_src[LOGIC_FSM] <= mac_done; // mac_done is a convenient pulse signal
                            int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[ENC_MHSA_OUT_MEM] + {3'd0, sender_id};
                            gen_reg_2b <= {gen_reg_2b[0], int_res_access_signals.read_req_src[LOGIC_FSM]};

                            // Sum with encoder's input as a residual connection
                            cim_add_refresh <= gen_reg_2b[1];
                            cim_add_input_q_1 <= mac_out;
                            cim_add_input_q_2 <= {{(N_COMP-N_STORAGE){int_res_read_data[N_STORAGE-1]}}, int_res_read_data}; // Sign extend

                            // Post-add cleanup
                            int_res_access_signals.write_req_src[LOGIC_FSM] <= cim_add_refresh; // add_refresh is a convenient pulse signal
                            int_res_access_signals.write_data[LOGIC_FSM] <= (add_output_q[N_COMP-1]) ? (~add_out_flipped[N_STORAGE-1:0]+1'd1) : add_output_q[N_STORAGE-1:0];

                            if ((bus_op_read == PATCH_LOAD_BROADCAST_START_OP) || (bus_op_read == PATCH_LOAD_BROADCAST_OP)) begin
                                is_ready_internal <= 'd0;
                            end else begin
                                is_ready_internal <= cim_add_refresh;
                            end

                            if (bus_op_read == PISTOL_START_OP) begin
                                $display("Finished post-MHSA dense step at time: %d", $time);
                                gen_reg_2b <= 'd0;
                                current_inf_step <= ENC_LAYERNORM_2_1ST_HALF_STEP_CIM;
                            end
                        end

                        MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP_CIM: begin
                            if ((bus_op_read == PATCH_LOAD_BROADCAST_START_OP) || (bus_op_read == PATCH_LOAD_BROADCAST_OP)) begin
                                is_ready_internal <= 'd0;
                            end else begin
                                is_ready_internal <= (word_snt_cnt >= 'd1);
                            end
                            current_inf_step <= (word_rec_cnt == NUM_SLEEP_STAGES) ? MLP_HEAD_SOFTMAX_STEP_CIM : MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP_CIM;
                        end

                        MLP_HEAD_SOFTMAX_STEP_CIM: begin
                            is_ready_internal <= 1'b0;
                            gen_cnt_7b_2_rst_n <= RST;

                            // Start softmax
                            softmax_len <= NUM_SLEEP_STAGES;
                            softmax_start_addr <= mem_map[MLP_HEAD_SOFTMAX_IN_MEM];
                            softmax_start <= (gen_cnt_7b_2_cnt < NUM_HEADS) && (ID < NUM_SLEEP_STAGES) && (~softmax_busy);

                            // Change step
                            if (softmax_done) begin
                                gen_reg_2b <= 'd0;
                                current_inf_step <= (ID == 0) ? POST_SOFTMAX_DIVIDE_STEP_CIM : INFERENCE_COMPLETE_CIM;
                                $display("Finished MLP Head Softmax step at time: %d", $time);
                            end
                        end

                        POST_SOFTMAX_DIVIDE_STEP_CIM: begin
                            gen_cnt_7b_inc <= {6'd0, logic_fsm_mul_refresh};
                            gen_cnt_7b_rst_n <= (gen_cnt_7b_cnt == NUM_SLEEP_STAGES) ? RST : RUN;
                            if (gen_cnt_7b_cnt < NUM_SLEEP_STAGES) begin
                                gen_reg_2b <= (int_res_access_signals.read_req_src[LOGIC_FSM] || logic_fsm_mul_refresh || (gen_reg_2b == 'd3)) ? gen_reg_2b + 'd1 : gen_reg_2b;
                            end else begin
                                gen_reg_2b <= 'd0;
                            end

                            // Load data for division
                            int_res_access_signals.read_req_src[LOGIC_FSM] <= (gen_reg_2b == 'd0);
                            int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[MLP_HEAD_SOFTMAX_IN_MEM] + {3'd0, gen_cnt_7b_cnt};

                            // Start division
                            logic_fsm_mul_input_1 <= {{(N_COMP-N_STORAGE){int_res_read_data[N_STORAGE-1]}}, int_res_read_data}; // Sign extend
                            logic_fsm_mul_input_2 <= INV_NUM_SAMPLES_OUT_AVG; // 1/NUM_SAMPLES_OUT_AVG in fixed-point
                            logic_fsm_mul_refresh <= (gen_reg_2b == 'd1);

                            // Save results
                            int_res_access_signals.write_req_src[LOGIC_FSM] <= logic_fsm_mul_refresh || (gen_reg_2b == 'd3);
                            int_res_access_signals.addr_table[LOGIC_FSM] <= (int_res_access_signals.write_req_src[LOGIC_FSM]) ? mem_map[SOFTMAX_AVG_SUM_MEM] + {3'd0, gen_cnt_7b_cnt} : mem_map[MLP_HEAD_SOFTMAX_IN_MEM] + {3'd0, gen_cnt_7b_cnt}; // Write to two locations
                            int_res_access_signals.write_data[LOGIC_FSM] <= (mult_output_q[N_COMP-1]) ? (~mult_out_flipped[N_STORAGE-1:0]+1'd1) : mult_output_q[N_STORAGE-1:0];

                            // Done
                            current_inf_step <= (gen_cnt_7b_cnt == NUM_SLEEP_STAGES) ? POST_SOFTMAX_AVERAGING_STEP_CIM : POST_SOFTMAX_DIVIDE_STEP_CIM;
                        end

                        POST_SOFTMAX_AVERAGING_STEP_CIM: begin
                            gen_cnt_7b_inc <= {6'd0, cim_add_refresh};
                            gen_cnt_7b_rst_n <= (gen_cnt_7b_cnt == NUM_SLEEP_STAGES) ? RST : RUN;
                            gen_cnt_7b_2_inc <= {6'd0, (gen_cnt_7b_cnt == NUM_SLEEP_STAGES)};
                            gen_cnt_7b_2_rst_n <= ((gen_cnt_7b_cnt == NUM_SLEEP_STAGES) && (gen_cnt_7b_2_cnt == (NUM_SAMPLES_OUT_AVG-1))) ? RST : RUN;
                            cim_add_refresh_delayed <= cim_add_refresh;

                            if (cim_add_refresh_delayed) begin
                                gen_reg_2b <= 'd0;
                            end else begin
                                gen_reg_2b <= ((int_res_access_signals.read_req_src[LOGIC_FSM] || int_res_access_signals.write_req_src[LOGIC_FSM]) && (gen_reg_2b != 'd3)) ? (gen_reg_2b + 'd1) : gen_reg_2b;
                            end

                            // Read previous softmax
                            int_res_access_signals.read_req_src[LOGIC_FSM] <= (gen_reg_2b == 'd0) || (gen_reg_2b == 'd3);
                            if (gen_reg_2b == 'd0) begin
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[PREV_SOFTMAX_OUTPUT_MEM] + {3'd0, gen_cnt_7b_cnt} + gen_cnt_7b_2_cnt*NUM_SLEEP_STAGES; // Move previous softmax result
                            end else if (gen_reg_2b == 'd1) begin
                                int_res_access_signals.addr_table[LOGIC_FSM] <= {3'd0, gen_cnt_7b_cnt};
                            end else if (gen_reg_2b == 'd2) begin
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[SOFTMAX_AVG_SUM_MEM] + {3'd0, gen_cnt_7b_cnt} - 'd1;
                            end else if (gen_reg_2b == 'd3) begin
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[SOFTMAX_AVG_SUM_MEM] + {3'd0, gen_cnt_7b_cnt};
                            end

                            // Save previous softmax and add module output
                            int_res_access_signals.write_req_src[LOGIC_FSM] <= (gen_reg_2b == 'd1) || (gen_reg_2b == 'd2);
                            int_res_access_signals.write_data[LOGIC_FSM] <= (gen_reg_2b == 'd1) ? int_res_read_data : ((add_output_q[N_COMP-1]) ? (~add_out_flipped[N_STORAGE-1:0]+1'd1) : add_output_q[N_STORAGE-1:0]);

                            // Read and start add
                            cim_add_refresh <= (gen_reg_2b == 'd3);
                            cim_add_input_q_1 <= (gen_reg_2b == 'd1) ? {{(N_COMP-N_STORAGE){int_res_read_data[N_STORAGE-1]}}, int_res_read_data} : cim_add_input_q_1; // Grab int_res[addr_prev_softmax] when you write it to memory
                            cim_add_input_q_2 <= (gen_reg_2b == 'd3) ? {{(N_COMP-N_STORAGE){int_res_read_data[N_STORAGE-1]}}, int_res_read_data} : cim_add_input_q_2;

                            // Done
                            current_inf_step <= ((gen_cnt_7b_cnt == NUM_SLEEP_STAGES) && (gen_cnt_7b_2_cnt == (NUM_SAMPLES_OUT_AVG-1))) ? POST_SOFTMAX_ARGMAX_STEP_CIM : POST_SOFTMAX_AVERAGING_STEP_CIM;
                        end

                        POST_SOFTMAX_ARGMAX_STEP_CIM: begin
                            gen_cnt_7b_inc <= {6'd0, int_res_access_signals.read_req_src[LOGIC_FSM] & (gen_cnt_7b_inc != 'd1)}; // Current index
                            gen_cnt_7b_rst_n <= (gen_cnt_7b_cnt == NUM_SLEEP_STAGES) ? RST : RUN;

                            int_res_access_signals.read_req_src[LOGIC_FSM] <= (gen_cnt_7b_cnt < NUM_SLEEP_STAGES);
                            int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[SOFTMAX_AVG_SUM_MEM] + {3'd0, gen_cnt_7b_cnt};

                            if ({6'd0, int_res_read_data} > compute_temp) begin // compute_temp holds the large softmax value
                                inferred_sleep_stage <= gen_cnt_7b_cnt[$clog2(NUM_SLEEP_STAGES)-1:0];
                                compute_temp <= {6'd0, int_res_read_data}; // int_res_read_data will always be positive (since it's a probability, so no need to sign-extend)
                                gen_reg_2b <= 3;
                            end

                            if ((gen_cnt_7b_cnt == NUM_SLEEP_STAGES) && (gen_cnt_7b_rst_n == RUN)) begin
                            end else begin
                            end

                            current_inf_step <= ((gen_cnt_7b_cnt == NUM_SLEEP_STAGES) && (gen_cnt_7b_rst_n == RST) && (gen_reg_2b == 3)) ? RETIRE_SOFTMAX_STEP_CIM : POST_SOFTMAX_ARGMAX_STEP_CIM;
                        end

                        RETIRE_SOFTMAX_STEP_CIM: begin
                            if (gen_reg_2b == 'd1) begin // Want to extend cycle 1 (transition between reads and writes) by 1 to give time to properly clock in the last read
                                gen_reg_2b <= (int_res_access_signals.read_req_src[LOGIC_FSM] == 1'b0) ? gen_reg_2b + 'd1 : gen_reg_2b;
                            end else begin
                                gen_reg_2b <= gen_reg_2b + 'd1; // Note: We enter this state with gen_reg_2b = 3
                            end
                            gen_cnt_7b_inc <= {6'd0, (gen_reg_2b == 'd2)};
                            gen_cnt_7b_2_inc <= {6'd0, (gen_cnt_7b_cnt == NUM_SLEEP_STAGES)};
                            gen_cnt_7b_rst_n <= (gen_cnt_7b_cnt == NUM_SLEEP_STAGES) ? RST : RUN;
                            gen_cnt_7b_2_rst_n <= RUN;

                            // Read from memory
                            int_res_access_signals.read_req_src[LOGIC_FSM] <= (gen_reg_2b == 'd3) || (gen_reg_2b == 'd0);
                            if (gen_reg_2b == 'd3) begin
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[PREV_SOFTMAX_OUTPUT_MEM] + {3'd0, gen_cnt_7b_cnt} + NUM_SLEEP_STAGES*{3'd0, gen_cnt_7b_cnt};
                            end else if (gen_reg_2b == 'd0) begin
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[MLP_HEAD_SOFTMAX_IN_MEM] + {3'd0, gen_cnt_7b_cnt};
                            end else if (gen_reg_2b == 'd1) begin
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[PREV_SOFTMAX_OUTPUT_MEM] + {3'd0, gen_cnt_7b_cnt};
                            end else if (gen_reg_2b == 'd2) begin
                                int_res_access_signals.addr_table[LOGIC_FSM] <= mem_map[PREV_SOFTMAX_OUTPUT_MEM] + {3'd0, gen_cnt_7b_cnt} + NUM_SLEEP_STAGES*({3'd0, gen_cnt_7b_cnt}+1);
                            end

                            // Save to temporary memory for later write
                            compute_temp <= (gen_reg_2b == 'd1) ? {{(N_COMP-N_STORAGE){int_res_read_data[N_STORAGE-1]}}, int_res_read_data} : compute_temp;                            

                            // Write to memory
                            int_res_access_signals.write_req_src[LOGIC_FSM] <= (gen_reg_2b == 'd1) || (gen_reg_2b == 'd2);
                            int_res_access_signals.write_data[LOGIC_FSM] <= (gen_reg_2b == 'd2) ? int_res_read_data : compute_temp[N_STORAGE-1:0];

                            // Done
                            if ((gen_cnt_7b_cnt == NUM_SLEEP_STAGES) && (gen_cnt_7b_2_cnt == (NUM_SAMPLES_OUT_AVG-2))) begin
                                current_inf_step <= INFERENCE_COMPLETE_CIM;
                                is_ready_internal <= 1'b1;
                            end
                        end

                        INFERENCE_COMPLETE_CIM: begin
                            gen_cnt_7b_rst_n <= RST;
                            gen_cnt_7b_2_rst_n <= RST;
                            word_rec_cnt_rst_n_logic <= RST;
                            // word_snt_cnt_rst_n <= RST;
                            if (bus_op_read == PISTOL_START_OP) begin
                                $display("Inference complete at time: %d", $time);
                                $display("Inferred sleep stage: %d", integer'(inferred_sleep_stage));
                                send_inference_res_start <= (ID == 0);
                            end
                        end

                        //synopsys translate_off
                        default: begin
                            $fatal("Invalid current_inf_step value in CIM_INFERENCE_RUNNING state");
                        end
                        //synopsys translate_on
                    endcase
                end
                CIM_INVALID: begin
                    //synopsys translate_off
                    $fatal("Invalid state in CIM!");
                    //synopsys translate_on
                end
                default: begin
                    cim_state <= CIM_IDLE;
                end
            endcase
        end
    end

    // Data fill and send FSM (small FSM to fill a register with data and send it to the bus)
    logic [1:0] addr_offset_counter, addr_offset_counter_delayed;
    enum logic [1:0] {FILL_INST_IDLE, FILL_INST, SEND_INFERENCE_RESULT} data_fill_state;
    always_ff @ (posedge clk) begin : data_inst_fill_and_send_fsm
        if (!rst_n) begin
            data_fill_state <= FILL_INST_IDLE;
        end else begin
            // Doing this here so we don't have to conflicting drives to these signals
            if (bus_target_or_sender_read == ID) begin
                if (bus_op_read == DENSE_BROADCAST_START_OP) begin
                    bus_op_write <= DENSE_BROADCAST_DATA_OP;
                end else if (bus_op_read == TRANS_BROADCAST_START_OP) begin
                    bus_op_write <= TRANS_BROADCAST_DATA_OP;
                end
            end
            unique case (data_fill_state)
                FILL_INST_IDLE: begin
                    if (start_inst_fill || (done_dense_save_broadcast && need_to_send_dense)) begin
                        int_res_access_signals.addr_table[DATA_FILL_FSM] <= data_fill_start_addr;
                        int_res_access_signals.read_req_src[DATA_FILL_FSM] <= 1'b1;
                        data_fill_state <= FILL_INST;
                    end else if (send_inference_res_start) begin
                        data_fill_state <= SEND_INFERENCE_RESULT;
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

                SEND_INFERENCE_RESULT: begin
                    bus_op_write <= INFERENCE_RESULT_OP;
                    bus_data_write[0] <= {13'd0, inferred_sleep_stage};
                    bus_target_or_sender_write <= ID;
                    bus_drive <= 1'b1;
                    data_fill_state <= FILL_INST_IDLE;
                end
                //synopsys translate_off
                default:
                    $fatal("Invalid data_fill_state value in data_inst_fill_and_send_fsm");
                //synopsys translate_on
            endcase
        end
    end

    // FSM to save words from dense broadcast instruction
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
                //synopsys translate_off
                default: begin
                    $fatal("Invalid dense_broadcast_save_state value in dense broadcast save data FSM");
                end
                //synopsys translate_on
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

    always_comb begin : control_signal_mux
        is_ready = is_ready_internal & is_ready_internal_comms & ~(layernorm_busy | div_busy | sqrt_busy | mac_busy | exp_busy | softmax_busy);
        word_rec_cnt_rst_n = word_rec_cnt_rst_n_logic & word_rec_cnt_rst_n_comms;
    end

endmodule

`endif
