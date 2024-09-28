`ifndef _cim_centralized_vh_
`define _cim_centralized_vh_

import Defines::*;

module cim_centralized (
    input logic clk,
    output InferenceStep_t _current_inf_step,
    SoCInterface.cim soc_ctrl_i,

    // ----- Memory ---- //
    MemoryInterface param_write_tb_i,
    MemoryInterface int_res_write_tb_i
);
    // ----- GLOBAL SIGNALS ----- //
    logic rst_n, done;
    logic [2:0] delay_line_3b;
    State_t cim_state;
    InferenceStep_t current_inf_step;

    // ----- CONSTANTS -----//
    assign rst_n = soc_ctrl_i.rst_n;
    assign _current_inf_step = current_inf_step;

    // ----- INSTANTIATION ----- //
    // Counters
    CounterInterface #(.WIDTH(4)) cnt_4b_i ();
    CounterInterface #(.WIDTH(7)) cnt_7b_i ();
    CounterInterface #(.WIDTH(9)) cnt_9b_i ();
    counter #(.WIDTH(4), .MODE(LEVEL_TRIGGERED)) cnt_4b_u (.clk(clk), .sig(cnt_4b_i));
    counter #(.WIDTH(7), .MODE(LEVEL_TRIGGERED)) cnt_7b_u (.clk(clk), .sig(cnt_7b_i));
    counter #(.WIDTH(9), .MODE(LEVEL_TRIGGERED)) cnt_9b_u (.clk(clk), .sig(cnt_9b_i));

    // Memory
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) param_read_i ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) param_read_mac_i ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) param_read_cim_i ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) param_read_ln_i ();

    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_read_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_write_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_read_cim_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_read_mac_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_read_ln_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_read_softmax_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_write_cim_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_write_ln_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_write_softmax_i ();

    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) casts_mac_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) casts_ln_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) casts_softmax_i ();

    params_mem params_u   (.clk, .rst_n, .write(param_write_tb_i),.read(param_read_i)); // Only testbench writes to params
    int_res_mem int_res_u (.clk, .rst_n, .write(int_res_write_i), .read(int_res_read_i));

    // Compute
    ComputeIPInterface add_io ();
    ComputeIPInterface add_io_cim ();
    ComputeIPInterface add_io_exp ();
    ComputeIPInterface add_io_mac ();
    ComputeIPInterface add_io_ln ();
    ComputeIPInterface add_io_softmax ();
    ComputeIPInterface mult_io ();
    ComputeIPInterface mult_io_cim ();
    ComputeIPInterface mult_io_exp ();
    ComputeIPInterface mult_io_mac ();
    ComputeIPInterface mult_io_ln ();
    ComputeIPInterface mult_io_softmax ();
    ComputeIPInterface div_io ();
    ComputeIPInterface div_io_mac ();
    ComputeIPInterface div_io_ln ();
    ComputeIPInterface div_io_softmax ();
    ComputeIPInterface exp_io ();
    ComputeIPInterface exp_io_mac ();
    ComputeIPInterface exp_io_softmax ();
    ComputeIPInterface sqrt_io ();
    ComputeIPInterface mac_io ();
    ComputeIPInterface mac_io_extra ();
    ComputeIPInterface ln_io ();
    ComputeIPInterface ln_io_extra ();
    ComputeIPInterface softmax_io ();
    ComputeIPInterface softmax_io_extra ();

    adder add       (.clk, .rst_n, .io(add_io));
    multiplier mult (.clk, .rst_n, .io(mult_io));
    divider div     (.clk, .rst_n, .io(div_io));
    exp exp         (.clk, .rst_n, .io(exp_io), .adder_io(add_io_exp), .mult_io(mult_io_exp));
    sqrt sqrt       (.clk, .rst_n, .io(sqrt_io));
    mac mac         (.clk, .rst_n, .io(mac_io), .io_extra(mac_io_extra),
                     .casts(casts_mac_i), .param_read(param_read_mac_i), .int_res_read(int_res_read_mac_i),
                     .add_io(add_io_mac), .mult_io(mult_io_mac), .div_io(div_io_mac), .exp_io(exp_io_mac));
    layernorm layernorm (.clk, .rst_n, .io(ln_io), .io_extra(ln_io_extra), .add_io(add_io_ln), .mult_io(mult_io_ln), .div_io(div_io_ln), .sqrt_io,
                         .casts(casts_ln_i), .param_read(param_read_ln_i), .int_res_read(int_res_read_ln_i), .int_res_write(int_res_write_ln_i));
    softmax softmax (.clk, .rst_n, .io(softmax_io), .io_extra(softmax_io_extra), .add_io(add_io_softmax), .mult_io(mult_io_softmax), .div_io(div_io_softmax), .exp_io(exp_io_softmax),
                     .casts(casts_softmax_i), .int_res_read(int_res_read_softmax_i), .int_res_write(int_res_write_softmax_i));

    // ----- BYPASS ----- //
    CompFx_t adder_in_1_reg;
    always_comb begin : adder_bypass
        if ((current_inf_step == POS_EMB_COMPRESSION_STEP) & delay_line_3b[0]) add_io_cim.in_1 = add_io.out;
        else add_io_cim.in_1 = adder_in_1_reg;
    end

    // ----- FSM ----- //
    always_ff @ (posedge clk) begin : main_fsm
        if (~rst_n) begin
            cim_state <= IDLE_CIM;
        end else begin
            unique case (cim_state)
                IDLE_CIM: begin
                    if (soc_ctrl_i.start_eeg_load) cim_state <= EEG_LOAD;
                    if (soc_ctrl_i.new_sleep_epoch) start_inference();
                end
                EEG_LOAD: begin
                    if (soc_ctrl_i.new_sleep_epoch) start_inference();
                end
                INFERENCE_RUNNING: begin
                    if (current_inf_step == INFERENCE_COMPLETE) begin
                        cim_state <= IDLE_CIM;
                        soc_ctrl_i.inference_complete <= 1'b1;
                    end
                end
                INVALID_CIM: begin
                    cim_state <= IDLE_CIM;
                end
                default: begin
                    cim_state <= IDLE_CIM;
                end
            endcase
        end
    end

    always_ff @ (posedge clk) begin : inference_fsm
        if (~rst_n) begin
            reset();
        end else begin
            set_default_values();
            if (cim_state == IDLE_CIM) begin
                delay_line_3b[1] <= 1;
            end else if (cim_state == EEG_LOAD) begin
                if (soc_ctrl_i.new_eeg_data) begin
                    /* cnt_7b_i holds current EEG data in patch
                    cnt_9b_i holds current patch */
                        IntResAddr_t addr = mem_map[EEG_INPUT_MEM] + IntResAddr_t'(int'(cnt_7b_i.cnt) + (int'(cnt_9b_i.cnt) << $clog2(PATCH_LEN)));
                        CompFx_t eeg_normalized = CompFx_t'({soc_ctrl_i.eeg, (Q_COMP-ADC_BITWIDTH)'(0)}); // Normalize to [0, 1]
                        write_int_res(addr, eeg_normalized, int_res_width[EEG_WIDTH], int_res_format[EEG_FORMAT]);
                        if (int'(cnt_7b_i.cnt) == PATCH_LEN-1) begin
                            cnt_7b_i.rst_n <= 1'b0;
                            if (int'(cnt_9b_i.cnt) == NUM_PATCHES-1) cnt_9b_i.rst_n <= 1'b0;
                            else cnt_9b_i.inc <= 1'b1;
                        end else cnt_7b_i.inc <= 1'b1;
                end
            end else if (cim_state == INFERENCE_RUNNING) begin
                unique case (current_inf_step)
                    PATCH_PROJ_STEP: begin : patch_proj
                        /* cnt_7b_i holds current parameters row
                        cnt_9b_i holds current patch */

                        delay_line_3b[0] <= cnt_7b_i.inc | cnt_9b_i.inc;
                        delay_line_3b[1] <= delay_line_3b[0];

                        if (delay_line_3b[1]) begin
                            IntResAddr_t patch_addr = mem_map[EEG_INPUT_MEM] + IntResAddr_t'(int'(cnt_9b_i.cnt) << $clog2(EMB_DEPTH));
                            ParamAddr_t param_addr  = param_addr_map[PATCH_PROJ_KERNEL_PARAMS] + ParamAddr_t'(int'(EMB_DEPTH*cnt_7b_i.cnt));
                            ParamAddr_t bias_addr   = param_addr_map_bias[PATCH_PROJ_BIAS] + ParamAddr_t'(cnt_7b_i.cnt);
                            start_mac(patch_addr, IntResAddr_t'(param_addr), bias_addr, MODEL_PARAM, LINEAR_ACTIVATION, VectorLen_t'(PATCH_LEN), VectorLen_t'(0), HORIZONTAL,
                                    int_res_format[EEG_FORMAT], int_res_width[EEG_WIDTH], params_format[PATCH_PROJ_PARAM_FORMAT]);
                        end

                        if (mac_io.done) begin
                            IntResAddr_t int_res_write_addr = mem_map[PATCH_MEM] + IntResAddr_t'(int'(cnt_7b_i.cnt) + int'(cnt_9b_i.cnt) << $clog2(PATCH_LEN)); // Left shift instead of multiply since PATCH_LEN is a power of 2
                            write_int_res(int_res_write_addr, mac_io.out, int_res_width[PATCH_PROJ_OUTPUT_WIDTH], int_res_format[PATCH_PROJ_OUTPUT_FORMAT]);

                            // Update index control
                            if (int'(cnt_7b_i.cnt) == EMB_DEPTH-1) begin
                                cnt_7b_i.rst_n <= 1'b0;
                                if (int'(cnt_9b_i.cnt) == NUM_PATCHES-1) begin
                                    cnt_9b_i.rst_n <= 1'b0;
                                    current_inf_step <= CLASS_TOKEN_CONCAT_STEP;
                                end else cnt_9b_i.inc <= 1'b1;
                            end else cnt_7b_i.inc <= 1'b1;
                        end
                    end
                    CLASS_TOKEN_CONCAT_STEP: begin : class_token_concat
                        cnt_7b_i.inc <= 1'b1;
                        delay_line_3b[0] <= cnt_7b_i.inc; // One cycle delay
                        cnt_9b_i.inc <= delay_line_3b[0];
                        if (int'(cnt_9b_i.cnt) == EMB_DEPTH) begin
                            delay_line_3b <= 'b0;
                            cnt_7b_i.rst_n <= 1'b0;
                            cnt_9b_i.rst_n <= 1'b0;
                            current_inf_step <= POS_EMB_STEP;
                        end else begin
                            ParamAddr_t read_addr = param_addr_map_bias[CLASS_TOKEN] + ParamAddr_t'(cnt_7b_i.cnt);
                            IntResAddr_t write_addr = mem_map[CLASS_TOKEN_MEM] + IntResAddr_t'(cnt_9b_i.cnt);
                            if (cnt_7b_i.inc) read_params(read_addr, params_format[CLASS_EMB_TOKEN_PARAM_FORMAT]);
                            if (cnt_9b_i.inc) write_int_res(write_addr, param_read_i.data, int_res_width[CLASS_EMB_TOKEN_WIDTH], int_res_format[CLASS_EMB_TOKEN_FORMAT]);
                        end
                    end
                    POS_EMB_STEP: begin : pos_emb
                        /* gen_cnt_7b holds the column
                        gen_cnt_9b holds the row */

                        // TODO: The variables are updated with blocking operators. They would be synthesized with non-blocking operators, so does int_res_addr_write need to be updated in a line before int_res_addr_read?

                        ParamAddr_t params_addr = param_addr_map[POS_EMB_PARAMS] + ParamAddr_t'(cnt_7b_i.cnt) + ParamAddr_t'(EMB_DEPTH*cnt_9b_i.cnt);
                        IntResAddr_t int_res_addr_read = mem_map[CLASS_TOKEN_MEM] + IntResAddr_t'(cnt_7b_i.cnt) + IntResAddr_t'(EMB_DEPTH*cnt_9b_i.cnt);
                        IntResAddr_t int_res_addr_write = int_res_addr_read - (mem_map[CLASS_TOKEN_MEM] - mem_map[POS_EMB_MEM]) - IntResAddr_t'('d4);

                        // Read
                        if (cnt_7b_i.inc) read_params(params_addr, params_format[POS_EMB_PARAM_FORMAT]);
                        if (cnt_7b_i.inc) read_int_res(int_res_addr_read, int_res_width[CLASS_EMB_TOKEN_WIDTH], int_res_format[CLASS_EMB_TOKEN_FORMAT]);

                        // Add
                        if (delay_line_3b[1]) start_add(param_read_i.data, int_res_read_i.data);

                        // Write
                        if (add_io_cim.done) write_int_res(int_res_addr_write, add_io_cim.out, int_res_width[POS_EMB_WIDTH], int_res_format[POS_EMB_FORMAT]);

                        // Counter control
                        cnt_7b_i.inc <= 1'b1;
                        delay_line_3b[0] <= param_read_cim_i.en;
                        delay_line_3b[1] <= delay_line_3b[0]; // One cycle delay
                        if (int'(cnt_7b_i.cnt) == EMB_DEPTH-2) begin
                            cnt_7b_i.rst_n <= 1'b0;
                            cnt_9b_i.inc <= 1'b1;
                        end

                        // Done
                        if (int_res_addr_write == IntResAddr_t'(int'(mem_map[POS_EMB_MEM]) + EMB_DEPTH*(NUM_PATCHES+1) - 1)) begin
                            current_inf_step <= ENC_LAYERNORM_1_1ST_HALF_STEP;
                            delay_line_3b <= 3'b001;
                            cnt_7b_i.rst_n <= 1'b0;
                            cnt_9b_i.rst_n <= 1'b0;
                            done <= 1'b0;
                        end
                    end
                    ENC_LAYERNORM_1_1ST_HALF_STEP,
                    ENC_LAYERNORM_2_1ST_HALF_STEP,
                    ENC_LAYERNORM_3_1ST_HALF_STEP: begin : layernorm_1st_half
                        int num_rows;
                        IntResAddr_t input_starting_addr, output_starting_addr;
                        if (current_inf_step == ENC_LAYERNORM_3_1ST_HALF_STEP) num_rows = 1;
                        else num_rows = NUM_PATCHES+1;

                        if (~done & (delay_line_3b[1] | ln_io.done)) begin
                            if (current_inf_step == ENC_LAYERNORM_1_1ST_HALF_STEP) begin
                                input_starting_addr = mem_map[POS_EMB_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_7b_i.cnt);
                                output_starting_addr = mem_map[ENC_LN1_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_7b_i.cnt);
                            end else if (current_inf_step == ENC_LAYERNORM_2_1ST_HALF_STEP) begin
                                input_starting_addr = mem_map[ENC_MHSA_OUT_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_7b_i.cnt);
                                output_starting_addr = mem_map[ENC_LN2_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_7b_i.cnt);
                            end else if (current_inf_step == ENC_LAYERNORM_3_1ST_HALF_STEP) begin
                                input_starting_addr = mem_map[ENC_MLP_OUT_MEM];
                                output_starting_addr = mem_map[ENC_LN3_MEM];
                            end
                            start_layernorm(FIRST_HALF, input_starting_addr, output_starting_addr, ParamAddr_t'(0), ParamAddr_t'(0),
                                            int_res_width[LN_INPUT_WIDTH], int_res_format[LN_INPUT_FORMAT], int_res_width[LN_OUTPUT_WIDTH],
                                            int_res_format[LN_OUTPUT_FORMAT], params_format[LN_PARAM_FORMAT]);
                        end

                        cnt_7b_i.inc <= ln_io.start;
                        delay_line_3b[0] <= 1'b0;
                        delay_line_3b[1] <= delay_line_3b[0];
                        done <= (ln_io.start & (int'(cnt_7b_i.cnt) == num_rows-1)) | done;

                        if (done & ln_io.done) begin
                            done <= 1'b0;
                            cnt_7b_i.rst_n <= 1'b0;
                            current_inf_step <= InferenceStep_t'(int'(current_inf_step) + 1);
                            delay_line_3b[0] <= 1'b1;
                        end
                    end
                    ENC_LAYERNORM_1_2ND_HALF_STEP,
                    ENC_LAYERNORM_2_2ND_HALF_STEP,
                    ENC_LAYERNORM_3_2ND_HALF_STEP: begin : layernorm_2nd_half
                        IntResAddr_t input_starting_addr, output_starting_addr;
                        ParamAddr_t beta_addr, gamma_addr;

                        delay_line_3b[0] <= 1'b0;
                        delay_line_3b[1] <= delay_line_3b[0];

                        if (~done & (delay_line_3b[1] | ln_io.done)) begin
                            if (current_inf_step == ENC_LAYERNORM_1_2ND_HALF_STEP) begin
                                input_starting_addr = mem_map[ENC_LN1_MEM] + IntResAddr_t'(cnt_7b_i.cnt);
                                beta_addr = param_addr_map_bias[ENC_LAYERNORM_1_BETA] + ParamAddr_t'(cnt_7b_i.cnt);
                                gamma_addr = param_addr_map_bias[ENC_LAYERNORM_1_GAMMA] + ParamAddr_t'(cnt_7b_i.cnt);
                            end else if (current_inf_step == ENC_LAYERNORM_2_2ND_HALF_STEP) begin
                                input_starting_addr = mem_map[ENC_LN2_MEM] + IntResAddr_t'(cnt_7b_i.cnt);
                                beta_addr = param_addr_map_bias[ENC_LAYERNORM_2_BETA] + ParamAddr_t'(cnt_7b_i.cnt);
                                gamma_addr = param_addr_map_bias[ENC_LAYERNORM_2_GAMMA] + ParamAddr_t'(cnt_7b_i.cnt);
                            end else if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP) begin
                                input_starting_addr = mem_map[ENC_LN3_MEM] + IntResAddr_t'(cnt_7b_i.cnt);
                                beta_addr = param_addr_map_bias[ENC_LAYERNORM_3_BETA] + ParamAddr_t'(cnt_7b_i.cnt);
                                gamma_addr = param_addr_map_bias[ENC_LAYERNORM_3_GAMMA] + ParamAddr_t'(cnt_7b_i.cnt);
                            end
                            output_starting_addr = input_starting_addr;
                            start_layernorm(SECOND_HALF, input_starting_addr, output_starting_addr, beta_addr, gamma_addr,
                                            int_res_width[LN_INPUT_WIDTH], int_res_format[LN_INPUT_FORMAT], int_res_width[LN_OUTPUT_WIDTH],
                                            int_res_format[LN_OUTPUT_FORMAT], params_format[LN_PARAM_FORMAT]);
                        end

                        cnt_7b_i.inc <= ln_io.start;
                        done <= (ln_io.done & (int'(cnt_7b_i.cnt) == EMB_DEPTH-1)) | done;
                        if (done & ln_io.done) begin
                            start_add(CompFx_t'(0), CompFx_t'(0));
                            cnt_7b_i.rst_n <= 1'b0;
                            delay_line_3b[2] <= 1'b1;
                            current_inf_step <= InferenceStep_t'(int'(current_inf_step) + 1);
                            done <= 1'b0;
                            start_add(CompFx_t'(0), CompFx_t'(0));
                        end
                    end
                    POS_EMB_COMPRESSION_STEP: begin : pos_emb_compression
                        delay_line_3b[0] <= 1'b1;
                        delay_line_3b[1] <= int_res_read_i.en;

                        if (delay_line_3b[0]) read_int_res(IntResAddr_t'(add_io.out), int_res_width[LN_OUTPUT_WIDTH], int_res_format[LN_OUTPUT_FORMAT]);
                        if (delay_line_3b[1]) write_int_res(IntResAddr_t'(add_io.out-2), int_res_read_i.data, int_res_width[POS_EMB_COMPRESSION_WIDTH], int_res_format[POS_EMB_COMPRESSION_FORMAT]); // TODO: Need to fine-tune format using fixed-point accuracy study

                        // Update index control (just add ADD)
                        start_add(add_io.out, CompFx_t'(1));

                        if (int'(add_io.out) == EMB_DEPTH*(NUM_PATCHES+1)+1) begin
                            current_inf_step <= ENC_MHSA_Q_STEP;
                            delay_line_3b <= 3'b100;
                        end
                    end
                    ENC_MHSA_Q_STEP,
                    ENC_MHSA_K_STEP,
                    ENC_MHSA_V_STEP,
                    MLP_DENSE_1_STEP,
                    MLP_HEAD_DENSE_1_STEP,
                    MLP_HEAD_DENSE_2_STEP: begin : mhsa_qkv
                        /* cnt_7b_i holds current parameters row
                        cnt_9b_i holds current patch */

                        // Variables
                        int input_height, kernel_width;
                        IntResAddr_t data_row_addr;
                        ParamAddr_t kernel_col_addr, bias_addr;
                        FxFormatIntRes_t input_format, output_format;
                        DataWidth_t input_width, output_width;
                        FxFormatParams_t input_params_format;
                        if (current_inf_step == MLP_DENSE_1_STEP) begin
                            kernel_width = MLP_DIM;
                            input_height = NUM_PATCHES + 1;
                            input_format = int_res_format[LN_OUTPUT_FORMAT];
                            input_width = int_res_width[LN_OUTPUT_WIDTH];
                            output_format = int_res_format[MLP_DENSE_1_OUTPUT_FORMAT];
                            output_width = int_res_width[MLP_DENSE_1_OUTPUT_WIDTH];
                            input_params_format = params_format[MLP_DENSE_1_PARAMS_FORMAT];
                            kernel_col_addr = param_addr_map[ENC_MLP_DENSE_1_PARAMS] + ParamAddr_t'(EMB_DEPTH*cnt_9b_i.cnt);
                            bias_addr = param_addr_map_bias[ENC_MLP_DENSE_1_BIAS] + ParamAddr_t'(cnt_9b_i.cnt);
                            data_row_addr = mem_map[ENC_LN2_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_7b_i.cnt);
                        end else if (current_inf_step == MLP_HEAD_DENSE_1_STEP) begin
                            kernel_width = MLP_DIM;
                            input_height = 1;
                            input_format = int_res_format[LN_OUTPUT_FORMAT];
                            input_width = int_res_width[LN_OUTPUT_WIDTH];
                            output_format = int_res_format[MLP_HEAD_DENSE_1_OUTPUT_FORMAT];
                            output_width = int_res_width[MLP_HEAD_DENSE_1_OUTPUT_WIDTH];
                            input_params_format = params_format[MLP_HEAD_DENSE_1_PARAMS_FORMAT];
                            kernel_col_addr = param_addr_map[MLP_HEAD_DENSE_1_PARAMS] + ParamAddr_t'(EMB_DEPTH*cnt_9b_i.cnt);
                            bias_addr = param_addr_map_bias[MLP_HEAD_DENSE_1_BIAS] + ParamAddr_t'(cnt_9b_i.cnt);
                            data_row_addr = mem_map[ENC_LN3_MEM];
                        end else if (current_inf_step == MLP_HEAD_DENSE_2_STEP) begin
                            kernel_width = NUM_SLEEP_STAGES;
                            input_height = 1;
                            input_format = int_res_format[MLP_HEAD_DENSE_1_OUTPUT_FORMAT];
                            input_width = int_res_width[MLP_HEAD_DENSE_1_OUTPUT_WIDTH];
                            output_format = int_res_format[MLP_HEAD_DENSE_2_OUTPUT_FORMAT];
                            output_width = int_res_width[MLP_HEAD_DENSE_2_OUTPUT_WIDTH];
                            input_params_format = params_format[MLP_HEAD_DENSE_2_PARAMS_FORMAT];
                            kernel_col_addr = param_addr_map[MLP_HEAD_DENSE_2_PARAMS] + ParamAddr_t'(MLP_DIM*cnt_9b_i.cnt);
                            bias_addr = param_addr_map_bias[MLP_HEAD_DENSE_2_BIAS] + ParamAddr_t'(cnt_9b_i.cnt);
                            data_row_addr = mem_map[MLP_HEAD_DENSE_1_OUT_MEM];
                        end else begin
                            kernel_width = EMB_DEPTH;
                            input_height = NUM_PATCHES + 1;
                            input_format = int_res_format[QKV_INPUT_FORMAT];
                            input_width = int_res_width[QKV_INPUT_WIDTH];
                            output_format = int_res_format[QKV_OUTPUT_FORMAT];
                            output_width = int_res_width[QKV_OUTPUT_WIDTH];
                            input_params_format = params_format[QKV_PARAMS_FORMAT];
                            data_row_addr = mem_map[ENC_LN1_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_7b_i.cnt);
                            if (current_inf_step == ENC_MHSA_Q_STEP) begin
                                kernel_col_addr = param_addr_map[ENC_Q_DENSE_PARAMS] + ParamAddr_t'(EMB_DEPTH*cnt_9b_i.cnt);
                                bias_addr = param_addr_map_bias[ENC_Q_DENSE_BIAS] + ParamAddr_t'(cnt_9b_i.cnt);
                            end else if (current_inf_step == ENC_MHSA_K_STEP) begin
                                kernel_col_addr = param_addr_map[ENC_K_DENSE_PARAMS] + ParamAddr_t'(EMB_DEPTH*cnt_9b_i.cnt);
                                bias_addr = param_addr_map_bias[ENC_K_DENSE_BIAS] + ParamAddr_t'(cnt_9b_i.cnt);
                            end else if (current_inf_step == ENC_MHSA_V_STEP) begin
                                kernel_col_addr = param_addr_map[ENC_V_DENSE_PARAMS] + ParamAddr_t'(EMB_DEPTH*cnt_9b_i.cnt);
                                bias_addr = param_addr_map_bias[ENC_V_DENSE_BIAS] + ParamAddr_t'(cnt_9b_i.cnt);
                            end
                        end

                        // Execute
                        delay_line_3b[0] <= cnt_7b_i.inc | cnt_9b_i.inc;
                        delay_line_3b[1] <= delay_line_3b[0] | delay_line_3b[2];
                        delay_line_3b[2] <= 1'b0;
                        if (delay_line_3b[1]) begin
                            if (current_inf_step == MLP_DENSE_1_STEP || current_inf_step == MLP_HEAD_DENSE_1_STEP) start_mac(data_row_addr, IntResAddr_t'(kernel_col_addr), bias_addr, MODEL_PARAM, SWISH_ACTIVATION, VectorLen_t'(EMB_DEPTH), VectorLen_t'(0), HORIZONTAL, input_format, input_width, input_params_format);
                            else if (current_inf_step == MLP_HEAD_DENSE_2_STEP) start_mac(data_row_addr, IntResAddr_t'(kernel_col_addr), bias_addr, MODEL_PARAM, LINEAR_ACTIVATION, VectorLen_t'(MLP_DIM), VectorLen_t'(0), HORIZONTAL, input_format, input_width, input_params_format);
                            else start_mac(data_row_addr, IntResAddr_t'(kernel_col_addr), bias_addr, MODEL_PARAM, LINEAR_ACTIVATION, VectorLen_t'(EMB_DEPTH), VectorLen_t'(0), HORIZONTAL, input_format, input_width, input_params_format);

                        end
                        // Control
                        if (mac_io.done) begin
                            IntResAddr_t int_res_write_addr;
                            if (current_inf_step == ENC_MHSA_Q_STEP) int_res_write_addr = mem_map[ENC_Q_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_7b_i.cnt + int'(cnt_9b_i.cnt));
                            else if (current_inf_step == ENC_MHSA_K_STEP) int_res_write_addr = mem_map[ENC_K_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_7b_i.cnt + int'(cnt_9b_i.cnt));
                            else if (current_inf_step == ENC_MHSA_V_STEP) int_res_write_addr = mem_map[ENC_V_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_7b_i.cnt + int'(cnt_9b_i.cnt));
                            else if (current_inf_step == MLP_DENSE_1_STEP) int_res_write_addr = mem_map[ENC_MLP_DENSE1_MEM] + IntResAddr_t'(MLP_DIM*cnt_7b_i.cnt + int'(cnt_9b_i.cnt));
                            else if (current_inf_step == MLP_HEAD_DENSE_1_STEP) int_res_write_addr = mem_map[MLP_HEAD_DENSE_1_OUT_MEM] + IntResAddr_t'(cnt_9b_i.cnt);
                            else if (current_inf_step == MLP_HEAD_DENSE_2_STEP) int_res_write_addr = mem_map[MLP_HEAD_DENSE_2_OUT_MEM] + IntResAddr_t'(cnt_9b_i.cnt);
                            write_int_res(int_res_write_addr, mac_io.out, output_width, output_format);

                            // Update index control
                            if (int'(cnt_7b_i.cnt) == input_height-1) begin
                                cnt_7b_i.rst_n <= 1'b0;
                                if (int'(cnt_9b_i.cnt) == kernel_width-1) begin
                                    cnt_9b_i.rst_n <= 1'b0;
                                    done <= 1'b1;
                                end else cnt_9b_i.inc <= 1'b1;
                            end else cnt_7b_i.inc <= 1'b1;
                        end

                        if (done) begin
                            done <= 1'b0;
                            delay_line_3b[0] <= 'b1;
                            current_inf_step <= InferenceStep_t'(int'(current_inf_step) + 1);
                        end
                    end
                    ENC_MHSA_QK_T_STEP: begin : mhsa_qk_t
                        /* cnt_7b holds x
                        cnt_9b holds y
                        cnt_4b holds z

                        for z in 0...(NUM_HEADS-1):
                            for y in 0...(NUM_PATCHES):
                                for x 0...(NUM_PATCHES):
                        */

                        read_params(param_addr_map_bias[ENC_INV_SQRT_NUM_HEADS], params_format[ENC_INV_SQRT_NUM_HEADS_FORMAT]);
                        delay_line_3b[0] <= mult_io_cim.start;
                        delay_line_3b[1] <= 'b0;

                        // Execute
                        if ((delay_line_3b[1] | delay_line_3b[0]) & ~done) begin
                            IntResAddr_t Q_addr   = mem_map[ENC_Q_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_9b_i.cnt + NUM_HEADS*cnt_4b_i.cnt);
                            IntResAddr_t K_T_addr = mem_map[ENC_K_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_7b_i.cnt + NUM_HEADS*cnt_4b_i.cnt);
                            start_mac(Q_addr, K_T_addr, ParamAddr_t'(0), INTERMEDIATE_RES, NO_ACTIVATION, VectorLen_t'(NUM_HEADS), VectorLen_t'(0), HORIZONTAL, int_res_format[QK_T_OUTPUT_FORMAT], int_res_width[QK_T_OUTPUT_WIDTH], FxFormatParams_t'(0));
                        end

                        if (mac_io.done) start_mult(mac_io.out, param_read_i.data); // Multiply by inverse of number of heads (read in previous step and remains on the params bus)

                        if (delay_line_3b[0] & ~done) begin
                            IntResAddr_t int_res_write_addr = mem_map[ENC_QK_T_MEM] + IntResAddr_t'(cnt_7b_i.cnt) + IntResAddr_t'((NUM_PATCHES+1)*cnt_9b_i.cnt) + IntResAddr_t'((NUM_PATCHES+1)*(NUM_PATCHES+1)*cnt_4b_i.cnt) - IntResAddr_t'(1); // -1 to account for the counter having been increment by one to start a new MAC
                            write_int_res(int_res_write_addr, mult_io.out, int_res_width[QK_T_OUTPUT_WIDTH], int_res_format[QK_T_OUTPUT_FORMAT]);
                        end

                        // Control
                        if (mac_io.done) begin
                            if (int'(cnt_7b_i.cnt) == NUM_PATCHES) begin
                                cnt_7b_i.rst_n <= 1'b0;
                                if (int'(cnt_9b_i.cnt) == NUM_PATCHES) begin
                                    cnt_9b_i.rst_n <= 1'b0;
                                    if (int'(cnt_4b_i.cnt) == NUM_HEADS-1) begin
                                        cnt_4b_i.rst_n <= 1'b0;
                                        done <= 1'b1;
                                    end else cnt_4b_i.inc <= 1'b1; // z++
                                end else cnt_9b_i.inc <= 1'b1; // y++
                            end else cnt_7b_i.inc <= 1'b1; // x++
                        end

                        // Exit control
                        if (done & mult_io.done) begin
                            IntResAddr_t int_res_write_addr = mem_map[ENC_QK_T_MEM] + IntResAddr_t'(NUM_PATCHES) + IntResAddr_t'((NUM_PATCHES+1)*60) + IntResAddr_t'((NUM_PATCHES+1)*(NUM_PATCHES+1)*(NUM_HEADS-1));
                            done <= 1'b0;
                            delay_line_3b[0] <= 'b1;
                            cnt_7b_i.rst_n <= 1'b0;
                            current_inf_step <= ENC_MHSA_SOFTMAX_STEP;
                            write_int_res(int_res_write_addr, mult_io.out, int_res_width[QK_T_OUTPUT_WIDTH], int_res_format[QK_T_OUTPUT_FORMAT]);
                        end
                    end
                    ENC_MHSA_SOFTMAX_STEP,
                    MLP_HEAD_SOFTMAX_STEP: begin : softmax
                        // Variables
                        int num_rows;
                        if (current_inf_step == ENC_MHSA_SOFTMAX_STEP) num_rows = NUM_HEADS*(NUM_PATCHES+1);
                        else num_rows = 1;

                        // Execute
                        delay_line_3b[0] <= 'b0;
                        if ((softmax_io.done | delay_line_3b[0]) & ~done) begin
                            if (current_inf_step == ENC_MHSA_SOFTMAX_STEP) begin
                                IntResAddr_t addr = mem_map[ENC_QK_T_MEM] + IntResAddr_t'((NUM_PATCHES+1)*cnt_9b_i.cnt);
                                start_softmax(addr, VectorLen_t'(NUM_PATCHES+1), int_res_format[QK_T_OUTPUT_FORMAT], int_res_width[QK_T_OUTPUT_WIDTH], int_res_format[MHSA_SOFTMAX_OUTPUT_FORMAT], int_res_width[MHSA_SOFTMAX_OUTPUT_WIDTH]);
                            end else start_softmax(mem_map[MLP_HEAD_DENSE_2_OUT_MEM], VectorLen_t'(NUM_SLEEP_STAGES), int_res_format[MLP_HEAD_DENSE_2_OUTPUT_FORMAT], int_res_width[MLP_HEAD_DENSE_2_OUTPUT_WIDTH], int_res_format[MLP_SOFTMAX_OUTPUT_FORMAT], int_res_width[MLP_SOFTMAX_OUTPUT_WIDTH]);
                        end

                        // Control
                        if (softmax_io.start) begin
                            if (int'(cnt_9b_i.cnt) == num_rows-1) begin
                                cnt_9b_i.rst_n <= 1'b0;
                                done <= 1'b1;
                            end else cnt_9b_i.inc <= 1'b1;
                        end

                        // Exit control
                        if (done & softmax_io.done) begin
                            current_inf_step <= InferenceStep_t'(int'(current_inf_step) + 1);
                            delay_line_3b[0] <= 'b1;
                            done <= 1'b0;
                            if (current_inf_step == MLP_HEAD_SOFTMAX_STEP) read_int_res(mem_map[MLP_HEAD_DENSE_2_OUT_MEM], int_res_width[MLP_SOFTMAX_OUTPUT_WIDTH], int_res_format[MLP_SOFTMAX_OUTPUT_FORMAT]);
                        end
                    end
                    ENC_MHSA_MULT_V_STEP: begin : mult_v
                    /* cnt_7b holds x
                        cnt_9b holds y
                        cnt_4b holds z

                        for z in 0...(NUM_HEADS-1):
                            for y in 0...(NUM_PATCHES):
                                for x 0...(EMB_DEPTH/NUM_HEADS-1):
                        */

                        // Execute
                        if (delay_line_3b[1]) begin
                            IntResAddr_t QK_T_addr = mem_map[ENC_QK_T_MEM] + IntResAddr_t'((NUM_PATCHES+1)*cnt_9b_i.cnt) + IntResAddr_t'((NUM_PATCHES+1)*(NUM_PATCHES+1)*cnt_4b_i.cnt);
                            IntResAddr_t V_addr    = mem_map[ENC_V_MEM]    + IntResAddr_t'(cnt_7b_i.cnt) + IntResAddr_t'(NUM_HEADS*cnt_4b_i.cnt);
                            start_mac(QK_T_addr, V_addr, ParamAddr_t'(0), INTERMEDIATE_RES, NO_ACTIVATION, VectorLen_t'(NUM_PATCHES+1), VectorLen_t'(EMB_DEPTH), VERTICAL, int_res_format[MHSA_SOFTMAX_OUTPUT_FORMAT], int_res_width[MHSA_SOFTMAX_OUTPUT_WIDTH], FxFormatParams_t'(0));
                        end

                        // Control
                        delay_line_3b[0] <= cnt_7b_i.inc | cnt_9b_i.inc | cnt_4b_i.inc;
                        delay_line_3b[1] <= delay_line_3b[0];

                        // Control
                        if (mac_io.done) begin
                            IntResAddr_t int_res_write_addr = mem_map[ENC_V_MULT_MEM] + IntResAddr_t'(cnt_7b_i.cnt) + IntResAddr_t'(EMB_DEPTH*cnt_9b_i.cnt) + IntResAddr_t'((EMB_DEPTH/NUM_HEADS)*cnt_4b_i.cnt);
                            write_int_res(int_res_write_addr, mac_io.out, int_res_width[MULT_V_OUTPUT_WIDTH], int_res_format[MULT_V_OUTPUT_FORMAT]);

                            if (int'(cnt_7b_i.cnt) == (EMB_DEPTH/NUM_HEADS-1)) begin
                                cnt_7b_i.rst_n <= 1'b0;
                                if (int'(cnt_9b_i.cnt) == NUM_PATCHES) begin
                                    cnt_9b_i.rst_n <= 1'b0;
                                    if (int'(cnt_4b_i.cnt) == NUM_HEADS-1) begin
                                        cnt_4b_i.rst_n <= 1'b0;
                                        done <= 1'b1;
                                    end else cnt_4b_i.inc <= 1'b1; // z++
                                end else cnt_9b_i.inc <= 1'b1; // y++
                            end else cnt_7b_i.inc <= 1'b1; // x++
                        end

                        // Exit control
                        if (done) begin
                            IntResAddr_t int_res_write_addr = mem_map[ENC_V_MULT_MEM] + IntResAddr_t'(EMB_DEPTH/NUM_HEADS-1) + IntResAddr_t'(EMB_DEPTH*NUM_PATCHES) + IntResAddr_t'((EMB_DEPTH/NUM_HEADS)*(NUM_HEADS-1));
                            write_int_res(int_res_write_addr, mac_io.out, int_res_width[MULT_V_OUTPUT_WIDTH], int_res_format[MULT_V_OUTPUT_FORMAT]);
                            done <= 1'b0;
                            delay_line_3b[0] <= 'b1;
                            cnt_7b_i.rst_n <= 1'b0;
                            current_inf_step <= ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP;
                        end
                    end
                    ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP,
                    MLP_DENSE_2_AND_SUM_STEP: begin : dense_and_input_sum
                        /*  cnt_7b_i holds x
                            cnt_9b_i holds y

                        for y in 0...(NUM_PATCHES):
                            for x 0...(EMB_DEPTH-1):
                        */

                        // Variables
                        int input_height;
                        if (current_inf_step == ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP) input_height = int'(NUM_PATCHES+1);
                        else if (current_inf_step == MLP_DENSE_2_AND_SUM_STEP) input_height = 1;

                        // Execute
                        delay_line_3b[0] <= int_res_write_cim_i.en;
                        if (delay_line_3b[0] & ~done) begin
                            if (current_inf_step == ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP) begin
                                IntResAddr_t input_addr = mem_map[ENC_V_MULT_MEM] + IntResAddr_t'(EMB_DEPTH*cnt_9b_i.cnt);
                                IntResAddr_t kernel_addr = param_addr_map[ENC_COMB_HEAD_PARAMS] + IntResAddr_t'(EMB_DEPTH*cnt_7b_i.cnt);
                                ParamAddr_t bias_addr = param_addr_map_bias[ENC_COMB_HEAD_BIAS] + ParamAddr_t'(cnt_7b_i.cnt);
                                start_mac(input_addr, kernel_addr, bias_addr, MODEL_PARAM, LINEAR_ACTIVATION, VectorLen_t'(EMB_DEPTH), VectorLen_t'(0), HORIZONTAL, int_res_format[MULT_V_OUTPUT_FORMAT], int_res_width[MULT_V_OUTPUT_WIDTH], params_format[POST_MHSA_PARAM_FORMAT]);
                            end else begin
                                IntResAddr_t input_addr = mem_map[ENC_MLP_DENSE1_MEM];
                                IntResAddr_t kernel_addr = param_addr_map[ENC_MLP_DENSE_2_PARAMS] + IntResAddr_t'(MLP_DIM*cnt_7b_i.cnt);
                                ParamAddr_t bias_addr = param_addr_map_bias[ENC_MLP_DENSE_2_BIAS] + ParamAddr_t'(cnt_7b_i.cnt);
                                start_mac(input_addr, kernel_addr, bias_addr, MODEL_PARAM, LINEAR_ACTIVATION, VectorLen_t'(MLP_DIM), VectorLen_t'(0), HORIZONTAL, int_res_format[MLP_DENSE_1_OUTPUT_FORMAT], int_res_width[MLP_DENSE_1_OUTPUT_WIDTH], params_format[MLP_DENSE_2_PARAMS_FORMAT]);
                            end
                        end

                        if (mac_io.done) begin
                            IntResAddr_t add_addr;
                            DataWidth_t width;
                            FxFormatIntRes_t format;
                            if (current_inf_step == ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP) begin
                                add_addr = mem_map[POS_EMB_MEM] + IntResAddr_t'(cnt_7b_i.cnt) + IntResAddr_t'(EMB_DEPTH*cnt_9b_i.cnt);
                                width = int_res_width[POS_EMB_COMPRESSION_WIDTH];
                                format = int_res_format[POS_EMB_COMPRESSION_FORMAT];
                            end else begin
                                add_addr = mem_map[ENC_MHSA_OUT_MEM] + IntResAddr_t'(cnt_7b_i.cnt);
                                width = int_res_width[MLP_DENSE_1_OUTPUT_WIDTH];
                                format = int_res_format[MLP_DENSE_1_OUTPUT_FORMAT];
                            end
                            read_int_res(add_addr, width, format);
                        end

                        delay_line_3b[1] <= int_res_read_cim_i.en;
                        if (delay_line_3b[1]) start_add(mac_io.out, int_res_read_i.data);

                        delay_line_3b[2] <= add_io_cim.start;
                        if (delay_line_3b[2]) begin // TODO: Need to delay this by one cycle
                            IntResAddr_t output_addr;
                            DataWidth_t width;
                            FxFormatIntRes_t format;
                            if (current_inf_step == ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP) begin
                                output_addr = mem_map[ENC_MHSA_OUT_MEM] + IntResAddr_t'(cnt_7b_i.cnt) + IntResAddr_t'(EMB_DEPTH*cnt_9b_i.cnt);
                                width = int_res_width[MHSA_SUM_OUTPUT_WIDTH];
                                format = int_res_format[MHSA_SUM_OUTPUT_FORMAT];
                            end else begin
                                output_addr = mem_map[ENC_MLP_OUT_MEM] + IntResAddr_t'(cnt_7b_i.cnt);
                                width = int_res_width[MLP_DENSE_2_OUTPUT_WIDTH];
                                format = int_res_format[MLP_DENSE_2_OUTPUT_FORMAT];
                            end
                            write_int_res(output_addr, add_io.out, width, format);

                            if (int'(cnt_7b_i.cnt) == EMB_DEPTH-1) begin
                                cnt_7b_i.rst_n <= 1'b0;
                                delay_line_3b[0] <= 1'b1;
                                if (int'(cnt_9b_i.cnt) == input_height-1) begin
                                    cnt_9b_i.rst_n <= 1'b0;
                                    done <= 1'b1;
                                end else cnt_9b_i.inc <= 1'b1;
                            end else cnt_7b_i.inc <= 1'b1;
                        end

                        if (done & int_res_write_cim_i.en) begin
                            done <= 1'b0;
                            delay_line_3b[0] <= 'b1;
                            current_inf_step <= InferenceStep_t'(int'(current_inf_step) + 1);
                        end
                    end
                    SOFTMAX_DIVIDE_STEP: begin : softmax_divide
                        /* cnt_4b_i internal step
                        cnt_7b_i holds sleep stage
                        */

                        if (cnt_4b_i.cnt == 0) begin
                            start_mult(int_res_read_i.data, NUM_SLEEP_STAGES_INVERSE_COMP_FX); // Divide by NUM_SLEEP_STAGES
                        end else if (cnt_4b_i.cnt == 2) begin
                            write_int_res(mem_map[MLP_HEAD_DENSE_2_OUT_MEM] + IntResAddr_t'(cnt_7b_i.cnt), mult_io.out, int_res_width[SOFTMAX_AVG_SUM_INV_WIDTH], int_res_format[SOFTMAX_AVG_SUM_INV_FORMAT]);
                        end else if (cnt_4b_i.cnt == 3) begin
                            write_int_res(mem_map[SOFTMAX_AVG_SUM_MEM] + IntResAddr_t'(cnt_7b_i.cnt), mult_io.out, int_res_width[SOFTMAX_AVG_SUM_INV_WIDTH], int_res_format[SOFTMAX_AVG_SUM_INV_FORMAT]);
                        end

                        // Execute
                        if (cnt_4b_i.cnt == 4) begin
                            cnt_4b_i.rst_n <= 1'b0;
                            read_int_res(mem_map[MLP_HEAD_DENSE_2_OUT_MEM] + IntResAddr_t'(cnt_7b_i.cnt), int_res_width[MLP_SOFTMAX_OUTPUT_WIDTH], int_res_format[MLP_SOFTMAX_OUTPUT_FORMAT]);
                            if (int'(cnt_7b_i.cnt) == NUM_SLEEP_STAGES) begin
                                cnt_7b_i.rst_n <= 1'b0;
                                start_add(CompFx_t'(0), CompFx_t'(0));
                                current_inf_step <= SOFTMAX_AVERAGING_STEP;
                            end
                        end else cnt_4b_i.inc <= 1'b1;

                        if (cnt_4b_i.cnt == 2) cnt_7b_i.inc <= 1'b1;
                    end
                    SOFTMAX_AVERAGING_STEP: begin : softmax_averaging
                        /* gen_cnt_7b holds the current sleep stage within an epoch's softmax
                        gen_cnt_9b holds the epoch
                        gen_cnt_4b holds internal step
                        */

                        IntResAddr_t addr_prev_softmax = mem_map[PREV_SOFTMAX_OUTPUT_MEM] + IntResAddr_t'(int'(cnt_7b_i.cnt) + int'(NUM_SLEEP_STAGES*cnt_9b_i.cnt));
                        IntResAddr_t addr_softmax_divide_sum = mem_map[SOFTMAX_AVG_SUM_MEM] + IntResAddr_t'(cnt_7b_i.cnt);

                        if (cnt_4b_i.cnt == 0) read_int_res(addr_prev_softmax, int_res_width[SOFTMAX_AVG_SUM_INV_WIDTH], int_res_format[SOFTMAX_AVG_SUM_INV_FORMAT]);
                        else if (cnt_4b_i.cnt == 1) read_int_res(addr_softmax_divide_sum, int_res_width[SOFTMAX_AVG_SUM_INV_WIDTH], int_res_format[SOFTMAX_AVG_SUM_INV_FORMAT]);
                        else if (cnt_4b_i.cnt == 2) start_add(int_res_read_i.data, CompFx_t'(0)); // Use add as a storage location for previous softmax
                        else if (cnt_4b_i.cnt == 4) start_add(int_res_read_i.data, add_io.out);
                        else if (cnt_4b_i.cnt == 6) write_int_res(addr_softmax_divide_sum, add_io.out, int_res_width[SOFTMAX_AVG_SUM_INV_WIDTH], int_res_format[SOFTMAX_AVG_SUM_INV_FORMAT]);

                        if (cnt_4b_i.cnt == 6) begin
                            cnt_4b_i.rst_n <= 1'b0;
                            start_add(CompFx_t'(0), CompFx_t'(0));
                            if (int'(cnt_7b_i.cnt) == NUM_SLEEP_STAGES-1) begin
                                cnt_7b_i.rst_n <= 1'b0;
                                if (int'(cnt_9b_i.cnt) == NUM_SAMPLES_OUT_AVG-2) begin
                                    cnt_9b_i.rst_n <= 1'b0;
                                    current_inf_step <= SOFTMAX_AVERAGE_ARGMAX_STEP;
                                    delay_line_3b <= 3'b000;
                                    start_mult(CompFx_t'(0), CompFx_t'(0));
                                end else cnt_9b_i.inc <= 1'b1;
                            end else cnt_7b_i.inc <= 1'b1;
                        end else cnt_4b_i.inc <= 1'b1;
                    end
                    SOFTMAX_AVERAGE_ARGMAX_STEP: begin : softmax_avg_argmax
                        /* Compute argmax of the averaged softmax.
                        * ADD out holds the current max softmax
                        * MULT out holds the current argmax
                        */

                        delay_line_3b[0] <= 1'b1;
                        delay_line_3b[1] <= int_res_read_i.en;

                        if (delay_line_3b[0]) read_int_res(mem_map[SOFTMAX_AVG_SUM_MEM] + IntResAddr_t'(cnt_9b_i.cnt), int_res_width[SOFTMAX_AVG_SUM_INV_WIDTH], int_res_format[SOFTMAX_AVG_SUM_INV_FORMAT]);

                        if (delay_line_3b[1] && cnt_9b_i.cnt < 7) begin
                            if (int_res_read_i.data > add_io.in_1) begin
                                start_add(int_res_read_i.data, CompFx_t'(0)); // Update max softmax found
                                start_mult(CompFx_t'(cnt_9b_i.cnt) - CompFx_t'(2), CompFx_t'(1 << Q_COMP)); // Update max argmax found
                            end
                        end

                        if (cnt_9b_i.cnt == 8) begin
                            soc_ctrl_i.inferred_sleep_stage <= SleepStage_t'(mult_io.out);
                            current_inf_step <= SOFTMAX_RETIRE_STEP;
                            cnt_9b_i.rst_n <= 1'b0;
                            delay_line_3b <= 3'b000;
                        end else cnt_9b_i.inc <= 1'b1;
                    end
                    SOFTMAX_RETIRE_STEP: begin : softmax_retire
                        /* Move dummy #0 into dummy #1's position and current softmax into dummy #0
                        gen_cnt_7b holds the current sleep stage within an epoch's softmax
                        gen_cnt_4b holds internal step
                        */
                        if (cnt_9b_i.cnt == 0) read_int_res(mem_map[PREV_SOFTMAX_OUTPUT_MEM] + IntResAddr_t'(cnt_7b_i.cnt), int_res_width[SOFTMAX_AVG_SUM_INV_WIDTH], int_res_format[SOFTMAX_AVG_SUM_INV_FORMAT]);
                        else if (cnt_9b_i.cnt == 2) write_int_res(mem_map[PREV_SOFTMAX_OUTPUT_MEM] + IntResAddr_t'(cnt_7b_i.cnt) + IntResAddr_t'(NUM_SLEEP_STAGES), int_res_read_i.data, int_res_width[PREV_SOFTMAX_OUTPUT_WIDTH], int_res_format[PREV_SOFTMAX_OUTPUT_FORMAT]);
                        else if (cnt_9b_i.cnt == 3) read_int_res(mem_map[MLP_HEAD_DENSE_2_OUT_MEM] + IntResAddr_t'(cnt_7b_i.cnt), int_res_width[SOFTMAX_AVG_SUM_INV_WIDTH], int_res_format[SOFTMAX_AVG_SUM_INV_FORMAT]);
                        else if (cnt_9b_i.cnt == 5) write_int_res(mem_map[PREV_SOFTMAX_OUTPUT_MEM] + IntResAddr_t'(cnt_7b_i.cnt), int_res_read_i.data, int_res_width[PREV_SOFTMAX_OUTPUT_WIDTH], int_res_format[PREV_SOFTMAX_OUTPUT_FORMAT]);
                    
                        if (cnt_9b_i.cnt == 5) begin
                            cnt_9b_i.rst_n <= 1'b0;
                            if (int'(cnt_7b_i.cnt) == NUM_SLEEP_STAGES-1) begin
                                cnt_7b_i.rst_n <= 1'b0;
                                current_inf_step <= INFERENCE_COMPLETE;
                            end else cnt_7b_i.inc <= 1'b1;
                        end else cnt_9b_i.inc <= 1'b1;
                    end
                    INFERENCE_COMPLETE: begin
                        current_inf_step <= PATCH_PROJ_STEP;
                        delay_line_3b <= 'b0;
                    end
                    default: begin
                    end
                endcase
            end
        end
    end

    // ----- MUX -----//
    always_latch begin : param_mem_MUX
        // Write: Only testbench writes to params to nothing to do here

        // Read
        param_read_i.data_width = SINGLE_WIDTH;
        param_read_i.en = param_read_mac_i.en | param_read_cim_i.en | param_read_ln_i.en;
        param_read_mac_i.data = param_read_i.data;
        param_read_ln_i.data = param_read_i.data;
        if (param_read_mac_i.en) begin // MAC
            param_read_i.addr = param_read_mac_i.addr;
            param_read_i.format = param_read_mac_i.format;
        end else if (param_read_cim_i.en) begin
            param_read_i.addr = param_read_cim_i.addr;
            param_read_i.format = param_read_cim_i.format;
        end else if (param_read_ln_i.en) begin
            param_read_i.addr = param_read_ln_i.addr;
            param_read_i.format = param_read_ln_i.format;
        end
    end

    always_latch begin : int_res_mem_MUX
        // Write
        int_res_write_i.chip_en = 1'b1;
        int_res_write_i.en = int_res_write_tb_i.en | int_res_write_cim_i.en | int_res_write_ln_i.en | int_res_write_softmax_i.en;
        if (int_res_write_tb_i.en) begin
            int_res_write_i.addr = int_res_write_tb_i.addr;
            int_res_write_i.data = int_res_write_tb_i.data;
            int_res_write_i.format = int_res_write_tb_i.format;
            int_res_write_i.data_width = int_res_write_tb_i.data_width;
        end else if (int_res_write_cim_i.en) begin
            int_res_write_i.addr = int_res_write_cim_i.addr;
            int_res_write_i.data = int_res_write_cim_i.data;
            int_res_write_i.format = int_res_write_cim_i.format;
            int_res_write_i.data_width = int_res_write_cim_i.data_width;
        end else if (int_res_write_ln_i.en) begin
            int_res_write_i.addr = int_res_write_ln_i.addr;
            int_res_write_i.data = int_res_write_ln_i.data;
            int_res_write_i.format = int_res_write_ln_i.format;
            int_res_write_i.data_width = int_res_write_ln_i.data_width;
        end else if (int_res_write_softmax_i.en) begin
            int_res_write_i.addr = int_res_write_softmax_i.addr;
            int_res_write_i.data = int_res_write_softmax_i.data;
            int_res_write_i.format = int_res_write_softmax_i.format;
            int_res_write_i.data_width = int_res_write_softmax_i.data_width;
        end

        // Read
        int_res_read_i.en = int_res_read_mac_i.en | int_res_read_cim_i.en | int_res_read_ln_i.en | int_res_read_softmax_i.en;
        int_res_read_mac_i.data = int_res_read_i.data;
        int_res_read_cim_i.data = int_res_read_i.data;
        int_res_read_ln_i.data = int_res_read_i.data;
        int_res_read_softmax_i.data = int_res_read_i.data;
        if (int_res_read_mac_i.en) begin // MAC
            int_res_read_i.addr = int_res_read_mac_i.addr;
            int_res_read_i.data_width = int_res_read_mac_i.data_width;
            int_res_read_i.format = int_res_read_mac_i.format;
        end else if (int_res_read_cim_i.en) begin
            int_res_read_i.addr = int_res_read_cim_i.addr;
            int_res_read_i.data_width = int_res_read_cim_i.data_width;
            int_res_read_i.format = int_res_read_cim_i.format;
        end else if (int_res_read_ln_i.en) begin
            int_res_read_i.addr = int_res_read_ln_i.addr;
            int_res_read_i.data_width = int_res_read_ln_i.data_width;
            int_res_read_i.format = int_res_read_ln_i.format;
        end else if (int_res_read_softmax_i.en) begin
            int_res_read_i.addr = int_res_read_softmax_i.addr;
            int_res_read_i.data_width = int_res_read_softmax_i.data_width;
            int_res_read_i.format = int_res_read_softmax_i.format;
        end
    end

    always_latch begin : add_io_MUX
        if (add_io_exp.start) begin
            add_io.in_1 = add_io_exp.in_1;
            add_io.in_2 = add_io_exp.in_2;
        end else if (add_io_mac.start) begin
            add_io.in_1 = add_io_mac.in_1;
            add_io.in_2 = add_io_mac.in_2;
        end else if (add_io_cim.start) begin
            add_io.in_1 = add_io_cim.in_1;
            add_io.in_2 = add_io_cim.in_2;
        end else if (add_io_ln.start) begin
            add_io.in_1 = add_io_ln.in_1;
            add_io.in_2 = add_io_ln.in_2;
        end else if (add_io_softmax.start) begin
            add_io.in_1 = add_io_softmax.in_1;
            add_io.in_2 = add_io_softmax.in_2;
        end

        add_io.start = add_io_exp.start | add_io_mac.start | add_io_cim.start | add_io_ln.start | add_io_softmax.start;
        add_io_exp.out = add_io.out;
        add_io_exp.done = add_io.done;
        add_io_mac.out = add_io.out;
        add_io_mac.done = add_io.done;
        add_io_cim.out = add_io.out;
        add_io_cim.done = add_io.done;
        add_io_ln.out = add_io.out;
        add_io_ln.done = add_io.done;
        add_io_softmax.out = add_io.out;
        add_io_softmax.done = add_io.done;
    end

    always_latch begin : mult_io_MUX
        if (mult_io_exp.start) begin
            mult_io.in_1 = mult_io_exp.in_1;
            mult_io.in_2 = mult_io_exp.in_2;
        end else if (mult_io_mac.start) begin
            mult_io.in_1 = mult_io_mac.in_1;
            mult_io.in_2 = mult_io_mac.in_2;
        end else if (mult_io_ln.start) begin
            mult_io.in_1 = mult_io_ln.in_1;
            mult_io.in_2 = mult_io_ln.in_2;
        end else if (mult_io_cim.start) begin
            mult_io.in_1 = mult_io_cim.in_1;
            mult_io.in_2 = mult_io_cim.in_2;
        end else if (mult_io_softmax.start) begin
            mult_io.in_1 = mult_io_softmax.in_1;
            mult_io.in_2 = mult_io_softmax.in_2;
        end

        mult_io.start = mult_io_exp.start | mult_io_mac.start | mult_io_ln.start | mult_io_cim.start | mult_io_softmax.start;
        mult_io_exp.out = mult_io.out;
        mult_io_exp.done = mult_io.done;
        mult_io_mac.out = mult_io.out;
        mult_io_mac.done = mult_io.done;
        mult_io_ln.out = mult_io.out;
        mult_io_ln.done = mult_io.done;
        mult_io_softmax.out = mult_io.out;
        mult_io_softmax.done = mult_io.done;
    end

    always_latch begin : div_io_MUX
        if (div_io_mac.start) begin
            div_io.in_1 = div_io_mac.in_1;
            div_io.in_2 = div_io_mac.in_2;
        end else if (div_io_ln.start) begin
            div_io.in_1 = div_io_ln.in_1;
            div_io.in_2 = div_io_ln.in_2;
        end else if (div_io_softmax.start) begin
            div_io.in_1 = div_io_softmax.in_1;
            div_io.in_2 = div_io_softmax.in_2;
        end

        div_io.start = div_io_mac.start | div_io_ln.start | div_io_softmax.start;
        div_io_mac.out = div_io.out;
        div_io_mac.busy = div_io.busy;
        div_io_mac.done = div_io.done;
        div_io_ln.out = div_io.out;
        div_io_ln.busy = div_io.busy;
        div_io_ln.done = div_io.done;
        div_io_softmax.out = div_io.out;
        div_io_softmax.busy = div_io.busy;
        div_io_softmax.done = div_io.done;
    end

    always_latch begin : exp_io_MUX
        if (exp_io_mac.start) exp_io.in_1 = exp_io_mac.in_1;
        else if (exp_io_softmax.start) exp_io.in_1 = exp_io_softmax.in_1;

        exp_io.in_2 = CompFx_t'(0);
        exp_io.start = exp_io_mac.start | exp_io_softmax.start;
        exp_io_mac.out = exp_io.out;
        exp_io_mac.busy = exp_io.busy;
        exp_io_mac.done = exp_io.done;
        exp_io_softmax.out = exp_io.out;
        exp_io_softmax.busy = exp_io.busy;
        exp_io_softmax.done = exp_io.done;
    end

    // ----- TASKS ----- //
    task automatic set_default_values();
        // Sets the default value for signals that do not persist cycle-to-cycle
        cnt_4b_i.inc <= 1'b0;
        cnt_7b_i.inc <= 1'b0;
        cnt_9b_i.inc <= 1'b0;
        cnt_4b_i.rst_n <= 1'b1;
        cnt_7b_i.rst_n <= 1'b1;
        cnt_9b_i.rst_n <= 1'b1;

        param_read_cim_i.en <= 1'b0;
        int_res_read_cim_i.en <= 1'b0;
        int_res_write_cim_i.en <= 1'b0;

        add_io_cim.start <= 1'b0;
        mult_io_cim.start <= 1'b0;
        mac_io.start <= 1'b0;
        ln_io.start <= 1'b0;
        softmax_io.start <= 1'b0;
    endtask

    task automatic reset();
        current_inf_step <= PATCH_PROJ_STEP;

        cnt_4b_i.inc <= 1'b0;
        cnt_7b_i.inc <= 1'b0;
        cnt_9b_i.inc <= 1'b0;
        cnt_4b_i.rst_n <= 1'b0;
        cnt_7b_i.rst_n <= 1'b0;
        cnt_9b_i.rst_n <= 1'b0;

        mac_io.start <= 1'b0;
        add_io_cim.start <= 1'b0;
    endtask

    task automatic start_inference();
        cim_state <= INFERENCE_RUNNING;
    endtask

    task automatic write_int_res(input IntResAddr_t addr, input CompFx_t data, input DataWidth_t width, input FxFormatIntRes_t int_res_format);
        int_res_write_cim_i.en <= 1'b1;
        int_res_write_cim_i.addr <= addr;
        int_res_write_cim_i.data <= data;
        int_res_write_cim_i.data_width <= width;
        int_res_write_cim_i.format <= int_res_format;
    endtask

    task automatic read_params(input ParamAddr_t addr, input FxFormatParams_t format);
        param_read_cim_i.en <= 1'b1;
        param_read_cim_i.addr <= addr;
        param_read_cim_i.format <= format;
    endtask

    task automatic read_int_res(input IntResAddr_t addr, input DataWidth_t width, input FxFormatIntRes_t int_res_format);
        int_res_read_cim_i.en <= 1'b1;
        int_res_read_cim_i.addr <= addr;
        int_res_read_cim_i.data_width <= width;
        int_res_read_cim_i.format <= int_res_format;
    endtask

    task automatic start_mac(input IntResAddr_t addr_1, input IntResAddr_t addr_2, input ParamAddr_t bias_addr, input ParamType_t param_type, input Activation_t act, input VectorLen_t len, input VectorLen_t matrix_width, input Direction_t dir, input FxFormatIntRes_t int_res_input_format, input DataWidth_t int_res_read_width, input FxFormatParams_t params_read_format);
        mac_io.start <= 1'b1;
        mac_io_extra.start_addr_1 <= addr_1;
        mac_io_extra.start_addr_2 <= addr_2;
        mac_io_extra.param_type <= param_type;
        mac_io_extra.activation <= act;
        mac_io_extra.len <= len;
        mac_io_extra.matrix_width <= matrix_width;
        mac_io_extra.bias_addr <= bias_addr;
        mac_io_extra.direction <= dir;
        casts_mac_i.int_res_read_format <= int_res_input_format;
        casts_mac_i.int_res_read_width <= int_res_read_width;
        casts_mac_i.params_read_format <= params_read_format;
    endtask

    task automatic start_add(CompFx_t in_1, CompFx_t in_2);
        add_io_cim.start <= 1'b1;
        adder_in_1_reg <= in_1;
        add_io_cim.in_2 <= in_2;
    endtask

    task automatic start_mult(CompFx_t in_1, CompFx_t in_2);
        mult_io_cim.start <= 1'b1;
        mult_io_cim.in_1 <= in_1;
        mult_io_cim.in_2 <= in_2;
    endtask

    task automatic start_layernorm(input HalfSelect_t half_select, input IntResAddr_t input_starting_addr, input IntResAddr_t output_starting_addr, input ParamAddr_t beta_addr, input ParamAddr_t gamma_addr, input DataWidth_t input_width, input FxFormatIntRes_t input_format, input DataWidth_t output_width, input FxFormatIntRes_t output_format, input FxFormatParams_t param_format);
        ln_io.start <= 1'b1;
        ln_io_extra.half_select <= half_select;
        ln_io_extra.start_addr_1 <= input_starting_addr;
        ln_io_extra.start_addr_2 <= output_starting_addr;
        ln_io_extra.start_addr_3 <= IntResAddr_t'(beta_addr);
        ln_io_extra.start_addr_4 <= IntResAddr_t'(gamma_addr);
        casts_ln_i.int_res_read_format <= input_format;
        casts_ln_i.int_res_write_format <= output_format;
        casts_ln_i.int_res_read_width <= input_width;
        casts_ln_i.int_res_write_width <= output_width;
        casts_ln_i.params_read_format <= param_format;
    endtask

    task automatic start_softmax(input IntResAddr_t input_starting_addr, input VectorLen_t len, input FxFormatIntRes_t input_format, input DataWidth_t input_width, input FxFormatIntRes_t output_format, input DataWidth_t output_width);
        softmax_io.start <= 1'b1;
        softmax_io_extra.start_addr_1 <= input_starting_addr;
        softmax_io_extra.len <= len;
        casts_softmax_i.int_res_read_format <= input_format;
        casts_softmax_i.int_res_write_format <= output_format;
        casts_softmax_i.int_res_read_width <= input_width;
        casts_softmax_i.int_res_write_width <= output_width;
    endtask

    // ----- ASSERTIONS ----- //
`ifdef ENABLE_ASSERTIONS
    always_ff @ (posedge clk) begin : compute_mux_assertions
        assert (~(int_res_write_tb_i.en & int_res_write_cim_i.en & int_res_write_ln_i.en & int_res_write_softmax_i.en)) else $fatal("More than one source are trying to write to intermediate result memory simulatenously!");
        assert (~(int_res_read_cim_i.en & int_res_read_mac_i.en & int_res_read_ln_i.en & int_res_read_softmax_i.en)) else $fatal("More than one source are trying to read from intermediate results memory simulatenously!");
        assert (~(param_read_cim_i.en & param_read_mac_i.en & param_read_ln_i.en)) else $fatal("More than one source are trying to read from parameters result memory simulatenously!");

        assert (~(add_io_exp.start & add_io_mac.start & add_io_cim.start & add_io_ln.start)) else $fatal("More than one source are trying to start an add!");
        assert (~(mult_io_exp.start & mult_io_mac.start & mult_io_ln.start & mult_io_cim.start)) else $fatal("More than one source are trying to start a mult!");
        assert (~(div_io_mac.start & div_io_ln.start)) else $fatal("More than one source are trying to start a div!");
        assert (~(exp_io_mac.start & 0)) else $fatal("More than one source are trying to start an exp!");
    end
`endif
endmodule

`endif // _cim_centralized_vh_
