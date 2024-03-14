// Includes
`include "master.svh"
`include "master_fcn.svh"
`include "../ip/counter/counter.sv"
`include "../types.svh"

module master (
    input wire clk,
    input wire rst_n,

    // Control signals
    input wire new_sleep_epoch, start_param_load, all_cims_ready,

    // Bus
    inout bus_t bus,

    // EEG
    input wire new_eeg_sample,
    input wire [EEG_SAMPLE_DEPTH-1:0] eeg_sample,

    // Signals to fake external memory containing the weights
    input wire ext_mem_data_valid,
    input logic signed [N_STORAGE-1:0] ext_mem_data,
    output logic ext_mem_data_read_pulse,
    output logic [$clog2(NUM_PARAMS)-1:0] ext_mem_addr
);
    // Initialize arrays
    `include "master_init.sv"

    // Read bus
    wire [BUS_OP_WIDTH-1:0] bus_op_read;
    wire signed [N_STORAGE-1:0] bus_data_0_read, bus_data_1_read, bus_data_2_read;
    wire [$clog2(NUM_CIMS)-1:0] bus_target_or_sender_read;
    assign {bus_op_read, bus_data_0_read, bus_data_1_read, bus_data_2_read, bus_target_or_sender_read} = bus;
    
    // Write bus
    logic bus_drive;
    logic [BUS_OP_WIDTH-1:0] bus_op_write;
    logic signed [2:0][N_STORAGE-1:0] bus_data_write;
    logic [$clog2(NUM_CIMS)-1:0] bus_target_or_sender_write;
    always_comb begin : bus_drive_comb
        bus.op = (bus_drive) ? bus_op_write : 'Z;
        bus.data[0] = (bus_drive) ? bus_data_write[0] : 'Z;
        bus.data[1] = (bus_drive) ? bus_data_write[1] : 'Z;
        bus.data[2] = (bus_drive) ? bus_data_write[2] : 'Z;
        bus.target_or_sender = (bus_drive) ? bus_target_or_sender_write : 'Z;
    end

    // Instantiate counters
    logic gen_cnt_2b_rst_n, gen_cnt_7b_rst_n, gen_cnt_7b_2_rst_n;
    logic [1:0] gen_cnt_2b_cnt, gen_cnt_2b_inc;
    logic [6:0] gen_cnt_7b_cnt, gen_cnt_7b_2_cnt, gen_cnt_7b_inc, gen_cnt_7b_2_inc;
    counter #(.WIDTH(2), .MODE(0)) gen_cnt_2b   (.clk(clk), .rst_n(gen_cnt_2b_rst_n), .inc(gen_cnt_2b_inc), .cnt(gen_cnt_2b_cnt));
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b   (.clk(clk), .rst_n(gen_cnt_7b_rst_n), .inc(gen_cnt_7b_inc), .cnt(gen_cnt_7b_cnt));
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b_2 (.clk(clk), .rst_n(gen_cnt_7b_2_rst_n), .inc(gen_cnt_7b_2_inc), .cnt(gen_cnt_7b_2_cnt));

    // Internal registers and wires
    logic new_cim, loading_params;
    logic [15:0] gen_reg_16b = 'd0;
    logic [15:0] gen_reg_16b_2 = 'd0;
    logic [15:0] gen_reg_16b_3 = 'd0;
    logic [3:0] params_curr_layer;
    logic [$clog2(NUM_PARAMS)-1:0] ext_mem_addr_prev;

    // Main FSM
    MASTER_STATE_T state = MASTER_STATE_IDLE;
    HIGH_LEVEL_INFERENCE_STEP_T high_level_inf_step = PRE_LAYERNORM_1_TRANS_STEP;
    always_ff @ (posedge clk or negedge rst_n) begin : master_main_fsm
        if (!rst_n) begin // Reset
            gen_reg_16b <= 'd0;
            gen_reg_16b_2 <= 'd0;
            gen_reg_16b_3 <= 'd0;
            gen_cnt_2b_rst_n <= RST;
            gen_cnt_7b_rst_n <= RST;
            gen_cnt_7b_2_rst_n <= RST;
            state <= MASTER_STATE_IDLE;
            high_level_inf_step <= PRE_LAYERNORM_1_TRANS_STEP;
        end else begin
            unique case (state)
                MASTER_STATE_IDLE: begin
                    if (start_param_load || loading_params) begin
                        state <= MASTER_STATE_PARAM_LOAD;
                        gen_cnt_2b_rst_n <= RST;
                        gen_cnt_7b_rst_n <= RST;
                        gen_cnt_7b_2_rst_n <= RST;
                        bus_drive <= 1'b1;
                        new_cim <= 1'b1;
                    end else if (new_sleep_epoch) begin
                        state <= MASTER_STATE_SIGNAL_LOAD;
                        bus_drive <= 1'b1;
                        // Clean up bus and counters
                        bus_data_write[0] <= 'd0;
                        bus_data_write[1] <= 'd0;
                        bus_data_write[2] <= 'd0;
                        bus_target_or_sender_write <= 'd0;
                        gen_cnt_2b_rst_n <= RST;
                        gen_cnt_7b_rst_n <= RST;
                        gen_cnt_7b_2_rst_n <= RST;
                        bus_op_write <= PATCH_LOAD_BROADCAST_START_OP;
                    end
                    gen_reg_16b <= 'd0;
                    gen_reg_16b_2 <= 'd0;
                    gen_reg_16b_3 <= 'd0;

                    // Take out of reset
                    gen_cnt_7b_rst_n <= RUN;
                    gen_cnt_7b_2_rst_n <= RUN;
                end

                MASTER_STATE_PARAM_LOAD: begin
                    case (params_curr_layer)
                        PATCH_PROJ_KERNEL_PARAMS,
                        POS_EMB_PARAMS,
                        ENC_Q_DENSE_KERNEL_PARAMS,
                        ENC_K_DENSE_KERNEL_PARAMS,
                        ENC_V_DENSE_KERNEL_PARAMS,
                        ENC_COMB_HEAD_KERNEL_PARAMS,
                        ENC_MLP_DENSE_2_KERNEL_PARAMS,
                        ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_KERNEL_PARAMS,
                        MLP_HEAD_DENSE_2_KERNEL_PARAMS: begin
                            /* Note:
                                -gen_cnt_2b_cnt is the element number of the current instruction
                                -gen_cnt_7b_cnt is the element number for the current CiM
                                -gen_cnt_7b_2_cnt is the current CiM number
                            */
                            loading_params <= 'd1;
                            gen_cnt_7b_inc <= {6'd0, ext_mem_data_valid};
                            gen_cnt_7b_rst_n <= (gen_cnt_7b_cnt == param_addr_map[params_curr_layer].len) ? RST : RUN;
                            gen_cnt_7b_2_inc <= {6'd0, (gen_cnt_7b_cnt == param_addr_map[params_curr_layer].len)};
                            gen_cnt_7b_2_rst_n <= (gen_cnt_7b_2_cnt == param_addr_map[params_curr_layer].num_rec) ? RST : RUN;
                            
                            params_curr_layer <= ((gen_cnt_7b_2_cnt == param_addr_map[params_curr_layer].num_rec) && (gen_cnt_7b_2_rst_n == RUN)) ? params_curr_layer + 'd1 : params_curr_layer;
                            state <= (gen_cnt_7b_2_cnt == param_addr_map[params_curr_layer].num_rec) ? MASTER_STATE_IDLE : MASTER_STATE_PARAM_LOAD;
                        end

                        SINGLE_PARAMS: begin
                            gen_cnt_7b_inc <= {6'd0, ext_mem_data_valid && (gen_cnt_2b_cnt == 'd2) && ext_mem_data_valid};
                            gen_cnt_7b_rst_n <= (gen_cnt_7b_cnt == 'd6) ? RST : RUN;
                            gen_cnt_7b_2_inc <= {6'd0, (gen_cnt_7b_cnt == 'd6)};
                            params_curr_layer <= ((gen_cnt_7b_2_cnt == param_addr_map[params_curr_layer].num_rec) && (gen_cnt_7b_2_rst_n == RUN)) ? params_curr_layer + 'd1 : params_curr_layer;
                        end

                        PARAM_LOAD_FINISHED: begin
                            loading_params <= 1'd0;
                            bus_drive <= 1'b0;
                            state <= MASTER_STATE_IDLE;
                        end

                        default:
                            $fatal("Invalid params_curr_layer");
                    endcase
                    gen_cnt_2b_inc <= {1'd0, ext_mem_data_valid};
                    gen_cnt_2b_rst_n <= (ext_mem_data_valid && (gen_cnt_2b_cnt == 'd2)) ? RST : RUN;

                    // Bus instruction
                    new_cim <= (gen_cnt_7b_2_cnt == param_addr_map[params_curr_layer].num_rec);
                    bus_target_or_sender_write <= gen_cnt_7b_2_cnt[5:0];
                    update_inst(bus_data_write, bus_op_write, ext_mem_data, gen_cnt_2b_cnt, gen_cnt_7b_cnt, ext_mem_data_valid, new_cim, gen_cnt_2b_rst_n);

                    // Don't update external memory if we are sending DATA_STREAM_START_OP because it will be garbage
                    ext_mem_addr <= (bus_op_write != DATA_STREAM_START_OP) ? param_ext_mem_addr(gen_cnt_7b_cnt, gen_cnt_7b_2_cnt, gen_cnt_2b_cnt, params_curr_layer) : ext_mem_addr;
                end

                MASTER_STATE_SIGNAL_LOAD: begin
                    bus_op_write <= (new_eeg_sample) ? PATCH_LOAD_BROADCAST_OP : NOP;
                    bus_data_write[0] <= { {(N_STORAGE-Q){1'd0}}, eeg_sample[EEG_SAMPLE_DEPTH-1 -: Q] }; // Select the upper Q bits as a way to normalize (divide by 15b) and convert to fixed-point
                    state <= (new_sleep_epoch) ? MASTER_STATE_SIGNAL_LOAD : MASTER_STATE_IDLE;
                end

                MASTER_STATE_INFERENCE_RUNNING: begin
                    unique case (high_level_inf_step)
                        PRE_LAYERNORM_1_TRANS_STEP,
                        INTRA_LAYERNORM_1_TRANS_STEP,
                        POST_LAYERNORM_1_TRANS_STEP,
                        ENC_MHSA_DENSE_STEP,
                        ENC_MHSA_Q_TRANS_STEP,
                        ENC_MHSA_K_TRANS_STEP,
                        ENC_MHSA_QK_T_STEP,
                        ENC_MHSA_PRE_SOFTMAX_TRANS_STEP,
                        ENC_MHSA_V_MULT_STEP,
                        ENC_MHSA_POST_V_TRANS_STEP,
                        ENC_MHSA_POST_V_DENSE_STEP,
                        PRE_LAYERNORM_2_TRANS_STEP,
                        INTRA_LAYERNORM_2_TRANS_STEP,
                        ENC_PRE_MLP_TRANSPOSE_STEP,
                        ENC_MLP_DENSE_1_STEP,
                        ENC_MLP_DENSE_2_TRANSPOSE_STEP,
                        ENC_MLP_DENSE_2_AND_SUM_STEP,
                        PRE_LAYERNORM_3_TRANS_STEP,
                        INTRA_LAYERNORM_3_TRANS_STEP,
                        PRE_MLP_HEAD_DENSE_TRANS_STEP,
                        MLP_HEAD_DENSE_1_STEP,
                        PRE_MLP_HEAD_DENSE_2_TRANS_STEP,
                        MLP_HEAD_DENSE_2_STEP,
                        MLP_HEAD_SOFTMAX_TRANS_STEP,
                        SOFTMAX_AVERAGING: begin
                        end

                        ENC_MHSA_SOFTMAX_STEP: begin
                        end

                        default:
                            $fatal("Invalid master high_level_inf_step!");
                    endcase
                end

                MASTER_STATE_BROADCAST_MANAGEMENT: begin
                end

                default:
                    $fatal("Invalid master state!");
            endcase
        end
    end

    // External memory
    always_ff @ (posedge clk) begin : ext_mem_read_pulse_gen
        ext_mem_data_read_pulse <= (ext_mem_addr != ext_mem_addr_prev);
        ext_mem_addr_prev <= ext_mem_addr;
    end

    // Basic asserts
    always_comb begin : basic_asserts
        assert (~(new_sleep_epoch && start_param_load)) else $fatal("new_sleep_epoch and start_param_load cannot be high at the same time!");
    end

endmodule
