`ifndef _master_sv_
`define _master_sv_

// Includes
`include "types.svh"
`include "master/master.svh"
`include "ip/counter/counter.sv"

module master #(
    parameter STANDALONE_TB = 0
)(
    input wire clk,
    input wire rst_n,

    // Control signals
    input wire new_sleep_epoch, start_param_load, all_cims_ready,

    // Bus
    inout wire [BUS_OP_WIDTH-1:0] bus_op,
    inout wire signed [2:0][N_STORAGE-1:0] bus_data,
    inout wire [$clog2(NUM_CIMS)-1:0] bus_target_or_sender,

    // EEG
    input wire new_eeg_sample,
    input wire [EEG_SAMPLE_DEPTH-1:0] eeg_sample,

    // Interface to external memory storing the weights
    input wire ext_mem_data_valid,
    input logic signed [N_STORAGE-1:0] ext_mem_data,
    output logic ext_mem_data_read_pulse,
    output logic [$clog2(NUM_PARAMS)-1:0] ext_mem_addr
);

    // Initialize arrays
    `include "top_init.sv"
    `include "master/master_init.sv"
    `include "master/master_fcn.svh"

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

    // Counters
    logic gen_cnt_2b_rst_n, gen_cnt_7b_rst_n, gen_cnt_7b_2_rst_n;
    logic [1:0] gen_cnt_2b_inc;
    logic [6:0] gen_cnt_7b_inc, gen_cnt_7b_2_inc;
    wire [1:0] gen_cnt_2b_cnt;
    wire [6:0] gen_cnt_7b_cnt, gen_cnt_7b_2_cnt;
    counter #(.WIDTH(2), .MODE(0)) gen_cnt_2b   (.clk(clk), .rst_n(gen_cnt_2b_rst_n), .inc(gen_cnt_2b_inc), .cnt(gen_cnt_2b_cnt));
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b   (.clk(clk), .rst_n(gen_cnt_7b_rst_n), .inc(gen_cnt_7b_inc), .cnt(gen_cnt_7b_cnt));
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b_2 (.clk(clk), .rst_n(gen_cnt_7b_2_rst_n), .inc(gen_cnt_7b_2_inc), .cnt(gen_cnt_7b_2_cnt));

    // Internal registers and wires
    logic new_cim, loading_params, gen_bit;
    logic [3:0] params_curr_layer;
    logic [$clog2(NUM_PARAMS)-1:0] ext_mem_addr_prev;

    // Main FSM
    MASTER_STATE_T state;
    HIGH_LEVEL_INFERENCE_STEP_T high_level_inf_step;
    always_ff @ (posedge clk) begin : master_main_fsm
        if (!rst_n) begin // Reset
            gen_bit <= 1'b0;
            gen_cnt_2b_rst_n <= RST;
            gen_cnt_7b_rst_n <= RST;
            gen_cnt_7b_2_rst_n <= RST;
            state <= MASTER_STATE_IDLE;
            high_level_inf_step <= PRE_LAYERNORM_1_TRANS_STEP;
        end else begin
            case (state)
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
                        bus_data_write[0] <= 'd0;
                        bus_data_write[1] <= 'd0;
                        bus_data_write[2] <= 'd0;
                        bus_target_or_sender_write <= 'd0;
                        gen_cnt_2b_rst_n <= RST;
                        gen_cnt_7b_rst_n <= RST;
                        gen_cnt_7b_2_rst_n <= RST;
                        bus_op_write <= PATCH_LOAD_BROADCAST_START_OP;
                    end
                    gen_bit <= 1'b0;
                    gen_cnt_7b_rst_n <= RUN;
                    gen_cnt_7b_2_rst_n <= RUN;
                    high_level_inf_step <= PRE_LAYERNORM_1_TRANS_STEP;
                end

                MASTER_STATE_PARAM_LOAD: begin
                    /* Note that there is a slight mismatch between model parameter loading behaviour in the RTL and the C++ functional simulation.
                       In the functional simulation, the master updates all three data word in the instruction on the same cycle and holds the instruction on the bus, while the CiM sequentially reads the words and writes to memory.
                       In the RTL, because we need to wait for external memory to send data to the master, we only update one word at a time. When waiting for data from external memory, the instruction is NOP. Instruction changes
                       to PARAM_STREAM_OP only when the external memory data is valid.
                    */
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

                        //synopsys translate_off
                        default:
                            $fatal("Invalid params_curr_layer");
                        //synopsys translate_on
                    endcase
                    gen_cnt_2b_inc <= {1'd0, ext_mem_data_valid};
                    gen_cnt_2b_rst_n <= (ext_mem_data_valid && (gen_cnt_2b_cnt == 'd2)) ? RST : RUN;

                    // Bus instruction
                    new_cim <= (gen_cnt_7b_2_cnt == param_addr_map[params_curr_layer].num_rec);
                    bus_target_or_sender_write <= gen_cnt_7b_2_cnt[5:0];
                    if (new_cim) begin
                        bus_op_write <= PARAM_STREAM_START_OP;
                        bus_data_write[0] <= {6'd0, param_addr_map[gen_cnt_7b_cnt[3:0]].addr};
                        bus_data_write[1] <= {9'd0, param_addr_map[gen_cnt_7b_cnt[3:0]].len};
                    end else begin
                        bus_op_write <= ((gen_cnt_2b_rst_n == RST) || ext_mem_data_valid) ? PARAM_STREAM_OP : NOP;
                        bus_data_write[0] <= (ext_mem_data_valid && (gen_cnt_2b_cnt == 'd0)) ? ext_mem_data : bus_data_write[0];
                        bus_data_write[1] <= (ext_mem_data_valid && (gen_cnt_2b_cnt == 'd1)) ? ext_mem_data : bus_data_write[1];
                        bus_data_write[2] <= (ext_mem_data_valid && (gen_cnt_2b_cnt == 'd2)) ? ext_mem_data : bus_data_write[2];
                    end

                    // Don't update external memory address if we are sending PARAM_STREAM_START_OP because it will be garbage
                    if (bus_op_write != PARAM_STREAM_START_OP) begin
                        ext_mem_addr <= param_ext_mem_addr(gen_cnt_7b_cnt, gen_cnt_7b_2_cnt, gen_cnt_2b_cnt, params_curr_layer);
                    end else begin
                        ext_mem_addr <= ext_mem_addr;
                    end
                end

                MASTER_STATE_SIGNAL_LOAD: begin
                    bus_op_write <= (new_eeg_sample) ? PATCH_LOAD_BROADCAST_OP : NOP;
                    bus_data_write[0] <= { {(N_STORAGE-Q){1'd0}}, eeg_sample[EEG_SAMPLE_DEPTH-1 -: Q] }; // Select the upper Q bits as a way to normalize (divide by 15b) and convert to fixed-point
                    state <= (new_sleep_epoch) ? MASTER_STATE_SIGNAL_LOAD : MASTER_STATE_INFERENCE_RUNNING;
                    bus_drive <= new_sleep_epoch;
                end

                MASTER_STATE_INFERENCE_RUNNING: begin
                    case (high_level_inf_step)
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
                            if (all_cims_ready && (bus_op_read == INFERENCE_RESULT_OP)) begin
                                state <= MASTER_STATE_DONE_INFERENCE;
                            end else if (high_level_inf_step != SOFTMAX_AVERAGING) begin
                                state <= MASTER_STATE_BROADCAST_MANAGEMENT;
                                gen_cnt_7b_rst_n <= RUN;
                                gen_cnt_7b_2_rst_n <= RUN;
                                bus_drive <= 1'b1;
                                gen_cnt_7b_inc <= 'd1;
                                bus_op_write <= broadcast_ops[high_level_inf_step].op;
                                bus_data_write[1] <= {{(N_STORAGE-$clog2(NUM_CIMS+1)){1'd0}}, broadcast_ops[high_level_inf_step].len};
                                case (high_level_inf_step)
                                    ENC_MHSA_QK_T_STEP: begin
                                        bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr + $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(gen_cnt_7b_2_cnt*NUM_HEADS)};
                                        bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr};
                                        bus_target_or_sender_write <= 'd0;
                                    end
                                    ENC_MHSA_V_MULT_STEP: begin
                                        bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr + $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(gen_cnt_7b_2_cnt*(NUM_PATCHES+1))};
                                        bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr};
                                        bus_target_or_sender_write <= 'd0;
                                    end
                                    ENC_MHSA_PRE_SOFTMAX_TRANS_STEP: begin
                                        bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr + $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(gen_cnt_7b_2_cnt*(NUM_PATCHES+1))};
                                        bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr + $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(gen_cnt_7b_2_cnt*(NUM_PATCHES+1))};
                                        bus_target_or_sender_write <= 'd0;
                                    end
                                    PRE_MLP_HEAD_DENSE_2_TRANS_STEP: begin
                                        bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr};
                                        bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr};
                                        bus_target_or_sender_write <= MLP_DIM;
                                    end
                                    default: begin
                                        bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr};
                                        bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr};
                                        bus_target_or_sender_write <= 'd0;
                                    end
                                endcase
                            end
                        end

                        ENC_MHSA_SOFTMAX_STEP: begin
                            high_level_inf_step <= (all_cims_ready) ? HIGH_LEVEL_INFERENCE_STEP_T'(high_level_inf_step + 5'd1) : high_level_inf_step;
                        end

                        //synopsys translate_off
                        default:
                            $fatal("Invalid master high_level_inf_step!");
                        //synopsys translate_on
                    endcase
                end

                MASTER_STATE_BROADCAST_MANAGEMENT: begin
                    // gen_cnt_7b counts # of cims sent to the bus
                    // gen_cnt_7b_2 counts position in Z-stack
                    automatic logic all_transactions_sent = all_cims_ready && (bus_op_read == NOP);
                    automatic logic all_cims_sent = (gen_cnt_7b_cnt == broadcast_ops[high_level_inf_step].num_cim);
                    automatic logic is_multi_head_op = (high_level_inf_step == ENC_MHSA_QK_T_STEP) || (high_level_inf_step == ENC_MHSA_V_MULT_STEP) || (high_level_inf_step == ENC_MHSA_PRE_SOFTMAX_TRANS_STEP);

                    if (all_transactions_sent) begin
                        bus_op_write <= broadcast_ops[high_level_inf_step].op;
                        bus_data_write[1] <= {{(N_STORAGE-$clog2(NUM_CIMS+1)){1'd0}}, broadcast_ops[high_level_inf_step].len};
                        case (high_level_inf_step)
                            ENC_MHSA_QK_T_STEP: begin
                                bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr + $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(gen_cnt_7b_2_cnt*NUM_HEADS)};
                                bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr};
                                bus_target_or_sender_write <= gen_cnt_7b_cnt[5:0];
                            end
                            ENC_MHSA_V_MULT_STEP: begin
                                bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr + $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(gen_cnt_7b_2_cnt*(NUM_PATCHES+1))};
                                bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr};
                                bus_target_or_sender_write <= gen_cnt_7b_cnt[5:0];
                            end
                            ENC_MHSA_PRE_SOFTMAX_TRANS_STEP: begin
                                bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr + $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(gen_cnt_7b_2_cnt*(NUM_PATCHES+1))};
                                bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr + $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(gen_cnt_7b_2_cnt*(NUM_PATCHES+1))};
                                bus_target_or_sender_write <= gen_cnt_7b_cnt[5:0];
                            end
                            PRE_MLP_HEAD_DENSE_2_TRANS_STEP: begin
                                bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr};
                                bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr};
                                bus_target_or_sender_write <= MLP_DIM + gen_cnt_7b_cnt[5:0];
                            end
                            default: begin
                                bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr};
                                bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr};
                                bus_target_or_sender_write <= gen_cnt_7b_cnt[5:0];
                            end
                        endcase
                        if (all_cims_sent) begin
                            gen_cnt_7b_2_inc <= 'd1;
                            bus_op_write <= PISTOL_START_OP;
                            gen_cnt_7b_rst_n <= RST;
                            state <= MASTER_STATE_INFERENCE_RUNNING;
                            if (~is_multi_head_op || (gen_cnt_7b_2_cnt == NUM_HEADS)) begin // Go to next inference step
                                gen_cnt_7b_2_rst_n <= RST;
                                high_level_inf_step <= HIGH_LEVEL_INFERENCE_STEP_T'(high_level_inf_step + 5'd1);
                            end
                        end else begin
                            gen_cnt_7b_2_inc <= 'd0;
                        end
                        bus_drive <= 'b1;
                    end else begin
                        bus_op_write <= NOP;
                        bus_drive <= 'b0;
                    end
                    gen_cnt_7b_inc <= {6'd0, all_transactions_sent};
                    gen_cnt_7b_rst_n <= (all_transactions_sent && all_cims_sent) ? RST : RUN;
                end

                MASTER_STATE_DONE_INFERENCE: begin
                    bus_drive <= 1'b0;
                    state <= MASTER_STATE_IDLE;
                end

                //synopsys translate_off
                default:
                    $fatal("Invalid master state!");
                //synopsys translate_on
            endcase
        end
    end

    // External memory
    always_ff @ (posedge clk) begin : ext_mem_read_pulse_gen
        ext_mem_data_read_pulse <= (ext_mem_addr != ext_mem_addr_prev);
        ext_mem_addr_prev <= ext_mem_addr;
    end

    // Basic asserts
    //synopsys translate_off
    always_comb begin : basic_asserts
        assert (~(new_sleep_epoch && start_param_load)) else $fatal("new_sleep_epoch and start_param_load cannot be high at the same time!");
    end
    //synopsys translate_on

endmodule

`endif
