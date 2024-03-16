`ifndef _master_fcn_vh_
`define _master_fcn_vh_

`include "../types.svh"

/*----- FUNCTIONS -----*/
function automatic logic [$clog2(NUM_PARAMS)-1:0] param_ext_mem_addr(input logic[6:0] param_num, input logic[6:0] cim_num, input logic[1:0] gen_cnt_2b_cnt, input logic [3:0] params_curr_layer);
    logic [$clog2(NUM_PARAMS)-1:0] addr;
    unique case (params_curr_layer)
        PATCH_PROJ_KERNEL_PARAMS: begin
            addr <= (({6'd0, ext_mem_param_addr_map[PATCH_PROJ_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
        end
        POS_EMB_PARAMS: begin
            addr <= (({6'd0, ext_mem_param_addr_map[POS_EMB_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
        end
        ENC_Q_DENSE_KERNEL_PARAMS: begin
            addr <= (({6'd0, ext_mem_param_addr_map[ENC_Q_DENSE_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
        end
        ENC_K_DENSE_KERNEL_PARAMS: begin
            addr <= (({6'd0, ext_mem_param_addr_map[ENC_K_DENSE_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
        end
        ENC_V_DENSE_KERNEL_PARAMS: begin
            addr <= (({6'd0, ext_mem_param_addr_map[ENC_V_DENSE_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
        end
        ENC_COMB_HEAD_KERNEL_PARAMS: begin
            addr <= (({6'd0, ext_mem_param_addr_map[ENC_COMB_HEAD_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
        end
        ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_KERNEL_PARAMS: begin
            if (cim_num < MLP_DIM) begin
                addr <= (({6'd0, ext_mem_param_addr_map[MLP_DENSE_1_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
            end else begin
                addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_DENSE_1_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
            end
        end
        ENC_MLP_DENSE_2_KERNEL_PARAMS: begin
            addr <= (({6'd0, ext_mem_param_addr_map[MLP_DENSE_2_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
        end
        MLP_HEAD_DENSE_2_KERNEL_PARAMS: begin
            addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_DENSE_SOFTMAX_KERNEL_EXT_MEM]} + {8'd0, cim_num}) << $clog2(64)) + {8'd0, param_num};
        end
        SINGLE_PARAMS: begin
            unique case (param_num)
                'd0: begin
                    if (gen_cnt_2b_cnt == 'd0)      addr <= (({6'd0, ext_mem_param_addr_map[PATCH_PROJ_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                    else if (gen_cnt_2b_cnt == 'd1) addr <= (({6'd0, ext_mem_param_addr_map[CLASS_EMB_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                    else if (gen_cnt_2b_cnt == 'd2) addr <= (({6'd0, ext_mem_param_addr_map[ENC_LAYERNORM_1_GAMMA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                end
                'd1: begin
                    if (gen_cnt_2b_cnt == 'd0)      addr <= (({6'd0, ext_mem_param_addr_map[ENC_LAYERNORM_1_BETA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                    else if (gen_cnt_2b_cnt == 'd1) addr <= (({6'd0, ext_mem_param_addr_map[ENC_Q_DENSE_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                    else if (gen_cnt_2b_cnt == 'd2) addr <= (({6'd0, ext_mem_param_addr_map[ENC_K_DENSE_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                end
                'd2: begin
                    if (gen_cnt_2b_cnt == 'd0)      addr <= (({6'd0, ext_mem_param_addr_map[ENC_V_DENSE_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                    else if (gen_cnt_2b_cnt == 'd1) addr <= (({6'd0, ext_mem_param_addr_map[SQRT_NUM_HEAD_EXT_MEM]}) << $clog2(64)) + MLP_DIM + NUM_SLEEP_STAGES;
                    else if (gen_cnt_2b_cnt == 'd2) addr <= (({6'd0, ext_mem_param_addr_map[ENC_COMB_HEAD_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                end
                'd3: begin
                    if (gen_cnt_2b_cnt == 'd0)      addr <= (({6'd0, ext_mem_param_addr_map[ENC_LAYERNORM_2_GAMMA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                    else if (gen_cnt_2b_cnt == 'd1) addr <= (({6'd0, ext_mem_param_addr_map[ENC_LAYERNORM_2_BETA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                    else if (gen_cnt_2b_cnt == 'd2) begin
                        if (cim_num < MLP_DIM)      addr <= (({6'd0, ext_mem_param_addr_map[MLP_DENSE_1_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                        else                        addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_DENSE_1_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num} - MLP_DIM;
                    end
                end
                'd4: begin
                    if (gen_cnt_2b_cnt == 'd0)      addr <= (({6'd0, ext_mem_param_addr_map[MLP_DENSE_2_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                    else if (gen_cnt_2b_cnt == 'd1) addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_LAYERNORM_GAMMA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                    else if (gen_cnt_2b_cnt == 'd2) addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_LAYERNORM_BETA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                end
                'd5: begin
                    // Note: Only one data field used here
                    if (cim_num < NUM_SLEEP_STAGES) begin
                        if (gen_cnt_2b_cnt == 'd0)  addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_DENSE_SOFTMAX_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num} + MLP_DIM;
                        else                        addr <= {13'd0, gen_cnt_2b_cnt};
                    end else begin
                        addr <= {13'd0, gen_cnt_2b_cnt}; // Dummy address to maintain the data_valid pulse from external memory coming
                    end
                end
            endcase
        end
        PARAM_LOAD_FINISHED: begin
            // Nothing to do
        end
        default: begin
            $fatal("Invalid params_curr_layer!");
        end
    endcase
    return addr;
endfunction

function automatic void update_inst(ref logic signed [2:0][N_STORAGE-1:0] bus_data_write, ref logic[3:0] bus_op_write, input logic signed [N_STORAGE-1:0] ext_mem_data, input logic[1:0] gen_cnt_2b_cnt, input logic[6:0] gen_cnt_7b_cnt, input logic ext_mem_data_valid, input logic new_cim, input logic gen_cnt_2b_rst_n);
    if (new_cim) begin
        bus_op_write <= DATA_STREAM_START_OP;
        bus_data_write[0] <= {6'd0, param_addr_map[gen_cnt_7b_cnt[3:0]].addr};
        bus_data_write[1] <= {9'd0, param_addr_map[gen_cnt_7b_cnt[3:0]].len};
    end else begin
        bus_op_write <= (gen_cnt_2b_rst_n == RST) ? DATA_STREAM_OP : NOP;
        bus_data_write[0] <= (ext_mem_data_valid && (gen_cnt_2b_cnt == 'd0)) ? ext_mem_data : bus_data_write[0];
        bus_data_write[1] <= (ext_mem_data_valid && (gen_cnt_2b_cnt == 'd1)) ? ext_mem_data : bus_data_write[1];
        bus_data_write[2] <= (ext_mem_data_valid && (gen_cnt_2b_cnt == 'd2)) ? ext_mem_data : bus_data_write[2];
    end
endfunction

function automatic void prepare_for_broadcast(input HIGH_LEVEL_INFERENCE_STEP_T high_level_inf_step, input logic [6:0] gen_cnt_7b_cnt, input logic [6:0] gen_cnt_7b_2_cnt, input BroadcastOpInfo_t broadcast_info, ref logic[3:0] bus_op_write, ref logic signed [2:0][N_STORAGE-1:0] bus_data_write, ref logic [$clog2(NUM_CIMS)-1:0] bus_target_or_sender_write);
    bus_op_write <= broadcast_info.op;
    bus_data_write[1] <= {{(N_STORAGE-$clog2(NUM_CIMS+1)){1'd0}}, broadcast_info.len};

    unique case (high_level_inf_step)
        ENC_MHSA_QK_T_STEP: begin
            bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr + 10'($rtoi($ceil(gen_cnt_7b_2_cnt*NUM_HEADS)))};
            bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr};
            bus_target_or_sender_write <= gen_cnt_7b_cnt[5:0];
        end
        ENC_MHSA_V_MULT_STEP: begin
            bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr + 10'($rtoi($ceil(gen_cnt_7b_2_cnt*(NUM_PATCHES+1))))};
            bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr};
            bus_target_or_sender_write <= gen_cnt_7b_cnt[5:0];
        end
        ENC_MHSA_PRE_SOFTMAX_TRANS_STEP: begin
            bus_data_write[0] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].tx_addr + 10'($rtoi($ceil(gen_cnt_7b_2_cnt*(NUM_PATCHES+1))))};
            bus_data_write[2] <= {{(N_STORAGE-$clog2(TEMP_RES_STORAGE_SIZE_CIM)){1'd0}}, broadcast_ops[high_level_inf_step].rx_addr + 10'($rtoi($ceil(gen_cnt_7b_2_cnt*(NUM_PATCHES+1))))};
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
    
endfunction
`endif
