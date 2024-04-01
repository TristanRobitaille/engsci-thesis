`ifndef _master_fcn_vh_
`define _master_fcn_vh_

`include "../types.svh"

/*----- CONSTANTS -----*/
parameter logic [$clog2(NUM_PARAMS/64)-1:0] ext_mem_param_addr_map[27] = {
    $clog2(NUM_PARAMS/64)'(0),
    $clog2(NUM_PARAMS/64)'(478),
    $clog2(NUM_PARAMS/64)'(64),
    $clog2(NUM_PARAMS/64)'(125),
    $clog2(NUM_PARAMS/64)'(479),
    $clog2(NUM_PARAMS/64)'(189),
    $clog2(NUM_PARAMS/64)'(480),
    $clog2(NUM_PARAMS/64)'(253),
    $clog2(NUM_PARAMS/64)'(481),
    $clog2(NUM_PARAMS/64)'(317),
    $clog2(NUM_PARAMS/64)'(482),
    $clog2(NUM_PARAMS/64)'(381),
    $clog2(NUM_PARAMS/64)'(483),
    $clog2(NUM_PARAMS/64)'(381),
    $clog2(NUM_PARAMS/64)'(491),
    $clog2(NUM_PARAMS/64)'(445),
    $clog2(NUM_PARAMS/64)'(484),
    $clog2(NUM_PARAMS/64)'(477),
    $clog2(NUM_PARAMS/64)'(485),
    $clog2(NUM_PARAMS/64)'(486),
    $clog2(NUM_PARAMS/64)'(487),
    $clog2(NUM_PARAMS/64)'(488),
    $clog2(NUM_PARAMS/64)'(489),
    $clog2(NUM_PARAMS/64)'(490),
    $clog2(NUM_PARAMS/64)'(492),
    $clog2(NUM_PARAMS/64)'(483),
    $clog2(NUM_PARAMS/64)'(483)
};

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
        bus_op_write <= PARAM_STREAM_START_OP;
        bus_data_write[0] <= {6'd0, param_addr_map[gen_cnt_7b_cnt[3:0]].addr};
        bus_data_write[1] <= {9'd0, param_addr_map[gen_cnt_7b_cnt[3:0]].len};
    end else begin
        bus_op_write <= ((gen_cnt_2b_rst_n == RST) || ext_mem_data_valid) ? PARAM_STREAM_OP : NOP;
        bus_data_write[0] <= (ext_mem_data_valid && (gen_cnt_2b_cnt == 'd0)) ? ext_mem_data : bus_data_write[0];
        bus_data_write[1] <= (ext_mem_data_valid && (gen_cnt_2b_cnt == 'd1)) ? ext_mem_data : bus_data_write[1];
        bus_data_write[2] <= (ext_mem_data_valid && (gen_cnt_2b_cnt == 'd2)) ? ext_mem_data : bus_data_write[2];
    end
endfunction

`endif
