`ifndef _master_fcn_vh_
`define _master_fcn_vh_

`include "types.svh"

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
// function automatic logic [$clog2(NUM_PARAMS)-1:0] param_ext_mem_addr(input logic[6:0] param_num, input logic[6:0] cim_num, input logic[1:0] gen_cnt_2b_cnt, input logic [3:0] params_curr_layer);
//     logic [$clog2(NUM_PARAMS)-1:0] addr;
//     unique case (params_curr_layer)
//         PATCH_PROJ_KERNEL_PARAMS: begin
//             addr <= (({6'd0, ext_mem_param_addr_map[PATCH_PROJ_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
//         end
//         POS_EMB_PARAMS: begin
//             addr <= (({6'd0, ext_mem_param_addr_map[POS_EMB_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
//         end
//         ENC_Q_DENSE_KERNEL_PARAMS: begin
//             addr <= (({6'd0, ext_mem_param_addr_map[ENC_Q_DENSE_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
//         end
//         ENC_K_DENSE_KERNEL_PARAMS: begin
//             addr <= (({6'd0, ext_mem_param_addr_map[ENC_K_DENSE_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
//         end
//         ENC_V_DENSE_KERNEL_PARAMS: begin
//             addr <= (({6'd0, ext_mem_param_addr_map[ENC_V_DENSE_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
//         end
//         ENC_COMB_HEAD_KERNEL_PARAMS: begin
//             addr <= (({6'd0, ext_mem_param_addr_map[ENC_COMB_HEAD_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
//         end
//         ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_KERNEL_PARAMS: begin
//             if (cim_num < MLP_DIM) begin
//                 addr <= (({6'd0, ext_mem_param_addr_map[MLP_DENSE_1_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
//             end else begin
//                 addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_DENSE_1_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
//             end
//         end
//         ENC_MLP_DENSE_2_KERNEL_PARAMS: begin
//             addr <= (({6'd0, ext_mem_param_addr_map[MLP_DENSE_2_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
//         end
//         MLP_HEAD_DENSE_2_KERNEL_PARAMS: begin
//             addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_DENSE_SOFTMAX_KERNEL_EXT_MEM]} + {8'd0, cim_num}) << $clog2(64)) + {8'd0, param_num};
//         end
//         SINGLE_PARAMS: begin
//             unique case (param_num)
//                 'd0: begin
//                     if (gen_cnt_2b_cnt == 'd0)      addr <= (({6'd0, ext_mem_param_addr_map[PATCH_PROJ_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                     else if (gen_cnt_2b_cnt == 'd1) addr <= (({6'd0, ext_mem_param_addr_map[CLASS_EMB_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                     else if (gen_cnt_2b_cnt == 'd2) addr <= (({6'd0, ext_mem_param_addr_map[ENC_LAYERNORM_1_GAMMA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                 end
//                 'd1: begin
//                     if (gen_cnt_2b_cnt == 'd0)      addr <= (({6'd0, ext_mem_param_addr_map[ENC_LAYERNORM_1_BETA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                     else if (gen_cnt_2b_cnt == 'd1) addr <= (({6'd0, ext_mem_param_addr_map[ENC_Q_DENSE_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                     else if (gen_cnt_2b_cnt == 'd2) addr <= (({6'd0, ext_mem_param_addr_map[ENC_K_DENSE_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                 end
//                 'd2: begin
//                     if (gen_cnt_2b_cnt == 'd0)      addr <= (({6'd0, ext_mem_param_addr_map[ENC_V_DENSE_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                     else if (gen_cnt_2b_cnt == 'd1) addr <= (({6'd0, ext_mem_param_addr_map[SQRT_NUM_HEAD_EXT_MEM]}) << $clog2(64)) + MLP_DIM + NUM_SLEEP_STAGES;
//                     else if (gen_cnt_2b_cnt == 'd2) addr <= (({6'd0, ext_mem_param_addr_map[ENC_COMB_HEAD_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                 end
//                 'd3: begin
//                     if (gen_cnt_2b_cnt == 'd0)      addr <= (({6'd0, ext_mem_param_addr_map[ENC_LAYERNORM_2_GAMMA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                     else if (gen_cnt_2b_cnt == 'd1) addr <= (({6'd0, ext_mem_param_addr_map[ENC_LAYERNORM_2_BETA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                     else if (gen_cnt_2b_cnt == 'd2) begin
//                         if (cim_num < MLP_DIM)      addr <= (({6'd0, ext_mem_param_addr_map[MLP_DENSE_1_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                         else                        addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_DENSE_1_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num} - MLP_DIM;
//                     end
//                 end
//                 'd4: begin
//                     if (gen_cnt_2b_cnt == 'd0)      addr <= (({6'd0, ext_mem_param_addr_map[MLP_DENSE_2_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                     else if (gen_cnt_2b_cnt == 'd1) addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_LAYERNORM_GAMMA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                     else if (gen_cnt_2b_cnt == 'd2) addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_LAYERNORM_BETA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
//                 end
//                 'd5: begin
//                     // Note: Only one data field used here
//                     if (cim_num < NUM_SLEEP_STAGES) begin
//                         if (gen_cnt_2b_cnt == 'd0)  addr <= (({6'd0, ext_mem_param_addr_map[MLP_HEAD_DENSE_SOFTMAX_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num} + MLP_DIM;
//                         else                        addr <= {13'd0, gen_cnt_2b_cnt};
//                     end else begin
//                         addr <= {13'd0, gen_cnt_2b_cnt}; // Dummy address to maintain the data_valid pulse from external memory coming
//                     end
//                 end
//             endcase
//         end
//         PARAM_LOAD_FINISHED: begin
//             // Nothing to do
//         end
//         //synopsys translate_off
//         default: begin
//             $fatal("Invalid params_curr_layer!");
//         end
//         //synopsys translate_on
//     endcase
//     return addr;
// endfunction

function automatic logic [$clog2(NUM_PARAMS)-1:0] param_ext_mem_addr(input logic [6:0] param_num, input logic [6:0] cim_num, input logic [1:0] gen_cnt_2b_cnt, input logic [3:0] params_curr_layer);
    logic [$clog2(NUM_PARAMS)-1:0] addr;
    logic [5:0] index;

    if (params_curr_layer == PATCH_PROJ_KERNEL_PARAMS) begin
        addr = (({6'd0, ext_mem_param_addr_map[PATCH_PROJ_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
    end
    else if (params_curr_layer == POS_EMB_PARAMS) begin
        addr = (({6'd0, ext_mem_param_addr_map[POS_EMB_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
    end
    else if (params_curr_layer == ENC_Q_DENSE_KERNEL_PARAMS) begin
        addr = (({6'd0, ext_mem_param_addr_map[ENC_Q_DENSE_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
    end
    else if (params_curr_layer == ENC_K_DENSE_KERNEL_PARAMS) begin
        addr = (({6'd0, ext_mem_param_addr_map[ENC_K_DENSE_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
    end
    else if (params_curr_layer == ENC_V_DENSE_KERNEL_PARAMS) begin
        addr = (({6'd0, ext_mem_param_addr_map[ENC_V_DENSE_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
    end
    else if (params_curr_layer == ENC_COMB_HEAD_KERNEL_PARAMS) begin
        addr = (({6'd0, ext_mem_param_addr_map[ENC_COMB_HEAD_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
    end
    else if (params_curr_layer == ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_KERNEL_PARAMS) begin
        if (cim_num < MLP_DIM) begin
            index = MLP_DENSE_1_KERNEL_EXT_MEM;
        end else begin
            index = MLP_HEAD_DENSE_1_KERNEL_EXT_MEM;
        end
        addr = (({6'd0, ext_mem_param_addr_map[index]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
    end
    else if (params_curr_layer == ENC_MLP_DENSE_2_KERNEL_PARAMS) begin
        addr = (({6'd0, ext_mem_param_addr_map[MLP_DENSE_2_KERNEL_EXT_MEM]} + {8'd0, param_num}) << $clog2(64)) + {8'd0, cim_num};
    end
    else if (params_curr_layer == MLP_HEAD_DENSE_2_KERNEL_PARAMS) begin
        addr = (({6'd0, ext_mem_param_addr_map[MLP_HEAD_DENSE_SOFTMAX_KERNEL_EXT_MEM]} + {8'd0, cim_num}) << $clog2(64)) + {8'd0, param_num};
    end
    else if (params_curr_layer == SINGLE_PARAMS) begin
        if (param_num == 0) begin
            if (gen_cnt_2b_cnt == 2'b00) begin
                index = PATCH_PROJ_BIAS_EXT_MEM;
            end
            else if (gen_cnt_2b_cnt == 2'b01) begin
                index = CLASS_EMB_EXT_MEM;
            end
            else if (gen_cnt_2b_cnt == 2'b10) begin
                index = ENC_LAYERNORM_1_GAMMA_EXT_MEM;
            end
            addr = ({6'd0, ext_mem_param_addr_map[index]} << 6) + {8'd0, cim_num};
        end
        else if (param_num == 1) begin
            if (gen_cnt_2b_cnt == 'd0) begin
                addr = (({6'd0, ext_mem_param_addr_map[ENC_LAYERNORM_1_BETA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
            end
            else if (gen_cnt_2b_cnt == 'd1) begin
                addr = (({6'd0, ext_mem_param_addr_map[ENC_Q_DENSE_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
            end
            else if (gen_cnt_2b_cnt == 'd2) begin
                addr = (({6'd0, ext_mem_param_addr_map[ENC_K_DENSE_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
            end
        end
        else if (param_num == 2) begin
            if (gen_cnt_2b_cnt == 'd0) begin
                addr = (({6'd0, ext_mem_param_addr_map[ENC_V_DENSE_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
            end
            else if (gen_cnt_2b_cnt == 'd1) begin
                addr = (({6'd0, ext_mem_param_addr_map[SQRT_NUM_HEAD_EXT_MEM]}) << $clog2(64)) + MLP_DIM + NUM_SLEEP_STAGES;
            end
            else if (gen_cnt_2b_cnt == 'd2) begin
                addr = (({6'd0, ext_mem_param_addr_map[ENC_COMB_HEAD_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
            end
        end
        else if (param_num == 3) begin
            if (gen_cnt_2b_cnt == 'd0) begin
                addr = (({6'd0, ext_mem_param_addr_map[ENC_LAYERNORM_2_GAMMA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
            end
            else if (gen_cnt_2b_cnt == 'd1) begin
                addr = (({6'd0, ext_mem_param_addr_map[ENC_LAYERNORM_2_BETA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
            end
            else if (gen_cnt_2b_cnt == 'd2) begin
                if (cim_num < MLP_DIM) begin
                    addr = (({6'd0, ext_mem_param_addr_map[MLP_DENSE_1_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
                end else begin
                    addr = (({6'd0, ext_mem_param_addr_map[MLP_HEAD_DENSE_1_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num} - MLP_DIM;
                end
            end
        end
        else if (param_num == 4) begin
            if (gen_cnt_2b_cnt == 'd0) begin
                addr = (({6'd0, ext_mem_param_addr_map[MLP_DENSE_2_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
            end
            else if (gen_cnt_2b_cnt == 'd1) begin
                addr = (({6'd0, ext_mem_param_addr_map[MLP_HEAD_LAYERNORM_GAMMA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
            end
            else if (gen_cnt_2b_cnt == 'd2) begin
                addr = (({6'd0, ext_mem_param_addr_map[MLP_HEAD_LAYERNORM_BETA_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num};
            end
        end
        else if (param_num == 5) begin
            // Note: Only one data field used here
            if (cim_num < NUM_SLEEP_STAGES) begin
                if (gen_cnt_2b_cnt == 'd0) begin
                    addr = (({6'd0, ext_mem_param_addr_map[MLP_HEAD_DENSE_SOFTMAX_BIAS_EXT_MEM]}) << $clog2(64)) + {8'd0, cim_num} + MLP_DIM;
                end else begin
                    addr = {13'd0, gen_cnt_2b_cnt};
                end
            end else begin
                addr = {13'd0, gen_cnt_2b_cnt}; // Dummy address to maintain the data_valid pulse from external memory coming
            end
        end
    end
    else if (params_curr_layer == PARAM_LOAD_FINISHED) begin
        // Nothing to do
    end
    else begin
        //synopsys translate_off
        $fatal("Invalid params_curr_layer!");
        //synopsys translate_on
    end

    return addr;
endfunction

`endif
