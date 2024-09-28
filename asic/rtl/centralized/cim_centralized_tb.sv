`ifndef _cim_centralized_tb_vh_
`define _cim_centralized_tb_vh_

import Defines::*;

module cim_centralized_tb (
    input logic clk, soc_ctrl_rst_n, soc_ctrl_start_eeg_load,
    input logic soc_ctrl_new_eeg_data, soc_ctrl_new_sleep_epoch,
    input AdcData_t soc_ctrl_eeg,
    output InferenceStep_t current_inf_step, // For testbench monitoring purposes only
    output logic soc_ctrl_inference_complete,
    output SleepStage_t soc_ctrl_inferred_sleep_stage,

    // ----- Memory ---- //
    // Params
    input logic param_write_en, param_chip_en,
    input ParamAddr_t param_write_addr,
    input CompFx_t param_write_data,
    input FxFormatParams_t param_write_format,

    // Intermediate results
    input logic int_res_write_en, int_res_chip_en,
    input FxFormatIntRes_t int_res_write_format,
    input IntResAddr_t int_res_write_addr,
    input CompFx_t int_res_write_data,
    input DataWidth_t int_res_write_data_width
);

    // ----- ASSIGNS ----- //
    // SoC Interface
    SoCInterface soc_ctrl_i ();
    always_comb begin : soc_ctrl_sign_assign
        soc_ctrl_i.rst_n = soc_ctrl_rst_n;
        soc_ctrl_i.start_eeg_load = soc_ctrl_start_eeg_load;
        soc_ctrl_i.new_eeg_data = soc_ctrl_new_eeg_data;
        soc_ctrl_i.new_sleep_epoch = soc_ctrl_new_sleep_epoch;
        soc_ctrl_i.eeg = soc_ctrl_eeg;
        soc_ctrl_inference_complete = soc_ctrl_i.inference_complete;
        soc_ctrl_inferred_sleep_stage = soc_ctrl_i.inferred_sleep_stage;
    end

    // Memory
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) param_write_tb_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_write_tb_i ();

    always_comb begin : tb_mem_sig
        int_res_write_tb_i.en = int_res_write_en;
        int_res_write_tb_i.chip_en = int_res_chip_en;
        int_res_write_tb_i.addr = int_res_write_addr;
        int_res_write_tb_i.data = int_res_write_data;
        int_res_write_tb_i.data_width = int_res_write_data_width;
        int_res_write_tb_i.format = int_res_write_format;

        param_write_tb_i.en = param_write_en;
        param_write_tb_i.chip_en = param_chip_en;
        param_write_tb_i.addr = param_write_addr;
        param_write_tb_i.data = param_write_data;
        param_write_tb_i.format = param_write_format;
    end

    cim_centralized cim_centralized (
        .clk(clk),
        .soc_ctrl_i(soc_ctrl_i),
        ._current_inf_step(current_inf_step),

        // ----- Memory ---- //
        .param_write_tb_i, .int_res_write_tb_i
    );
endmodule

`endif
