module cim_centralized_tb (
    input logic clk, soc_ctrl_rst_n,
    input logic soc_ctrl_new_sleep_epoch,
    output logic soc_ctrl_inference_complete,

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
        soc_ctrl_i.new_sleep_epoch = soc_ctrl_new_sleep_epoch;
        soc_ctrl_inference_complete = soc_ctrl_i.inference_complete;        
    end

    // Memory
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) tb_param_write_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) tb_int_res_write_i ();

    always_comb begin : tb_mem_sig
        tb_int_res_write_i.en = int_res_write_en;
        tb_int_res_write_i.chip_en = int_res_chip_en;
        tb_int_res_write_i.addr = int_res_write_addr;
        tb_int_res_write_i.data = int_res_write_data;
        tb_int_res_write_i.data_width = int_res_write_data_width;
        tb_int_res_write_i.format = int_res_write_format;

        tb_param_write_i.en = param_write_en;
        tb_param_write_i.chip_en = param_chip_en;
        tb_param_write_i.addr = param_write_addr;
        tb_param_write_i.data = param_write_data;
        tb_param_write_i.format = param_write_format;
    end

    cim_centralized cim_centralized (
        .clk(clk),
        .soc_ctrl_i(soc_ctrl_i),

        // ----- Memory ---- //
        .tb_param_write_i, .tb_int_res_write_i
    );
endmodule
