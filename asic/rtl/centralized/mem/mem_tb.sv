import Defines::*;

module mem_tb (
    input logic clk, rst_n,

    // Parameters memory signals
    input logic param_read_en, param_write_en, param_chip_en,
    input FxFormatParams_t param_read_format, param_write_format,
    input ParamAddr_t param_read_addr, param_write_addr,
    input CompFx_t param_write_data,
    output CompFx_t param_read_data,

    // Intermediate results memory signals
    input logic int_res_read_en, int_res_write_en, int_res_chip_en,
    input FxFormatIntRes_t int_res_read_format, int_res_write_format,
    input DataWidth_t int_res_read_data_width, int_res_write_data_width,
    input IntResAddr_t int_res_read_addr, int_res_write_addr,
    input CompFx_t int_res_write_data,
    output CompFx_t int_res_read_data
);

    // Parameters memory
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) param_read_sig ();
    assign param_read_sig.en            = param_read_en;
    assign param_read_sig.addr          = param_read_addr;
    assign param_read_sig.format        = param_read_format;
    assign param_read_sig.data_width    = SINGLE_WIDTH;
    assign param_read_data              = param_read_sig.data;

    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) param_write_sig ();
    assign param_write_sig.en           = param_write_en;
    assign param_write_sig.chip_en      = param_chip_en;
    assign param_write_sig.format       = param_write_format;
    assign param_write_sig.data         = param_write_data;
    assign param_write_sig.addr         = param_write_addr;
    assign param_write_sig.data_width   = SINGLE_WIDTH;

    params_mem params (.clk, .rst_n, .write(param_write_sig), .read(param_read_sig));

    // Intermediate results memory
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_read_sig ();
    assign int_res_read_sig.en          = int_res_read_en;
    assign int_res_read_sig.addr        = int_res_read_addr;
    assign int_res_read_sig.data_width  = int_res_read_data_width;
    assign int_res_read_sig.format      = (int_res_read_data_width == SINGLE_WIDTH) ? INT_RES_SW_FX_5_X : INT_RES_DW_FX; // Default for testbench
    assign int_res_read_data            = int_res_read_sig.data;
    assign int_res_read_sig.format      = int_res_read_format;

    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_write_sig ();
    assign int_res_write_sig.en         = int_res_write_en;
    assign int_res_write_sig.chip_en    = int_res_chip_en;
    assign int_res_write_sig.data       = int_res_write_data;
    assign int_res_write_sig.addr       = int_res_write_addr;
    assign int_res_write_sig.data_width = int_res_write_data_width;
    assign int_res_write_sig.format     = int_res_write_format; // Default for testbench

    int_res_mem int_res (.clk, .rst_n, .write(int_res_write_sig), .read(int_res_read_sig));
endmodule