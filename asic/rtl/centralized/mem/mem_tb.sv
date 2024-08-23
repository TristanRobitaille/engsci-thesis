import Defines::*;

module mem_tb (
    input logic clk, rst_n,

    // Parameters memory signals
    input logic param_read_en, param_write_en, param_chip_en,
    input DataWidth_t param_read_data_width, param_write_data_width,
    input ParamAddr_t param_read_addr, param_write_addr,
    input Param_t param_write_data,
    output Param_t param_read_data,

    // Intermediate results memory signals
    input logic int_res_read_en, int_res_write_en, int_res_chip_en,
    input DataWidth_t int_res_read_data_width, int_res_write_data_width,
    input IntResAddr_t int_res_read_addr, int_res_write_addr,
    input IntResDouble_t int_res_write_data,
    output IntResDouble_t int_res_read_data
);

    // Parameters memory
    MemoryInterface #(Param_t, ParamAddr_t) param_read_sig ();
    assign param_read_sig.en            = param_read_en;
    assign param_read_sig.addr          = param_read_addr;
    assign param_read_sig.data_width    = param_read_data_width;
    assign param_read_data              = param_read_sig.data;

    MemoryInterface #(Param_t, ParamAddr_t) param_write_sig ();
    assign param_write_sig.en           = param_write_en;
    assign param_write_sig.chip_en      = param_chip_en;
    assign param_write_sig.data         = param_write_data;
    assign param_write_sig.addr         = param_write_addr;
    assign param_write_sig.data_width   = param_write_data_width;

    params_mem params (.clk, .rst_n, .write(param_write_sig), .read(param_read_sig));

    // Intermediate results memory
    MemoryInterface #(IntResDouble_t, IntResAddr_t) int_res_read_sig ();
    assign int_res_read_sig.en          = int_res_read_en;
    assign int_res_read_sig.addr        = int_res_read_addr;
    assign int_res_read_sig.data_width  = int_res_read_data_width;
    assign int_res_read_data            = int_res_read_sig.data;

    MemoryInterface #(IntResDouble_t, IntResAddr_t) int_res_write_sig ();
    assign int_res_write_sig.en         = int_res_write_en;
    assign int_res_write_sig.chip_en    = int_res_chip_en;
    assign int_res_write_sig.data       = int_res_write_data;
    assign int_res_write_sig.addr       = int_res_write_addr;
    assign int_res_write_sig.data_width = int_res_write_data_width;

    int_res_mem int_res (.clk, .rst_n, .write(int_res_write_sig), .read(int_res_read_sig));
endmodule