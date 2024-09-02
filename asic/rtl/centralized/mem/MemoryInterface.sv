`ifndef _memory_interface_vh_
`define _memory_interface_vh_

interface MemoryInterface # (
    parameter type Data_t,
    parameter type Addr_t,
    parameter type FxFormat_t
);

    // Signals
    logic en, chip_en;
    DataWidth_t data_width;
    Data_t data;
    Addr_t addr;
    FxFormat_t format;

    // To memory controllers
    modport input_read (
        input en,
        input addr,
        input data_width,
        input format,
        output data
    );

    modport input_write (
        input en, chip_en,
        input data,
        input addr,
        input format,
        input data_width
    );

    // From compute
    modport output_read (
        output en,
        output addr,
        output data_width,
        output format,
        input data
    );

    modport output_write (
        output en, chip_en,
        output data,
        output addr,
        output format,
        output data_width
    );

    // To memory banks
    modport input_read_bank (
        input en,
        input addr,
        output data
    );

    modport input_write_bank (
        input en, chip_en,
        input data,
        input addr
    );

    // To compute
    FxFormatIntRes_t int_res_read_format, int_res_write_format;
    DataWidth_t int_res_read_width, int_res_write_width;
    FxFormatParams_t params_read_format, params_write_format;
    modport casts (
        input int_res_read_format, int_res_write_format,
        input int_res_read_width, int_res_write_width,
        input params_read_format, params_write_format
    );

endinterface
`endif
