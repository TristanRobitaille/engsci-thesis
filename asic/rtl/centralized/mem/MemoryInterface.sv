import Defines::*;

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
endinterface
