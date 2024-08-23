import Defines::*;

interface MemoryInterface # (
    parameter type Data_t,
    parameter type Addr_t
);

    // Signals
    logic en, chip_en;
    DataWidth_t data_width;
    Data_t data;
    Addr_t addr;

    modport data_out (
        input en,
        input addr,
        input data_width,
        output data
    );

    modport data_in (
        input en, chip_en,
        input data,
        input addr,
        input data_width
    );

    modport data_out_bank (
        input en,
        input addr,
        output data
    );

    modport data_in_bank (
        input en, chip_en,
        input data,
        input addr
    );
endinterface
