import Defines::*;

module mac_tb # () (
    input wire clk, rst_n,

    // MAC control
    input wire start,
    input VectorLen_t len,
    input ParamAddr_t bias_addr,
    input ParamType_t param_type,
    input Activation_t activation,
    input IntResAddr_t start_addr_1, start_addr_2,
    output wire done, busy,
    output CompFx_t computation_result,

    // Memory access signals
    input logic param_write_en, param_chip_en,
    input ParamAddr_t param_write_addr,
    input CompFx_t param_write_data,
    input FxFormatParams_t param_write_format,

    // Intermediate results memory signals
    input logic int_res_write_en, int_res_chip_en,
    input FxFormatIntRes_t int_res_write_format,
    input IntResAddr_t int_res_write_addr,
    input CompFx_t int_res_write_data,
    input DataWidth_t int_res_write_data_width
);

    always_comb begin : tb_mem_sig_assign
        tb_param_write.en = param_write_en;
        tb_param_write.chip_en = param_chip_en;
        tb_param_write.format = param_write_format;
        tb_param_write.addr = param_write_addr;
        tb_param_write.data = param_write_data;

        tb_int_res_write.en = int_res_write_en;
        tb_int_res_write.chip_en = int_res_chip_en;
        tb_int_res_write.addr = int_res_write_addr;
        tb_int_res_write.data = int_res_write_data;
        tb_int_res_write.data_width = int_res_write_data_width;
        tb_int_res_write.format = int_res_write_format;
    end

    // Memory
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t)  param_read ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t)  param_write ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t)  tb_param_write ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_read ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_write ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) tb_int_res_write ();
    params_mem params   (.clk, .rst_n, .write(param_write),     .read(param_read));
    int_res_mem int_res (.clk, .rst_n, .write(int_res_write),   .read(int_res_read));

    always_latch begin : param_mem_MUX
        // Read control
        param_read.data_width = SINGLE_WIDTH;
        if (MAC_param_read.en) begin // MAC reads
            param_read.addr = MAC_param_read.addr;
            param_read.format = MAC_param_read.format;
        end

        // Data reads
        MAC_param_read.data = param_read.data;

        // Write control
        param_write.chip_en = 1'b1;
        param_write.data_width = SINGLE_WIDTH;
        if (tb_param_write.en) begin // Testbench writes
            param_write.addr = tb_param_write.addr;
            param_write.data = tb_param_write.data;
            param_write.format = tb_param_write.format;
        end

        param_write.en = tb_param_write.en; // Only testbench writes
        param_read.en = MAC_param_read.en; // Only MAC reads
    end

    always_latch begin : int_res_mem_MUX
        // Read control
        if (MAC_int_res_read.en) begin // MAC reads
            int_res_read.addr = MAC_int_res_read.addr;
            int_res_read.data_width = MAC_int_res_read.data_width;
            int_res_read.format = MAC_int_res_read.format;
        end

        // Data read casting
        MAC_int_res_read.data = int_res_read.data;

        // Write control
        int_res_write.chip_en = 1'b1;
        if (tb_int_res_write.en) begin // Testbench writes
            int_res_write.addr = tb_int_res_write.addr;
            int_res_write.data = tb_int_res_write.data;
            int_res_write.format = tb_int_res_write.format;
            int_res_write.data_width = tb_int_res_write.data_width;
        end

        int_res_read.en = MAC_int_res_read.en; // Only MAC reads
        int_res_write.en = tb_int_res_write.en; // Only testbench writes
    end

    // Compute
    ComputeIPInterface add_io();
    ComputeIPInterface mult_io();
    ComputeIPInterface exp_add_io();
    ComputeIPInterface exp_mult_io();

    adder add       (.clk, .rst_n, .io(add_io));
    multiplier mult (.clk, .rst_n, .io(mult_io));
    divider div     (.clk, .rst_n, .io(MAC_div_io));
    exp exp         (.clk, .rst_n, .io(MAC_exp_io), .adder_io(exp_add_io), .mult_io(exp_mult_io));

    // Signal MUXing
    always_latch begin : add_MUX
        if (MAC_add_io.start) begin // MAC requests an add operation
            add_io.in_1 = MAC_add_io.in_1;
            add_io.in_2 = MAC_add_io.in_2;
        end else if (exp_add_io.start) begin // Exp requests an add operation
            add_io.in_1 = exp_add_io.in_1;
            add_io.in_2 = exp_add_io.in_2;
        end
        add_io.start = (MAC_add_io.start || exp_add_io.start);
        MAC_add_io.out = add_io.out;
        exp_add_io.out = add_io.out;
    end

    always_latch begin : mult_MUX
        if (MAC_mult_io.start) begin // MAC requests a mult operation
            mult_io.in_1 = MAC_mult_io.in_1;
            mult_io.in_2 = MAC_mult_io.in_2;
        end else if (exp_mult_io.start) begin // Exp requests a mult operation
            mult_io.in_1 = exp_mult_io.in_1;
            mult_io.in_2 = exp_mult_io.in_2;
        end
        mult_io.start = (MAC_mult_io.start || exp_mult_io.start);
        MAC_mult_io.out = mult_io.out;
        MAC_mult_io.done = mult_io.done;
        exp_mult_io.out = mult_io.out;
        exp_mult_io.done = mult_io.done;
    end

    // MAC module instantiation
    ComputeIPInterface MAC_add_io ();
    ComputeIPInterface MAC_mult_io ();
    ComputeIPInterface MAC_exp_io ();
    ComputeIPInterface MAC_div_io ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) MAC_int_res_read ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t)  MAC_param_read ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t)  casts ();

    always_comb begin : mem_casts_assigns
        casts.int_res_read_width = int_res_write_data_width;
        casts.int_res_read_format = int_res_write_format;
        casts.params_read_format = param_write_format;
    end

    ComputeIPInterface io ();
    ComputeIPInterface io_extra ();
    always_comb begin : tb_io_sig_assign
        io.start = start;
        io_extra.param_type = param_type;
        io_extra.activation = activation;
        io_extra.len = len;
        io_extra.start_addr_1 = start_addr_1;
        io_extra.start_addr_2 = start_addr_2;
        io_extra.bias_addr = bias_addr;
        done = io.done;
        busy = io.busy;
        computation_result = io.out;
    end

    mac mac (
        .clk, .rst_n,
        .casts, .param_read(MAC_param_read), .int_res_read(MAC_int_res_read),
        .io, .io_extra,
        .add_io(MAC_add_io), .mult_io(MAC_mult_io), .div_io(MAC_div_io), .exp_io(MAC_exp_io)
    );

    // Assertions
    always_ff @ (posedge clk) begin : compute_mux_assertions
        assert (~(MAC_add_io.start & exp_add_io.start)) else $fatal("Both MAC and exp are asserting adder start signal simultaneously!");
        assert (~(MAC_mult_io.start & exp_mult_io.start)) else $fatal("Both MAC and exp are asserting multiplier start signal simultaneously!");
        assert (~(MAC_param_read.en & tb_param_write.en)) else $fatal("MAC and testbench are trying to read/write from parameters memory simultaneously!");
        assert (~(MAC_int_res_read.en & tb_int_res_write.en)) else $fatal("MAC and testbench are trying to read/write from intermediate results memory simultaneously!");
    end
endmodule
