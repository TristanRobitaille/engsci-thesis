module layernorm_tb # () (
    input wire clk, rst_n,

    // LayerNorm control
    input wire start,
    input VectorLen_t len,
    input ParamAddr_t beta_addr, gamma_addr,
    input IntResAddr_t start_addr,
    input HalfSelect_t half_select,
    output wire done, busy,

    // Memory access signals
    input logic param_write_en, param_chip_en,
    input ParamAddr_t param_write_addr,
    input CompFx_t param_write_data,
    input FxFormatParams_t param_write_format,

    // Intermediate results memory signals
    input logic int_res_write_en, int_res_read_en, int_res_chip_en,
    input FxFormatIntRes_t int_res_write_format, int_res_read_format,
    input IntResAddr_t int_res_write_addr, int_res_read_addr,
    input CompFx_t int_res_write_data,
    input DataWidth_t int_res_read_data_width, int_res_write_data_width,
    output CompFx_t int_res_read_data
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

        tb_int_res_read.en = int_res_read_en;
        tb_int_res_read.addr = int_res_read_addr;
        int_res_read_data = tb_int_res_read.data;

        // Since testbench writes what LayerNorm reads, we assign *_read_* here and vice-versa
        tb_int_res_write.data_width = int_res_read_data_width;
        tb_int_res_write.format = int_res_read_format;
        tb_int_res_read.data_width = int_res_write_data_width;
        tb_int_res_read.format = int_res_write_format;
    end

    // Memory
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t)  param_read ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t)  param_write ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t)  tb_param_write ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_read ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_write ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) tb_int_res_write ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) tb_int_res_read ();
    params_mem params   (.clk, .rst_n, .write(param_write),     .read(param_read));
    int_res_mem int_res (.clk, .rst_n, .write(int_res_write),   .read(int_res_read));

    always_latch begin : param_mem_MUX
        // Read control
        param_read.data_width = SINGLE_WIDTH;
        if (LN_param_read.en) begin // MAC reads
            param_read.addr = LN_param_read.addr;
            param_read.format = LN_param_read.format;
        end

        // Data reads
        LN_param_read.data = param_read.data;

        // Write control
        param_write.chip_en = 1'b1;
        param_write.data_width = SINGLE_WIDTH;
        if (tb_param_write.en) begin // Testbench writes
            param_write.addr = tb_param_write.addr;
            param_write.data = tb_param_write.data;
            param_write.format = tb_param_write.format;
        end

        param_write.en = tb_param_write.en; // Only testbench writes
        param_read.en = LN_param_read.en; // Only LayerNorm reads
    end

    always_latch begin : int_res_mem_MUX
        // Read control
        if (LN_int_res_read.en) begin // LayerNorm reads
            int_res_read.addr = LN_int_res_read.addr;
            int_res_read.data_width = LN_int_res_read.data_width;
            int_res_read.format = LN_int_res_read.format;
        end else if (tb_int_res_read.en) begin // Testbench reads
            int_res_read.addr = tb_int_res_read.addr;
            int_res_read.data_width = tb_int_res_read.data_width;
            int_res_read.format = tb_int_res_read.format;
        end

        // Data read
        LN_int_res_read.data = int_res_read.data;
        tb_int_res_read.data = int_res_read.data;

        // Write control
        int_res_write.chip_en = 1'b1;
        if (LN_int_res_write.en) begin // LayerNorm reads
            int_res_write.addr = LN_int_res_write.addr;
            int_res_write.data = LN_int_res_write.data;
            int_res_write.format = LN_int_res_write.format;
            int_res_write.data_width = LN_int_res_write.data_width;
        end else if (tb_int_res_write.en) begin // Testbench writes
            int_res_write.addr = tb_int_res_write.addr;
            int_res_write.data = tb_int_res_write.data;
            int_res_write.format = tb_int_res_write.format;
            int_res_write.data_width = tb_int_res_write.data_width;
        end

        int_res_read.en = (LN_int_res_read.en | tb_int_res_read.en);
        int_res_write.en = (LN_int_res_write.en | tb_int_res_write.en);
    end

    // Compute
    ComputeIPInterface add_io();
    ComputeIPInterface mult_io();
    ComputeIPInterface div_io();
    ComputeIPInterface sqrt_io();

    adder add       (.clk, .rst_n, .io(add_io));
    multiplier mult (.clk, .rst_n, .io(mult_io));
    divider div     (.clk, .rst_n, .io(div_io));
    sqrt sqrt       (.clk, .rst_n, .io(sqrt_io));

    ComputeIPInterface io();
    ComputeIPInterface io_extra ();
    always_comb begin : tb_io_sig_assign
        io.start = start;
        done = io.done;
        busy = io.busy;
        io_extra.half_select = half_select;
        io_extra.start_addr_1 = IntResAddr_t'(start_addr);
        io_extra.start_addr_2 = IntResAddr_t'(beta_addr);
        io_extra.start_addr_3 = IntResAddr_t'(gamma_addr);
    end

    always_comb begin : tb_mem_casts_assign
        LN_mem_casts.int_res_read_format = int_res_read_format;
        LN_mem_casts.int_res_write_format = int_res_write_format;
        LN_mem_casts.int_res_read_width = int_res_read_data_width;
        LN_mem_casts.int_res_write_width = int_res_write_data_width;
        LN_mem_casts.params_read_format = param_write_format;
    end

    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) LN_int_res_read ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) LN_int_res_write ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t)  LN_param_read ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t)  LN_mem_casts ();

    layernorm layernom (
        .clk, .rst_n,
        .param_read(LN_param_read), .int_res_read(LN_int_res_read), .int_res_write(LN_int_res_write), .casts(LN_mem_casts),
        .io, .io_extra, .add_io, .mult_io, .div_io, .sqrt_io
    );

    // Assertions
    always_ff @ (posedge clk) begin : mem_mux_assertions
        assert (~(LN_int_res_read.en & tb_int_res_read.en)) else $fatal("Both LayerNorm and testbench are trying to read int res memory simultaneously!");
        assert (~(LN_int_res_write.en & tb_int_res_write.en)) else $fatal("Both LayerNorm and testbench are trying to write to int res memory simultaneously!");
    end

endmodule;
