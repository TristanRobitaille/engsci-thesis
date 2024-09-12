module softmax_tb # () (
    input wire clk, rst_n,

    // Softmax control
    input wire start,
    input VectorLen_t len,
    input IntResAddr_t start_addr,
    output wire done, busy,

    // Intermediate results memory signals
    input logic int_res_write_en, int_res_read_en, int_res_chip_en,
    input FxFormatIntRes_t int_res_write_format, int_res_read_format,
    input IntResAddr_t int_res_write_addr, int_res_read_addr,
    input CompFx_t int_res_write_data,
    input DataWidth_t int_res_write_data_width, int_res_read_data_width,
    output CompFx_t int_res_read_data
);

    always_comb begin : tb_mem_sig_assign
        tb_int_res_write.en = int_res_write_en;
        tb_int_res_write.chip_en = int_res_chip_en;
        tb_int_res_write.addr = int_res_write_addr;
        tb_int_res_write.data = int_res_write_data;
        tb_int_res_write.data_width = int_res_read_data_width;
        tb_int_res_write.format = int_res_read_format;

        tb_int_res_read.en = int_res_read_en;
        tb_int_res_read.addr = int_res_read_addr;
        tb_int_res_read.data_width = int_res_write_data_width;
        tb_int_res_read.format = int_res_write_format;
        int_res_read_data = tb_int_res_read.data;
    end

    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_read ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_write ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) tb_int_res_write ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) tb_int_res_read ();
    int_res_mem int_res (.clk, .rst_n, .write(int_res_write), .read(int_res_read));

    always_latch begin : int_res_mem_MUX
        // Read control
        if (softmax_int_res_read.en) begin // Softmax reads
            int_res_read.addr = softmax_int_res_read.addr;
            int_res_read.data_width = softmax_int_res_read.data_width;
            int_res_read.format = softmax_int_res_read.format;
        end else if (tb_int_res_read.en) begin // Testbench reads
            int_res_read.addr = tb_int_res_read.addr;
            int_res_read.data_width = tb_int_res_read.data_width;
            int_res_read.format = tb_int_res_read.format;
        end

        // Data read casting
        softmax_int_res_read.data = int_res_read.data;
        tb_int_res_read.data = int_res_read.data;

        // Write control
        int_res_write.chip_en = 1'b1;
        if (tb_int_res_write.en) begin // Testbench writes
            int_res_write.addr = tb_int_res_write.addr;
            int_res_write.data = tb_int_res_write.data;
            int_res_write.format = tb_int_res_write.format;
            int_res_write.data_width = tb_int_res_write.data_width;
        end else if (softmax_int_res_write.en) begin // Softmax writes
            int_res_write.addr = softmax_int_res_write.addr;
            int_res_write.data = softmax_int_res_write.data;
            int_res_write.format = softmax_int_res_write.format;
            int_res_write.data_width = softmax_int_res_write.data_width;
        end

        int_res_read.en = softmax_int_res_read.en | tb_int_res_read.en;
        int_res_write.en = tb_int_res_write.en | softmax_int_res_write.en;
    end

    // Compute
    ComputeIPInterface add_io();
    ComputeIPInterface mult_io();
    ComputeIPInterface exp_add_io();
    ComputeIPInterface exp_mult_io();

    adder add       (.clk, .rst_n, .io(add_io));
    multiplier mult (.clk, .rst_n, .io(mult_io));
    divider div     (.clk, .rst_n, .io(div_io));
    exp exp         (.clk, .rst_n, .io(exp_io), .adder_io(exp_add_io), .mult_io(exp_mult_io));

    // Signal MUXing
    always_latch begin : add_MUX
        if (softmax_add_io.start) begin // Softmax requests an add operation
            add_io.in_1 = softmax_add_io.in_1;
            add_io.in_2 = softmax_add_io.in_2;
        end else if (exp_add_io.start) begin // Exp requests an add operation
            add_io.in_1 = exp_add_io.in_1;
            add_io.in_2 = exp_add_io.in_2;
        end
        add_io.start = (softmax_add_io.start || exp_add_io.start);
        softmax_add_io.out = add_io.out;
        exp_add_io.out = add_io.out;
    end

    always_latch begin : mult_MUX
        if (softmax_mult_io.start) begin // Softmax requests a mult operation
            mult_io.in_1 = softmax_mult_io.in_1;
            mult_io.in_2 = softmax_mult_io.in_2;
        end else if (exp_mult_io.start) begin // Exp requests a mult operation
            mult_io.in_1 = exp_mult_io.in_1;
            mult_io.in_2 = exp_mult_io.in_2;
        end
        mult_io.start = (softmax_mult_io.start || exp_mult_io.start);
        softmax_mult_io.done = mult_io.done;
        exp_mult_io.done = mult_io.done;
        softmax_mult_io.out = mult_io.out;
        exp_mult_io.out = mult_io.out;
    end

    // Softmax module instantiation
    ComputeIPInterface softmax_add_io();
    ComputeIPInterface softmax_mult_io();
    ComputeIPInterface div_io();
    ComputeIPInterface exp_io();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) softmax_int_res_read ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) softmax_int_res_write ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t)  casts ();

    always_comb begin : mem_casts_assigns
        casts.int_res_read_width = int_res_read_data_width;
        casts.int_res_read_format = int_res_read_format;
        casts.int_res_write_width = int_res_write_data_width;
        casts.int_res_write_format = int_res_write_format;
    end

    ComputeIPInterface io();
    ComputeIPInterface io_extra ();
    always_comb begin : tb_io_sig_assign
        io.start = start;
        io_extra.len = len;
        io_extra.start_addr_1 = start_addr;
        done = io.done;
        busy = io.busy;
    end

    softmax softmax (.clk, .rst_n, 
                     .int_res_read(softmax_int_res_read), .int_res_write(softmax_int_res_write), .casts(casts),
                     .io, .io_extra, .add_io(softmax_add_io), .mult_io(softmax_mult_io), .div_io, .exp_io);

    // Assertions
    always_ff @ (posedge clk) begin : compute_mux_assertions
        assert (~(softmax_add_io.start & exp_add_io.start)) else $fatal("Both softmax and exp are asserting adder start signal simultaneously!");
        assert (~(softmax_mult_io.start & exp_mult_io.start)) else $fatal("Both softmax and exp are asserting multiplier start signal simultaneously!");
        assert (~(softmax_int_res_read.en & tb_int_res_write.en)) else $fatal("Softmax and testbench are trying to read/write from intermediate results memory simultaneously!");
    end
endmodule
