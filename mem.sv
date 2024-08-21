params_mem.sv


module params_mem (
    input logic clk, rst_n,
    input MemoryAccessSignals.data_in write,
    output MemoryAccessSignals.data_out read
);

    //Theory of operation: Params are always SINGLE_WIDTH so both banks form essentially a contiguous block of memory, with params_1 mapping the upper addresses

    // Signals
    logic params_0_read_en_prev, params_1_read_en_prev;
    MemoryAccessSignals #(Param_t, ParamBankAddr_t) params_0_read ();
    MemoryAccessSignals #(Param_t, ParamBankAddr_t) params_0_write ();
    MemoryAccessSignals #(Param_t, ParamBankAddr_t) params_1_read ();
    MemoryAccessSignals #(Param_t, ParamBankAddr_t) params_1_write ();
    params_bank #(.DEPTH(CIM_PARAMS_BANK_SIZE_NUM_WORD)) params_0 (.rst, .clk, .read(params_0_read), .write(params_0_write));
    params_bank #(.DEPTH(CIM_PARAMS_BANK_SIZE_NUM_WORD)) params_1 (.rst, .clk, .read(params_1_read), .write(params_1_write));

    // Constants
    assign params_0_write.cen = write.cen;
    assign params_1_write.cen = write.cen;

    // Read logic
    always_comb begin : param_mem_ctrl_read_comb
        if (read.enable) begin
            if (read.addr < CIM_PARAMS_BANK_SIZE_NUM_WORD) begin // Bank 0
                params_0_read.addr = ParamBankAddr_t'(read.addr);
                params_0_read.enable = read.enable;
                params_1_read.enable = 'b0;
            end else begin // Bank 1
                params_1_read.addr = ParamBankAddr_t'(read.addr - CIM_PARAMS_BANK_SIZE_NUM_WORD);
                params_1_read.enable = read.enable;
                params_0_read.enable = 'b0;
            end
        end else begin // Avoid latch
            params_0_read.addr = ParamBankAddr_t'(0);
            params_0_read.data = Param_t'(0);
            params_0_read.enable = 'b0;
            params_1_read.addr = ParamBankAddr_t'(0);
            params_1_read.data = Param_t'(0);
            params_1_read.enable = 'b0;
        end
    end

    // To have the correct data on the output, considering that the memory is single-cycle access, we need to latch which memory bank we read from
    always_ff @ (posedge clk) begin : param_mem_ctrl_read_ff
        params_0_read_en_prev <= params_0_read.enable;
        params_1_read_en_prev <= params_1_read.enable;
    end

    always_comb begin : param_mem_read_data_sel
        if (params_0_read_en_prev)      read.data = params_0_read.data;
        else if (params_1_read_en_prev) read.data = params_1_read.data;
        else read.data = Param_t'(0); // Avoid latch
    end

    // Write logic
    always_comb begin : param_mem_ctrl_write_comb
        if (write.enable) begin
            if (write.addr < CIM_PARAMS_BANK_SIZE_NUM_WORD) begin // Bank 0
                params_0_write.addr = ParamBankAddr_t'(write.addr);
                params_0_write.data = write.data;
                params_0_write.enable = read.enable;
                params_1_write.enable = 'b0;
            end else begin // Bank 1
                params_1_write.addr = ParamBankAddr_t'(write.addr - CIM_PARAMS_BANK_SIZE_NUM_WORD);
                params_1_write.data = write.data;
                params_1_write.enable = read.enable;
                params_0_write.enable = 'b0;
            end
        end else begin // Avoid latch
            params_0_write.addr = ParamBankAddr_t'(0);
            params_0_write.data = Param_t'(0);
            params_0_write.enable = 'b0;
            params_1_write.addr = ParamBankAddr_t'(0);
            params_1_write.data = Param_t'(0);
            params_1_write.enable = 'b0;
        end
    end

    // Assertions
    assert (!(params_0_read.enable & params_1_read.enable)) else $error("Trying to read from both banks of parameters memory simultaneously!");
    assert (!(params_0_write.enable & params_1_write.enable)) else $error("Trying to write to both banks of parameters memory simultaneously!");
endmodule

// TODO: Review waveforms, especially delayed read



























int_res_mem.sv

module int_res_mem (
    input clk, rst_n,
    input MemoryAccessSignals.data_in write,
    output MemoryAccessSignals.data_out read
);

    /*  Theory of operation
        The intermediate results memory is split into 4 banks, and supports two data width: SINGLE_WIDTH and DOUBLE_WIDTH. The later uses two
        locations to represent the upper and lower halves. Each bank is single-port. To permit single-cycle accesses for DOUBLE_WIDTH data, the upper and lower halves
        are always stored at the same address, but in different banks. Banks 0 and 2 hold the two halves of a double word (bank 0 storing the Most-Significant Half).
        Banks 1 and 3 hold the two halves of a word (bank 1 storing the Most-Significant Half). Apart from that, the memory looks contiguous to the rest of the system.
        The data is passed as DOUBLE_WIDTH; if the data is SINGLE_WIDTH, is it aligned to the LSH rather than aligned to the decimal point (since it changes).
    */

    // Signals to individual memory banks instantiated in the memory controller
    MemoryAccessSignals #(IntResSingle_t, IntResBankAddr_t) int_res_0_read ();
    MemoryAccessSignals #(IntResSingle_t, IntResBankAddr_t) int_res_0_write ();
    MemoryAccessSignals #(IntResSingle_t, IntResBankAddr_t) int_res_1_read ();
    MemoryAccessSignals #(IntResSingle_t, IntResBankAddr_t) int_res_1_write ();
    MemoryAccessSignals #(IntResSingle_t, IntResBankAddr_t) int_res_2_read ();
    MemoryAccessSignals #(IntResSingle_t, IntResBankAddr_t) int_res_2_write ();
    MemoryAccessSignals #(IntResSingle_t, IntResBankAddr_t) int_res_3_read ();
    MemoryAccessSignals #(IntResSingle_t, IntResBankAddr_t) int_res_3_write ();

    params_bank #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_0 (.rst, .clk, .read(int_res_0_read), .write(int_res_0_write));
    params_bank #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_1 (.rst, .clk, .read(int_res_1_read), .write(int_res_1_write));
    params_bank #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_2 (.rst, .clk, .read(int_res_2_read), .write(int_res_2_write));
    params_bank #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_3 (.rst, .clk, .read(int_res_3_read), .write(int_res_3_write));

    logic [1:0] bank_read_current, bank_read_prev;
    logic [1:0] bank_write_current;
    IntResBankAddr_t read_base_addr, write_base_addr;

    // ----- TASKS -----//
    task automatic read_single_from_bank(input int bank, input IntResBankAddr_t addr);
        int_res_0_read.addr = (bank == 0) ? addr : IntResBankAddr_t'(0);
        int_res_1_read.addr = (bank == 1) ? addr : IntResBankAddr_t'(0);
        int_res_2_read.addr = (bank == 2) ? addr : IntResBankAddr_t'(0);
        int_res_3_read.addr = (bank == 3) ? addr : IntResBankAddr_t'(0);
        int_res_0_read.enable = (bank == 0);
        int_res_1_read.enable = (bank == 1);
        int_res_2_read.enable = (bank == 2);
        int_res_3_read.enable = (bank == 3);
    endtask

    task automatic write_single_from_bank(input int bank, input IntResBankAddr_t addr, IntResDouble_t data);
        int_res_0_write.addr = (bank == 0) ? addr : IntResBankAddr_t'(0);
        int_res_1_write.addr = (bank == 1) ? addr : IntResBankAddr_t'(0);
        int_res_2_write.addr = (bank == 2) ? addr : IntResBankAddr_t'(0);
        int_res_3_write.addr = (bank == 3) ? addr : IntResBankAddr_t'(0);

        // Note: When casting from IntResDouble_t to IntResSingle_t, we discard the upper bits. Thus, this assumes the valid, single-width data is on the lower bits.
        int_res_0_write.data = (bank == 0) ? IntResSingle_t'(data) : IntResSingle_t'(0);
        int_res_1_write.data = (bank == 1) ? IntResSingle_t'(data) : IntResSingle_t'(0);
        int_res_2_write.data = (bank == 2) ? IntResSingle_t'(data) : IntResSingle_t'(0);
        int_res_3_write.data = (bank == 3) ? IntResSingle_t'(data) : IntResSingle_t'(0);

        int_res_0_write.enable = (bank == 0);
        int_res_1_write.enable = (bank == 1);
        int_res_2_write.enable = (bank == 2);
        int_res_3_write.enable = (bank == 3);
    endtask

    // Read logic
    always_comb begin : int_res_mem_ctrl_read_bank_sel
        if (read.addr < CIM_INT_RES_BANK_SIZE_NUM_WORD) begin
            bank_read_current = 0;
            read_base_addr = read.addr;
        end else if (read.addr < 2*CIM_INT_RES_BANK_SIZE_NUM_WORD) begin
            bank_read_current = 1;
            read_base_addr = read.addr - CIM_INT_RES_BANK_SIZE_NUM_WORD;
        end else if (read.addr < 3*CIM_INT_RES_BANK_SIZE_NUM_WORD) begin
            bank_read_current = 2;
            read_base_addr = read.addr - 2*CIM_INT_RES_BANK_SIZE_NUM_WORD;
        end else if (read.addr < 4*CIM_INT_RES_BANK_SIZE_NUM_WORD) begin
            bank_read_current = 3;
            read_base_addr = read.addr - 3*CIM_INT_RES_BANK_SIZE_NUM_WORD;
        end
    end

    logic read_data_width_prev;
    always_comb begin : int_res_mem_ctrl_read_comb
        if (read.enable) begin
            if (read.data_width == SINGLE_WIDTH) begin
                read_single_from_bank(bank_read_current, read_base_addr);
            end else if (read.data_width == DOUBLE_WIDTH) begin
                // If the requested address maps to banks 0 or 2 --> Bank 0 holds the MSH
                // If the requested address maps to banks 1 or 3 --> Bank 1 holds the MSH
                if (bank_read_current == 0 || bank_read_current == 2) begin
                    int_res_0_read.addr = read_base_addr;
                    int_res_2_read.addr = read_base_addr;
                    int_res_1_read.addr = IntResBankAddr_t'(0);
                    int_res_3_read.addr = IntResBankAddr_t'(0);
                    int_res_0_read.enable = 'b1;
                    int_res_2_read.enable = 'b1;
                    int_res_1_read.enable = 'b0;
                    int_res_3_read.enable = 'b0;
                end else begin
                    int_res_1_read.addr = read_base_addr;
                    int_res_3_read.addr = read_base_addr;
                    int_res_0_read.addr = IntResBankAddr_t'(0);
                    int_res_2_read.addr = IntResBankAddr_t'(0);
                    int_res_1_read.enable = 'b1;
                    int_res_3_read.enable = 'b1;
                    int_res_0_read.enable = 'b0;
                    int_res_2_read.enable = 'b0;
                end
            end
        end else begin
            int_res_0_read.enable = 'b0;
            int_res_1_read.enable = 'b0;
            int_res_2_read.enable = 'b0;
            int_res_3_read.enable = 'b0;
        end
    end

    always_comb begin : int_res_mem_ctrl_read_sel
        if (read_data_width_prev == SINGLE_WIDTH) begin
            if (bank_read_prev == 0)        read.data = IntResDouble_t'(int_res_0_read.data);
            else if (bank_read_prev == 1)   read.data = IntResDouble_t'(int_res_1_read.data);
            else if (bank_read_prev == 2)   read.data = IntResDouble_t'(int_res_2_read.data);
            else if (bank_read_prev == 3)   read.data = IntResDouble_t'(int_res_3_read.data);
        end else if (read_data_width_prev == DOUBLE_WIDTH) begin
            if (bank == 0 || bank == 2) begin
                read.data = IntResDouble_t'({int_res_0_read.data, int_res_2_read.data});
            end else begin
                read.data = IntResDouble_t'({int_res_1_read.data, int_res_3_read.data});
            end
        end
    end

    always_ff @ (posedge clk) begin : int_res_mem_ctrl_read_ff
        read_data_width_prev <= read.data_width;
        bank_read_prev <= bank_read_current;
    end

    // Write logic
    always_comb begin : int_res_mem_ctrl_write_bank_sel
        if (write.addr < CIM_INT_RES_BANK_SIZE_NUM_WORD) begin
            bank_write_current = 0;
            write_base_addr = write.addr;
        end else if (write.addr < 2*CIM_INT_RES_BANK_SIZE_NUM_WORD) begin
            bank_write_current = 1;
            write_base_addr = write.addr - CIM_INT_RES_BANK_SIZE_NUM_WORD;
        end else if (write.addr < 3*CIM_INT_RES_BANK_SIZE_NUM_WORD) begin
            bank_write_current = 2;
            write_base_addr = write.addr - 2*CIM_INT_RES_BANK_SIZE_NUM_WORD;
        end else if (write.addr < 4*CIM_INT_RES_BANK_SIZE_NUM_WORD) begin
            bank_write_current = 3;
            write_base_addr = write.addr - 3*CIM_INT_RES_BANK_SIZE_NUM_WORD;
        end
    end

    always_comb begin : int_res_mem_ctrl_write
        if (write.enable) begin
            if (write.data_width == SINGLE_WIDTH) begin
                write_single_from_bank(bank_write_current, write.addr - bank_write_current*CIM_INT_RES_BANK_SIZE_NUM_WORD, write.data);
            end else if (write.data_width == DOUBLE_WIDTH) begin
                if (bank_write_current == 0 || bank_write_current == 2) begin
                    int_res_0_write.addr = base_addr;
                    int_res_2_write.addr = base_addr;
                    int_res_1_write.addr = IntResBankAddr_t'(0);
                    int_res_3_write.addr = IntResBankAddr_t'(0);
                    int_res_0_write.enable = 'b1;
                    int_res_2_write.enable = 'b1;
                    int_res_1_write.enable = 'b0;
                    int_res_3_write.enable = 'b0;
                    int_res_0_write.data = IntResDouble_t'(write.data[2*N_STO_INT_RES-1:N_STO_INT_RES]); // MSH
                    int_res_2_write.data = IntResDouble_t'(write.data[N_STO_INT_RES-1:0]); // LSH
                    int_res_1_write.data = IntResSingle_t'(0);
                    int_res_3_write.data = IntResSingle_t'(0);
                end else begin
                    int_res_0_write.addr = IntResBankAddr_t'(0);
                    int_res_2_write.addr = IntResBankAddr_t'(0);
                    int_res_1_write.addr = base_addr;
                    int_res_3_write.addr = base_addr;
                    int_res_0_write.enable = 'b0;
                    int_res_2_write.enable = 'b0;
                    int_res_1_write.enable = 'b1;
                    int_res_3_write.enable = 'b1;
                    int_res_0_write.data = IntResSingle_t'(0);
                    int_res_2_write.data = IntResSingle_t'(0);
                    int_res_1_write.data = IntResDouble_t'(write.data[2*N_STO_INT_RES-1:N_STO_INT_RES]); // MSH
                    int_res_3_write.data = IntResDouble_t'(write.data[N_STO_INT_RES-1:0]); // LSH
                end
            end
        end else begin
            int_res_0_write.enable = 'b0;
            int_res_1_write.enable = 'b0;
            int_res_2_write.enable = 'b0;
            int_res_3_write.enable = 'b0;
        end
    end

endmodule


TODO: Review
















MemoryAccessSignals.sv


interface MemoryAccessSignals # (
    parameter type data_t,
    parameter type addr_t
);

    // Signals
    logic en;
    DataWidth_t data_width;
    logic [$bits(data_t)-1:0] data;
    logic [$bits(addr_t)-1:0] addr;

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




// High-level memory signals
// Instantiation
MemoryAccessSignals #(Param_t, ParamAddr_t) param_read ();
MemoryAccessSignals #(Param_t, ParamAddr_t) param_write ();
MemoryAccessSignals #(IntResDouble_t, TempResAddr_t) int_res_read ();
MemoryAccessSignals #(IntResDouble_t, TempResAddr_t) int_res_write ();

// Signals to memory controller
params_mem params (.clk, .rst_n, .write(param_read), .read(param_write));
int_res_mem int_res (.clk, .rst_n, .write(param_read), .read(param_write));











mem_model.sv


module mem_model #(
    parameter int DEPTH
)(
    input logic clk, rst_n,
    input MemoryAccessSignals.data_in_bank write,
    output MemoryAccessSignals.data_out_bank read
);

    logic WEN, CEN;
    logic [$bits(write.data)-1:0] memory [DEPTH-1:0];

    assign WEN = write.en & ~read.en;
    assign CEN = write.chip_en;

    always_ff @ (posedge clk) begin : mem_write
        if (WEN & CEN & ~rst_n) begin
            memory[write.addr] <= write.data;
        end
    end

    always_ff @ (posedge clk) begin : mem_read
        if (~rst_n) begin
            read.data <= 'd0;
        end else if (~WEN & CEN) begin
            read.data <= memory[read.addr];
        end
    end

    // Assertions
    assert (!(write.en & read.en)) else $error("Tried to read and write to memory simultaneously!"); // Never try to read and write simulteanously
    assert (!((write.en | read.en) & ~write.chip_enable)) else $error("Tried to write or read from memory while it was not enabled!"); // Never try to read or write while memory not enabled

endmodule
