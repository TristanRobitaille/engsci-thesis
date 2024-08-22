import Defines::*;

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

    mem_model #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_0 (.clk, .rst_n, .read(int_res_0_read), .write(int_res_0_write));
    mem_model #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_1 (.clk, .rst_n, .read(int_res_1_read), .write(int_res_1_write));
    mem_model #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_2 (.clk, .rst_n, .read(int_res_2_read), .write(int_res_2_write));
    mem_model #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_3 (.clk, .rst_n, .read(int_res_3_read), .write(int_res_3_write));

    logic [1:0] bank_read_current, bank_read_prev;
    logic [1:0] bank_write_current;
    IntResBankAddr_t read_base_addr, write_base_addr;

    // Constants
    assign int_res_0_write.chip_en = write.chip_en;
    assign int_res_1_write.chip_en = write.chip_en;
    assign int_res_2_write.chip_en = write.chip_en;
    assign int_res_3_write.chip_en = write.chip_en;

    // Read logic
    always_comb begin : int_res_mem_ctrl_read_bank_sel
        if (read.addr < IntResAddr_t'(CIM_INT_RES_BANK_SIZE_NUM_WORD)) begin
            bank_read_current = 0;
            read_base_addr = IntResBankAddr_t'(read.addr);
        end else if (read.addr < IntResAddr_t'(2*CIM_INT_RES_BANK_SIZE_NUM_WORD)) begin
            bank_read_current = 1;
            read_base_addr = IntResBankAddr_t'(read.addr - IntResAddr_t'(CIM_INT_RES_BANK_SIZE_NUM_WORD));
        end else if (read.addr < IntResAddr_t'(3*CIM_INT_RES_BANK_SIZE_NUM_WORD)) begin
            bank_read_current = 2;
            read_base_addr = IntResBankAddr_t'(read.addr - IntResAddr_t'(2*CIM_INT_RES_BANK_SIZE_NUM_WORD));
        end else if (read.addr < IntResAddr_t'(4*CIM_INT_RES_BANK_SIZE_NUM_WORD)) begin
            bank_read_current = 3;
            read_base_addr = IntResBankAddr_t'(read.addr - IntResAddr_t'(3*CIM_INT_RES_BANK_SIZE_NUM_WORD));
        end else begin
            bank_read_current = 0;
            read_base_addr = IntResBankAddr_t'(0);
        end
    end

    logic read_data_width_prev;
    always_comb begin : int_res_mem_ctrl_read_comb
        if (read.en) begin
            if (read.data_width == SINGLE_WIDTH) begin
                int_res_0_read.addr = (bank_read_current == 0) ? read_base_addr : IntResBankAddr_t'(0);
                int_res_1_read.addr = (bank_read_current == 1) ? read_base_addr : IntResBankAddr_t'(0);
                int_res_2_read.addr = (bank_read_current == 2) ? read_base_addr : IntResBankAddr_t'(0);
                int_res_3_read.addr = (bank_read_current == 3) ? read_base_addr : IntResBankAddr_t'(0);
                int_res_0_read.en = (bank_read_current == 0);
                int_res_1_read.en = (bank_read_current == 1);
                int_res_2_read.en = (bank_read_current == 2);
                int_res_3_read.en = (bank_read_current == 3);
            end else begin
                // If the requested address maps to banks 0 or 2 --> Bank 0 holds the MSH
                // If the requested address maps to banks 1 or 3 --> Bank 1 holds the MSH
                if (bank_read_current == 0 || bank_read_current == 2) begin
                    int_res_0_read.addr = read_base_addr;
                    int_res_2_read.addr = read_base_addr;
                    int_res_1_read.addr = IntResBankAddr_t'(0);
                    int_res_3_read.addr = IntResBankAddr_t'(0);
                    int_res_0_read.en = 'b1;
                    int_res_2_read.en = 'b1;
                    int_res_1_read.en = 'b0;
                    int_res_3_read.en = 'b0;
                end else begin
                    int_res_1_read.addr = read_base_addr;
                    int_res_3_read.addr = read_base_addr;
                    int_res_0_read.addr = IntResBankAddr_t'(0);
                    int_res_2_read.addr = IntResBankAddr_t'(0);
                    int_res_1_read.en = 'b1;
                    int_res_3_read.en = 'b1;
                    int_res_0_read.en = 'b0;
                    int_res_2_read.en = 'b0;
                end
            end
        end else begin
            int_res_0_read.addr = 'b0;
            int_res_1_read.addr = 'b0;
            int_res_2_read.addr = 'b0;
            int_res_3_read.addr = 'b0;
            int_res_0_read.en = 'b0;
            int_res_1_read.en = 'b0;
            int_res_2_read.en = 'b0;
            int_res_3_read.en = 'b0;
        end
    end

    always_comb begin : int_res_mem_ctrl_read_sel
        if (read_data_width_prev == SINGLE_WIDTH) begin
            if (bank_read_prev == 0)        read.data = IntResDouble_t'(int_res_0_read.data);
            else if (bank_read_prev == 1)   read.data = IntResDouble_t'(int_res_1_read.data);
            else if (bank_read_prev == 2)   read.data = IntResDouble_t'(int_res_2_read.data);
            else                            read.data = IntResDouble_t'(int_res_3_read.data);
        end else begin
            if (bank_read_prev == 0 || bank_read_prev == 2) begin
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
        if (write.addr < IntResAddr_t'(CIM_INT_RES_BANK_SIZE_NUM_WORD)) begin
            bank_write_current = 0;
            write_base_addr = IntResBankAddr_t'(write.addr);
        end else if (write.addr < IntResAddr_t'(2*CIM_INT_RES_BANK_SIZE_NUM_WORD)) begin
            bank_write_current = 1;
            write_base_addr = IntResBankAddr_t'(write.addr - IntResAddr_t'(CIM_INT_RES_BANK_SIZE_NUM_WORD));
        end else if (write.addr < IntResAddr_t'(3*CIM_INT_RES_BANK_SIZE_NUM_WORD)) begin
            bank_write_current = 2;
            write_base_addr = IntResBankAddr_t'(write.addr - IntResAddr_t'(2*CIM_INT_RES_BANK_SIZE_NUM_WORD));
        end else if (write.addr < IntResAddr_t'(4*CIM_INT_RES_BANK_SIZE_NUM_WORD)) begin
            bank_write_current = 3;
            write_base_addr = IntResBankAddr_t'(write.addr - IntResAddr_t'(3*CIM_INT_RES_BANK_SIZE_NUM_WORD));
        end else begin
            bank_write_current = 0;
            write_base_addr = IntResBankAddr_t'(0);
        end
    end

    always_comb begin : int_res_mem_ctrl_write
        if (write.en) begin
            if (write.data_width == SINGLE_WIDTH) begin
                int_res_0_write.addr = (bank_write_current == 0) ? IntResBankAddr_t'(write.addr) : IntResBankAddr_t'(0);
                int_res_1_write.addr = (bank_write_current == 1) ? IntResBankAddr_t'(write.addr - IntResAddr_t'(CIM_INT_RES_BANK_SIZE_NUM_WORD)): IntResBankAddr_t'(0);
                int_res_2_write.addr = (bank_write_current == 2) ? IntResBankAddr_t'(write.addr - IntResAddr_t'(2*CIM_INT_RES_BANK_SIZE_NUM_WORD)): IntResBankAddr_t'(0);
                int_res_3_write.addr = (bank_write_current == 3) ? IntResBankAddr_t'(write.addr - IntResAddr_t'(3*CIM_INT_RES_BANK_SIZE_NUM_WORD)): IntResBankAddr_t'(0);

                // Note: When casting from IntResDouble_t to IntResSingle_t, we discard the upper bits. Thus, this assumes the valid, single-width data is on the lower bits.
                int_res_0_write.data = (bank_write_current == 0) ? IntResSingle_t'(write.data) : IntResSingle_t'(0);
                int_res_1_write.data = (bank_write_current == 1) ? IntResSingle_t'(write.data) : IntResSingle_t'(0);
                int_res_2_write.data = (bank_write_current == 2) ? IntResSingle_t'(write.data) : IntResSingle_t'(0);
                int_res_3_write.data = (bank_write_current == 3) ? IntResSingle_t'(write.data) : IntResSingle_t'(0);

                int_res_0_write.en = (bank_write_current == 0);
                int_res_1_write.en = (bank_write_current == 1);
                int_res_2_write.en = (bank_write_current == 2);
                int_res_3_write.en = (bank_write_current == 3);
            end else begin
                if (bank_write_current == 0 || bank_write_current == 2) begin
                    int_res_0_write.addr = write_base_addr;
                    int_res_2_write.addr = write_base_addr;
                    int_res_1_write.addr = IntResBankAddr_t'(0);
                    int_res_3_write.addr = IntResBankAddr_t'(0);
                    int_res_0_write.en = 'b1;
                    int_res_2_write.en = 'b1;
                    int_res_1_write.en = 'b0;
                    int_res_3_write.en = 'b0;
                    int_res_0_write.data = IntResSingle_t'(write.data[2*N_STO_INT_RES-1:N_STO_INT_RES]); // MSH
                    int_res_2_write.data = IntResSingle_t'(write.data[N_STO_INT_RES-1:0]); // LSH
                    int_res_1_write.data = IntResSingle_t'(0);
                    int_res_3_write.data = IntResSingle_t'(0);
                end else begin
                    int_res_0_write.addr = IntResBankAddr_t'(0);
                    int_res_2_write.addr = IntResBankAddr_t'(0);
                    int_res_1_write.addr = write_base_addr;
                    int_res_3_write.addr = write_base_addr;
                    int_res_0_write.en = 'b0;
                    int_res_2_write.en = 'b0;
                    int_res_1_write.en = 'b1;
                    int_res_3_write.en = 'b1;
                    int_res_0_write.data = IntResSingle_t'(0);
                    int_res_2_write.data = IntResSingle_t'(0);
                    int_res_1_write.data = IntResSingle_t'(write.data[2*N_STO_INT_RES-1:N_STO_INT_RES]); // MSH
                    int_res_3_write.data = IntResSingle_t'(write.data[N_STO_INT_RES-1:0]); // LSH
                end
            end
        end else begin
            int_res_0_write.en = 'b0;
            int_res_1_write.en = 'b0;
            int_res_2_write.en = 'b0;
            int_res_3_write.en = 'b0;
            int_res_0_write.addr = 'b0;
            int_res_1_write.addr = 'b0;
            int_res_2_write.addr = 'b0;
            int_res_3_write.addr = 'b0;
            int_res_0_write.data = IntResSingle_t'(0);
            int_res_1_write.data = IntResSingle_t'(0);
            int_res_2_write.data = IntResSingle_t'(0);
            int_res_3_write.data = IntResSingle_t'(0);
        end
    end

endmodule
