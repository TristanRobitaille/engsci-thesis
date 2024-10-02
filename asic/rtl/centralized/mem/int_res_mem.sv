import Defines::*;

module int_res_mem (
    input clk, rst_n,
    MemoryInterface.write_in write,
    MemoryInterface.read_in read
);

    /*  Theory of operation
    *   The intermediate results memory is split into 4 banks, and supports 2 data width: SINGLE_WIDTH and DOUBLE_WIDTH. The later uses two
    *   locations to represent the upper and lower halves. Each bank is single-port. To permit single-cycle accesses for DOUBLE_WIDTH data, the upper and lower halves
    *   are always stored at the same address, but in different banks. Banks 0 and 2 hold the two halves of a double word (bank 0 storing the Most-Significant Half).
    *   Banks 1 and 3 hold the two halves of a word (bank 1 storing the Most-Significant Half). Apart from that, the memory looks contiguous to the rest of the system.
    *   The data is passed as DOUBLE_WIDTH; if the data is SINGLE_WIDTH, is it aligned to the LSH rather than aligned to the decimal point (since it changes).
    */

    // Typedef
    typedef enum logic {
        MSH,
        LSH
    } HalfType_e;

    // Functions
    function automatic CompFx_t cast_to_CompFx_t(input IntResDouble_t data, input FxFormatIntRes_t format);
        // Casts a IntResDouble_t number to CompFx_t, taking care of proper sign extension in both the upper bits and lower bits
        bit sign = (format == INT_RES_DW_FX) ? data[2*N_STO_INT_RES-1] : data[N_STO_INT_RES-1];

        case (format)
            INT_RES_SW_FX_1_X: return {{(N_COMP - Q_COMP - 1){sign}}, data[N_STO_INT_RES-1:0], {(Q_COMP - (N_STO_INT_RES - 1)){sign}}};
            INT_RES_SW_FX_2_X: return {{(N_COMP - Q_COMP - 2){sign}}, data[N_STO_INT_RES-1:0], {(Q_COMP - (N_STO_INT_RES - 2)){sign}}};
            INT_RES_SW_FX_4_X: return {{(N_COMP - Q_COMP - 4){sign}}, data[N_STO_INT_RES-1:0], {(Q_COMP - (N_STO_INT_RES - 4)){sign}}};
            INT_RES_SW_FX_5_X: return {{(N_COMP - Q_COMP - 5){sign}}, data[N_STO_INT_RES-1:0], {(Q_COMP - (N_STO_INT_RES - 5)){sign}}};
            INT_RES_SW_FX_6_X: return {{(N_COMP - Q_COMP - 6){sign}}, data[N_STO_INT_RES-1:0], {(Q_COMP - (N_STO_INT_RES - 6)){sign}}};
            INT_RES_DW_FX:     return {{(N_COMP - Q_COMP - (2*N_STO_INT_RES-Q_STO_INT_RES_DOUBLE)){sign}}, data, {(Q_COMP - Q_STO_INT_RES_DOUBLE){sign}}};
            default: return {{(N_COMP - Q_COMP - 5){sign}}, data[N_STO_INT_RES-1:0], {(Q_COMP - (N_STO_INT_RES - 5)){sign}}};
        endcase
    endfunction

    function automatic IntResSingle_t cast_to_IntResSingle_t(input CompFx_t data, input FxFormatIntRes_t format);
        bit sign = data[N_COMP-1];
        case (format)
            INT_RES_SW_FX_1_X: return {sign, data[Q_COMP-1 : Q_COMP - (N_STO_INT_RES - 1)]};
            INT_RES_SW_FX_2_X: return {sign, data[Q_COMP+0 : Q_COMP - (N_STO_INT_RES - 2)]};
            INT_RES_SW_FX_4_X: return {sign, data[Q_COMP+2 : Q_COMP - (N_STO_INT_RES - 4)]};
            INT_RES_SW_FX_5_X: return {sign, data[Q_COMP+3 : Q_COMP - (N_STO_INT_RES - 5)]};
            INT_RES_SW_FX_6_X: return {sign, data[Q_COMP+4 : Q_COMP - (N_STO_INT_RES - 6)]};
            default: return {sign, data[Q_COMP+3 : Q_COMP - (N_STO_INT_RES - 5)]};
        endcase
    endfunction

    function automatic IntResSingle_t cast_to_HalfIntResDouble_t(input CompFx_t data, input HalfType_e half_type);
        bit sign = data[N_COMP-1];
        IntResDouble_t data_cast = {sign, data[Q_COMP+(2*N_STO_INT_RES-Q_STO_INT_RES_DOUBLE)-2 : Q_COMP - Q_STO_INT_RES_DOUBLE]};

        if (half_type == MSH) return data_cast[2*N_STO_INT_RES-1:N_STO_INT_RES];
        else return data_cast[N_STO_INT_RES-1:0];
    endfunction

    // Signals to individual memory banks instantiated in the memory controller
    MemoryInterface #(IntResSingle_t, IntResBankAddr_t, FxFormat_Unused_t) int_res_0_read ();
    MemoryInterface #(IntResSingle_t, IntResBankAddr_t, FxFormat_Unused_t) int_res_0_write ();
    MemoryInterface #(IntResSingle_t, IntResBankAddr_t, FxFormat_Unused_t) int_res_1_read ();
    MemoryInterface #(IntResSingle_t, IntResBankAddr_t, FxFormat_Unused_t) int_res_1_write ();
    MemoryInterface #(IntResSingle_t, IntResBankAddr_t, FxFormat_Unused_t) int_res_2_read ();
    MemoryInterface #(IntResSingle_t, IntResBankAddr_t, FxFormat_Unused_t) int_res_2_write ();
    MemoryInterface #(IntResSingle_t, IntResBankAddr_t, FxFormat_Unused_t) int_res_3_read ();
    MemoryInterface #(IntResSingle_t, IntResBankAddr_t, FxFormat_Unused_t) int_res_3_write ();

`ifdef USE_MEM_MODEL
    mem_model #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_0 (.clk, .rst_n, .read(int_res_0_read), .write(int_res_0_write));
    mem_model #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_1 (.clk, .rst_n, .read(int_res_1_read), .write(int_res_1_write));
    mem_model #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_2 (.clk, .rst_n, .read(int_res_2_read), .write(int_res_2_write));
    mem_model #(.DEPTH(CIM_INT_RES_BANK_SIZE_NUM_WORD)) int_res_3 (.clk, .rst_n, .read(int_res_3_read), .write(int_res_3_write));
`else
    IntResBankAddr_t int_res_0_addr, int_res_1_addr, int_res_2_addr, int_res_3_addr;
    assign int_res_0_addr = (int_res_0_write.en) ? int_res_0_write.addr : int_res_0_read.addr;
    assign int_res_1_addr = (int_res_1_write.en) ? int_res_1_write.addr : int_res_1_read.addr;
    assign int_res_2_addr = (int_res_2_write.en) ? int_res_2_write.addr : int_res_2_read.addr;
    assign int_res_3_addr = (int_res_3_write.en) ? int_res_3_write.addr : int_res_3_read.addr;
    int_res_14336x8 int_res_0 (.Q(int_res_0_read.data), .CLK(clk), .CEN(int_res_0_write.chip_en), .WEN(int_res_0_write.en), .A(int_res_0_addr), .D(int_res_0_write.data), .EMA(3'b000), .RETN(RETENTION_DISABLED), .PGEN(POWER_GATING_ENABLED));
    int_res_14336x8 int_res_1 (.Q(int_res_1_read.data), .CLK(clk), .CEN(int_res_1_write.chip_en), .WEN(int_res_1_write.en), .A(int_res_1_addr), .D(int_res_1_write.data), .EMA(3'b000), .RETN(RETENTION_DISABLED), .PGEN(POWER_GATING_ENABLED));
    int_res_14336x8 int_res_2 (.Q(int_res_2_read.data), .CLK(clk), .CEN(int_res_2_write.chip_en), .WEN(int_res_2_write.en), .A(int_res_2_addr), .D(int_res_2_write.data), .EMA(3'b000), .RETN(RETENTION_DISABLED), .PGEN(POWER_GATING_ENABLED));
    int_res_14336x8 int_res_3 (.Q(int_res_3_read.data), .CLK(clk), .CEN(int_res_3_write.chip_en), .WEN(int_res_3_write.en), .A(int_res_3_addr), .D(int_res_3_write.data), .EMA(3'b000), .RETN(RETENTION_DISABLED), .PGEN(POWER_GATING_ENABLED));
`endif

    logic read_en_prev;
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

    always_latch begin : int_res_mem_ctrl_read_sel
        if (read_en_prev) begin
            if (read_data_width_prev == SINGLE_WIDTH) begin
                if (bank_read_prev == 0)        read.data = cast_to_CompFx_t({IntResSingle_t'(0), IntResSingle_t'(int_res_0_read.data)}, read.format);
                else if (bank_read_prev == 1)   read.data = cast_to_CompFx_t({IntResSingle_t'(0), IntResSingle_t'(int_res_1_read.data)}, read.format);
                else if (bank_read_prev == 2)   read.data = cast_to_CompFx_t({IntResSingle_t'(0), IntResSingle_t'(int_res_2_read.data)}, read.format);
                else                            read.data = cast_to_CompFx_t({IntResSingle_t'(0), IntResSingle_t'(int_res_3_read.data)}, read.format);
            end else begin
                if (bank_read_prev == 0 || bank_read_prev == 2) begin
                    read.data = cast_to_CompFx_t(IntResDouble_t'({int_res_0_read.data, int_res_2_read.data}), read.format);
                end else begin
                    read.data = cast_to_CompFx_t(IntResDouble_t'({int_res_1_read.data, int_res_3_read.data}), read.format);
                end
            end
        end
    end

    always_ff @ (posedge clk) begin : int_res_mem_ctrl_read_ff
        read_data_width_prev <= read.data_width;
        bank_read_prev <= bank_read_current;
        read_en_prev <= read.en;
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
                int_res_0_write.data = (bank_write_current == 0) ? cast_to_IntResSingle_t(write.data, write.format) : IntResSingle_t'(0);
                int_res_1_write.data = (bank_write_current == 1) ? cast_to_IntResSingle_t(write.data, write.format) : IntResSingle_t'(0);
                int_res_2_write.data = (bank_write_current == 2) ? cast_to_IntResSingle_t(write.data, write.format) : IntResSingle_t'(0);
                int_res_3_write.data = (bank_write_current == 3) ? cast_to_IntResSingle_t(write.data, write.format) : IntResSingle_t'(0);
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
                    int_res_0_write.data = cast_to_HalfIntResDouble_t(write.data, MSH);
                    int_res_2_write.data = cast_to_HalfIntResDouble_t(write.data, LSH);
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
                    int_res_1_write.data = cast_to_HalfIntResDouble_t(write.data, MSH);
                    int_res_3_write.data = cast_to_HalfIntResDouble_t(write.data, LSH);
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

`ifdef ENABLE_ASSERTIONS
    // Assertions
    always_ff @ (posedge clk) begin : int_res_mem_assertions
        if (read.data_width == DOUBLE_WIDTH) assert (read.format == INT_RES_DW_FX) else $error("Intermediate results memory: Read format must be INT_RES_DW_FX if width is DOUBLE_WIDTH");
        if (write.data_width == DOUBLE_WIDTH) assert (write.format == INT_RES_DW_FX) else $error("Intermediate results memory: Write format must be INT_RES_DW_FX if width is DOUBLE_WIDTH");
    end
`endif
endmodule
