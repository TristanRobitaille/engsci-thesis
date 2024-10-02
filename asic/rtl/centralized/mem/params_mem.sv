import Defines::*;

module params_mem (
    input logic clk, rst_n,
    MemoryInterface.write_in write,
    MemoryInterface.read_in read
);

    /* Theory of operation
    * -Params are always SINGLE_WIDTH so both banks form essentially a contiguous block of memory, with params_1 mapping the upper addresses.
    * -This module also performs casting of the memory data to CompFx_t, which is the format used by the rest of the CiM.
    */

    // Functions
    function automatic CompFx_t cast_to_CompFx_t(input Param_t data, input FxFormatParams_t format);
        // Casts a Param_t number to CompFx_t, taking care of proper sign extension in both the upper bits and lower bits
        bit sign = data[N_STO_PARAMS-1];
        case (format)
            PARAMS_FX_2_X: return {{(N_COMP - Q_COMP - 2){sign}}, data, {(Q_COMP - (N_STO_PARAMS - 2)){sign}}};
            PARAMS_FX_3_X: return {{(N_COMP - Q_COMP - 3){sign}}, data, {(Q_COMP - (N_STO_PARAMS - 3)){sign}}};
            PARAMS_FX_4_X: return {{(N_COMP - Q_COMP - 4){sign}}, data, {(Q_COMP - (N_STO_PARAMS - 4)){sign}}};
            PARAMS_FX_5_X: return {{(N_COMP - Q_COMP - 5){sign}}, data, {(Q_COMP - (N_STO_PARAMS - 5)){sign}}};
            default: return {{(N_COMP - Q_COMP - 4){sign}}, data, {(Q_COMP - (N_STO_PARAMS - 4)){sign}}};
        endcase
    endfunction

    function automatic Param_t cast_to_Param_t(input CompFx_t data, input FxFormatParams_t format);
        bit sign = data[N_COMP-1];
        case (format)
            PARAMS_FX_2_X: return {sign, data[Q_COMP+0 : Q_COMP - (N_STO_PARAMS - 2)]};
            PARAMS_FX_3_X: return {sign, data[Q_COMP+1 : Q_COMP - (N_STO_PARAMS - 3)]};
            PARAMS_FX_4_X: return {sign, data[Q_COMP+2 : Q_COMP - (N_STO_PARAMS - 4)]};
            PARAMS_FX_5_X: return {sign, data[Q_COMP+3 : Q_COMP - (N_STO_PARAMS - 5)]};
            default: return {sign, data[Q_COMP+2 : Q_COMP - (N_STO_PARAMS - 4)]};
        endcase
    endfunction

    // Signals
    logic params_0_read_en_prev, params_1_read_en_prev;
    MemoryInterface #(Param_t, ParamBankAddr_t, FxFormat_Unused_t) params_0_read ();
    MemoryInterface #(Param_t, ParamBankAddr_t, FxFormat_Unused_t) params_0_write ();
    MemoryInterface #(Param_t, ParamBankAddr_t, FxFormat_Unused_t) params_1_read ();
    MemoryInterface #(Param_t, ParamBankAddr_t, FxFormat_Unused_t) params_1_write ();

`ifdef USE_MEM_MODEL
    mem_model #(.DEPTH(CIM_PARAMS_BANK_SIZE_NUM_WORD)) params_0 (.rst_n, .clk, .read(params_0_read), .write(params_0_write));
    mem_model #(.DEPTH(CIM_PARAMS_BANK_SIZE_NUM_WORD)) params_1 (.rst_n, .clk, .read(params_1_read), .write(params_1_write));
`else
    ParamBankAddr_t params_0_addr, params_1_addr;
    assign params_0_addr = (params_0_write.en) ? params_0_write.addr : params_0_read.addr;
    assign params_1_addr = (params_1_write.en) ? params_1_write.addr : params_1_read.addr;
    params_15872x8 params_0 (.Q(params_0_read.data), .CLK(clk), .CEN(params_0_write.chip_en), .WEN(params_0_write.en), .A(params_0_addr), .D(params_0_write.data), .EMA(3'b000), .RETN(RETENTION_ENABLED), .PGEN(POWER_GATING_DISABLED));
    params_15872x8 params_1 (.Q(params_1_read.data), .CLK(clk), .CEN(params_1_write.chip_en), .WEN(params_1_write.en), .A(params_1_addr), .D(params_1_write.data), .EMA(3'b000), .RETN(RETENTION_ENABLED), .PGEN(POWER_GATING_DISABLED));
`endif

    // Constants
    assign params_0_write.chip_en = write.chip_en;
    assign params_1_write.chip_en = write.chip_en;

    // Read logic
    always_comb begin : param_mem_ctrl_read_comb
        if (read.en) begin
            if (read.addr < ParamAddr_t'(CIM_PARAMS_BANK_SIZE_NUM_WORD)) begin // Bank 0
                params_0_read.addr = ParamBankAddr_t'(read.addr);
                params_0_read.en = read.en;
                params_1_read.en = 'b0;
            end else begin // Bank 1
                params_1_read.addr = ParamBankAddr_t'(read.addr - ParamAddr_t'(CIM_PARAMS_BANK_SIZE_NUM_WORD));
                params_1_read.en = read.en;
                params_0_read.en = 'b0;
            end
        end else begin // Avoid latch
            params_0_read.addr = ParamBankAddr_t'(0);
            params_0_read.en = 'b0;
            params_1_read.addr = ParamBankAddr_t'(0);
            params_1_read.en = 'b0;
        end
    end

    // To have the correct data on the output, considering that the memory is single-cycle access, we need to latch which memory bank we read from
    always_ff @ (posedge clk) begin : param_mem_ctrl_read_ff
        params_0_read_en_prev <= params_0_read.en;
        params_1_read_en_prev <= params_1_read.en;
    end

    always_comb begin : param_mem_read_data_sel
        if (params_0_read_en_prev)      read.data = cast_to_CompFx_t(params_0_read.data, read.format);
        else if (params_1_read_en_prev) read.data = cast_to_CompFx_t(params_1_read.data, read.format);
        else read.data = CompFx_t'(0); // Avoid latch
    end

    // Write logic
    always_comb begin : param_mem_ctrl_write_comb
        if (write.en) begin
            if (write.addr < ParamAddr_t'(CIM_PARAMS_BANK_SIZE_NUM_WORD)) begin // Bank 0
                params_0_write.addr = ParamBankAddr_t'(write.addr);
                params_0_write.data = cast_to_Param_t(write.data, write.format);
                params_0_write.en = 'b1;
                params_1_write.en = 'b0;
            end else begin // Bank 1
                params_1_write.addr = ParamBankAddr_t'(write.addr - ParamAddr_t'(CIM_PARAMS_BANK_SIZE_NUM_WORD));
                params_1_write.data = cast_to_Param_t(write.data, write.format);
                params_1_write.en = 'b1;
                params_0_write.en = 'b0;
            end
        end else begin // Avoid latch
            params_0_write.addr = ParamBankAddr_t'(0);
            params_0_write.data = Param_t'(0);
            params_0_write.en = 'b0;
            params_1_write.addr = ParamBankAddr_t'(0);
            params_1_write.data = Param_t'(0);
            params_1_write.en = 'b0;
        end
    end

`ifdef ENABLE_ASSERTIONS
    // Assertions
    always_ff @ (posedge clk) begin : params_mem_assertions
        assert (!(params_0_read.en & params_1_read.en)) else $error("Trying to read from both banks of parameters memory simultaneously!");
        assert (!(params_0_write.en & params_1_write.en)) else $error("Trying to write to both banks of parameters memory simultaneously!");        
    end
`endif
endmodule
