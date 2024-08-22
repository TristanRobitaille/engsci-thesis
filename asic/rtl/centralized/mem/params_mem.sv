import Defines::*;

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
    mem_model #(.DEPTH(CIM_PARAMS_BANK_SIZE_NUM_WORD)) params_0 (.rst_n, .clk, .read(params_0_read), .write(params_0_write));
    mem_model #(.DEPTH(CIM_PARAMS_BANK_SIZE_NUM_WORD)) params_1 (.rst_n, .clk, .read(params_1_read), .write(params_1_write));

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
        if (params_0_read_en_prev)      read.data = params_0_read.data;
        else if (params_1_read_en_prev) read.data = params_1_read.data;
        else read.data = Param_t'(0); // Avoid latch
    end

    // Write logic
    always_comb begin : param_mem_ctrl_write_comb
        if (write.en) begin
            if (write.addr < ParamAddr_t'(CIM_PARAMS_BANK_SIZE_NUM_WORD)) begin // Bank 0
                params_0_write.addr = ParamBankAddr_t'(write.addr);
                params_0_write.data = write.data;
                params_0_write.en = 'b1;
                params_1_write.en = 'b0;
            end else begin // Bank 1
                params_1_write.addr = ParamBankAddr_t'(write.addr - ParamAddr_t'(CIM_PARAMS_BANK_SIZE_NUM_WORD));
                params_1_write.data = write.data;
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

    // Assertions
    always_ff @ (posedge clk) begin : params_mem_assertions
        assert (!(params_0_read.en & params_1_read.en)) else $error("Trying to read from both banks of parameters memory simultaneously!");
        assert (!(params_0_write.en & params_1_write.en)) else $error("Trying to write to both banks of parameters memory simultaneously!");        
    end
endmodule
