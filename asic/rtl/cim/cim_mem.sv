`ifndef _cim_mem_sv_
`define _cim_mem_sv_

module cim_mem # (
    parameter int SRAM = 1 // Choose whether to synthesize general-purpose memory or 
)(
    input wire clk,

    // Access signals
    MemAccessSignals int_res_access_signals,
    MemAccessSignals params_access_signals,

    // Data
    output STORAGE_WORD_T int_res_read_data,
    output STORAGE_WORD_T params_read_data
);

    logic int_res_wen, params_wen;
    STORAGE_WORD_T int_res_write_data, params_write_data;
    TEMP_RES_ADDR_T int_res_addr;
    PARAMS_ADDR_T params_addr;

    always_comb begin : memory_wen
        // Memory in write mode when WEN = 0, else in read mode
        int_res_wen = ~(int_res_access_signals.write_req_src > 0); // One bit is set in the one-hot signal
        params_wen = ~(params_access_signals.write_req_src > 0); // One bit is set in the one-hot signal
    end

    // MUXing
    always_latch begin : memory_addr_MUX
        // Intermediate results memory
        if (int_res_access_signals.read_req_src[BUS_FSM] || int_res_access_signals.write_req_src[BUS_FSM]) begin
            int_res_addr = int_res_access_signals.addr_table[BUS_FSM];
        end else if (int_res_access_signals.read_req_src[LOGIC_FSM] || int_res_access_signals.write_req_src[LOGIC_FSM]) begin
            int_res_addr = int_res_access_signals.addr_table[LOGIC_FSM];
        end else if (int_res_access_signals.read_req_src[MAC] || int_res_access_signals.write_req_src[MAC]) begin
            int_res_addr = int_res_access_signals.addr_table[MAC];
        end else if (int_res_access_signals.read_req_src[LAYERNORM] || int_res_access_signals.write_req_src[LAYERNORM]) begin
            int_res_addr = int_res_access_signals.addr_table[LAYERNORM];
        end else if (int_res_access_signals.read_req_src[DATA_FILL_FSM] || int_res_access_signals.write_req_src[DATA_FILL_FSM]) begin
            int_res_addr = int_res_access_signals.addr_table[DATA_FILL_FSM];
        end else if (int_res_access_signals.read_req_src[DENSE_BROADCAST_SAVE_FSM] || int_res_access_signals.write_req_src[DENSE_BROADCAST_SAVE_FSM]) begin
            int_res_addr = int_res_access_signals.addr_table[DENSE_BROADCAST_SAVE_FSM];
        end

        if (params_access_signals.read_req_src[BUS_FSM] || params_access_signals.write_req_src[BUS_FSM]) begin
            params_addr = params_access_signals.addr_table[BUS_FSM];
        end else if (params_access_signals.read_req_src[LOGIC_FSM] || params_access_signals.write_req_src[LOGIC_FSM]) begin
            params_addr = params_access_signals.addr_table[LOGIC_FSM];
        end else if (params_access_signals.read_req_src[MAC] || params_access_signals.write_req_src[MAC]) begin
            params_addr = params_access_signals.addr_table[MAC];
        end else if (params_access_signals.read_req_src[LAYERNORM] || params_access_signals.write_req_src[LAYERNORM]) begin
            params_addr = params_access_signals.addr_table[LAYERNORM];
        end else if (params_access_signals.read_req_src[DATA_FILL_FSM] || params_access_signals.write_req_src[DATA_FILL_FSM]) begin
            params_addr = params_access_signals.addr_table[DATA_FILL_FSM];
        end else if (params_access_signals.read_req_src[DENSE_BROADCAST_SAVE_FSM] || params_access_signals.write_req_src[DENSE_BROADCAST_SAVE_FSM]) begin
            params_addr = params_access_signals.addr_table[DENSE_BROADCAST_SAVE_FSM];
        end
    end

    always_latch begin : memory_data_mux
        case (int_res_access_signals.write_req_src)
            7'b0000001: int_res_write_data = int_res_access_signals.write_data[0];
            7'b0000010: int_res_write_data = int_res_access_signals.write_data[1];
            7'b0000100: int_res_write_data = int_res_access_signals.write_data[2];
            7'b0001000: int_res_write_data = int_res_access_signals.write_data[3];
            7'b0010000: int_res_write_data = int_res_access_signals.write_data[4];
            7'b0100000: int_res_write_data = int_res_access_signals.write_data[5];
            7'b1000000: int_res_write_data = int_res_access_signals.write_data[6];
            default: ; // Add appropriate default handling here
        endcase

        case (params_access_signals.write_req_src)
            7'b0000001: params_write_data = params_access_signals.write_data[0];
            7'b0000010: params_write_data = params_access_signals.write_data[1];
            7'b0000100: params_write_data = params_access_signals.write_data[2];
            7'b0001000: params_write_data = params_access_signals.write_data[3];
            7'b0010000: params_write_data = params_access_signals.write_data[4];
            7'b0100000: params_write_data = params_access_signals.write_data[5];
            7'b1000000: params_write_data = params_access_signals.write_data[6];
            default: ; // Add appropriate default handling here
        endcase
    end

    // Memory instantiation (will get optimized away if SRAM == 1)
    STORAGE_WORD_T params [PARAMS_STORAGE_SIZE_CIM-1:0];
    STORAGE_WORD_T int_res [TEMP_RES_STORAGE_SIZE_CIM-1:0];

    if (SRAM == 1) begin : sram_instantiation
        mem_528x16 params_inst (
            .Q(params_read_data), // Output data
            .CLK(clk), // Clock
            .CEN(1'b0), //Active-low chip enable
            .WEN(params_wen), // Active-low write enable
            .A(params_addr), // Address
            .D(params_write_data), // Input data
            .EMA(3'b011), // Extra Margin Adjustment --> Increases delay of internal timing pulses for extra margin
            .RETN(1'b1), // Active-low Retention Mode Enable 
            .PGEN(1'b1) // Active-low Power Down Mode Enable
        );
        mem_848x16 int_res_inst (
            .Q(int_res_read_data), .CLK(clk), .CEN(1'b0), .WEN(int_res_wen), .A(int_res_addr), .D(int_res_write_data), .EMA(3'b011), .RETN(1'b1), .PGEN(1'b1)
        );
    end else begin : reg_array_access
        always_ff @ (posedge clk) begin
            if (~int_res_wen) begin // Write
                int_res[int_res_addr] <= int_res_write_data;
            end else begin
                int_res_read_data <= int_res[int_res_addr];
            end

            if (~params_wen) begin // Write
                params[params_addr] <= params_write_data;
            end else begin // Read
                params_read_data <= params[params_addr];
            end
        end
    end
    //synopsys translate_off
    // Assertions
    always_ff @ (posedge clk) begin : mem_assertions
        // Only one request at a time
        if ($countones({int_res_access_signals.read_req_src, int_res_access_signals.write_req_src}) > 1)
            $display("Got more than one read/write request for intermediate results memory: %b at time %d", {int_res_access_signals.read_req_src, int_res_access_signals.write_req_src}, $time);
        if ($countones({params_access_signals.read_req_src, params_access_signals.write_req_src}) > 1)
            $display("Got more than one read/write request for params memory: %b at time %d", {params_access_signals.read_req_src, params_access_signals.write_req_src}, $time);

        assert ($countones({int_res_access_signals.read_req_src, int_res_access_signals.write_req_src}) <= 1) else $display("Got more than one read/write request for intermediate results memory");
        assert ($countones({params_access_signals.read_req_src, params_access_signals.write_req_src}) <= 1) else $display("Got more than one read/write request for params memory");
    end
    //synopsys translate_on
endmodule

`endif
