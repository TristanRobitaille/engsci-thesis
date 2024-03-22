`ifndef _cim_mem_sv_
`define _cim_mem_sv_

module cim_mem (
    input wire clk,

    // Access signals
    input MemAccessSignals int_res_access_signals,
    input MemAccessSignals params_access_signals,

    // Data
    output logic [N_STORAGE-1:0] int_res_read_data,
    output logic [N_STORAGE-1:0] params_read_data
);

    logic int_res_wen, params_wen;
    logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] int_res_addr;
    logic [$clog2(PARAMS_STORAGE_SIZE_CIM)-1:0] params_addr;

    // TODO: Replace with actual (single port) generated memory
    logic [N_STORAGE-1:0] params [PARAMS_STORAGE_SIZE_CIM-1:0];
    logic [N_STORAGE-1:0] int_res [TEMP_RES_STORAGE_SIZE_CIM-1:0];

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
        end

        if (params_access_signals.read_req_src[BUS_FSM] || params_access_signals.write_req_src[BUS_FSM]) begin
            params_addr = params_access_signals.addr_table[BUS_FSM];
        end else if (params_access_signals.read_req_src[LOGIC_FSM] || params_access_signals.write_req_src[LOGIC_FSM]) begin
            params_addr = params_access_signals.addr_table[LOGIC_FSM];
        end else if (params_access_signals.read_req_src[MAC] || params_access_signals.write_req_src[MAC]) begin
            params_addr = params_access_signals.addr_table[MAC];
        end else if (params_access_signals.read_req_src[LAYERNORM] || params_access_signals.write_req_src[LAYERNORM]) begin
            params_addr = params_access_signals.addr_table[LAYERNORM];
        end
    end

    always_comb begin : memory_wen
        // Memory in write mode when WEN = 1, else in read mode
        int_res_wen = (int_res_access_signals.write_req_src > 0); // One bit is set in the one-hot signal
        params_wen = (params_access_signals.write_req_src > 0); // One bit is set in the one-hot signal
    end

    always_ff @ (posedge clk) begin : memory_access
        if (int_res_wen) begin // Write
            int_res[int_res_addr] <= int_res_access_signals.write_data[$clog2(int_res_access_signals.write_req_src)];
        end else begin // Read
            int_res_read_data <= int_res[int_res_addr];
        end

        if (params_wen) begin // Write
            params[params_addr] <= params_access_signals.write_data[$clog2(params_access_signals.write_req_src)];
        end else begin // Read
            params_read_data <= params[params_addr];
        end
    end

    // Assertions
    always_ff @ (posedge clk) begin : mem_assertions
        // MAC is not allowed to write to memory
        assert (!int_res_access_signals.write_req_src[MAC]) else $fatal("MAC is not allowed to write to intermediate results memory");
        assert (!params_access_signals.write_req_src[MAC]) else $fatal("MAC is not allowed to write to model parameters memory");

        // Only one request at a time
        assert ($countones({int_res_access_signals.read_req_src, int_res_access_signals.write_req_src}) <= 1) else $fatal("Got more than one read/write request for intermediate results memory");
        assert ($countones({params_access_signals.read_req_src, params_access_signals.write_req_src}) <= 1) else $fatal("Got more than one read/write request for params memory");
    end
endmodule

`endif
