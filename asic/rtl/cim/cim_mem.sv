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
        case ({int_res_access_signals.read_req_src, int_res_access_signals.write_req_src})
            6'b000001, // Bus FSM write
            6'b001000: // Bus FSM read
                int_res_addr = int_res_access_signals.addr_table[BUS_FSM];
            6'b000010, // Logic FSM write
            6'b010000: // Logic FSM read
                int_res_addr = int_res_access_signals.addr_table[LOGIC_FSM];
            6'b000100, // MAC write
            6'b100000: // MAC read
                int_res_addr = int_res_access_signals.addr_table[MAC];
            default:
                int_res_addr = int_res_addr; // No change
        endcase

        // Model parameters memory
        case ({params_access_signals.read_req_src, params_access_signals.write_req_src})
            6'b000001, // Bus FSM write
            6'b001000: // Bus FSM read
                params_addr = params_access_signals.addr_table[BUS_FSM];
            6'b000010, // Logic FSM write
            6'b010000: // Logic FSM read
                params_addr = params_access_signals.addr_table[LOGIC_FSM];
            6'b000100, // MAC write
            6'b100000: // MAC read
                params_addr = params_access_signals.addr_table[MAC];
            default:
                params_addr = params_addr; // No change
        endcase
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

        // TODO: Only one request at a 1time
    end
endmodule

`endif
