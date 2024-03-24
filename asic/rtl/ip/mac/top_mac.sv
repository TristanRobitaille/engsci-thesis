`include "../../types.svh"
`include "../../cim/cim.svh"

module top_mac # () (
    input wire clk, rst_n
);
    `include "../../top_init.sv"

    // CiM memory
    logic [N_STORAGE-1:0] params [PARAMS_STORAGE_SIZE_CIM-1:0];
    logic [N_STORAGE-1:0] int_res [TEMP_RES_STORAGE_SIZE_CIM-1:0];

    logic [N_STORAGE-1:0] param_data, int_res_data = 'd0;
    MemAccessSignals params_access_signals();
    MemAccessSignals int_res_access_signals();
    always_ff @ (posedge clk) begin : memory_ctrl
        if (!rst_n) begin
            param_data <= 'd0;
            int_res_data <= 'd0;
        end else begin
            param_data <= (params_access_signals.read_req_src[MAC]) ? params[params_access_signals.addr_table[MAC]] : param_data;
            int_res_data <= (int_res_access_signals.read_req_src[MAC]) ? int_res[int_res_access_signals.addr_table[MAC]] : int_res_data;
        end
    end

    // Control signals
    logic start, param_type;
    logic [1:0] activation = 'd0;
    logic [$clog2(MAC_MAX_LEN+1)-1:0] len = 'd0;
    logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] start_addr1 = 'd0;
    logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] start_addr2 = 'd0;

    // MAC outputs
    wire busy, done, int_res_mem_access_req, params_mem_access_req;
    wire [N_COMP-1:0] computation_result;

    // Adder and multiplier
    wire add_overflow, mul_overflow, add_refresh, mul_refresh;
    wire signed [N_COMP-1:0] mul_input_q_1, mul_input_q_2, mul_output_q;
    wire signed [N_COMP-1:0] add_input_q_1, add_input_q_2, add_output_q;
    adder       add (.clk(clk), .rst_n(rst_n), .refresh(add_refresh), .input_q_1(add_input_q_1), .input_q_2(add_input_q_2), .output_q(add_output_q), .overflow(add_overflow));
    multiplier  mul (.clk(clk), .rst_n(rst_n), .refresh(mul_refresh), .input_q_1(mul_input_q_1), .input_q_2(mul_input_q_2), .output_q(mul_output_q), .overflow(mul_overflow));

    mac mac_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .param_type(param_type),
        .len(len),
        .start_addr1(start_addr1),
        .start_addr2(start_addr2),
        .bias_addr(param_addr_map[SINGLE_PARAMS].addr + PATCH_PROJ_BIAS_OFF),
        .activation(activation),
        .busy(busy),
        .done(done),
        .param_data(param_data),
        .int_res_data(int_res_data),
        .params_access_signals(params_access_signals),
        .int_res_access_signals(int_res_access_signals),
        .computation_result(computation_result),
        .add_output_q(add_output_q),
        .add_input_q_1(add_input_q_1),
        .add_input_q_2(add_input_q_2),
        .add_refresh(add_refresh),
        .mult_output_q(mul_output_q),
        .mult_input_q_1(mul_input_q_1),
        .mult_input_q_2(mul_input_q_2),
        .mult_refresh(mul_refresh)
    );

endmodule
