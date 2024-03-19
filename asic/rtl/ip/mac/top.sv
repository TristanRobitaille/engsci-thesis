`include "../adder/adder.sv"
`include "../multiplier/multiplier.sv"

module top # (
    parameter N_COMP = 22, // 22b total
    parameter N_STORAGE = 16, // 16b total
    parameter Q = 10, // 10b fractional
    parameter TEMP_RES_STORAGE_SIZE_CIM = 848, // # of elements in the intermediate result storage
    parameter PARAMS_STORAGE_SIZE_CIM = 528, // # of elements in the model parameters storage
    parameter MAX_LEN = 64 // Maximum length of the vectors to MAC
) (
    input wire clk, rst_n
);

    // CiM memory
    logic [N_STORAGE-1:0] params [PARAMS_STORAGE_SIZE_CIM-1:0];
    logic [N_STORAGE-1:0] intermediate_res [TEMP_RES_STORAGE_SIZE_CIM-1:0];

    logic [N_STORAGE-1:0] param_data, intermediate_res_data = 'd0;
    wire [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] param_addr;
    wire [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] intermediate_res_addr;
    always_comb begin
    end
    always_ff @ (posedge clk) begin : memory_ctrl
        if (!rst_n) begin
            param_data <= 'd0;
            intermediate_res_data <= 'd0;
        end else begin
            param_data <= (params_mem_access_req) ? params[param_addr] : param_data;
            intermediate_res_data <= (int_res_mem_access_req) ? intermediate_res[intermediate_res_addr] : intermediate_res_data;
        end
    end

    // Control signals
    logic start, param_type;
    logic [1:0] activation = 'd0;
    logic [$clog2(MAX_LEN+1)-1:0] len = 'd0;
    logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] start_addr1 = 'd0;
    logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] start_addr2 = 'd0;

    // Compute signals
    logic signed [N_COMP-1:0] compute_temp, compute_temp_2, computation_result;

    // MAC outputs
    wire busy, done, int_res_mem_access_req, params_mem_access_req;

    // Adder and multiplier
    wire add_overflow, add_refresh, mul_refresh;
    wire signed [N_COMP-1:0] mul_input_q_1, mul_input_q_2, mul_output_q;
    wire signed [N_COMP-1:0] add_input_q_1, add_input_q_2, add_output_q;
    adder       #(.N(N_COMP))           add (.clk(clk), .rst_n(rst_n), .refresh(add_refresh), .input_q_1(add_input_q_1), .input_q_2(add_input_q_2), .output_q(add_output_q), .overflow(add_overflow));
    multiplier  #(.N(N_COMP), .Q(Q))    mul (.clk(clk), .rst_n(rst_n), .refresh(mul_refresh), .input_q_1(mul_input_q_1), .input_q_2(mul_input_q_2), .output_q(mul_output_q));

    mac #(
        .N_STORAGE(N_STORAGE), .N_COMP(N_COMP),
        .TEMP_RES_STORAGE_SIZE_CIM(TEMP_RES_STORAGE_SIZE_CIM),
        .MAX_LEN(MAX_LEN)
    ) mac_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .param_type(param_type),
        .len(len),
        .start_addr1(start_addr1),
        .start_addr2(start_addr2),
        .activation(activation),
        .busy(busy),
        .done(done),
        .int_res_mem_access_req(int_res_mem_access_req),
        .params_mem_access_req(params_mem_access_req),
        .param_data(param_data),
        .intermediate_res_data(intermediate_res_data),
        .param_addr(param_addr),
        .intermediate_res_addr(intermediate_res_addr),
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
