`include "../../types.svh"
`include "../../cim/cim.svh"

module top_layernorm # () (
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
            // Write
            if (params_access_signals.write_req_src[LAYERNORM]) begin
                params[params_access_signals.addr_table[LAYERNORM]] <= params_access_signals.write_data[LAYERNORM];
            end
            if (int_res_access_signals.write_req_src[LAYERNORM]) begin
                int_res[int_res_access_signals.addr_table[LAYERNORM]] <= int_res_access_signals.write_data[LAYERNORM];
            end

            // Read
            if (params_access_signals.read_req_src[LAYERNORM]) begin
                param_data <= params[params_access_signals.addr_table[LAYERNORM]];
            end
            if (int_res_access_signals.read_req_src[LAYERNORM]) begin
                int_res_data <= int_res[int_res_access_signals.addr_table[LAYERNORM]];
            end     
        end
    end

    // Control signals
    logic start, half_select;
    logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] start_addr = 'd0;

    // LayerNorm outputs
    wire busy, done, int_res_mem_access_req, params_mem_access_req;
    wire [N_COMP-1:0] computation_result;

    // Compute modules
    wire add_overflow, mul_overflow, add_refresh, mul_refresh, div_start, div_done, div_busy, div_dbz, div_overflow, sqrt_neg_rad, sqrt_start, sqrt_done, sqrt_busy;
    wire signed [N_COMP-1:0] mul_input_q_1, mul_input_q_2, mul_output_q;
    wire signed [N_COMP-1:0] add_input_q_1, add_input_q_2, add_output_q;
    wire signed [N_COMP-1:0] div_dividend, divisor, div_output_q;
    logic signed [N_COMP-1:0] div_out_flipped;
    wire signed [N_COMP-1:0] sqrt_rad_q, sqrt_root_q;
    adder       add (.clk(clk), .rst_n(rst_n), .refresh(add_refresh), .input_q_1(add_input_q_1), .input_q_2(add_input_q_2), .output_q(add_output_q), .overflow(add_overflow));
    multiplier  mul (.clk(clk), .rst_n(rst_n), .refresh(mul_refresh), .input_q_1(mul_input_q_1), .input_q_2(mul_input_q_2), .output_q(mul_output_q), .overflow(mul_overflow));
    divider     div (.clk(clk), .rst_n(rst_n), .start(div_start), .dividend(div_dividend), .divisor(divisor), .done(div_done), .busy(div_busy), .output_q(div_output_q), .dbz(div_dbz), .overflow(div_overflow));
    sqrt        sqrt(.clk(clk), .rst_n(rst_n), .start(sqrt_start), .done(sqrt_done), .busy(sqrt_busy), .rad_q(sqrt_rad_q), .root_q(sqrt_root_q), .neg_rad(sqrt_neg_rad));

    layernorm layernorm_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .half_select(half_select),
        .start_addr(start_addr),
        .busy(busy),
        .done(done),
        .int_res_access_signals(int_res_access_signals),
        .params_access_signals(params_access_signals),
        .param_data(param_data),
        .int_res_data(int_res_data),
        .add_output_q(add_output_q),
        .add_input_q_1(add_input_q_1),
        .add_input_q_2(add_input_q_2),
        .mult_refresh(mul_refresh),
        .mult_output_q(mul_output_q),
        .mult_input_q_1(mul_input_q_1),
        .mult_input_q_2(mul_input_q_2),
        .add_refresh(add_refresh),
        .div_done(div_done),
        .div_busy(div_busy),
        .div_output_q(div_output_q),
        .div_output_flipped(div_out_flipped),
        .div_dividend(div_dividend),
        .div_divisor(divisor),
        .div_start(div_start),
        .sqrt_done(sqrt_done),
        .sqrt_busy(sqrt_busy),
        .sqrt_start(sqrt_start),
        .sqrt_rad_q(sqrt_rad_q),
        .sqrt_root_q(sqrt_root_q)
    );

    always_comb begin : computation_twos_comp_flip
        div_out_flipped = ~div_output_q + 'd1;
    end

endmodule;
