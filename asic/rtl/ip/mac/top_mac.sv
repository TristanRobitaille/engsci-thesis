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
    TEMP_RES_ADDR_T start_addr1 = 'd0;
    TEMP_RES_ADDR_T start_addr2 = 'd0;

    // MAC outputs
    wire busy, done, int_res_mem_access_req, params_mem_access_req;
    wire [N_COMP-1:0] computation_result;

    // Compute module instantiation
    wire add_overflow, mul_overflow, exp_add_refresh, mac_add_refresh, exp_mul_refresh, mac_mul_refresh, div_start, div_done, div_busy, div_dbz, div_overflow, exp_start, exp_done, exp_busy;
    logic add_refresh, mul_refresh;
    COMP_WORD_T add_input_q_1, add_input_q_2, exp_add_input_1, exp_add_input_2, mac_add_input_1, mac_add_input_2, add_output_q, add_out_flipped;
    COMP_WORD_T mul_input_q_1, mul_input_q_2, exp_mul_input_1, exp_mul_input_2, mac_mul_input_1, mac_mul_input_2, mul_output_q;
    COMP_WORD_T div_dividend, div_divisor, div_output_q;
    COMP_WORD_T exp_input, exp_output_q;
    adder       add (.clk(clk), .rst_n(rst_n), .refresh(add_refresh), .input_q_1(add_input_q_1), .input_q_2(add_input_q_2), .output_q(add_output_q), .overflow(add_overflow));
    multiplier  mul (.clk(clk), .rst_n(rst_n), .refresh(mul_refresh), .input_q_1(mul_input_q_1), .input_q_2(mul_input_q_2), .output_q(mul_output_q), .overflow(mul_overflow));
    divider     div (.clk(clk), .rst_n(rst_n), .start(div_start), .dividend(div_dividend), .divisor(div_divisor), .done(div_done), .busy(div_busy), .output_q(div_output_q), .dbz(div_dbz), .overflow(div_overflow));
    exp         exp (.clk(clk), .rst_n(rst_n), .start(exp_start), .adder_output(add_output_q), .adder_refresh(exp_add_refresh), .adder_input_1(exp_add_input_1), .adder_input_2(exp_add_input_2),
                     .mult_output(mul_output_q), .mult_refresh(exp_mul_refresh), .mult_input_1(exp_mul_input_1), .mult_input_2(exp_mul_input_2), .input_q(exp_input), .busy(exp_busy), .done(exp_done), .output_q(exp_output_q));

    // Signal MUXing
    always_latch begin : add_MUX
        if (exp_add_refresh) begin
            add_input_q_1 = exp_add_input_1;
            add_input_q_2 = exp_add_input_2;
        end else if (mac_add_refresh) begin
            add_input_q_1 = mac_add_input_1;
            add_input_q_2 = mac_add_input_2;
        end
        add_refresh = (exp_add_refresh || mac_add_refresh);
    end

    always_latch begin : mult_MUX
        if (exp_mul_refresh) begin
            mul_input_q_1 = exp_mul_input_1;
            mul_input_q_2 = exp_mul_input_2;
        end else if (mac_mul_refresh) begin
            mul_input_q_1 = mac_mul_input_1;
            mul_input_q_2 = mac_mul_input_2;
        end
        mul_refresh = (exp_mul_refresh || mac_mul_refresh);
    end

    always_ff @ (posedge clk) begin : mux_assertions
        assert ($countones({exp_add_refresh, mac_add_refresh}) <= 1) else $fatal("Multiple add_refresh signals are asserted simultaneously!");
        assert ($countones({exp_mul_refresh, mac_mul_refresh}) <= 1) else $fatal("Multiple mul_refresh signals are asserted simultaneously!");
    end

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
        .add_out_flipped(add_out_flipped),
        .add_input_q_1(mac_add_input_1),
        .add_input_q_2(mac_add_input_2),
        .add_refresh(mac_add_refresh),
        .mult_output_q(mul_output_q),
        .mult_input_q_1(mac_mul_input_1),
        .mult_input_q_2(mac_mul_input_2),
        .mult_refresh(mac_mul_refresh),
        .div_busy(div_busy),
        .div_done(div_done),
        .div_output_q(div_output_q),
        .div_dividend(div_dividend),
        .div_divisor(div_divisor),
        .div_start(div_start),
        .exp_busy(exp_busy),
        .exp_done(exp_done),
        .exp_output_q(exp_output_q),
        .exp_input(exp_input),
        .exp_start(exp_start)
    );

    // Miscellanous combinational logic
    always_comb begin : computation_twos_comp_flip
        add_out_flipped = ~add_output_q + 'd1;
    end

endmodule
