`include "../../types.svh"
`include "../../cim/cim.svh"

module top_softmax # () (
    input wire clk, rst_n
);

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
            if (int_res_access_signals.write_req_src[SOFTMAX]) begin
                int_res[int_res_access_signals.addr_table[SOFTMAX]] <= int_res_access_signals.write_data[SOFTMAX];
            end

            // Read
            if (int_res_access_signals.read_req_src[SOFTMAX]) begin
                int_res_data <= int_res[int_res_access_signals.addr_table[SOFTMAX]];
            end     
        end
    end

    // Instantiation
    logic exp_add_refresh, softmax_add_refresh, add_refresh, add_overflow, exp_mul_refresh, softmax_mul_refresh;
    logic exp_busy, exp_done, exp_start;
    logic mul_refresh, mul_overflow;
    logic div_start, div_done, div_busy, div_dbz, div_overflow;
    logic start, done, busy;
    logic [$clog2(SOFTMAX_MAX_LEN+1)-1:0] len;
    TEMP_RES_ADDR_T start_addr;
    COMP_WORD_T add_input_q_1, add_input_q_2, add_out_flipped, softmax_add_input_1, softmax_add_input_2;
    COMP_WORD_T mul_input_q_1, mul_input_q_2, mul_output_q, mult_out_flipped, softmax_mul_input_1, softmax_mul_input_2;
    COMP_WORD_T add_output_q, div_output_q, exp_output_q;
    COMP_WORD_T exp_add_input_1, exp_add_input_2, exp_mul_input_1, exp_mul_input_2;
    COMP_WORD_T div_dividend, div_divisor;
    COMP_WORD_T exp_input, exp_out_flipped;

    adder       add (.clk(clk), .rst_n(rst_n), .refresh(add_refresh), .input_q_1(add_input_q_1), .input_q_2(add_input_q_2), .output_q(add_output_q), .overflow(add_overflow));
    multiplier  mul (.clk(clk), .rst_n(rst_n), .refresh(mul_refresh), .input_q_1(mul_input_q_1), .input_q_2(mul_input_q_2), .output_q(mul_output_q), .overflow(mul_overflow));    
    divider     div (.clk(clk), .rst_n(rst_n), .start(div_start), .dividend(div_dividend), .divisor(div_divisor), .done(div_done), .busy(div_busy), .output_q(div_output_q), .dbz(div_dbz), .overflow(div_overflow));
    exp         exp (.clk(clk), .rst_n(rst_n), .start(exp_start), .adder_output(add_output_q), .adder_refresh(exp_add_refresh), .adder_input_1(exp_add_input_1), .adder_input_2(exp_add_input_2),
                     .mult_output(mul_output_q), .mult_refresh(exp_mul_refresh), .mult_input_1(exp_mul_input_1), .mult_input_2(exp_mul_input_2), .input_q(exp_input), .busy(exp_busy), .done(exp_done), .output_q(exp_output_q));

    softmax     softmax(.clk(clk), .rst_n(rst_n), .start(start), .len(len), .start_addr(start_addr), .busy(busy), .done(done),
                        .int_res_access_signals(int_res_access_signals), .int_res_data(int_res_data),
                        .add_output_q(add_output_q), .add_out_flipped(add_out_flipped), .add_input_q_1(softmax_add_input_1), .add_input_q_2(softmax_add_input_2), .add_refresh(softmax_add_refresh),
                        .mult_output_q(mul_output_q), .mult_out_flipped(mult_out_flipped), .mult_input_q_1(softmax_mul_input_1), .mult_input_q_2(softmax_mul_input_2), .mult_refresh(softmax_mul_refresh),
                        .div_busy(div_busy), .div_done(div_done), .div_output_q(div_output_q), .div_dividend(div_dividend), .div_divisor(div_divisor), .div_start(div_start),
                        .exp_busy(exp_busy), .exp_done(exp_done), .exp_output_q(exp_output_q), .exp_out_flipped(exp_out_flipped), .exp_input_q(exp_input), .exp_start(exp_start));

    // Signal MUXing
    always_latch begin : add_input_mux
        if (exp_add_refresh) begin
            add_input_q_1 = exp_add_input_1;
            add_input_q_2 = exp_add_input_2;
        end else if (softmax_add_refresh) begin
            add_input_q_1 = softmax_add_input_1;
            add_input_q_2 = softmax_add_input_2;
        end
        add_refresh = (exp_add_refresh | softmax_add_refresh);

    end
    always_latch begin : mul_input_mux
        if (exp_mul_refresh) begin
            mul_input_q_1 = exp_mul_input_1;
            mul_input_q_2 = exp_mul_input_2;
        end else if (softmax_mul_refresh) begin
            mul_input_q_1 = softmax_mul_input_1;
            mul_input_q_2 = softmax_mul_input_2;
        end
        mul_refresh = (exp_mul_refresh | softmax_mul_refresh);
    end

    always_ff @ (posedge clk) begin : compute_mux_assertions
        assert ($countones({exp_add_refresh, softmax_add_refresh}) <= 1) else $fatal("Multiple add_refresh signals are asserted simultaneously!");
        assert ($countones({exp_mul_refresh, softmax_mul_refresh}) <= 1) else $fatal("Multiple mul_refresh signals are asserted simultaneously!");
    end

    always_comb begin: comp_out_flipped
        add_out_flipped = ~add_output_q + 'd1;
        mult_out_flipped = ~mul_output_q + 'd1;
        exp_out_flipped = ~exp_output_q + 'd1;
    end

endmodule
