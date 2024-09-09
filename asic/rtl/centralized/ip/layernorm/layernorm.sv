`ifndef _layernorm_sv_
`define _layernorm_sv_

/* Notes:
- Performs fixed-point LayerNorm over a given vector.
- Seperated in the two-halves (normalization, and centering/scaling)
- Done signal is a single-cycle pulse
- Uses external adder, multiplier, divider and sqrt modules to be shared with other modules in the CiM.
*/

import Defines::*;

module layernorm (
    input wire clk, rst_n,

    // Memory access signals
    input MemoryInterface.casts casts,
    output MemoryInterface.output_read  param_read,
    output MemoryInterface.output_read  int_res_read,
    output MemoryInterface.output_write int_res_write,

    // Compute IO signals
    input ComputeIPInterface.basic_in   io,
    input ComputeIPInterface.extra      io_extra,
    output ComputeIPInterface.basic_out add_io,
    output ComputeIPInterface.basic_out mult_io,
    output ComputeIPInterface.basic_out div_io,
    output ComputeIPInterface.basic_out sqrt_io
);

    /*----- TASKS -----*/
    task set_default_values();
        param_read.en <= 1'b0;
        int_res_read.en <= 1'b0;
        int_res_write.en <= 1'b0;
        add_io.start <= 1'b0;
        mult_io.start <= 1'b0;
        sqrt_io.start <= 1'b0;
        div_io.start <= 1'b0;
    endtask

    task start_add(input CompFx_t in_1, input CompFx_t in_2);
        add_io.in_1 <= in_1;
        add_io.in_2 <= in_2;
        add_io.start <= 1'b1;
    endtask

    task start_mult(input CompFx_t in_1, input CompFx_t in_2);
        mult_io.in_1 <= in_1;
        mult_io.in_2 <= in_2;
        mult_io.start <= 1'b1;
    endtask

    task start_div(input CompFx_t dividend, input CompFx_t divisor);
        div_io.in_1 <= dividend;
        div_io.in_2 <= divisor;
        div_io.start <= 1'b1;
    endtask

    task start_sqrt(input CompFx_t rad);
        sqrt_io.in_1 <= rad;
        sqrt_io.start <= 1'b1;
    endtask

    task read_int_res(input IntResAddr_t addr, input DataWidth_t width, input FxFormatIntRes_t int_res_format);
        int_res_read.en <= 1'b1;
        int_res_read.addr <= addr;
        int_res_read.data_width <= width;
        int_res_read.format <= int_res_format;
    endtask

    task write_int_res(input IntResAddr_t addr, input CompFx_t data, input DataWidth_t width, input FxFormatIntRes_t int_res_format);
        int_res_write.en <= 1'b1;
        int_res_write.addr <= addr;
        int_res_write.data <= data;
        int_res_write.data_width <= width;
        int_res_write.format <= int_res_format;
    endtask

    task read_param(input ParamAddr_t addr, input FxFormatParams_t param_format);
        param_read.en <= 1'b1;
        param_read.addr <= addr;
        param_read.format <= param_format;
    endtask

    // TODO as of 2024/09/08: Change output indexing for second half to increment by EMB_DEPTH instead of 1 to match C++ centralized

    /*----- LOGIC -----*/
    localparam  LEN_FIRST_HALF = 64, // LEN_FIRST_HALF must be a power of 2 since we divide by bit shifting
                LEN_SECOND_HALF = 61;
    logic [2:0] gen_reg_3b;
    VectorLen_t index;
    CompFx_t compute_temp, compute_temp_2, compute_temp_3;

    enum logic [1:0] {MEAN_SUM, VARIANCE, PARTIAL_NORM_FIRST_HALF} loop_sum_type;
    enum logic [3:0] {IDLE, LOOP_SUM, VARIANCE_LOOP, VARIANCE_SQRT, NORM_FIRST_HALF, BETA_LOAD, GAMMA_LOAD, SECOND_HALF_LOOP, DONE} state;

    always_ff @ (posedge clk) begin : layernorm_fsm
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            set_default_values();
            unique case (state)
                IDLE : begin
                    if (io.start) begin
                        state <= (io_extra.half_select == FIRST_HALF) ? LOOP_SUM : GAMMA_LOAD;
                        if (io_extra.half_select == SECOND_HALF) read_param(ParamAddr_t'(io_extra.start_addr_4), casts.params_read_format);
                        io.busy <= 1'b1;
                    end else begin
                        io.done <= 1'b0;
                        io.busy <= 1'b0;
                        loop_sum_type <= MEAN_SUM;
                        index <= 'd0;
                        gen_reg_3b <= 'd0;
                        compute_temp <= 'd0;
                        compute_temp_2 <= 'd0;
                        compute_temp_3 <= 'd0;
                    end
                end
                LOOP_SUM : begin // Can reuse this state whenever we need to sum over LEN_FIRST_HALF addresses
                    if (index < LEN_FIRST_HALF) begin
                        if (gen_reg_3b == 'd0) begin
                            read_int_res(io_extra.start_addr_1 + IntResAddr_t'(index), casts.int_res_read_width, casts.int_res_read_format);
                            if (int_res_read.en) gen_reg_3b <= 'd1;
                        end else if (gen_reg_3b == 'd1) begin
                            CompFx_t data_2 = ((loop_sum_type == VARIANCE) || (loop_sum_type == PARTIAL_NORM_FIRST_HALF)) ? (~compute_temp + 1'd1) : compute_temp; // Can subtract by adding the flipped version (data is 2's complement) 
                            start_add(int_res_read.data, data_2);
                            if (add_io.start) gen_reg_3b <= 'd2;
                        end else if (gen_reg_3b == 'd2) begin
                            if (loop_sum_type == MEAN_SUM) begin // Move to next
                                compute_temp <= add_io.out;
                                gen_reg_3b <= 'd0;
                                index <= index + 'd1;
                            end else if (loop_sum_type == VARIANCE) begin
                                // To continue variance computation, need to square and sum the result
                                start_mult(add_io.out, add_io.out);
                                if (mult_io.start) gen_reg_3b <= 'd3;
                            end else if (loop_sum_type == PARTIAL_NORM_FIRST_HALF) begin
                                CompFx_t input_2 = (mult_io.start) ? mult_io.in_2 : add_io.out; // Only output when mult_io.start is 0 to avoid false positive transient overflows
                                start_mult(compute_temp_3, input_2);
                                gen_reg_3b <= (mult_io.start) ? 'd4 : 'd2;
                            end
                        end else if (gen_reg_3b == 'd3) begin
                            start_add(compute_temp_2, mult_io.out);
                            if (add_io.start) gen_reg_3b <= 'd4;
                        end else if (gen_reg_3b == 'd4) begin
                            gen_reg_3b <= 'd0;
                            index <= index + 'd1;
                            compute_temp_2 <= add_io.out;
                            if (mult_io.start) write_int_res(io_extra.start_addr_2 + IntResAddr_t'(index), mult_io.out, casts.int_res_write_width, casts.int_res_write_format);
                        end
                    end else begin
                        index <= 'd0;
                        if (loop_sum_type == MEAN_SUM) begin
                            state <= VARIANCE_LOOP;
                            compute_temp <= (compute_temp >>> $clog2(LEN_FIRST_HALF)); // Divide by LEN_FIRST_HALF (shift right by log2(LEN_FIRST_HALF) bits) to obtain mean
                        end else if (loop_sum_type == VARIANCE) begin
                            compute_temp_2 <= (compute_temp_2 >>> $clog2(LEN_FIRST_HALF)); // Divide by LEN_FIRST_HALF (shift right by log2(LEN_FIRST_HALF) bits to obtain variance
                            state <= VARIANCE_SQRT;
                            start_mult(compute_temp_3, compute_temp_3);
                        end else if (loop_sum_type == PARTIAL_NORM_FIRST_HALF) begin
                            state <= DONE;
                        end
                    end
                end
                VARIANCE_LOOP : begin
                    loop_sum_type <= VARIANCE;
                    state <= LOOP_SUM;
                end
                VARIANCE_SQRT : begin
                    if (~sqrt_io.busy & ~sqrt_io.done) start_sqrt(compute_temp_2);
                    state <= (sqrt_io.done) ? NORM_FIRST_HALF : VARIANCE_SQRT;
                end
                NORM_FIRST_HALF : begin
                    // Compute 1.0/variance such that we can multiply by that instead of dividing
                    compute_temp_3 <= div_io.out;
                    state <= (div_io.done) ? LOOP_SUM : NORM_FIRST_HALF;
                    if (~div_io.done & ~div_io.busy) start_div(('d1 << Q_COMP), sqrt_io.out);
                    loop_sum_type <= PARTIAL_NORM_FIRST_HALF;
                end
                // Following states are for second-half
                GAMMA_LOAD: begin
                    gen_reg_3b <= gen_reg_3b + 'd1;
                    if (gen_reg_3b == 'd1) begin
                        compute_temp_2 <= param_read.data; // Gamma
                        read_param(ParamAddr_t'(io_extra.start_addr_3), casts.params_read_format);
                        state <= BETA_LOAD;
                    end
                end
                BETA_LOAD: begin
                    gen_reg_3b <= 'd0;
                    if (gen_reg_3b == 'd0) begin
                        compute_temp_3 <= param_read.data; // Beta
                        state <= SECOND_HALF_LOOP;
                    end
                end
                SECOND_HALF_LOOP: begin
                    if (index < LEN_SECOND_HALF) begin
                        if (gen_reg_3b == 'd0) begin
                            read_int_res(io_extra.start_addr_2 + IntResAddr_t'(index*EMB_DEPTH), casts.int_res_read_width, casts.int_res_read_format);
                            if (int_res_read.en) gen_reg_3b <= 'd1;
                        end else if (gen_reg_3b == 'd1) begin
                            start_mult(compute_temp_2, int_res_read.data); // Gamma
                            if (mult_io.start) gen_reg_3b <= 'd2;
                        end else if (gen_reg_3b == 'd2) begin
                            start_add(mult_io.out, compute_temp_3); // Beta
                            if (add_io.start) gen_reg_3b <= 'd3;
                        end else if (gen_reg_3b == 'd3) begin
                            write_int_res(io_extra.start_addr_2 + IntResAddr_t'(index*EMB_DEPTH), add_io.out, casts.int_res_write_width, casts.int_res_write_format);
                            gen_reg_3b <= 'd4;
                        end else begin
                            gen_reg_3b <= 'd0;
                            index <= index + 'd1;
                        end
                    end else begin
                        state <= DONE;
                    end
                end
                DONE: begin
                    io.done <= 1'b1;
                    state <= IDLE;
                end
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule

`endif
