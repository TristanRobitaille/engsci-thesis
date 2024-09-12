`ifndef _mac_sv_
`define _mac_sv_

/* Notes:
- Fixed-point MAC
- Done signal is a single-cycle pulse
- Uses external adder, multiplier, divider and exp modules to be shared with other modules in the CiM.
- Implements three different activations: No activation (no bias), linear (adds bias to MAC result) and SWISH (adds bias to MAC result, then applies SWISH activation function)
*/

import Defines::*;

module mac (
    input wire clk, rst_n,

    // Memory access signals
    input MemoryInterface.casts         casts,
    output MemoryInterface.output_read  param_read,
    output MemoryInterface.output_read  int_res_read,

    // Compute IO signals
    input ComputeIPInterface.basic_in   io,
    input ComputeIPInterface.extra      io_extra,
    output ComputeIPInterface.basic_out add_io,
    output ComputeIPInterface.basic_out mult_io,
    output ComputeIPInterface.basic_out div_io,
    output ComputeIPInterface.basic_out exp_io
);

    /*----- TASKS -----*/
    task set_default_values();
        param_read.en <= 1'b0;
        int_res_read.en <= 1'b0;
        mult_io.start <= 1'b0;
        add_io.start <= 1'b0;
        exp_io.start <= 1'b0;
        div_io.start <= 1'b0;
    endtask

    task read_int_res(input IntResAddr_t addr, input DataWidth_t width, input FxFormatIntRes_t int_res_format);
        int_res_read.en <= 1'b1;
        int_res_read.addr <= addr;
        int_res_read.data_width <= width;
        int_res_read.format <= int_res_format;
    endtask

    task read_param(input ParamAddr_t addr, input FxFormatParams_t param_format);
        param_read.en <= 1'b1;
        param_read.addr <= addr;
        param_read.format <= param_format;
    endtask

    task start_mult(input CompFx_t in_1, input CompFx_t in_2);
        mult_io.in_1 <= in_1;
        mult_io.in_2 <= in_2;
        mult_io.start <= 1'b1;
    endtask

    task start_add(input CompFx_t in_1, input CompFx_t in_2);
        add_io.in_1 <= in_1;
        add_in_2_reg <= in_2;
        add_io.start <= 1'b1;
    endtask

    task start_div(input CompFx_t in_1, input CompFx_t in_2);
        div_io.in_1 <= in_1;
        div_io.in_2 <= in_2;
        div_io.start <= 1'b1;
    endtask

    task start_exp(input CompFx_t in);
        exp_io.in_1 <= in;
        exp_io.start <= 1'b1;
    endtask

    /*----- BYPASS -----*/
    always_comb begin : adder_bypass
        if ((state == BASIC_MAC_MODEL_PARAM) | (state == BASIC_MAC_INTERMEDIATE_RES)) add_io.in_2 = add_io.out;
        else add_io.in_2 = add_in_2_reg;
    end

    /*----- LOGIC -----*/
    logic input_currently_reading;
    logic [3:0] delay_signal;
    VectorLen_t index;
    CompFx_t compute_temp, compute_temp_2, add_in_2_reg;

    enum logic [3:0] {IDLE, BASIC_MAC_INTERMEDIATE_RES, BASIC_MAC_MODEL_PARAM, BASIC_MAC_DONE, LINEAR_ACTIVATION_COMP,
                      SWISH_ACTIVATION_ADD, SWISH_ACTIVATION_EXP, SWISH_ACTIVATION_DIV, SWISH_ACTIVATION_FINAL_ADD} state;

    // Flip add
    CompFx_t add_out_flipped;
    always_comb begin : add_flip
        add_out_flipped = ~add_io.out + 'd1;
    end

    always_ff @ (posedge clk) begin : mac_fsm
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            set_default_values();

            unique case (state)
                IDLE: begin
                    if (io.start) begin
                        state <= (io_extra.param_type == INTERMEDIATE_RES) ? BASIC_MAC_INTERMEDIATE_RES : BASIC_MAC_MODEL_PARAM;
                        compute_temp <= 'd0;
                        compute_temp_2 <= 'd0;
                        io.busy <= 1'b1;
                    end else begin
                        io.done <= io.busy;
                        io.busy <= 1'b0;
                        index <= 'd0;
                        delay_signal <= 'b0;
                    end
                end
                BASIC_MAC_MODEL_PARAM: begin
                    index <= index + 1;
                    delay_signal[0] <= 1'b1;
                    delay_signal[1] <= delay_signal[0];
                    delay_signal[2] <= mult_io.start;
                    if (index <= io_extra.len) read_int_res(io_extra.start_addr_1 + IntResAddr_t'(index), casts.int_res_read_width, casts.int_res_read_format);
                    if (index <= io_extra.len) read_param(ParamAddr_t'(io_extra.start_addr_2 + IntResAddr_t'(index)), casts.params_read_format);
                    if (delay_signal[1] && (index != 0)) start_mult(param_read.data, int_res_read.data);
                    if (delay_signal[2] & (index != io_extra.len+4)) start_add(mult_io.out, add_io.out);
                    if (index == (io_extra.len+4)) state <= BASIC_MAC_DONE;
                end
                BASIC_MAC_INTERMEDIATE_RES: begin
                    // Pipelined MAC for multiplying two intermediate results vectors
                    delay_signal[2] <= delay_signal[1] | delay_signal[2]; // Latch high 
                    delay_signal[3] <= mult_io.done | delay_signal[3]; // Latch high 

                    if (~delay_signal[0]) begin // Read input 1
                        delay_signal[0] <= 1'b1;
                        delay_signal[1] <= 1'b0;
                        compute_temp <= int_res_read.data;
                        if (index <= io_extra.len) read_int_res(io_extra.start_addr_1 + IntResAddr_t'(index), casts.int_res_read_width, casts.int_res_read_format);
                    end else begin // Read input 2
                        IntResAddr_t addr = (io_extra.direction == HORIZONTAL) ? io_extra.start_addr_2 + IntResAddr_t'(index): io_extra.start_addr_2 + IntResAddr_t'(index*io_extra.matrix_width);
                        index <= index + 1;
                        delay_signal[0] <= 1'b0;
                        delay_signal[1] <= 1'b1;
                        compute_temp_2 <= int_res_read.data;
                        if (index <= io_extra.len) read_int_res(addr, casts.int_res_read_width, casts.int_res_read_format);
                    end

                    if (delay_signal[2] & delay_signal[1]) begin
                        if (index == (io_extra.len+3)) begin
                            state <= BASIC_MAC_DONE;
                            compute_temp <= add_io.out;
                        end else start_mult(compute_temp, compute_temp_2);
                    end

                    if (mult_io.start & delay_signal[3] & (index <= io_extra.len+3)) start_add(mult_io.out, add_io.out);
                end
                BASIC_MAC_DONE: begin
                    if (io_extra.activation == NO_ACTIVATION) begin
                        io.out <= add_io.out;
                        state <= IDLE;
                        start_add('d0, 'd0); // Reset
                        start_mult('d0, 'd0); // Reset
                    end else begin // Linear activation or SWISH activation
                        read_param(io_extra.bias_addr, casts.params_read_format);
                        if (delay_signal[1]) begin
                            delay_signal <= 'd0; // Reset signal
                            state <= (io_extra.activation == LINEAR_ACTIVATION) ? LINEAR_ACTIVATION_COMP : SWISH_ACTIVATION_ADD;
                        end else delay_signal[1:0] <= {delay_signal[0], 1'b1};
                    end
                end
                LINEAR_ACTIVATION_COMP: begin
                    read_param(io_extra.bias_addr, casts.params_read_format);                 
                    start_add(add_io.out, CompFx_t'(param_read.data));
                    delay_signal[0] <= 1; // Need to delay signal by 1 cycle while we wait for param read
                    delay_signal[1] <= delay_signal[0]; // Need to delay signal by 1 cycle while we wait for read
                    delay_signal[2] <= delay_signal[1]; // Need to delay signal by 1 cycle while we wait for add
                    if (add_io.start & delay_signal[2]) begin
                        io.out <= add_io.out;
                        state <= IDLE;
                        start_add('d0, 'd0); // Reset
                        start_mult('d0, 'd0); // Reset
                    end
                end
                SWISH_ACTIVATION_ADD: begin
                    start_add(add_io.out, CompFx_t'(param_read.data));
                    compute_temp <= add_io.out; // Capture result of MAC for last step
                    state <= (add_io.start) ? SWISH_ACTIVATION_EXP : SWISH_ACTIVATION_ADD;
                end
                SWISH_ACTIVATION_EXP: begin
                    delay_signal[0] <= 'd1;
                    if (delay_signal[0]) start_exp(add_out_flipped);
                    if (delay_signal[0]) state <= SWISH_ACTIVATION_DIV;
                end
                SWISH_ACTIVATION_DIV: begin
                    delay_signal[0] <= 'd0;
                    delay_signal[1] <= add_io.start;
                    if (exp_io.start) compute_temp_2 <= add_io.out;
                    if (exp_io.done) start_add(exp_io.out, ('d1 << Q_COMP)); // Add 1 to exp result
                    if (delay_signal[1]) start_div(compute_temp_2, add_io.out);
                    if (div_io.done) begin
                        state <= SWISH_ACTIVATION_FINAL_ADD;
                        start_add(compute_temp, div_io.out);
                    end
                end
                SWISH_ACTIVATION_FINAL_ADD: begin
                    delay_signal[1:0] <= {delay_signal[0], add_io.start};
                    io.out <= add_io.out;
                    if (delay_signal[1]) begin
                        start_add('d0, 'd0); // Reset
                        start_mult('d0, 'd0); // Reset
                        state <= IDLE;
                    end
                end
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule

`endif
