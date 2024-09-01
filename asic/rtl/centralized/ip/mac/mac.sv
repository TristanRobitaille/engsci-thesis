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
        add_io.in_2 <= in_2;
        add_io.start <= 1'b1;
    endtask

    /*----- LOGIC -----*/
    logic input_currently_reading;
    logic [1:0] delay_signal;
    VectorLen_t index;
    CompFx_t compute_temp, compute_temp_2;

    enum logic [2:0] {IDLE, BASIC_MAC, BASIC_MAC_DONE, LINEAR_ACTIVATION_COMP, SWISH_ACTIVATION_ADD,
                      SWISH_ACTIVATION_EXP, SWISH_ACTIVATION_DIV, SWISH_ACTIVATION_FINAL_ADD} state;

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
                        state <= BASIC_MAC;
                        compute_temp <= 'd0;
                        compute_temp_2 <= 'd0;
                        input_currently_reading <= 1'b0;
                        io.busy <= 1'b1;
                        read_int_res(io_extra.start_addr_1, casts.int_res_read_width, casts.int_res_read_format);
                    end else begin
                        io.done <= 1'b0;
                        io.busy <= 1'b0;
                        index <= 'd0;
                        delay_signal <= 2'b0;
                        mult_io.start <= 1'b0;
                        add_io.start <= 1'b0;
                        if (io.busy) begin // Reset their output
                            start_add('d0, 'd0);
                            start_mult('d0, 'd0);
                        end
                    end
                end
                BASIC_MAC: begin
                    // Highly-pipelined MAC
                    if (input_currently_reading == 0) begin // Read input 2
                        index <= index + 1;
                        input_currently_reading <= 1'b1;
                        if (io_extra.param_type == INTERMEDIATE_RES) mult_io.in_2 <= int_res_read.data;
                        else if (io_extra.param_type == MODEL_PARAM) mult_io.in_2 <= param_read.data;

                        if (index <= io_extra.len) begin
                            if (io_extra.param_type == INTERMEDIATE_RES) read_int_res(io_extra.start_addr_2 + IntResAddr_t'(index), casts.int_res_read_width, casts.int_res_read_format);
                            else if (io_extra.param_type == MODEL_PARAM) read_param(ParamAddr_t'(io_extra.start_addr_2 + IntResAddr_t'(index)), casts.params_read_format);
                        end
                    end else begin // Read input 1
                        input_currently_reading <= 1'b0;
                        mult_io.in_1 <= int_res_read.data;
                        if (index <= io_extra.len) read_int_res(io_extra.start_addr_1 + IntResAddr_t'(index), casts.int_res_read_width, casts.int_res_read_format);
                    end

                    if (input_currently_reading == 1'b0 && index != 0) begin
                        mult_io.start <= 1'b1;
                        if (index == (io_extra.len+2)) state <= BASIC_MAC_DONE;
                    end

                    if (input_currently_reading == 1'b1 && index != 0) begin
                        start_add(mult_io.out, add_io.out);
                    end
                end
                BASIC_MAC_DONE: begin
                    if (io_extra.activation == NO_ACTIVATION) begin
                        io.out <= add_io.out;
                        io.done <= 1'b1;
                        state <= IDLE;
                    end else begin // Linear activation or SWISH activation
                        read_param(io_extra.bias_addr, casts.params_read_format);
                        if (delay_signal[0]) begin
                            delay_signal <= (io_extra.activation == LINEAR_ACTIVATION) ? delay_signal : 'd0; // Reset signal
                            state <= (io_extra.activation == LINEAR_ACTIVATION) ? LINEAR_ACTIVATION_COMP : SWISH_ACTIVATION_ADD;
                        end else delay_signal <= {delay_signal[0], 1'b1}; // Need to delay signal by 1 cycle while we wait for bias
                    end
                end
                LINEAR_ACTIVATION_COMP: begin
                    start_add(add_io.out, CompFx_t'(param_read.data));
                    delay_signal <= {delay_signal[0], 1'b0}; // Need to delay signal by 1 cycle while we wait for add
                    if (add_io.start & ~delay_signal[1]) begin
                        io.out <= add_io.out;
                        io.done <= 1'b1;
                        state <= IDLE;
                    end
                end
                SWISH_ACTIVATION_ADD: begin
                    start_add(add_io.out, CompFx_t'(param_read.data));
                    compute_temp <= add_io.out; // Capture result of MAC for last step
                    state <= (add_io.start) ? SWISH_ACTIVATION_EXP : SWISH_ACTIVATION_ADD;
                end
                SWISH_ACTIVATION_EXP: begin
                    compute_temp_2 <= add_io.out;
                    exp_io.start <= 'd1;
                    exp_io.in_1 <= add_out_flipped;
                    state <= SWISH_ACTIVATION_DIV;
                end
                SWISH_ACTIVATION_DIV: begin
                    if (exp_io.done) begin
                        div_io.in_1 <= compute_temp_2;
                        div_io.in_2 <= exp_io.out + /* +1 */('d1 << Q_COMP);
                        div_io.start <= 1'b1;
                    end
                    if (div_io.done) begin
                        state <= SWISH_ACTIVATION_FINAL_ADD;
                        start_add(compute_temp, div_io.out);
                    end
                end
                SWISH_ACTIVATION_FINAL_ADD: begin
                    io.out <= add_io.out;
                    delay_signal <= {delay_signal[0], add_io.start};
                    io.done <= (delay_signal[1]);
                    state <= (delay_signal[1]) ? IDLE : SWISH_ACTIVATION_FINAL_ADD;
                end
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule

`endif
