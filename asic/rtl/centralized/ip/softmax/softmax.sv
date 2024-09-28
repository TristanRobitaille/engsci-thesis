`ifndef _softmax_sv_
`define _softmax_sv_

/* Notes:
- Performs fixed-point Softmax over a given vector.
- Done signal is a single-cycle pulse.
- Uses external adder, multiplier, divider and exp modules to be shared with other modules in the CiM.
*/

import Defines::*;

module softmax (
    input wire clk, rst_n,

    // Memory access signals
    MemoryInterface.casts_in casts,
    MemoryInterface.read_out int_res_read,
    MemoryInterface.write_out int_res_write,

    // Compute IO signals
    ComputeIPInterface.basic_in io,
    ComputeIPInterface.extra_in io_extra,
    ComputeIPInterface.basic_out add_io,
    ComputeIPInterface.basic_out mult_io,
    ComputeIPInterface.basic_out div_io,
    ComputeIPInterface.basic_out exp_io
);

    /*----- MEMORY -----*/
    CompFx_t softmax_exp_int_res [0:NUM_PATCHES+1-1];

    /*----- TASKS -----*/
    task set_default_values();
        int_res_read.en <= 1'b0;
        int_res_write.en <= 1'b0;
        add_io.start <= 1'b0;
        mult_io.start <= 1'b0;
        div_io.start <= 1'b0;
        exp_io.start <= 1'b0;
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

    /*----- LOGIC -----*/
    logic [2:0] delay_line_3b;
    VectorLen_t index;
    CompFx_t compute_temp;
    enum logic [2:0] {IDLE, EXP, INVERT_DIV, DIV} state;

    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            compute_temp <= 'd0;
        end else begin
            set_default_values();
            
            unique case (state)
                IDLE: begin
                    if (io.start) read_int_res(io_extra.start_addr_1, casts.int_res_read_width, casts.int_res_read_format);
                    state <= (io.start) ? EXP : IDLE;
                    index <= 'd0;
                    delay_line_3b <= 'd0;
                    io.busy <= io.start;
                    io.done <= 1'b0;
                    compute_temp <= 'd0;
                end
                EXP: begin
                    IntResAddr_t addr = (delay_line_3b[1]) ? io_extra.start_addr_1 + IntResAddr_t'(index) : int_res_read.addr;
                    if (exp_io.done) softmax_exp_int_res[index[5:0]] <= exp_io.out;
                    if (delay_line_3b[1]) read_int_res(addr, casts.int_res_read_width, casts.int_res_read_format);

                    // Read next word
                    delay_line_3b[2] <= int_res_read.en;
                    
                    if ((index == io_extra.len) & delay_line_3b[0]) begin // Done
                        index <= 'd0;
                        state <= INVERT_DIV;
                    end else index <= (exp_io.done & (index < io_extra.len)) ? index + 'd1 : index;

                    // Start exponent once we've read new word
                    delay_line_3b[1] <= exp_io.done;
                    exp_io.start <= delay_line_3b[2] & (index < io_extra.len);
                    exp_io.in_1 <= int_res_read.data;

                    // Start add to compute_temp once we've finished exp
                    delay_line_3b[0] <= add_io.start;
                    if (exp_io.done) start_add(compute_temp, exp_io.out);
                    if (delay_line_3b[0]) compute_temp <= add_io.out;
                end
                INVERT_DIV: begin
                    // Compute 1/compute_temp so we can multiply by that in the next step rather than doing a division for each element in the vector
                    if (~div_io.busy & ~div_io.done) start_div(('d1 << Q_COMP), compute_temp);
                    if (div_io.done) state <= DIV;
                end
                DIV: begin
                    start_mult(div_io.out, softmax_exp_int_res[index[5:0]]);
                    if (mult_io.done) write_int_res(io_extra.start_addr_1 + IntResAddr_t'(int'(index) - int'(2)), mult_io.out, casts.int_res_write_width, casts.int_res_write_format);

                    delay_line_3b[0] <= mult_io.done;
                    index <= index + 'd1;
                    state <= (index == io_extra.len+2) ? IDLE : DIV;
                    io.done <= (index == io_extra.len+2);
                end
                //synopsys translate_off
                default: begin
                    $fatal("Softmax in unexpected state %d", state);
                end
                //synopsys translate_on
            endcase
        end
    end

endmodule

`endif
