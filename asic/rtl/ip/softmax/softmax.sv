`ifndef _softmax_sv_
`define _softmax_sv_

`include "../../types.svh"

module softmax
(
    input wire clk,
    input wire rst_n,

    // Control signals
    input wire start,
    input wire [$clog2(MAC_MAX_LEN+1)-1:0] len,
    input TEMP_RES_ADDR_T start_addr, // Either intermediate res or params
    output logic busy, done,

    // Memory access signals
    MemAccessSignals int_res_access_signals,
    input STORAGE_WORD_T int_res_data,

    // Adder signals
    input COMP_WORD_T add_output_q, add_out_flipped,
    output COMP_WORD_T add_input_q_1, add_input_q_2,
    output logic add_refresh,

    // Multiplier signals
    input COMP_WORD_T mult_output_q, mult_out_flipped,
    output COMP_WORD_T mult_input_q_1, mult_input_q_2,
    output logic mult_refresh,

    // Divide signals
    input wire div_busy, div_done,
    input COMP_WORD_T div_output_q,
    output COMP_WORD_T div_dividend, div_divisor,
    output logic div_start,

    // Exponential signals
    input wire exp_busy, exp_done,
    input COMP_WORD_T exp_output_q, exp_out_flipped,
    output COMP_WORD_T exp_input_q,
    output logic exp_start
);

    logic add_refresh_delayed, exp_done_delayed, read_req_delayed, mult_refresh_delayed;
    logic [1:0] div_loop_timekeep;
    logic [$clog2(SOFTMAX_MAX_LEN+1)-1:0] index;
    COMP_WORD_T compute_temp; // TODO: Consider if it should be shared with other modules in CiM
    enum logic [2:0] {IDLE, EXP, INVERT_DIV, DIV} state;

    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            compute_temp <= 'd0;
        end else begin
            unique case (state)
                IDLE: begin
                    int_res_access_signals.read_req_src[SOFTMAX] <= start;
                    int_res_access_signals.write_req_src[SOFTMAX] <= 1'b0;
                    int_res_access_signals.addr_table[SOFTMAX] <= (start) ? start_addr : int_res_access_signals.addr_table[SOFTMAX];
                    state <= (start) ? EXP : IDLE;
                    busy <= start;
                    done <= 1'b0;
                    index <= 'd0;
                    div_loop_timekeep <= 'd0;
                    mult_refresh <= 1'b0;
                    read_req_delayed <= 1'b1; // Needed when entering EXP
                end
                EXP: begin
                    // Write computed result
                    int_res_access_signals.write_req_src[SOFTMAX] <= exp_done;
                    int_res_access_signals.write_data[SOFTMAX] <= (exp_output_q[N_COMP-1]) ? (~exp_out_flipped[N_STORAGE-1:0]+1'd1) : exp_output_q[N_STORAGE-1:0];

                    // Read next word
                    read_req_delayed <= int_res_access_signals.read_req_src[SOFTMAX];
                    int_res_access_signals.read_req_src[SOFTMAX] <= exp_done_delayed;
                    int_res_access_signals.addr_table[SOFTMAX] <= (exp_done_delayed) ? start_addr + {3'd0, index} : int_res_access_signals.addr_table[SOFTMAX];
                    
                    if ((index == len) && add_refresh_delayed) begin // Done
                        index <= 'd0;
                        state <= INVERT_DIV;
                    end else begin
                        index <= (exp_done && (index < len)) ? index + 'd1 : index;
                    end

                    // Start exponent once we've read new word
                    exp_done_delayed <= exp_done;
                    exp_start <= read_req_delayed && (index < len);
                    exp_input_q <= {{(N_COMP-N_STORAGE){int_res_data[N_STORAGE-1]}}, int_res_data}; // Sign extend

                    // Start add to compute_temp once we've finished exp
                    add_refresh <= exp_done;
                    add_input_q_1 <= compute_temp;
                    add_input_q_2 <= exp_output_q;
                    add_refresh_delayed <= add_refresh;
                    compute_temp <= (add_refresh_delayed) ?  add_output_q : compute_temp;
                end

                INVERT_DIV: begin
                    // Compute 1/compute_temp so we can multiply by that in the next step rather than doing a division for each element in the vector
                    div_start <= (div_loop_timekeep == 'd0);
                    div_dividend <= 'd1 << Q; // 1 in fixed-point
                    div_divisor <= compute_temp;
                    state <= (int_res_access_signals.read_req_src[SOFTMAX]) ? DIV : INVERT_DIV;
                    int_res_access_signals.addr_table[SOFTMAX] <= start_addr;
                    int_res_access_signals.read_req_src[SOFTMAX] <= div_done;
                    div_loop_timekeep <= 'd2; // To start cycle in DIV
                end

                DIV: begin
                    // Note that we arrive here with div_loop_timekeep == 'd2
                    div_loop_timekeep <= (div_loop_timekeep == 'd2) ? 'd0 : div_loop_timekeep + 'd1;
                    index <= (div_loop_timekeep == 'd1) ? index + 'd1 : index;

                    int_res_access_signals.addr_table[SOFTMAX] <= (div_loop_timekeep == 'd2) ? start_addr + {3'd0, index} : int_res_access_signals.addr_table[SOFTMAX];
                    int_res_access_signals.read_req_src[SOFTMAX] <= (div_loop_timekeep == 'd2);

                    int_res_access_signals.write_req_src[SOFTMAX] <= (div_loop_timekeep == 'd1);
                    mult_refresh <= (div_loop_timekeep == 'd0);
                    mult_input_q_1 <= div_output_q;
                    mult_input_q_2 <= {{(N_COMP-N_STORAGE){int_res_data[N_STORAGE-1]}}, int_res_data}; // Sign extend

                    int_res_access_signals.write_data[SOFTMAX] <= (mult_output_q[N_COMP-1]) ? (~mult_out_flipped[N_STORAGE-1:0]+1'd1) : mult_output_q[N_STORAGE-1:0];

                    state <= (index == len) ? IDLE : DIV;
                    done <= (index == len);
                end
                default: begin
                    $fatal("Softmax in unexpected state %d", state);
                end
            endcase
        end
    end

endmodule

`endif
