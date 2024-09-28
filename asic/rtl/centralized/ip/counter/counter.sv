`ifndef _counter_sv_
`define _counter_sv_

/* Note:
    In posedge-triggered mode, the counter increments on the rising edge of the increment input. 
    In level-triggered mode, the counter increments as long as the increment input is high.
*/

import Defines::*;

module counter # (
    parameter int WIDTH = 10,
    parameter CounterMode_t MODE = POSEDGE_TRIGGERED // 0: posedge-triggered, 1: level-triggered
) (
    input wire clk,
    CounterInterface.data_in sig
);

    logic inc_prev;

    always_ff @ (posedge clk) begin : main_cnt_block
        if (~sig.rst_n) begin
            sig.cnt <= 0;
        end else begin
            if (MODE == POSEDGE_TRIGGERED) begin // Posedge-triggered
                sig.cnt <= (sig.inc & ~inc_prev) ? (sig.cnt + 1) : sig.cnt;
                inc_prev <= sig.inc;
            end else if (MODE == LEVEL_TRIGGERED) begin // Level-triggered
                sig.cnt <= (sig.inc) ? (sig.cnt + 1) : sig.cnt;
            end
        end
    end

endmodule // counter

`endif
