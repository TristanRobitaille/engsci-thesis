`ifndef _counter_sv_
`define _counter_sv_

/* Note:
    In posedge-triggered mode, the counter increments on the rising edge of the increment input. 
    In level-triggered mode, the counter increments as long as the increment input is high.
*/

module counter # (
    parameter int WIDTH = 10,
    parameter int MODE = 0 // 0: posedge-triggered, 1: level-triggered
) (
    input wire clk,
    CounterInterface sig
);

    logic inc_prev;

    always_ff @ (posedge clk) begin : main_cnt_block
        if (!sig.rst_n) begin
            sig.cnt <= 0;
        end else begin
            if (MODE == 0) begin // Posedge-triggered
                sig.cnt <= (sig.inc & ~inc_prev) ? (sig.cnt + 1) : sig.cnt;
                inc_prev <= sig.inc;
            end else if (MODE == 1) begin // Level-triggered
                sig.cnt <= (sig.inc) ? (sig.cnt + 1) : sig.cnt;
            end
        end
    end

endmodule // counter

`endif