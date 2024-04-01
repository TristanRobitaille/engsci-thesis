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
  input wire rst_n,
  input wire [WIDTH-1:0] inc,
  output logic [WIDTH-1:0] cnt
);

    logic [WIDTH-1:0] inc_prev;

    always_ff @ (posedge clk) begin : main_cnt_block
        if (!rst_n) begin
            cnt <= 0;
        end else begin
            if (MODE == 0) begin // Posedge-triggered
                cnt <= (inc != inc_prev) ? (cnt + inc) : cnt;
                inc_prev <= inc;
            end else if (MODE == 1) begin // Level-triggered
                cnt <= cnt + inc;
            end
        end
    end

endmodule // counter

`endif
