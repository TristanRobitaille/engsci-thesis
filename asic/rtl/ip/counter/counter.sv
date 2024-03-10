/* Note:
    In posedge-triggered mode, the counter increments on the rising edge of the increment input. 
    In level-triggered mode, the counter increments as long as the increment input is high.
*/

module counter # (
    parameter int WIDTH = 10,
    parameter int MODE = 0 // 0: posedge-triggered, 1: level-triggered
) (
  input logic clk,
  input logic rst_n,
  input logic inc,
  output logic [WIDTH-1:0] cnt
);

    logic inc_prev;

    always_ff @ (posedge clk or negedge rst_n) begin : main_cnt_block
        if (!rst_n) begin
            cnt <= 0;
        end else begin
            if (MODE == 0) begin // Posedge-triggered
                cnt <= (inc & ~inc_prev) ? (cnt + 'd1) : cnt;
                inc_prev <= inc;
            end else if (MODE == 1) begin // Level-triggered
                cnt <= (inc) ? (cnt + 'd1) : cnt;
            end
        end
    end

endmodule // counter
