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
    `ifdef DISTRIBUTED_ARCH
        input wire [WIDTH-1:0] inc,
    `elsif CENTRALIZED_ARCH
        input logic inc,
    `endif
    output logic [WIDTH-1:0] cnt
);

    `ifdef DISTRIBUTED_ARCH
        logic [WIDTH-1:0] inc_prev;
    `elsif CENTRALIZED_ARCH
        logic inc_prev;
    `endif

    always_ff @ (posedge clk) begin : main_cnt_block
        if (!rst_n) begin
            cnt <= 0;
        end else begin
            if (MODE == 0) begin // Posedge-triggered
                `ifdef DISTRIBUTED_ARCH
                    cnt <= (inc != inc_prev) ? (cnt + inc) : cnt;
                `elsif CENTRALIZED_ARCH
                    cnt <= (inc != inc_prev) ? (cnt + 1) : cnt;
                `endif
                inc_prev <= inc;
            end else if (MODE == 1) begin // Level-triggered
                `ifdef DISTRIBUTED_ARCH
                    cnt <= cnt + inc;
                `elsif CENTRALIZED_ARCH
                    cnt <= (inc) ? (cnt + 1) : cnt;
                `endif
            end
        end
    end

endmodule // counter

`endif
