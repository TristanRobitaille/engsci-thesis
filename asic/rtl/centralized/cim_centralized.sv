module cim_centralized #()
(
    input wire clk, rst_n
);

// ----- TASKS ----- //
task automatic set_default_values();
    // Sets the default value for signals that do not persist cycle-to-cycle
    cnt_4b_rst_n = 1'b1;
    cnt_4b_inc = 1'b0;
    cnt_7b_rst_n = 1'b1;
    cnt_7b_inc = 1'b0;
    cnt_9b_rst_n = 1'b1;
    cnt_9b_inc = 1'b0;
endtask

task automatic reset();
    cnt_4b_rst_n = 1'b0;
    cnt_4b_inc = 1'b0;
    cnt_7b_rst_n = 1'b0;
    cnt_7b_inc = 1'b0;
    cnt_9b_rst_n = 1'b0;
    cnt_9b_inc = 1'b0;
endtask

// ----- INSTANTIATION ----- //
logic cnt_4b_rst_n, cnt_4b_inc;
logic [3:0] cnt_4b_cnt;
counter #(.WIDTH(4), .MODE(0)) cnt_4b (.clk(clk), .rst_n(cnt_4b_rst_n), .inc(cnt_4b_inc), .cnt(cnt_4b_cnt));

logic cnt_7b_rst_n, cnt_7b_inc;
logic [6:0] cnt_7b_cnt;
counter #(.WIDTH(7), .MODE(0)) cnt_7b (.clk(clk), .rst_n(cnt_7b_rst_n), .inc(cnt_7b_inc), .cnt(cnt_7b_cnt));

logic cnt_9b_rst_n, cnt_9b_inc;
logic [8:0] cnt_9b_cnt;
counter #(.WIDTH(9), .MODE(0)) cnt_9b (.clk(clk), .rst_n(cnt_9b_rst_n), .inc(cnt_9b_inc), .cnt(cnt_9b_cnt));

// ----- FSM ----- //
always_ff @ (posedge clk) begin : main_fsm
    if (~rst_n) begin
        reset();
    end else begin
        set_default_values();
    end
end

endmodule