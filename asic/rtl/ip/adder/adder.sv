/* Note:
- Single-cycle fixed-point adder
- Output is updated when refresh is set
- Provides notice of overflow
*/
module adder (
    input wire clk,
    input wire rst_n,
    input refresh, // Update output when this is set

    // Data in 2's complement format
    input COMP_WORD_T input_q_1,
    input COMP_WORD_T input_q_2,
    output COMP_WORD_T output_q,
    output wire overflow
);

    logic refresh_delayed;
    always_ff @ (posedge clk) begin : refresh_delayed_blk
        refresh_delayed <= refresh;
    end

    always_ff @(posedge clk) begin : adder_logic
        if (!rst_n) begin
            output_q <= '0;
        end else if (refresh) begin
            output_q <= input_q_1 + input_q_2;
        end
    end

    assign overflow = refresh_delayed & (input_q_1[N_COMP-1] == input_q_2[N_COMP-1]) & (input_q_1[N_COMP-1] != output_q[N_COMP-1]);

    // Note: Verilator does not support assertions
    always @(posedge clk) begin : adder_assertions
        if (overflow) begin
            $display("Overflow detected in adder at time %d!", $time);
        end
    end
endmodule // adder
