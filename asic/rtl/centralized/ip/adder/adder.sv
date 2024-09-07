`ifndef _adder_sv_
`define _adder_sv_

/* Note:
- Single-cycle fixed-point adder
- Output is updated when start is set
- Provides notice of overflow
*/

module adder (
    input wire clk, rst_n,
    input ComputeIPInterface.basic_in io
);

    CompFx_t in_1_q, in_2_q;
    logic [N_COMP:0] sum;
    always_ff @(posedge clk) begin : adder_logic
        if (!rst_n) begin
            sum <= '0;
            io.done <= 1'b0;
        end else begin
            sum <= (io.start) ? (io.in_1 + io.in_2) : sum;
            io.done <= io.start;
            in_1_q <= io.in_1;
            in_2_q <= io.in_2;
        end
    end

    assign io.out = sum[N_COMP-1:0]; // Essentially AP_TRN (truncation towards minus infinity)
    assign io.overflow = io.done & (in_1_q[N_COMP-1] == in_2_q[N_COMP-1]) & (in_1_q[N_COMP-1] != io.out[N_COMP-1]);

    // Note: Verilator does not support assertions well
    always @(posedge clk) begin : adder_assertions
        if (io.overflow) begin
            $display("Overflow detected in adder at time %d (in_1: %d, in_2: %d, out: %d)", $time, io.in_1, io.in_2, io.out);
        end
    end
endmodule // adder

`endif
