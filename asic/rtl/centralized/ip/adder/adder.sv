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

    assign io.overflow = io.done & (in_1_q[N_COMP-1] == in_2_q[N_COMP-1]) & (in_1_q[N_COMP-1] != sum[N_COMP-1]);

    always_comb begin
        if (io.overflow) begin // Implement AP_SAT_SYM in case of overflow
            if (sum[N_COMP-1]) 
                io.out = {1'b0, {N_COMP-1{1'b1}}}; // Positive overflow --> Saturate to max. positive value
            else 
                io.out = {1'b1, {N_COMP-1{1'b0}}}; // Negative overflow --> Saturate to min. negative value
        end else 
        io.out = sum[N_COMP-1:0]; // Essentially AP_TRN (truncation towards minus infinity)
    end
    
    `ifdef OVERFLOW_WARNING
        always @(posedge clk) begin : adder_assertions
            if (io.overflow) begin
                $display("Overflow detected in adder at time %d (in_1: %d, in_2: %d, out: %d)", $time, io.in_1, io.in_2, io.out);
            end
        end
    `endif
endmodule // adder

`endif
