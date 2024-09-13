`ifndef _sqrt_sv_
`define _sqrt_sv_

/* Note:
    - A pulse on start will initiate the sqrt process.
    - Signal done (pulse) is asserted when the computation is complete.
    - Signal busy is asserted during the sqrt process.
    - Signal neg_rad is asserted when the input radicand is negative, and sqrt computation is avoided.
    - Inputs must remain stable until done is asserted.
    - Computation time is ((N+Q)//2 + 1) cycles, where N is the total number of bits, and Q is the number of fractional bits.
*/

module sqrt (
    input wire clk, rst_n,
    input ComputeIPInterface.basic_in io
);

    logic neg_rad;      // Negative radicand flag
    CompFx_t x, x_next; // Radicand copy
    CompFx_t q, q_next; // Intermediate root (quotient)
    logic [N_COMP+1:0] acc, acc_next; // Accumulator (2 bits wider)
    logic [N_COMP+1:0] test_res;      // Sign test result (2 bits wider)

    always_comb begin : sqrt_compute
        test_res = acc - {q, 2'b01};
        if (test_res[N_COMP+1] == 0) begin
            {acc_next, x_next} = {test_res[N_COMP-1:0], x, 2'b0};
            q_next = {q[N_COMP-2:0], 1'b1};
        end else begin
            {acc_next, x_next} = {acc[N_COMP-1:0], x, 2'b0};
            q_next = q << 1;
        end
    end

    always_comb begin : neg_radicand_check
        neg_rad = (io.in_1[N_COMP-1] == 1);
        io.overflow = io.done && io.out[N_COMP-1]; // Overflow occurs when the output is negative since sqrt(x) is always positive
    end

    enum {IDLE, CALC} state;
    typedef logic [$clog2(N_COMP)-1:0] count_t;
    localparam CNT_THRESHOLD = (N_COMP + Q_COMP) >> 1;
    count_t count;

    always_ff @ (posedge clk) begin : sqrt_FSM
        if (!rst_n) begin
            io.busy <= 0;
            io.done <= 0;
            io.out <= 0;
            q <= 0;
            count <= 0;
        end else begin
            case (state)
                IDLE: begin
                    io.done <= 0;
`ifdef ALLOW_NEG_RAD_SQRT
                    io.busy <= io.start;
`else
                    io.busy <= (io.start & ~neg_rad);
`endif
                    q <= 0;
                    count <= 0;
                    {acc, x} <= {{N_COMP{1'b0}}, io.in_1, 2'b0};
`ifdef ALLOW_NEG_RAD_SQRT
                    state <= (io.start) ? CALC : IDLE;
`else
                    state <= (io.start & ~neg_rad) ? CALC : IDLE;
`endif
                end
                CALC: begin
                    if (count == count_t'(CNT_THRESHOLD-1)) begin  // We're done
                        io.busy <= 0;
                        io.done <= 1;
                        io.out <= q_next;
                        state <= IDLE;
                    end else begin  // next iteration
                        count <= count + 1;
                        x <= x_next;
                        acc <= acc_next;
                        q <= q_next;
                    end
                end
            endcase
        end
    end

    // synopsys translate_off
    always_ff @ (posedge clk) begin : sqrt_assertions
        assert (~io.in_1[N_COMP-1]) else $fatal("Error from sqrt: Negative radicand detected.");
    end
    // synopsys translate_on
endmodule

`endif
