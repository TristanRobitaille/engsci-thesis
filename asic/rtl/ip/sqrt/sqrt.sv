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

`include "../../parameters.svh"
`include "../../types.svh"

module sqrt (
    input wire clk,
    input wire rst_n,
    input wire start,

    input wire signed [N_COMP-1:0] rad_q,
    output logic busy, done, neg_rad,
    output logic signed [N_COMP-1:0] root_q
);

    logic [N_COMP-1:0] x, x_next; // Radicand copy
    logic [N_COMP-1:0] q, q_next; // Intermediate root (quotient)
    logic [N_COMP+1:0] acc, acc_next; // Accumulator (2 bits wider)
    logic [N_COMP+1:0] test_res; // Sign test result (2 bits wider)

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
        neg_rad = (rad_q[N_COMP-1] == 1);
    end

    enum {IDLE, CALC} state;
    localparam CNT_THRESHOLD = (N_COMP+Q) >> 1;
    logic [$clog2(N_COMP)-1:0] count;

    always_ff @ (posedge clk) begin : sqrt_FSM
        if (!rst_n) begin
            busy <= 0;
            done <= 0;
            root_q <= 0;
        end else begin
            case (state)
                IDLE: begin
                    state <= (start & ~neg_rad) ? CALC : IDLE;
                    done <= 0;
                    busy <= (start & ~neg_rad);
                    q <= 0;
                    count <= 0;
                    {acc, x} <= {{N_COMP{1'b0}}, rad_q, 2'b0};
                end
                CALC: begin
                    if (count == CNT_THRESHOLD-1) begin  // we're done
                        busy <= 0;
                        done <= 1;
                        root_q <= q_next;
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

always_ff @ (posedge clk) begin : sqrt_assertions
    assert (~rad_q[N_COMP-1]) else $fatal("Error from sqrt: Negative radicand detected.");
end
endmodule

`endif
