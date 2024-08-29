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
    logic [N_COMP-1:0] x, x_next; // Radicand copy
    logic [N_COMP-1:0] q, q_next; // Intermediate root (quotient)
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
                    io.busy <= (io.start & ~neg_rad);
                    q <= 0;
                    count <= 0;
                    {acc, x} <= {{N_COMP{1'b0}}, io.in_1, 2'b0};
                    state <= (io.start & ~neg_rad) ? CALC : IDLE;
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


//     logic [N_COMP-1:0] x, x_next;    // radicand copy
//     logic [N_COMP-1:0] q, q_next;    // intermediate root (quotient)
//     logic [N_COMP+1:0] ac, ac_next;  // accumulator (2 bits wider)
//     logic [N_COMP+1:0] test_res;     // sign test result (2 bits wider)

//     typedef logic [$clog2(N_COMP)-1:0] count_t;
//     localparam ITER = (N_COMP + Q_COMP) >> 1;
//     count_t i;            // iteration counter

//     always_comb begin
//         test_res = ac - {q, 2'b01};
//         if (test_res[N_COMP+1] == 0) begin  // test_res ≥0? (check MSB)
//             {ac_next, x_next} = {test_res[N_COMP-1:0], x, 2'b0};
//             q_next = {q[N_COMP-2:0], 1'b1};
//         end else begin
//             {ac_next, x_next} = {ac[N_COMP-1:0], x, 2'b0};
//             q_next = q << 1;
//         end
//     end

//     always_ff @(posedge clk) begin
//         if (io.start) begin
//             io.busy <= 1;
//             io.done <= 0;
//             i <= 0;
//             q <= 0;
//             {ac, x} <= {{N_COMP{1'b0}}, io.in_1, 2'b0};
//         end else if (io.busy) begin
//             if (i == count_t'(ITER-1)) begin  // we're done
//                 io.busy <= 0;
//                 io.done <= 1;
//                 io.out <= q_next;
//             // rem <= ac_next[N_COMP+1:2];  // undo final shift
//             end else begin  // next iteration
//                 i <= i + 1;
//                 x <= x_next;
//                 ac <= ac_next;
//                 q <= q_next;
//             end
//         end
//     end
// endmodule
`endif
