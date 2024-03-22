module layernorm (
    input wire clk,
    input wire rst_n,

    // Control signals
    input wire start, half_select,
    input wire [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] start_addr,
    output logic busy, done,

    // Memory access signals
    input MemAccessSignals int_res_access_signals,
    input MemAccessSignals params_access_signals,
    input wire signed [N_STORAGE-1:0] param_data,
    input wire signed [N_STORAGE-1:0] int_res_data,

    // Adder signals
    input wire signed [N_COMP-1:0] add_output_q,
    output logic signed [N_COMP-1:0] add_input_q_1, add_input_q_2,
    output logic add_refresh,

    // Multiplier signals
    input wire signed [N_COMP-1:0] mult_output_q,
    output logic signed [N_COMP-1:0] mult_input_q_1, mult_input_q_2,
    output logic mult_refresh,

    // Divider signals
    input wire div_done, div_busy,
    input wire signed [N_COMP-1:0] div_output_q, div_output_flipped,
    output logic div_start,
    output logic signed [N_COMP-1:0] div_dividend, div_divisor,

    // Sqrt signals
    input wire sqrt_done, sqrt_busy,
    input wire signed [N_COMP-1:0] sqrt_root_q,
    output logic sqrt_start,
    output logic signed [N_COMP-1:0] sqrt_rad_q
);

    /*----- LOGIC -----*/
    localparam len = 64; // len must be a power of 2 since we divide by bit shifting
    logic [2:0] gen_reg_3b;
    logic [$clog2(EMB_DEPTH+1)-1:0] index;
    logic signed [N_COMP-1:0] compute_temp, compute_temp_2, compute_temp_3; // TODO: Consider if it should be shared with other modules in CiM

    enum logic [1:0] {MEAN_SUM, VARIANCE, PARTIAL_NORM_FIRST_HALF} loop_sum_type;
    enum logic [2:0] {IDLE, LOOP_SUM, VARIANCE_LOOP, VARIANCE_SQRT, NORM_FIRST_HALF, NORM_SECOND_HALF, DONE} state;

    always_ff @ (posedge clk) begin : layernorm_fsm
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            unique case (state)
                IDLE : begin
                    if (start) begin
                        state <= (half_select == FIRST_HALF) ? LOOP_SUM : NORM_SECOND_HALF;
                        busy <= 1'b1;
                    end else begin
                        done <= 1'b0;
                        loop_sum_type <= MEAN_SUM;
                        index <= 'd0;
                        gen_reg_3b <= 'd0;
                        compute_temp <= 'd0;
                        compute_temp_2 <= 'd0;
                        compute_temp_3 <= 'd0;
                        add_refresh <= 1'b0;
                        mult_refresh <= 1'b0;
                        div_start <= 1'b0;
                        sqrt_start <= 1'b0;
                        busy <= 1'b0;
                    end
                end
                LOOP_SUM : begin // Can reuse this state whenever we need to sum over len addresses
                    if (index < len) begin
                        if (gen_reg_3b == 'd0) begin // Set int_res addr and read request
                            int_res_access_signals.addr_table[LAYERNORM] <= start_addr + {3'd0, index};
                            int_res_access_signals.read_req_src[LAYERNORM] <= 1'b1;
                            if (int_res_access_signals.read_req_src[LAYERNORM] == 1'b1) gen_reg_3b <= 'd1;
                        end else if (gen_reg_3b == 'd1) begin
                            add_input_q_1 <= {{(N_COMP-N_STORAGE){1'b0}}, int_res_data};
                            add_input_q_2 <= ((loop_sum_type == VARIANCE) || (loop_sum_type == PARTIAL_NORM_FIRST_HALF)) ? (~compute_temp + 1'd1) : compute_temp; // Can subtract by adding the flipped version (data is 2's complement)
                            int_res_access_signals.read_req_src[LAYERNORM] <= 1'b0;
                            add_refresh <= 1'b1;
                            if (add_refresh) begin
                                gen_reg_3b <= 'd2;
                            end
                        end else if (gen_reg_3b == 'd2) begin
                            if (loop_sum_type == MEAN_SUM) begin // Move to next 
                                compute_temp <= add_output_q;
                                gen_reg_3b <= 'd0;
                                index <= index + 'd1;
                            end else if (loop_sum_type == VARIANCE) begin
                                // To continue variance computation, need to square and sum the result
                                add_refresh <= 1'b0;
                                mult_refresh <= 1'b1;
                                mult_input_q_1 <= add_output_q;
                                mult_input_q_2 <= add_output_q;
                                if (mult_refresh) begin
                                    gen_reg_3b <= 'd3;
                                end
                            end else if (loop_sum_type == PARTIAL_NORM_FIRST_HALF) begin
                                div_dividend <= add_output_q;
                                div_divisor <= sqrt_root_q;
                                div_start <=  ~div_busy & ~div_done;
                                gen_reg_3b <= (div_done) ? 'd4 : gen_reg_3b;
                                int_res_access_signals.write_req_src[LAYERNORM] <= div_done;
                                int_res_access_signals.write_data[LAYERNORM] <= (div_output_q[N_COMP-1]) ? (~div_output_flipped[N_STORAGE-1:0] + {{(N_STORAGE-1){1'b0}}, 1'd1}) : div_output_q[N_STORAGE-1:0];
                            end
                            add_refresh <= 1'b0;
                        end else if (gen_reg_3b == 'd3) begin
                            mult_refresh <= 1'b0;
                            add_refresh <= 1'b1;
                            add_input_q_1 <= compute_temp_2;
                            add_input_q_2 <= mult_output_q;
                            gen_reg_3b <= 'd4;
                        end else if (gen_reg_3b == 'd4) begin
                            add_refresh <= 1'b0;
                            if (~add_refresh) begin
                                gen_reg_3b <= 'd0;
                                index <= index + 'd1;
                                compute_temp_2 <= add_output_q;
                            end
                            int_res_access_signals.write_req_src[LAYERNORM] <= 1'd0;
                            div_start <= 1'b0;
                        end
                    end else begin
                        index <= 'd0;
                        if (loop_sum_type == MEAN_SUM) begin
                            state <= VARIANCE_LOOP;
                            compute_temp <= (compute_temp >>> $clog2(len)); // Divide by len (shift right by log2(len) bits.
                        end else if (loop_sum_type == VARIANCE) begin
                            compute_temp_2 <= (compute_temp_2 >>> $clog2(len)); // Divide by len (shift right by log2(len) bits
                            state <= VARIANCE_SQRT;
                            mult_input_q_1 <= compute_temp_3;
                            mult_input_q_2 <= compute_temp_3;
                            mult_refresh <= 1'b1;
                        end else if (loop_sum_type == PARTIAL_NORM_FIRST_HALF) begin
                            state <= DONE;
                        end
                    end
                end
                VARIANCE_LOOP : begin
                    loop_sum_type <= VARIANCE;
                    state <= LOOP_SUM;
                end
                VARIANCE_SQRT : begin
                    sqrt_rad_q <= compute_temp_2;
                    sqrt_start <= ~sqrt_busy & ~sqrt_done;
                    state <= (sqrt_done) ? NORM_FIRST_HALF : VARIANCE_SQRT;
                end
                NORM_FIRST_HALF : begin
                    state <= LOOP_SUM;
                    loop_sum_type <= PARTIAL_NORM_FIRST_HALF;
                end
                NORM_SECOND_HALF: begin
                    $display("NORM_SECOND_HALF in LayerNorm Not implemented yet");
                end
                DONE: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
