// Includes
`include "cim.svh"

/*TODO
    - Consider using a function to start MAC computation for more compact code
*/

module cim # (
    /* verilator lint_off UNUSEDPARAM */
    parameter ID = 0,
    parameter STANDALONE_TB = 1
    /* verilator lint_on UNUSEDPARAM */
)(
    input wire clk,
    input wire rst_n,

    // Bus
    inout wire [BUS_OP_WIDTH-1:0] bus_op,
    inout wire signed [2:0][N_STORAGE-1:0] bus_data,
    inout wire [$clog2(NUM_CIMS)-1:0] bus_target_or_sender,

    output logic is_ready
);

    // Initialize arrays
    `include "top_init.sv"
    `include "cim_init.sv"

    // Memory
    MemAccessSignals params_access_signals();
    MemAccessSignals int_res_access_signals();
    wire [N_STORAGE-1:0] int_res_read_data, params_read_data;
    cim_mem mem (   .clk(clk), .params_access_signals(params_access_signals), .int_res_access_signals(int_res_access_signals),
                    .int_res_read_data(int_res_read_data), .params_read_data(params_read_data));

    // Bus
    logic bus_drive, bus_drive_delayed;
    logic [BUS_OP_WIDTH-1:0] bus_op_write;
    logic signed [2:0][N_STORAGE-1:0] bus_data_write;
    logic [$clog2(NUM_CIMS)-1:0] bus_target_or_sender_write;
    assign bus_op = (bus_drive) ? bus_op_write : 'Z;
    assign bus_data = (bus_drive) ? bus_data_write : 'Z;
    assign bus_target_or_sender = (bus_drive) ? bus_target_or_sender_write : 'Z;

    wire [BUS_OP_WIDTH-1:0] bus_op_read;
    wire signed [2:0][N_STORAGE-1:0] bus_data_read;
    wire [$clog2(NUM_CIMS)-1:0] bus_target_or_sender_read;
    if (STANDALONE_TB == 0) begin : bus_read_assignment // Cannot drive inout's in CocoTB testbench, so we drive bus_x_read directly and do not assign to them
        assign bus_op_read = bus_op;
        assign bus_data_read = bus_data;
        assign bus_target_or_sender_read = bus_target_or_sender;
    end

    // Internal signals
    wire MAC_compute_in_progress;
    logic [1:0] gen_reg_2b;
    logic [$clog2(NUM_CIMS+1)-1:0] sender_id, data_len;
    logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] tx_addr, rx_addr;
    logic [N_COMP-1:0] compute_temp, compute_temp_2, compute_temp_3, computation_result;

    // Counters
    logic gen_cnt_7b_rst_n, gen_cnt_7b_2_rst_n, word_rec_cnt_rst_n, word_snt_cnt_rst_n;
    logic [6:0] gen_cnt_7b_inc, gen_cnt_7b_2_inc, word_rec_cnt_inc, word_snt_cnt_inc;
    wire [6:0] gen_cnt_7b_cnt, gen_cnt_7b_2_cnt, word_rec_cnt, word_snt_cnt;
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b_inst      (.clk(clk), .rst_n(gen_cnt_7b_rst_n),   .inc(gen_cnt_7b_inc),   .cnt(gen_cnt_7b_cnt));
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b_2_inst    (.clk(clk), .rst_n(gen_cnt_7b_2_rst_n), .inc(gen_cnt_7b_2_inc), .cnt(gen_cnt_7b_2_cnt));
    counter #(.WIDTH(7), .MODE(0)) word_rec_cnt_inst    (.clk(clk), .rst_n(word_rec_cnt_rst_n), .inc(word_rec_cnt_inc), .cnt(word_rec_cnt));
    counter #(.WIDTH(7), .MODE(0)) word_snt_cnt_inst    (.clk(clk), .rst_n(word_snt_cnt_rst_n), .inc(word_snt_cnt_inc), .cnt(word_snt_cnt));

    // Adder module
    wire add_overflow;
    logic add_refresh;
    logic [N_COMP-1:0] add_input_q_1, add_input_q_2, add_output_q;
    adder add_inst (.clk(clk), .rst_n(rst_n), .refresh(add_refresh), .overflow(add_overflow),
                    .input_q_1(add_input_q_1), .input_q_2(add_input_q_2), .output_q(add_output_q));

    // Multiplier module
    logic mult_refresh;
    logic [N_COMP-1:0] mult_input_q_1, mult_input_q_2, mult_output_q;
    multiplier mult_inst (  .clk(clk), .rst_n(rst_n), .refresh(mult_refresh),
                            .input_q_1(mult_input_q_1), .input_q_2(mult_input_q_2), .output_q(mult_output_q));

    // MAC module
    logic mac_start, mac_done, mac_compute_in_progress;
    logic [$clog2(MAC_MAX_LEN+1)-1:0] mac_len;
    logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] mac_start_addr1, mac_start_addr2;
    logic [$clog2(PARAMS_STORAGE_SIZE_CIM)-1:0] mac_params_addr, mac_bias_addr;
    logic [N_COMP-1:0] mac_out, mac_out_flipped;
    PARAM_TYPE_T mac_param_type;
    ACTIVATION_TYPE_T mac_activation;
    mac mac_inst    (.clk(clk), .rst_n(rst_n), .start(mac_start), .done(mac_done), .busy(mac_compute_in_progress), .param_type(mac_param_type), .len(mac_len), .activation(mac_activation), 
                     .start_addr1(mac_start_addr1), .start_addr2(mac_start_addr2), .bias_addr(mac_bias_addr),
                     .param_addr(params_access_signals.addr_table[MAC]), .intermediate_res_addr(int_res_access_signals.addr_table[MAC]),
                     .param_data(params_read_data), .intermediate_res_data(int_res_read_data), .computation_result(mac_out),
                     .int_res_mem_access_req(int_res_access_signals.read_req_src[MAC]), .params_mem_access_req(params_access_signals.read_req_src[MAC]),
                     .add_input_q_1(add_input_q_1), .add_input_q_2(add_input_q_2), .add_output_q(add_output_q), .add_refresh(add_refresh), 
                     .mult_input_q_1(mult_input_q_1), .mult_input_q_2(mult_input_q_2), .mult_output_q(mult_output_q), .mult_refresh(mult_refresh));

    CIM_STATE_T cim_state;
    INFERENCE_STEP_T current_inf_step;
    // Comms FSM
    always_ff @ (posedge clk) begin : cim_comms_fsm
        if (!rst_n) begin
            word_rec_cnt_rst_n <= RST;
            word_snt_cnt_rst_n <= RST;
            sender_id <= 'd0;
            data_len <= 'd0;
            tx_addr <= 'd0;
            rx_addr <= 'd0;
        end else begin
            unique case (bus_op_read)
                PATCH_LOAD_BROADCAST_START_OP: begin
                    word_rec_cnt_rst_n <= RST;
                    gen_cnt_7b_2_rst_n <= RST;
                end
                PATCH_LOAD_BROADCAST_OP: begin
                    int_res_access_signals.addr_table[BUS_FSM] <= {3'd0, word_rec_cnt};
                    int_res_access_signals.write_data[BUS_FSM] <= bus_data_read[0];
                    int_res_access_signals.write_req_src[BUS_FSM] <= 1'b1;
                    word_rec_cnt_inc <= 'd1;
                end
                DENSE_BROADCAST_START_OP,
                TRANS_BROADCAST_START_OP: begin
                end
                DENSE_BROADCAST_DATA_OP,
                TRANS_BROADCAST_DATA_OP: begin
                end
                DATA_STREAM_START_OP: begin
                end
                DATA_STREAM_OP: begin
                end
                NOP: begin
                    int_res_access_signals.write_req_src[BUS_FSM] <= 1'b0;
                    word_rec_cnt_inc <= 'd0;
                end
                PISTOL_START_OP,
                INFERENCE_RESULT_OP: begin
                end
                default: begin
                end
            endcase
        end
    end

    // Compute control FSM
    always_ff @ (posedge clk) begin : cim_compute_control_fsm
        if (!rst_n) begin
            cim_state <= CIM_IDLE;
            current_inf_step <= CLASS_TOKEN_CONCAT_STEP;
            gen_cnt_7b_rst_n <= RST;
            gen_cnt_7b_2_rst_n <= RST;
            compute_temp <= 'd0;
            compute_temp_2 <= 'd0;
            compute_temp_3 <= 'd0;
            computation_result <= 'd0;
            // TODO: Reset intermediate_res
        end else begin
            unique case (cim_state)
                CIM_IDLE: begin
                    if (bus_op_read == PATCH_LOAD_BROADCAST_START_OP) begin
                        cim_state <= CIM_PATCH_LOAD;
                    end
                end
                CIM_PATCH_LOAD: begin
                    unique case (gen_reg_2b) 
                        'd0: begin
                            if (~MAC_compute_in_progress && (word_rec_cnt == PATCH_LEN)) begin
                                gen_reg_2b <= 'd1;
                                gen_cnt_7b_2_inc <= 'd0;
                                mac_start <= 1'b1;
                                mac_start_addr1 <= 'd0;
                                mac_start_addr2 <= 'd0;
                                mac_len <= PATCH_LEN;
                                mac_param_type <= MODEL_PARAM;
                                mac_activation <= LINEAR_ACTIVATION;
                                mac_bias_addr <= param_addr_map[SINGLE_PARAMS].addr + PATCH_PROJ_BIAS_PARAMS;
                            end
                        end
                        'd1: begin
                            mac_start <= 1'b0;
                            if (mac_done) begin
                                int_res_access_signals.addr_table[LOGIC_FSM] <= {3'd0, gen_cnt_7b_2_cnt} + mem_map[PATCH_MEM];
                                int_res_access_signals.write_data[LOGIC_FSM] <= (mac_out[N_COMP-1]) ? mac_out_flipped[N_STORAGE-1:0] : mac_out[N_STORAGE-1:0]; // Selecting the bottom N_STORAGE bits requires converting to positive two's complement if the number is negative, selecting, and converting back
                                gen_reg_2b <= 'd0;
                                gen_cnt_7b_2_inc <= 'd1;
                                // TODO: Verify computation?
                            end
                        end
                    endcase
                    word_rec_cnt_rst_n <= (~MAC_compute_in_progress && (word_rec_cnt == PATCH_LEN)) ? RST : RUN;
                    int_res_access_signals.write_req_src[LOGIC_FSM] <= (mac_done);
                    is_ready <= (gen_cnt_7b_2_cnt == NUM_PATCHES);
                    cim_state <= (gen_cnt_7b_2_cnt == NUM_PATCHES) ? CIM_INFERENCE_RUNNING : cim_state;
                    gen_cnt_7b_2_rst_n <= (gen_cnt_7b_2_cnt == NUM_PATCHES) ? RST : RUN;
                end
                default: begin
                    cim_state <= CIM_IDLE;
                end
            endcase
        end
    end

    // Miscellanous combinational logic
    always_comb begin : computation_twos_comp_flip
        mac_out_flipped = ~mac_out + 'd1;
    end
    
endmodule
