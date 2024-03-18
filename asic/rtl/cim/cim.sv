// Includes
`include "cim.svh"
`include "../types.svh"
`include "../ip/counter/counter.sv"

module cim (
    input wire clk,
    input wire rst_n,

    output logic is_ready
);

    // TODO: Replace with actual generated memory
    logic [N_STORAGE-1:0] params [PARAMS_STORAGE_SIZE_CIM-1:0];
    logic [N_STORAGE-1:0] intermediate_res [TEMP_RES_STORAGE_SIZE_CIM-1:0];

    // Internal signals
    logic compute_in_progress;
    logic [2:0] gen_cnt_3b;
    logic [$clog2(NUM_CIMS+1)-1:0] sender_id, data_len;
    logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] tx_addr, rx_addr;
    logic [N_COMP-1:0] compute_temp, compute_temp_2, compute_temp_3, computation_result;

    // Modules
    logic gen_cnt_7b_rst_n, gen_cnt_7b_2_rst_n, word_rec_cnt_rst_n, word_snt_cnt_rst_n;
    logic [6:0] gen_cnt_7b_inc, gen_cnt_7b_2_inc, word_rec_cnt_inc, word_snt_cnt_inc;
    wire [6:0] gen_cnt_7b_cnt, gen_cnt_7b_2_cnt, word_rec_cnt, word_snt_cnt;
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b_inst      (.clk(clk), .rst_n(gen_cnt_7b_rst_n),   .inc(gen_cnt_7b_inc),   .cnt(gen_cnt_7b_cnt));
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b_2_inst    (.clk(clk), .rst_n(gen_cnt_7b_2_rst_n), .inc(gen_cnt_7b_2_inc), .cnt(gen_cnt_7b_2_cnt));
    counter #(.WIDTH(7), .MODE(0)) word_rec_cnt_inst    (.clk(clk), .rst_n(word_rec_cnt_rst_n), .inc(word_rec_cnt_inc), .cnt(word_rec_cnt));
    counter #(.WIDTH(7), .MODE(0)) word_snt_cnt_inst    (.clk(clk), .rst_n(word_snt_cnt_rst_n), .inc(word_snt_cnt_inc), .cnt(word_snt_cnt));

    // FSM
    CIM_STATE_T cim_state;
    INFERENCE_STEP_T current_inf_step;
    always_ff @ (posedge clk or negedge rst_n) begin : cim_main_fsm
        if (!rst_n) begin
            cim_state <= CIM_IDLE;
            current_inf_step <= CLASS_TOKEN_CONCAT_STEP;
        end else begin
        end
    end

endmodule
