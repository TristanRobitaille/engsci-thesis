// Includes
`include "master.svh"
`include "master_fcn.svh"
`include "../ip/counter/counter.sv"
`include "../types.svh"

module master (
    input wire clk,
    input wire rst_n,

    // Control signals
    input wire new_sleep_epoch, start_param_load, all_cims_ready,

    // Bus
    inout bus_t bus
);

    // Read bus
    wire [BUS_OP_WIDTH-1:0] bus_op_read;
    wire signed [BUS_DATA_WIDTH-1:0] bus_data_read;
    wire [$clog2(NUM_CIMS)-1:0] bus_target_or_sender_read;
    assign {bus_op_read, bus_data_read, bus_target_or_sender_read} = bus;
    
    // Write bus
    logic bus_drive;
    reg [BUS_OP_WIDTH-1:0] bus_op_write;
    reg signed [BUS_DATA_WIDTH-1:0] bus_data_write;
    reg [$clog2(NUM_CIMS)-1:0] bus_target_or_sender_write;
    always_comb begin : bus_drive_comb
        bus.op = (bus_drive) ? bus_op_write : 'Z;
        bus.data = (bus_drive) ? bus_data_write : 'Z;
        bus.target_or_sender = (bus_drive) ? bus_target_or_sender_write : 'Z;
    end

    // Instantiate counters
    logic gen_cnt_7b_inc, gen_cnt_7b_2_inc;
    logic [6:0] gen_cnt_7b_cnt, gen_cnt_7b_2_cnt;
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b   (.clk(clk), .rst_n(rst_n), .inc(gen_cnt_7b_inc),  .cnt(gen_cnt_7b_cnt));
    counter #(.WIDTH(7), .MODE(0)) gen_cnt_7b_2 (.clk(clk), .rst_n(rst_n), .inc(gen_cnt_7b_2_inc), .cnt(gen_cnt_7b_2_cnt));

    // Internal registers and wires
    logic [15:0] gen_reg_16b = 'd0;
    logic [15:0] gen_reg_16b_2 = 'd0;
    logic [15:0] gen_reg_16b_3 = 'd0;

    // Main FSM
    MASTER_STATE_T state = MASTER_STATE_IDLE;
    HIGH_LEVEL_INFERENCE_STEP_T high_level_inf_step = PRE_LAYERNORM_1_TRANS_STEP;
    always_ff @ (posedge clk or negedge rst_n) begin : master_main_fsm
        if (!rst_n) begin // Reset
            gen_reg_16b <= 'd0;
            gen_reg_16b_2 <= 'd0;
            gen_reg_16b_3 <= 'd0;
            state <= MASTER_STATE_IDLE;
        end else begin
            unique case (state)
                MASTER_STATE_IDLE: begin
                    if (start_param_load)
                        state <= MASTER_STATE_PARAM_LOAD;
                    else if (new_sleep_epoch)
                        state <= MASTER_STATE_SIGNAL_LOAD;
                    gen_reg_16b <= 'd0;
                    gen_reg_16b_2 <= 'd0;
                    gen_reg_16b_3 <= 'd0;
                end

                MASTER_STATE_PARAM_LOAD: begin
                    {bus_op_write, bus_data_write, bus_target_or_sender_write} = {PATCH_LOAD_BROADCAST_START_OP, 66'd0, 6'd0}; // TODO: Dummy
                    state <= MASTER_STATE_IDLE;
                end

                MASTER_STATE_SIGNAL_LOAD: begin
                end

                default:
                    state <= MASTER_STATE_IDLE;
            endcase
        end
    end

endmodule
