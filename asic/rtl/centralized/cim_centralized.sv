import Defines::*;

module cim_centralized #()(
    input wire clk,
    SoCInterface soc_ctrl
);

// ----- TASKS ----- //
task automatic set_default_values();
    // Sets the default value for signals that do not persist cycle-to-cycle
    cnt_4b.inc = 1'b0;
    cnt_7b.inc = 1'b0;
    cnt_9b.inc = 1'b0;
    cnt_4b.rst_n = 1'b1;
    cnt_7b.rst_n = 1'b1;
    cnt_9b.rst_n = 1'b1;

    param_read.en = 1'b0;
    param_write.en = 1'b0;
    int_res_read.en = 1'b0;
    int_res_write.en = 1'b0;
endtask

task automatic reset();
    cim_state <= IDLE_CIM;
    current_inf_step <= PATCH_PROJ_STEP;

    cnt_4b.inc = 1'b0;
    cnt_7b.inc = 1'b0;
    cnt_9b.inc = 1'b0;
    cnt_4b.rst_n = 1'b0;
    cnt_7b.rst_n = 1'b0;
    cnt_9b.rst_n = 1'b0;
endtask

// ----- INSTANTIATION ----- //
// Counters
CounterInterface #(.WIDTH(4)) cnt_4b ();
CounterInterface #(.WIDTH(7)) cnt_7b ();
CounterInterface #(.WIDTH(9)) cnt_9b ();
counter #(.WIDTH(4), .MODE(0)) cnt_4b_u (.clk(clk), .sig(cnt_4b));
counter #(.WIDTH(7), .MODE(0)) cnt_7b_u (.clk(clk), .sig(cnt_7b));
counter #(.WIDTH(9), .MODE(0)) cnt_9b_u (.clk(clk), .sig(cnt_9b));

// Memory
MemoryInterface #(Param_t, ParamAddr_t) param_read ();
MemoryInterface #(Param_t, ParamAddr_t) param_write ();
MemoryInterface #(IntResDouble_t, IntResAddr_t) int_res_read ();
MemoryInterface #(IntResDouble_t, IntResAddr_t) int_res_write ();
params_mem params (.clk, .rst_n, .write(param_write), .read(param_read));
int_res_mem int_res (.clk, .rst_n, .write(int_res_write), .read(int_res_read));

// ----- GLOBAL SIGNALS ----- //
logic rst_n;
State_t cim_state;
InferenceStep_t current_inf_step;

// ----- CONSTANTS -----//
assign rst_n = soc_ctrl.rst_n;
assign param_write.chip_en = 'b1;
assign int_res_write.chip_en = 'b1;

// ----- FSM ----- //
always_ff @ (posedge clk) begin : main_fsm
    if (~rst_n) begin
        reset();
    end else begin
        set_default_values();

        unique case (cim_state)
            IDLE_CIM: begin
                if (soc_ctrl.new_sleep_epoch) begin
                    cim_state <= INFERENCE_RUNNING;
                    current_inf_step <= PATCH_PROJ_STEP;
                end
            end
            INFERENCE_RUNNING: begin
                if (current_inf_step == INFERENCE_COMPLETE) begin
                    cim_state <= IDLE_CIM;
                end
            end
            INVALID_CIM: begin
                cim_state <= IDLE_CIM;
            end
            default: begin
                cim_state <= IDLE_CIM;
            end
        endcase
    end
end

endmodule
