`ifndef _cim_centralized_vh_
`define _cim_centralized_vh_

import Defines::*;

/* Todo:
*   - Load patch stream (instead of just starting with patch projection)
*/

module cim_centralized #()(
    input wire clk,
    SoCInterface soc_ctrl_i,

    // ----- Memory ---- //
    output MemoryInterface.input_write  tb_param_write_i,
    output MemoryInterface.input_write tb_int_res_write_i
);
    // ----- INSTANTIATION ----- //
    // Counters
    CounterInterface #(.WIDTH(4)) cnt_4b_i ();
    CounterInterface #(.WIDTH(7)) cnt_7b_i ();
    CounterInterface #(.WIDTH(9)) cnt_9b_i ();
    counter #(.WIDTH(4), .MODE(0)) cnt_4b_u (.clk(clk), .sig(cnt_4b_i));
    counter #(.WIDTH(7), .MODE(0)) cnt_7b_u (.clk(clk), .sig(cnt_7b_i));
    counter #(.WIDTH(9), .MODE(0)) cnt_9b_u (.clk(clk), .sig(cnt_9b_i));

    // Memory
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) param_write_i ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) param_read_i ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) mac_param_read_i ();
    MemoryInterface #(CompFx_t, ParamAddr_t, FxFormatParams_t) cim_param_read_i ();

    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_read_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) int_res_write_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) cim_int_res_read_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) mac_int_res_read_i ();
    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) cim_int_res_write_i ();

    MemoryInterface #(CompFx_t, IntResAddr_t, FxFormatIntRes_t) mac_casts_i ();

    params_mem params_u     (.clk, .rst_n, .write(tb_param_write_i),.read(param_read_i)); // Only testbench writes to params
    int_res_mem int_res_u   (.clk, .rst_n, .write(int_res_write_i), .read(int_res_read_i));

    // Compute
    ComputeIPInterface add_io();
    ComputeIPInterface add_io_exp();
    ComputeIPInterface add_io_mac();
    ComputeIPInterface mult_io();
    ComputeIPInterface mult_io_exp();
    ComputeIPInterface mult_io_mac();
    ComputeIPInterface div_io();
    ComputeIPInterface div_io_mac();
    ComputeIPInterface exp_io();
    ComputeIPInterface exp_io_mac();
    ComputeIPInterface mac_io();
    ComputeIPInterface mac_io_extra();

    adder add       (.clk, .rst_n, .io(add_io));
    multiplier mult (.clk, .rst_n, .io(mult_io));
    divider div     (.clk, .rst_n, .io(div_io));
    exp exp         (.clk, .rst_n, .io(exp_io), .adder_io(add_io_exp), .mult_io(mult_io_exp));
    mac mac         (.clk, .rst_n, .io(mac_io), .io_extra(mac_io_extra),
                     .casts(mac_casts_i), .param_read(mac_param_read_i), .int_res_read(mac_int_res_read_i),
                     .add_io(add_io_mac), .mult_io(mult_io_mac), .div_io(div_io_mac), .exp_io(exp_io_mac));

    // ----- GLOBAL SIGNALS ----- //
    logic rst_n;
    State_t cim_state;
    InferenceStep_t current_inf_step;

    // ----- CONSTANTS -----//
    assign rst_n = soc_ctrl_i.rst_n;

    // ----- FSM ----- //
    always_ff @ (posedge clk) begin : main_fsm
        if (~rst_n) begin
            reset();
        end else begin
            unique case (cim_state)
                IDLE_CIM: begin
                    if (soc_ctrl_i.new_sleep_epoch) begin
                        cim_state <= INFERENCE_RUNNING;
                        current_inf_step <= PATCH_PROJ_STEP;
                    end
                end
                INFERENCE_RUNNING: begin
                    if (current_inf_step == CLASS_TOKEN_CONCAT_STEP) begin
                        cim_state <= IDLE_CIM;
                        soc_ctrl_i.inference_complete <= 1'b1;
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

    always_ff @ (posedge clk) begin : inference_fsm
        set_default_values();
        if (cim_state == INFERENCE_RUNNING) begin
            unique case (current_inf_step)
                PATCH_PROJ_STEP: begin
                    /* cnt_7b_i holds current parameters row
                    cnt_9b_i holds current patch */

                    if (mac_io.done || (cnt_7b_i.cnt == 0 && ~mac_io.busy)) begin
                        IntResAddr_t patch_addr = mem_map[EEG_INPUT_MEM] + IntResAddr_t'(int'(cnt_9b_i.cnt) << $clog2(EMB_DEPTH));
                        ParamAddr_t param_addr  = param_addr_map[PATCH_PROJ_KERNEL_PARAMS] + ParamAddr_t'(int'(cnt_7b_i.cnt) << $clog2(EMB_DEPTH));
                        ParamAddr_t bias_addr   = param_addr_map_bias[PATCH_PROJ_BIAS_OFF] + ParamAddr_t'(cnt_7b_i.cnt);
                        start_mac(patch_addr, IntResAddr_t'(param_addr), bias_addr, MODEL_PARAM, LINEAR_ACTIVATION, VectorLen_t'(PATCH_LEN), int_res_format[PATCH_PROJ_INPUT_FORMAT],
                                int_res_width[PATCH_PROJ_INPUT_WIDTH], params_format[PATCH_PROJ_PARAM_FORMAT]);
                    end

                    if (mac_io.done) begin
                        IntResAddr_t addr = mem_map[PATCH_MEM] + IntResAddr_t'(cnt_7b_i.cnt) + IntResAddr_t'(int'(cnt_9b_i.cnt) << $clog2(PATCH_LEN)); // Left shift instead of multiply since PATCH_LEN is a power of 2
                        $display("Addr: %d.", addr);
                        write_int_res(addr, mac_io.out, int_res_width[PATCH_PROJ_OUTPUT_WIDTH], int_res_format[PATCH_PROJ_OUTPUT_FORMAT]);
        
                        // Update index control
                        if (int'(cnt_7b_i.cnt) == EMB_DEPTH-1) begin
                            cnt_7b_i.rst_n <= 1'b0;
                            if (int'(cnt_9b_i.cnt) == NUM_PATCHES-1) begin
                                cnt_9b_i.rst_n <= 1'b0;
                                current_inf_step <= CLASS_TOKEN_CONCAT_STEP;
                            end else cnt_9b_i.inc <= 1'b1;
                        end else cnt_7b_i.inc <= 1'b1;
                    end
                end
                CLASS_TOKEN_CONCAT_STEP: begin
                end
                default: begin
                end
            endcase
        end
    end

    // ----- MUX -----//
    always_comb begin : param_mem_MUX // TODO: Why does it claim that no latch is inferred if I use always_latch?
        // Write (only testbench writes to params)
        param_write_i.data_width = SINGLE_WIDTH;
        param_write_i.chip_en = 1'b1;
        param_write_i.en = tb_param_write_i.en;
        param_write_i.addr = tb_param_write_i.addr;
        param_write_i.data = tb_param_write_i.data;
        param_write_i.format = tb_param_write_i.format;

        // Read
        param_write_i.data_width = SINGLE_WIDTH;
        param_read_i.en = mac_param_read_i.en;
        mac_param_read_i.data = param_read_i.data;
        if (mac_param_read_i.en) begin // MAC
            param_read_i.addr = mac_param_read_i.addr;
            param_read_i.data_width = mac_param_read_i.data_width;
            param_read_i.format = mac_param_read_i.format;
        end
    end

    always_latch begin : int_res_mem_MUX
        // Write
        int_res_write_i.chip_en = 1'b1;
        int_res_write_i.en = tb_int_res_write_i.en | cim_int_res_write_i.en;
        if (tb_int_res_write_i.en) begin // Testbench
            int_res_write_i.addr = tb_int_res_write_i.addr;
            int_res_write_i.data = tb_int_res_write_i.data;
            int_res_write_i.format = tb_int_res_write_i.format;
            int_res_write_i.data_width = tb_int_res_write_i.data_width;
        end else if (cim_int_res_write_i.en) begin // CiM
            int_res_write_i.addr = cim_int_res_write_i.addr;
            int_res_write_i.data = cim_int_res_write_i.data;
            int_res_write_i.format = cim_int_res_write_i.format;
            int_res_write_i.data_width = cim_int_res_write_i.data_width;
        end

        // Read
        int_res_read_i.en = mac_int_res_read_i.en;
        mac_int_res_read_i.data = int_res_read_i.data;
        if (mac_int_res_read_i.en) begin // MAC
            int_res_read_i.addr = mac_int_res_read_i.addr;
            int_res_read_i.data_width = mac_int_res_read_i.data_width;
            int_res_read_i.format = mac_int_res_read_i.format;
        end
    end

    always_latch begin : add_io_MUX
        if (add_io_exp.start) begin
            add_io.in_1 = add_io_exp.in_1;
            add_io.in_2 = add_io_exp.in_2;
        end else if (add_io_mac.start) begin
            add_io.in_1 = add_io_mac.in_1;
            add_io.in_2 = add_io_mac.in_2;
        end

        add_io.start = add_io_exp.start | add_io_mac.start;
        add_io_exp.out = add_io.out;
        add_io_exp.done = add_io.done;
        add_io_mac.out = add_io.out;
        add_io_mac.done = add_io.done;
    end

    always_latch begin : mult_io_MUX
        if (mult_io_exp.start) begin
            mult_io.in_1 = mult_io_exp.in_1;
            mult_io.in_2 = mult_io_exp.in_2;
        end else if (mult_io_mac.start) begin
            mult_io.in_1 = mult_io_mac.in_1;
            mult_io.in_2 = mult_io_mac.in_2;
        end

        mult_io.start = mult_io_exp.start | mult_io_mac.start;
        mult_io_exp.out = mult_io.out;
        mult_io_exp.done = mult_io.done;
        mult_io_mac.out = mult_io.out;
        mult_io_mac.done = mult_io.done;
    end

    always_latch begin : div_io_MUX
        if (div_io_mac.start) begin
            div_io.in_1 = div_io_mac.in_1;
            div_io.in_2 = div_io_mac.in_2;
        end

        div_io.start = div_io_mac.start;
        div_io_mac.out = div_io.out;
        div_io_mac.busy = div_io.busy;
        div_io_mac.done = div_io.done;
    end

    always_latch begin : exp_io_MUX
        if (exp_io_mac.start) begin
            exp_io.in_1 = exp_io_mac.in_1;
        end

        exp_io.in_2 = CompFx_t'(0);
        exp_io.start = exp_io_mac.start;
        exp_io_mac.out = exp_io.out;
        exp_io_mac.busy = exp_io.busy;
        exp_io_mac.done = exp_io.done;
    end

    // ----- TASKS ----- //
    task automatic set_default_values();
        // Sets the default value for signals that do not persist cycle-to-cycle
        cnt_4b_i.inc <= 1'b0;
        cnt_7b_i.inc <= 1'b0;
        cnt_9b_i.inc <= 1'b0;
        cnt_4b_i.rst_n <= 1'b1;
        cnt_7b_i.rst_n <= 1'b1;
        cnt_9b_i.rst_n <= 1'b1;

        cim_int_res_read_i.en <= 1'b0;
        cim_param_read_i.en <= 1'b0;
        cim_int_res_write_i.en <= 1'b0;

        mac_io.start <= 1'b0;
    endtask

    task automatic reset();
        cim_state <= IDLE_CIM;
        current_inf_step <= PATCH_PROJ_STEP;

        cnt_4b_i.inc <= 1'b0;
        cnt_7b_i.inc <= 1'b0;
        cnt_9b_i.inc <= 1'b0;
        cnt_4b_i.rst_n <= 1'b0;
        cnt_7b_i.rst_n <= 1'b0;
        cnt_9b_i.rst_n <= 1'b0;
    endtask

    task write_int_res(input IntResAddr_t addr, input CompFx_t data, input DataWidth_t width, input FxFormatIntRes_t int_res_format);
        cim_int_res_write_i.en <= 1'b1;
        cim_int_res_write_i.addr <= addr;
        cim_int_res_write_i.data <= data;
        cim_int_res_write_i.data_width <= width;
        cim_int_res_write_i.format <= int_res_format;
    endtask

    task start_mac(input IntResAddr_t addr_1, input IntResAddr_t addr_2, input ParamAddr_t bias_addr, input ParamType_t param_type, input Activation_t act, input VectorLen_t len, input FxFormatIntRes_t int_res_input_format, input DataWidth_t int_res_read_width, input FxFormatParams_t params_read_format);
        mac_io.start <= 1'b1;
        mac_io_extra.start_addr_1 <= addr_1;
        mac_io_extra.start_addr_2 <= addr_2;
        mac_io_extra.param_type <= param_type;
        mac_io_extra.activation <= act;
        mac_io_extra.len <= len;
        mac_io_extra.bias_addr <= bias_addr;
        mac_casts_i.int_res_read_format <= int_res_input_format;
        mac_casts_i.int_res_read_width <= int_res_read_width;
        mac_casts_i.params_read_format <= params_read_format;
    endtask

    // ----- ASSERTIONS ----- //
    always_ff @ (posedge clk) begin : compute_mux_assertions
        assert (~(tb_int_res_write_i.en & cim_int_res_write_i.en)) else $fatal("More than one source is trying to write to intermediate result memory simulatenously!");
        assert (~(cim_int_res_read_i.en & 0)) else $fatal("More than one source is trying to read from intermediate result memory simulatenously!");

        assert (~(add_io_exp.start & add_io_mac.start)) else $fatal("More than one source is trying to start an add!");
        assert (~(mult_io_exp.start & mult_io_mac.start)) else $fatal("More than one source is trying to start a mult!");
        assert (~(div_io_mac.start & 0)) else $fatal("More than one source is trying to start a div!");
        assert (~(exp_io_mac.start & 0)) else $fatal("More than one source is trying to start an exp!");
    end
endmodule

`endif // _cim_centralized_vh_
