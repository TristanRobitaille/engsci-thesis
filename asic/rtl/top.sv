`ifndef _top_sv_
`define _top_sv_

`include "master/master.sv"
`include "cim/cim.sv"
`include "types.svh"

module top (
    input wire clk, rst_n,

    // RISC-V control signals
    input wire new_sleep_epoch, start_param_load, 

    // EEG
    input wire new_eeg_sample,
    input wire [EEG_SAMPLE_DEPTH-1:0] eeg_sample,

    // External memory interface
    input wire ext_mem_data_valid,
    input logic signed [N_STORAGE-1:0] ext_mem_data,
    output logic ext_mem_data_read_pulse,
    output logic [$clog2(NUM_PARAMS)-1:0] ext_mem_addr

    // Bus //TODO: Temporarily here because we need Synopsys to keep them
    output logic [BUS_OP_WIDTH-1:0] bus_op,
    output logic signed [2:0][N_STORAGE-1:0] bus_data,
    output logic [$clog2(NUM_CIMS)-1:0] bus_target_or_sender
);

logic all_cims_ready, cim_is_ready;
wire [BUS_OP_WIDTH-1:0] bus_op;
wire signed [2:0][N_STORAGE-1:0] bus_data;
wire [$clog2(NUM_CIMS)-1:0] bus_target_or_sender;

master master ( .clk(clk), .rst_n(rst_n),
                .new_sleep_epoch(new_sleep_epoch), .start_param_load(start_param_load), // Control signals from RISC-V
                .new_eeg_sample(new_eeg_sample), .eeg_sample(eeg_sample), // EEG from ADC
                .ext_mem_data_valid(ext_mem_data_valid), .ext_mem_data(ext_mem_data), .ext_mem_data_read_pulse(ext_mem_data_read_pulse), .ext_mem_addr(ext_mem_addr), // External memory interface
                .bus_op(bus_op), .bus_data(bus_data), .bus_target_or_sender(bus_target_or_sender), .all_cims_ready(cim_is_ready) // Bus interface
              );

// cim #(.ID(0), .SRAM(0)) cim (   .clk(clk), .rst_n(rst_n), .is_ready(cim_is_ready),
//                                 .bus_op(bus_op), .bus_data(bus_data), .bus_target_or_sender(bus_target_or_sender) // Bus interface
//                             );

//synopsys translate_off
always_ff @ (posedge clk) begin : top_assertions
    if ($countones(EMB_DEPTH) != 1) begin
        $fatal(1, "EMB_DEPTH must be a power of two");
    end
end
//synopsys translate_on

endmodule

`endif
