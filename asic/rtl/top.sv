`include "ip/counter/counter.sv"

module top # (
    parameter N = 22, // 22b total
    parameter Q = 10  // 10b fractional
) (
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
);

master master ( .clk(clk), .rst_n(rst_n),
                .new_sleep_epoch(new_sleep_epoch), .start_param_load(start_param_load), .all_cims_ready(), // Control signals from RISC-V
                .new_eeg_sample(new_eeg_sample), .eeg_sample(eeg_sample), // EEG from ADC
                .ext_mem_data_valid(ext_mem_data_valid), .ext_mem_data(ext_mem_data), .ext_mem_data_read_pulse(ext_mem_data_read_pulse), .ext_mem_addr(ext_mem_addr), // External memory interface
                .bus_op(bus_op), .bus_data(bus_data), .bus_target_or_sender(bus_target_or_sender), // Bus interface
             );

cim cim (   .clk(clk), .rst_n(rst_n), .is_ready(cim_is_ready),
            .bus_op(bus_op), .bus_data(bus_data), .bus_target_or_sender(bus_target_or_sender), // Bus interface
);

endmodule
