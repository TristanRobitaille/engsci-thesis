`ifndef _mem_model_sv_
`define _mem_model_sv_

module mem_model #(
    parameter int DEPTH
)(
    input logic clk, rst_n,
    input MemoryInterface.input_write_bank write,
    output MemoryInterface.input_read_bank read
);

    logic WEN, CEN;
    logic [$bits(write.data)-1:0] memory [DEPTH-1:0];

    assign WEN = write.en & ~read.en;
    assign CEN = write.chip_en;

    always_ff @ (posedge clk) begin : mem_write
        if (WEN & CEN & rst_n) begin
            memory[write.addr] <= write.data;
        end
    end

    always_ff @ (posedge clk) begin : mem_read
        if (~rst_n) begin
            read.data <= 'd0;
        end else if (~WEN & CEN & rst_n) begin
            read.data <= memory[read.addr];
        end
    end

    // Assertions
    always_ff @ (posedge clk) begin : mem_model_assertions
        assert (!(write.en & read.en)) else $error("Tried to read and write to memory simultaneously!"); // Never try to read and write simulteanously
        assert (!((write.en | read.en) & ~write.chip_en)) else $error("Tried to write or read from memory while it was not enabled!"); // Never try to read or write while memory not enabled 
    end
endmodule
`endif
