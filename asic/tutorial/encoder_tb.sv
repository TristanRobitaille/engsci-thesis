`timescale 1ps/1ps

module encoder_tb ();

localparam encoderWidth = 16;
logic clk; // Clock
logic clk_en; // Clock Enable
logic rst_n; // Asynchronous reset active low
logic [encoderWidth-1:0] encoder_in; // Encoder input
logic [$clog2(encoderWidth)-1:0] encoder_out; // Encoder output

encoder uut (.*);

localparam h_period = 5; //duration for each bit = 5ns
localparam period = 10;

// Clock Generation
always
    begin
    clk = 1'b1;
    #h_period;
    clk = 1'b0;
    #h_period;
end

initial
begin
    clk_en = 1'b1;
    rst_n = 1'b0;
    #period;
    rst_n = 1'b1;

    $display("***** TEST 1 *****");
    encoder_in = 'h0010; // 4
    #period;
    $display("Encoder in: ",encoder_in);
    $display("Encoder out: ",encoder_out);

    $display("***** TEST 2 *****");
    encoder_in = 'h0100; // 8
    #period;
    $display("Encoder in: ",encoder_in);
    $display("Encoder out: ",encoder_out);

    $display("***** TEST 3 *****");
    encoder_in = 'h8000; // 15
    #period;
    $display("Encoder in: ",encoder_in);
    $display("Encoder out: ",encoder_out);
end
endmodule