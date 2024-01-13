module encoder
#(
    parameter encoderWidth = 16 // Max Width of the input
)
(
    input clk, // Clock
    input clk_en, // Clock Enable
    input rst_n, // Asynchronous reset active low
    input [encoderWidth-1:0] encoder_in, // Encoder input
    
    // Encoder output
    output logic [$clog2(encoderWidth)-1:0] encoder_out
);
always_ff @(posedge clk)
begin
    if (~rst_n)
    begin
        encoder_out = 'd0;
    end
    else if (clk_en)
    begin
        for (int i = 0; i < encoderWidth; i++)
        begin
            if (encoder_in[i])
            begin
                encoder_out = i;
            end
        end
    end
end
endmodule
