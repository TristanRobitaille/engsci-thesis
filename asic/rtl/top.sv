module top (
    input clk,
    input clk_en,
    input rst_n,
    
    output logic toggle
);

    always_ff @ (clk) begin
        toggle <= ~toggle;        
    end
endmodule
