module cim_centralized_tb (
    input wire clk, rst_n
);

    cim_centralized cim_centralized (
        .clk(clk),
        .rst_n(rst_n)
    );
endmodule
