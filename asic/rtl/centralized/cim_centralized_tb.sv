module cim_centralized_tb (
    input wire clk, soc_ctrl_rst_n,
    input wire soc_ctrl_new_sleep_epoch
);

    SoCInterface soc_ctrl ();
    assign soc_ctrl.rst_n = soc_ctrl_rst_n;
    assign soc_ctrl.new_sleep_epoch = soc_ctrl_new_sleep_epoch;

    cim_centralized cim_centralized (
        .clk(clk),
        .soc_ctrl(soc_ctrl)
    );
endmodule
