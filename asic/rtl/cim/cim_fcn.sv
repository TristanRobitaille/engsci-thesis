`ifndef _cim_fcn_vh_
`define _cim_fcn_vh_

`include "../types.svh"

/*----- FUNCTIONS -----*/

function automatic has_my_data(input logic [6:0] word_snt_cnt, input logic [6:0] ID);
    return (ID >= word_snt_cnt) && (ID < (word_snt_cnt+'d3));
endfunction
`endif
