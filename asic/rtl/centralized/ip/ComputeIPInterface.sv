import Defines::*;

interface ComputeIPInterface;
    // Signals
    logic start, busy, done;
    logic overflow;
    CompFx_t in_1, in_2;
    CompFx_t out;
    modport basic_in (
        input start,
        input in_1, in_2,
        output busy, done,
        output out,
        output overflow
    );

    modport basic_out (
        input busy, done,
        input out,
        input overflow,
        output start,
        output in_1, in_2
    );

    // Extra signals for composite IP
    logic [$clog2(MAC_MAX_LEN+1)-1:0] len;
    ParamType_t param_type;
    Activation_t activation;
    ParamAddr_t bias_addr;
    IntResAddr_t start_addr_1, start_addr_2;
    modport extra (
        input len,
        input param_type,
        input activation,
        input bias_addr,
        input start_addr_1, start_addr_2
    );

endinterface
