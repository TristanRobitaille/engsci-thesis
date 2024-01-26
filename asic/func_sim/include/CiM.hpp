#ifndef CIM_H
#define CIM_H

#include <iostream>

#include <Misc.hpp>
#include <Param_Layer_Mapping.hpp>

/*----- DEFINE -----*/
#define CIM_PARAMS_STORAGE_SIZE_KB 3072
#define CIM_INT_RES_SIZE_KB 512

/*----- CLASS -----*/
class CiM {
    private:
        enum state {
            IDLE,
            RESET,
            INVALID = -1
        };

        int16_t id; // ID of the CiM
        int16_t gen_reg_16b; // General-purpose register
        float params[CIM_PARAMS_STORAGE_SIZE_KB / sizeof(float)];
        float intermediate_res[CIM_INT_RES_SIZE_KB / sizeof(float)];
        state state;
        op prev_bus_op;
        Counter gen_cnt_10b;
        Counter gen_cnt_10b_2;

    public:
        CiM() : id(-1), gen_cnt_10b(10), gen_cnt_10b_2(10) {}
        CiM(const int16_t cim_id);
        int reset();
        int run(struct ext_signals* ext_sigs, Bus* bus);
        float MAC(uint16_t input_start_addr, uint16_t params_start_addr, uint16_t len);
};

#endif //CIM_H
