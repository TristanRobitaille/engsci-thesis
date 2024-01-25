#ifndef CIM_H
#define CIM_H

#include <iostream>

#include <Misc.hpp>

/*----- DEFINE -----*/
#define NUM_CIM 64
#define CIM_WEIGHTS_STORAGE_SIZE_KB 2048
#define CIM_TEMP_STORAGE_SIZE_KB 512

/*----- CLASS -----*/
class CiM {
    private:
        enum state {
            IDLE,
            RESET,
            INVALID = -1
        };

        int16_t id;
        uint16_t data_and_param[CIM_WEIGHTS_STORAGE_SIZE_KB / sizeof(uint16_t)];
        float storage[CIM_TEMP_STORAGE_SIZE_KB / sizeof(float)];
        state state;
        op prev_bus_op;
        Counter gen_cnt_10b;

    public:
        CiM() : id(-1), gen_cnt_10b(10) {}
        CiM(const int16_t cim_id);
        int reset();
        int run(struct ext_signals* ext_sigs, Bus* bus);
};

#endif //CIM_H
