#include <queue>

#ifndef MISC_H
#define MISC_H

/*----- DEFINE -----*/
#define PATCH_LENGTH_NUM_SAMPLES 256
#define NUM_PATCHES 30

/*----- ENUM -----*/
enum bus_direction {
    MASTER_TO_CIM,
    CIM_TO_MASTER
};

enum op {
    PATCH_LOAD,
    INVALID
};

enum system_state {
    RUNNING,
    DONE
};

/*----- STRUCT -----*/
struct instruction {
    /* Instructions between master controller and CiM */
    bus_direction direction;
    op op;
    uint16_t target_cim;
    uint16_t data;
};

struct ext_signals {
    /* Signals external to the master controller and CiM, coming from peripherals or the RISC-V processor */
    bool master_nrst;
    bool new_sleep_epoch;
};

/*----- CLASS -----*/
/* Counter with overflow behaviour and reset */
class Counter {
    private:
        uint8_t width;
        uint64_t val;
    public:
        Counter(int width) : width(width), val(0) {}
        int inc() {
            val++;
            if (val == std::pow(2, width)){ val = 0; } // Overflow
            return 0;
        }
        int reset() {
            val = 0;
            return 0;
        }
        uint64_t get_cnt() { return val; }
};

/* Bus */
class Bus {
    private:
        bool new_trans;
        struct instruction inst;
        std::queue<struct instruction> q;

    public:
        bool is_trans_new() { return new_trans; };
        struct instruction get_inst() { return inst; };
        int push_inst(struct instruction inst) {
            q.push(inst);
            return 0;
        };
        int reset(){
            while (q.size() > 0) { q.pop(); }
            inst = {
                /* direction */  MASTER_TO_CIM,
                /* op */         INVALID,
                /* target_cim */ 0,
                /* data */       0};
            return 0;
        };
        int run() {
            if (q.size() == 0) { new_trans = false; }
            else {
                inst = q.front();
                q.pop();
                new_trans = true;
            }
            return 0;
        };
};

#endif //MISC_H