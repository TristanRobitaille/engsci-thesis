#include <queue>

#ifndef MISC_H
#define MISC_H

/*----- DEFINE -----*/
#define NUM_CIM 64
#define PATCH_LENGTH_NUM_SAMPLES 64
#define NUM_PATCHES 60
#define EMBEDDING_DEPTH 64
#define NUM_ENCODERS 2

/*----- MACROS -----*/
#define UPPER_8b_OF_16b(x) ((x) >> 8)
#define LOWER_8b_OF_16b(x) ((x) & 0x00FF)

/*----- ENUM -----*/
enum OP {
    PATCH_LOAD_BROADCAST_OP, // Broadcast current patch to all CiM, which perform vector-matrix multiplication after each patch
    DATA_STREAM_START_OP, // Indicates that the next x data transmission will contain only parameters data (except op field, so 3B)
    DATA_STREAM, // This instruction contains three bytes of data, and follow DATA_STREAM_START_OP
    NOP // Represents the no tranmission
};

enum SYSTEM_STATE {
    RUNNING,
    INFERENCE_FINISHED
};

/*----- STRUCT -----*/
struct instruction {
    /* Instructions between master controller and CiM */
    OP op;
    int target_cim; // Should be 6 bits
    std::array<float, 2> data; // 2 bytes in ASIC
    float extra_fields; // Opcode-dependent data arrangement
};

struct ext_signals {
    /* Signals external to the master controller and CiM, coming from peripherals or the RISC-V processor */
    bool master_nrst;
    bool new_sleep_epoch;
    bool start_param_load;
};

/*----- CLASS -----*/
/* Counter with overflow behaviour and reset */
class Counter {
    private:
        uint8_t width;
        uint64_t val;
    public:
        Counter(int width) : width(width), val(0) {}
        int inc(uint64_t increment=1) {
            val = val + increment;
            if (val >= (1 << width)) {
                val &= (1 << width) - 1;  // Wrap within width using bitwise AND
            }
            return val;
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
        struct instruction inst;
        std::queue<struct instruction> q;

    public:
        struct instruction get_inst() { return inst; };
        int push_inst(struct instruction inst) {
            q.push(inst);
            return 0;
        };
        struct instruction reset(){
            while (q.size() > 0) { q.pop(); }
            inst = {
                /* op */            NOP,
                /* target_cim */    0,
                /* data */          {0,0},
                /* extra_fields */  0};
            return inst;
        };
        int run() {
            if (q.size() == 0) { inst = reset(); } // Send NOP on bus
            else {
                inst = q.front();
                q.pop();
            }
            return 0;
        };
};

#endif //MISC_H