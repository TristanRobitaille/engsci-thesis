#ifndef MISC_H
#define MISC_H

#include <queue>

/*----- DEFINE -----*/
#define NUM_CIM 64
#define PATCH_LEN 64 // Patch length (in number of samples)
#define NUM_PATCHES 60
#define EMB_DEPTH 64
#define MLP_DIM 32
#define NUM_HEADS 8
#define NUM_SLEEP_STAGES 5
#define NUM_SAMPLES_OUT_AVG 3 // Number of samples in output averaging filter
#define EEG_SCALE_FACTOR 65535 // Normalize from 16b

/*----- MACROS -----*/
#define NUM_TRANS(x) ceil((x)/3.0f) // Returns the number of transactions each CiM will send (3 elements per transaction)

/*----- ENUM -----*/
enum OP {
    PATCH_LOAD_BROADCAST_START_OP, // Broadcast the start of a new patch to all CiM
    PATCH_LOAD_BROADCAST_OP, // Broadcast current patch to all CiM, which perform vector-matrix multiplication after each patch
    DENSE_BROADCAST_START_OP, // Tell the target CiM that it needs to broadcast its data starting at a given addr and length. Non-target CiM will then listen to the broadcast and perform MAC once the full vector is received.
    DENSE_BROADCAST_DATA_OP, // Sent from a CiM. Contains 3 bytes of data
    DATA_STREAM_START_OP, // Indicates that the next x data transmission will contain only parameters data (except op field, so 3B)
    DATA_STREAM_OP, // This instruction contains three bytes of data, and follow DATA_STREAM_START_OP
    TRANS_BROADCAST_START_OP, // Tell the target CiM that it needs to broadcast its data starting at a given addr and length. Non-target CiM will then listen to the broadcast and grab the data they need.
    TRANS_BROADCAST_DATA_OP, // Sent from a CiM. Contains 3 bytes of data
    PISTOL_START_OP, // Used to instruct CiMs to move to their next step in the inference pipeline
    INFERENCE_RESULT_OP, // Sent from CiM #0 to master. Contains inference result.
    NOP // Represents the no tranmission
};

enum SYSTEM_STATE {
    RUNNING,
    EVERYTHING_FINISHED
};

/*----- STRUCT -----*/
struct instruction {
    /* Instructions between master controller and CiM */
    OP op;
    int target_or_sender; // Should be 6 bits (represents target CiM for all ops except TRANSPOSE_BROADCAST_DATA_OP)
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
        uint16_t width;
        int val;
        int val_unbounded;
        int max_val_seen; // Keep track of the maximum value seen by the counter to determine the number of bits needed to represent the counter
    public:
        Counter(int width) : width(width), val(0), max_val_seen(0) {}
        int inc(int increment=1) {
            val += increment;
            val_unbounded += increment;
            if (val_unbounded > max_val_seen) { max_val_seen = val_unbounded; }
            if (val >= (1 << width)) {
                std::cout << "Counter overflow (width: " << width << ")!" << std::endl;
                val &= (1 << width) - 1; // Wrap within width using bitwise AND
            }
            return val;
        }

        int dec(int decrement=1) {
            val -= decrement;
            val_unbounded -= decrement;
            if (val < 0) {
                std::cout << "Counter underflow (width: " << width << "!\n" << std::endl;
                val = (1 << width) + val; // Wrap around
            }
            return val;
        }

        int set_val(int new_val) {
            val = new_val;
            val_unbounded = new_val;
            return val;
        }

        int reset() {
            val = 0;
            val_unbounded = 0;
            return 0;
        }

        int get_cnt() { return val;}
};

/* Bus */
class Bus {
    private:
        struct instruction inst;
        std::queue<struct instruction> q;
        uint32_t _num_transpose_data_op_sent = 0; // Used to track the number of transpose data ops (for debugging purposes)
        uint32_t _num_dense_broadcast_data_op_sent = 0; // Used to track the number of dense transpose data ops (for debugging purposes)

    public:
        struct instruction get_inst() { return inst; };
        int push_inst(struct instruction inst) {
            if (inst.target_or_sender > NUM_CIM) throw std::runtime_error("Invalid target or sender (out of range)");
            q.push(inst);
            return 0;
        };
        struct instruction reset(){
            while (q.size() > 0) { q.pop(); }
            inst = {
                /* op */                NOP,
                /* target_or_sender */  0,
                /* data */              {0,0},
                /* extra_fields */      0};
            return inst;
        };
        int run() {
            if (q.size() > 1) throw std::runtime_error("Bus queue overflow (more than 1 instruction, which would be a short in ASIC)");
            if (q.size() == 0) { inst = reset(); } // Send NOP on bus
            else {
                inst = q.front();
                q.pop();
                if (inst.op == TRANS_BROADCAST_DATA_OP) { _num_transpose_data_op_sent++; }
                if (inst.op == DENSE_BROADCAST_DATA_OP) { _num_dense_broadcast_data_op_sent++; }
            }
            return 0;
        };
};

#endif //MISC_H