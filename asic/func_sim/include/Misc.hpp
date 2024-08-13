#ifndef MISC_H
#define MISC_H

#include <array>
#include <queue>
#include <ap_fixed.h>

/*----- DEFINE -----*/
#define NUM_CIM                     64
#define PATCH_LEN                   64
#define NUM_PATCHES                 60
#define EMB_DEPTH                   64
#define MLP_DIM                     32
#define NUM_HEADS                   8
#define NUM_SLEEP_STAGES            5
#define NUM_SAMPLES_OUT_AVG         3     // Number of samples in output averaging filter
#define EEG_SCALE_FACTOR            65535 // Normalize from 16b
#define DATA_BASE_DIR               "../fixed_point_accuracy_study/reference_data/"
#define N_COMP 		                38             
#define Q_COMP                      21
#define NUM_TERMS_EXP_TAYLOR_APPROX 6
#define PRINT_INF_PROGRESS          false

#if DISTRIBUTED_ARCH
#define CIM_PARAMS_STORAGE_SIZE_NUM_ELEM 528
#define CIM_INT_RES_SIZE_NUM_ELEM 886
#elif CENTRALIZED_ARCH
#define CIM_PARAMS_STORAGE_SIZE_NUM_ELEM 31648
#define CIM_INT_RES_SIZE_NUM_ELEM 57116
#endif


/*----- TYPEDEFS -----*/
// All fixed-point types are signed and all except for comp_fx_t have the same number of bits. We adjust the fix point format on each layer.
// Xilinx AP_FIXED format: ap_fixed<width, integer, quantization_mode, overflow_mode, num. sat. bit> (https://docs.amd.com/r/en-US/ug1399-vitis-hls/Arbitrary-Precision-Fixed-Point-Data-Types)
// TODO: Evaluate best quantization and overflow modes
using comp_fx_t         = ap_fixed<N_COMP, N_COMP-Q_COMP, AP_RND_CONV, AP_SAT_SYM>;
using softmax_exp_fx_t  = ap_fixed<N_COMP, N_COMP-Q_COMP, AP_RND_CONV, AP_SAT_SYM>;

using params_fx_2_x_t   = ap_fixed<N_STO_PARAMS, 2, AP_RND_CONV, AP_SAT_SYM>; // ]-2, 2[
using params_fx_3_x_t   = ap_fixed<N_STO_PARAMS, 3, AP_RND_CONV, AP_SAT_SYM>; // ]-4, 4[
using params_fx_4_x_t   = ap_fixed<N_STO_PARAMS, 4, AP_RND_CONV, AP_SAT_SYM>; // ]-8, 8[
using params_fx_5_x_t   = ap_fixed<N_STO_PARAMS, 5, AP_RND_CONV, AP_SAT_SYM>; // ]-16, 16[

using sw_fx_1_x_t       = ap_fixed<N_STO_INT_RES, 1, AP_RND_CONV, AP_SAT_SYM>; // ]-1, 1[
using sw_fx_2_x_t       = ap_fixed<N_STO_INT_RES, 2, AP_RND_CONV, AP_SAT_SYM>; // ]-2, 2[
using sw_fx_5_x_t       = ap_fixed<N_STO_INT_RES, 5, AP_RND_CONV, AP_SAT_SYM>; // ]-16, 16[
using sw_fx_6_x_t       = ap_fixed<N_STO_INT_RES, 6, AP_RND_CONV, AP_SAT_SYM>; // ]-32, 32[
using dw_fx_x_t         = ap_fixed<2*N_STO_INT_RES, 8, AP_RND_CONV, AP_SAT_SYM>; // TODO: Reduce num integer bits

/*----- MACROS -----*/
#if DISTRIBUTED_ARCH
#define NUM_TRANS(x) ceil((x)/3.0f) // Returns the number of transactions each CiM will send (3 elements per transaction)
#endif //DISTRIBUTED_ARCH

/*----- ENUM -----*/
#if DISTRIBUTED_ARCH
enum OP {
    NOP, // Represents the no tranmission
    PATCH_LOAD_BROADCAST_START_OP, // Broadcast the start of a new patch to all CiM
    PATCH_LOAD_BROADCAST_OP, // Broadcast current patch to all CiM, which perform vector-matrix multiplication after each patch
    DENSE_BROADCAST_START_OP, // Tell the target CiM that it needs to broadcast its data starting at a given addr and length. Non-target CiM will then listen to the broadcast and perform MAC once the full vector is received.
    DENSE_BROADCAST_DATA_OP, // Sent from a CiM. Contains 3 bytes of data
    PARAM_STREAM_START_OP, // Indicates that the next x data transmission will contain only parameters data (except op field, so 3B)
    PARAM_STREAM_OP, // This instruction contains three bytes of data, and follow PARAM_STREAM_START_OP
    TRANS_BROADCAST_START_OP, // Tell the target CiM that it needs to broadcast its data starting at a given addr and length. Non-target CiM will then listen to the broadcast and grab the data they need.
    TRANS_BROADCAST_DATA_OP, // Sent from a CiM. Contains 3 bytes of data
    PISTOL_START_OP, // Used to instruct CiMs to move to their next step in the inference pipeline
    INFERENCE_RESULT_OP // Sent from CiM #0 to master. Contains inference result.
};
#endif //DISTRIBUTED_ARCH

enum SYSTEM_STATE {
    IDLE,
    RUNNING,
    EVERYTHING_FINISHED
};

enum DATA_WIDTH {
    SINGLE_WIDTH = 1,
    DOUBLE_WIDTH = 2
};

enum DIRECTION {
    VERTICAL,
    HORIZONTAL
};

/*----- STRUCT -----*/
#if DISTRIBUTED_ARCH
struct instruction {
    /* Instructions between master controller and CiM */
    OP op;
    int target_or_sender; // Should be 6 bits (represents target CiM for all ops except TRANSPOSE_BROADCAST_DATA_OP)
    std::array<float, 3> data; // 3 words in ASIC
    DATA_WIDTH data_width;
};
#endif //DISTRIBUTED_ARCH

struct ext_signals {
    /* Signals external to the master controller and CiM, coming from peripherals or the RISC-V processor */
    bool master_nrst;
    bool start_param_load;
    bool new_sleep_epoch;
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

        int reset() {
            val = 0;
            val_unbounded = 0;
            return 0;
        }

        int get_cnt() { return val;}
};

/* Bus */
#if DISTRIBUTED_ARCH
class Bus {
    private:
        bool hold_on_bus = false; // Allows instruction to remain on bus until next instruction is pushed
        struct instruction inst;
        std::queue<struct instruction> q;
        uint32_t _num_transpose_data_op_sent = 0; // Used to track the number of transpose data ops (for debugging purposes)
        uint32_t _num_dense_broadcast_data_op_sent = 0; // Used to track the number of dense transpose data ops (for debugging purposes)

    public:
        struct instruction get_inst() { return inst; };
        int push_inst(struct instruction inst, bool hold=false) {
            if (inst.target_or_sender > NUM_CIM) {
                std::cout << "Invalid target or sender (" << inst.target_or_sender << "; out of range). Exiting." << std::endl;
                exit(-1);
            }
            if (hold_on_bus == true) { q.pop(); } // If we were already holding an instruction on the bus, need to pop it so we can push the new one
            hold_on_bus = hold;
            q.push(inst);
            return 0;
        };
        struct instruction reset(){
            while (q.size() > 0) { q.pop(); }
            inst = {
                /* op */                NOP,
                /* target_or_sender */  0,
                /* data */              {0,0,0}};
            return inst;
        };
        int run() {
            if (q.size() > 1) throw std::runtime_error("Bus queue overflow (more than 1 instruction, which would be a short in ASIC)");
            if (q.size() == 0) { inst = reset(); } // Send NOP on bus
            else {
                inst = q.front();
                if (hold_on_bus == false) { q.pop(); }
                if (inst.op == TRANS_BROADCAST_DATA_OP) { _num_transpose_data_op_sent++; }
                if (inst.op == DENSE_BROADCAST_DATA_OP) { _num_dense_broadcast_data_op_sent++; }
            }
            return 0;
        };
};
#endif //DISTRIBUTED_ARCH

#endif //MISC_H
