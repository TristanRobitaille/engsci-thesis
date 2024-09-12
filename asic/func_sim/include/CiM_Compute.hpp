#ifndef CIM_COMPUTE_H
#define CIM_COMPUTE_H

#include <Misc.hpp>
#include <Compute_Verification.hpp>

using namespace std;

/*----- DEFINE -----*/
#define COMPUTE_CNT_THRESHOLD 3 // Used to simulate the delay in the computation to match the real hardware

/*----- CLASS -----*/
class CiM_Compute {
    private:
        comp_fx_t compute_temp_fp_1;
        comp_fx_t compute_temp_fp_2;
        comp_fx_t compute_temp_fp_3;
        uint16_t _num_compute_done; // [Not in ASIC] Counter used to track the number of computations done in a given inference step
        uint16_t _compute_process_cnt; // [Not in ASIC] Counter used to track the progress of the current computation (used to simulate the delay in the computation to match the real hardware)
        float softmax_exp_int_res[NUM_PATCHES+1];

    public:
        /*----- ENUM -----*/
        enum INPUT_TYPE { // Type of input for a given computation
            MODEL_PARAM,
            INTERMEDIATE_RES,
            IMMEDIATE_VAL,
            ADC_INPUT
        };

        enum ACTIVATION {
            NO_ACTIVATION, // Used for simple matrix multiplies (no bias)
            LINEAR_ACTIVATION,
            SWISH_ACTIVATION
        };

        enum DIRECTION { // Use in MAC to decide indexing direction for the dot-product
            VERTICAL,
            HORIZONTAL
        };

        /*----- VARIABLES ----*/
        float computation_result;
        bool compute_in_progress;
        bool compute_done;
#if DISTRIBUTED_ARCH
        float params[CIM_PARAMS_BANK_SIZE_NUM_WORD];
        float int_res[CIM_INT_RES_BANK_SIZE_NUM_WORD];
#elif CENTRALIZED_ARCH
        float params_0[CIM_PARAMS_BANK_SIZE_NUM_WORD];
        float params_1[CIM_PARAMS_BANK_SIZE_NUM_WORD];
        static float int_res_0[CIM_INT_RES_BANK_SIZE_NUM_WORD];
        static float int_res_1[CIM_INT_RES_BANK_SIZE_NUM_WORD];
        static float int_res_2[CIM_INT_RES_BANK_SIZE_NUM_WORD];
        static float int_res_3[CIM_INT_RES_BANK_SIZE_NUM_WORD];
#endif
        // Metrics
        uint32_t _neg_exp_cnt; // [Not in ASIC] Track # of exponentials that have a negative argument
        uint32_t _total_exp_cnt; // [Not in ASIC] Track the # of exponentials performed
        comp_fx_t _max_exp_input_arg; // [Not in ASIC] Track the maximum input argument to the exponential
        comp_fx_t _min_exp_input_arg; // [Not in ASIC] Track the minimum input argument to the exponential

        /*----- DEFINITION -----*/
        void reset_comp() {
            compute_temp_fp_1 = comp_fx_t { 0 };
            compute_temp_fp_2 = comp_fx_t { 0 };
            compute_temp_fp_3 = comp_fx_t { 0 };
            computation_result = 0.0f;
            _num_compute_done = 0;
            _compute_process_cnt = 0;
            _neg_exp_cnt = 0;
            _total_exp_cnt = 0;
            _max_exp_input_arg = comp_fx_t(0);
            _min_exp_input_arg = comp_fx_t(0);
            compute_in_progress = false;
        }

        void int_res_write(float data, uint32_t addr, DATA_WIDTH data_width) {
#if DISTRIBUTED_ARCH
            int_res[addr] = data;
            if (data_width == DOUBLE_WIDTH) { int_res[addr + 1] = data; }
#elif CENTRALIZED_ARCH
            // To maintain single-cycle access, we split the int_res into 4 banks and write the two halves of the double-width data in different banks (0/2 and 1/3)
            // If a word is double-width, the MSH (most-significant half) is always in bank 0 or 1 and the LSH (least-significant half) is always in bank 2 or 3
            // Note: For the purposes of the functional simulation, we simply write the same float in both banks for double-width data
            if (addr < CIM_INT_RES_BANK_SIZE_NUM_WORD) {
                int_res_0[addr] = data;
                if (data_width == DOUBLE_WIDTH) { int_res_2[addr] = data; } // LSH in bank 2
            } else if (addr < 2 * CIM_INT_RES_BANK_SIZE_NUM_WORD) {
                int_res_1[addr - CIM_INT_RES_BANK_SIZE_NUM_WORD] = data;
                if (data_width == DOUBLE_WIDTH) { int_res_3[addr - CIM_INT_RES_BANK_SIZE_NUM_WORD] = data; } // LSH in bank 3
            } else if (addr < 3 * CIM_INT_RES_BANK_SIZE_NUM_WORD) {
                int_res_2[addr - 2*CIM_INT_RES_BANK_SIZE_NUM_WORD] = data;
                if (data_width == DOUBLE_WIDTH) { int_res_0[addr - 2*CIM_INT_RES_BANK_SIZE_NUM_WORD] = data; } // MSH in bank 0
            } else if (addr < 4 * CIM_INT_RES_BANK_SIZE_NUM_WORD) {
                int_res_3[addr - 3*CIM_INT_RES_BANK_SIZE_NUM_WORD] = data;
                if (data_width == DOUBLE_WIDTH) { int_res_1[addr - 3*CIM_INT_RES_BANK_SIZE_NUM_WORD] = data; } // MSH in bank 1
            } else { throw runtime_error("Invalid address for int res write!"); }
#endif
        }

#if DISTRIBUTED_ARCH
        float int_res_read(uint32_t addr) {
            return int_res[addr];
#elif CENTRALIZED_ARCH
        static float int_res_read(uint32_t addr) {
            if (addr < CIM_INT_RES_BANK_SIZE_NUM_WORD) { return int_res_0[addr]; }
            else if (addr < 2 * CIM_INT_RES_BANK_SIZE_NUM_WORD) { return int_res_1[addr - CIM_INT_RES_BANK_SIZE_NUM_WORD]; }
            else if (addr < 3 * CIM_INT_RES_BANK_SIZE_NUM_WORD) { return int_res_2[addr - 2*CIM_INT_RES_BANK_SIZE_NUM_WORD]; }
            else if (addr < 4 * CIM_INT_RES_BANK_SIZE_NUM_WORD) { return int_res_3[addr - 3*CIM_INT_RES_BANK_SIZE_NUM_WORD]; }
            else { throw runtime_error("Invalid address for int res read!"); }
#endif
        }

        void param_write(float data, uint32_t addr) {
#if DISTRIBUTED_ARCH
            params[addr] = data;
#elif CENTRALIZED_ARCH
            if (addr < CIM_PARAMS_BANK_SIZE_NUM_WORD) { params_0[addr] = data; }
            else if (addr < 2 * CIM_PARAMS_BANK_SIZE_NUM_WORD) { params_1[addr - CIM_PARAMS_BANK_SIZE_NUM_WORD] = data; }
            else { throw runtime_error("Invalid address for parameter write!"); }
#endif
        }

        float param_read(uint32_t addr) {
#if DISTRIBUTED_ARCH
            return params[addr];
#elif CENTRALIZED_ARCH
            if (addr < CIM_PARAMS_BANK_SIZE_NUM_WORD) { return params_0[addr]; }
            else if (addr < 2 * CIM_PARAMS_BANK_SIZE_NUM_WORD) { return params_1[addr - CIM_PARAMS_BANK_SIZE_NUM_WORD]; }
            else { throw runtime_error("Invalid address for parameter read!"); }
#endif
        }

        template <typename in1_storage_fx_t, typename in2_storage_fx_t>
#if DISTRIBUTED_ARCH
        void MAC(uint32_t in1_start_addr, uint32_t in2_start_addr, uint16_t len, uint32_t bias_addr, INPUT_TYPE param_type, ACTIVATION activation, DATA_WIDTH data_width, DIRECTION dir_in2=HORIZONTAL, uint16_t matrix_width=1) {
#elif CENTRALIZED_ARCH
        void MAC(uint32_t in1_start_addr, uint32_t in2_start_addr, uint16_t len, uint32_t bias_addr, INPUT_TYPE param_type, ACTIVATION activation, DIRECTION dir_in2=HORIZONTAL, uint16_t matrix_width=1) {
#endif
            /* Dot-product between two vectors with selectable activation function. data_width only applies to intermediate results. width is used when dir_in2 is VERTICAL.*/
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start MAC!"); }
            if (param_type != INTERMEDIATE_RES && param_type != MODEL_PARAM) { throw runtime_error("Invalid parameter type for MAC!"); }

            compute_in_progress = true;

            compute_temp_fp_1 = comp_fx_t { 0 }; // Accumulator
            compute_temp_fp_2 = comp_fx_t { static_cast<in2_storage_fx_t>(param_read(bias_addr)) };

#if DISTRIBUTED_ARCH
            uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;
#elif CENTRALIZED_ARCH
            uint16_t stride = 1;
#endif
            // MAC
            uint16_t params_cnt = 0;
            for (uint16_t i = 0; i < stride*len; i += stride) {
                uint32_t in1_addr = in1_start_addr + i;
                comp_fx_t input1 = comp_fx_t { static_cast<in1_storage_fx_t>(int_res_read(in1_addr)) };
                comp_fx_t input2;

                if (param_type == INTERMEDIATE_RES) {
                    uint32_t in2_addr = (dir_in2 == HORIZONTAL) ? in2_start_addr + i : in2_start_addr + i*matrix_width;
                    input2 = comp_fx_t { static_cast<in2_storage_fx_t>(int_res_read(in2_addr)) };
                } else if (param_type == MODEL_PARAM) { // Params are always stored in single-width
                    uint32_t in2_addr = in2_start_addr + params_cnt;
                    input2 = comp_fx_t { static_cast<in2_storage_fx_t>(param_read(in2_addr)) };
                }
                compute_temp_fp_1 += input1 * input2;
                params_cnt++;
            }

            // Activation
            switch (activation) {
            case LINEAR_ACTIVATION:
                compute_temp_fp_1 += compute_temp_fp_2;
                break;
            case SWISH_ACTIVATION:
                if ((-compute_temp_fp_1 - compute_temp_fp_2) < comp_fx_t {0.0}) { _neg_exp_cnt++; }
                _total_exp_cnt++;
                if ((-compute_temp_fp_1 - compute_temp_fp_2) < _min_exp_input_arg) { _min_exp_input_arg = -compute_temp_fp_1 - compute_temp_fp_2; }
                if ((-compute_temp_fp_1 - compute_temp_fp_2) > _max_exp_input_arg) { _max_exp_input_arg = -compute_temp_fp_1 - compute_temp_fp_2; }
                compute_temp_fp_1 = compute_temp_fp_1 + compute_temp_fp_2;
                compute_temp_fp_1 = compute_temp_fp_1 / (comp_fx_t {1.0} + EXP_APPROX(-compute_temp_fp_1));
                break;
            case NO_ACTIVATION:
            default:
                break;
            }

            computation_result = static_cast<float>(compute_temp_fp_1);
        }

        void DIV(uint16_t num_addr, uint16_t in2, INPUT_TYPE in2_type) {
            /* Return division of two inputs. */
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start DIV!"); }
            compute_in_progress = true;

            compute_temp_fp_1 = comp_fx_t { 0 };

            if (in2_type == MODEL_PARAM) { compute_temp_fp_1 = comp_fx_t { int_res_read(num_addr) } / comp_fx_t { param_read(in2) }; } // Second input is a model parameter
            else if (in2_type == IMMEDIATE_VAL) { compute_temp_fp_1 = comp_fx_t { int_res_read(num_addr) } / comp_fx_t { in2 }; } // Second input is an immediate value
            else if (in2_type == ADC_INPUT) { compute_temp_fp_1 = comp_fx_t { int_res_read(num_addr) / in2 }; } // Second input is a ADC input (16b)
            else {
                cout << "Received unknown parameter type " << in2_type << endl;
                exit(-1);
            }

            computation_result = static_cast<float>(compute_temp_fp_1);
        }

        template <typename in1_storage_fx_t, typename in2_storage_fx_t>
        void ADD(uint16_t in1_addr, uint16_t in2_addr, INPUT_TYPE in2_type) {
            /* Return sum of two inputs. */
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start ADD!"); }
            compute_in_progress = true;

            // Get data as would be stored in memory
            compute_temp_fp_1 = comp_fx_t { static_cast<in1_storage_fx_t>(int_res_read(in1_addr)) };

            if (in2_type == INTERMEDIATE_RES) { compute_temp_fp_2 = comp_fx_t { static_cast<in2_storage_fx_t>(int_res_read(in2_addr)) }; }
            else if (in2_type == MODEL_PARAM) { compute_temp_fp_2 = comp_fx_t { static_cast<in2_storage_fx_t>(param_read(in2_addr)) }; }
            else {
                cout << "Received unknown parameter type " << in2_type << endl;
                exit(-1);
            }
            compute_temp_fp_1 += compute_temp_fp_2;
            computation_result = static_cast<float>(compute_temp_fp_1);
        }

        template <typename storage_fx_t>
#if DISTRIBUTED_ARCH
        void LAYERNORM_1ST_HALF(uint8_t id, uint16_t input_addr) {
#elif CENTRALIZED_ARCH
        void LAYERNORM_1ST_HALF(uint32_t input_addr, uint32_t output_addr) {
#endif
            /* 1st half of Layer normalization of input. Input is in intermediate storage location. Note: All LayerNorms are done over a row of EMB_DEPTH length. */
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start LAYERNORM_1ST_HALF!"); }
            compute_in_progress = true; // Using this compute_in_progress signal as it will be used in the CiM (in ASIC, this compute is multi-cycle, so leaving this here for visibility)

            compute_temp_fp_1 = comp_fx_t { 0.0f }; // Mean
            compute_temp_fp_2 = comp_fx_t { 0.0f }; // Variance
            compute_temp_fp_3 = comp_fx_t { 0.0f }; // Variance

            // Mean
#if DISTRIBUTED_ARCH
            for (uint16_t i = 0; i < DOUBLE_WIDTH*EMB_DEPTH; i += DOUBLE_WIDTH) { compute_temp_fp_1 += comp_fx_t { static_cast<storage_fx_t>(int_res_read(input_addr+i)) }; }
            compute_temp_fp_1 /= EMB_DEPTH;
            verify_result(MEAN, static_cast<float>(compute_temp_fp_1), int_res, input_addr, EMB_DEPTH, id, DOUBLE_WIDTH);
#elif CENTRALIZED_ARCH
            for (uint16_t i = 0; i < EMB_DEPTH; i++) { compute_temp_fp_1 += comp_fx_t { static_cast<storage_fx_t>(int_res_read(input_addr+i)) }; }
            compute_temp_fp_1 /= EMB_DEPTH;
            verify_result(MEAN, static_cast<float>(compute_temp_fp_1), int_res_read, input_addr, EMB_DEPTH);
#endif
            // Variance
#if DISTRIBUTED_ARCH
            for (uint16_t i = 0; i < DOUBLE_WIDTH*EMB_DEPTH; i += DOUBLE_WIDTH) {
#elif CENTRALIZED_ARCH
            for (uint16_t i = 0; i < EMB_DEPTH; i++) {
#endif
                compute_temp_fp_3 = comp_fx_t { static_cast<storage_fx_t>(int_res_read(input_addr+i)) } - compute_temp_fp_1; // Subtract
                compute_temp_fp_3 *= compute_temp_fp_3; // Square
                compute_temp_fp_2 += compute_temp_fp_3; // Sum
            }

            compute_temp_fp_2 /= EMB_DEPTH;
#if DISTRIBUTED_ARCH
            verify_result(VARIANCE, static_cast<float> (compute_temp_fp_2), int_res, input_addr, EMB_DEPTH, id, DOUBLE_WIDTH);
#elif CENTRALIZED_ARCH
            verify_result(VARIANCE, static_cast<float> (compute_temp_fp_2), int_res_read, input_addr, EMB_DEPTH);
#endif
            compute_temp_fp_2 = SQRT(compute_temp_fp_2); // Standard deviation
            if (compute_temp_fp_2 == comp_fx_t {0.0f}) { compute_temp_fp_3 = POW(comp_fx_t{2}, N_COMP); } // Avoid division by zero by saturating
            else { compute_temp_fp_3 = comp_fx_t {1.0f} / compute_temp_fp_2; } // Inverse standard deviation (so we can simply multiply instead of divide in the next step)

            // Partial normalization (excludes gamma and beta, which are applied in LAYERNORM_2ND_HALF() since they need to be applied column-wise)
#if DISTRIBUTED_ARCH
            for (uint16_t i = 0; i < DOUBLE_WIDTH*EMB_DEPTH; i += DOUBLE_WIDTH) {
#elif CENTRALIZED_ARCH
            for (uint16_t i = 0; i < EMB_DEPTH; i++) {
#endif
                float result = static_cast<float>((comp_fx_t { static_cast<storage_fx_t>(int_res_read(input_addr+i)) } - compute_temp_fp_1) * compute_temp_fp_3);
#if DISTRIBUTED_ARCH
                int_res_write(result, input_addr+i, DOUBLE_WIDTH);
#elif CENTRALIZED_ARCH
                int_res_write(result, output_addr+i, SINGLE_WIDTH);
#endif
            }
        }

        template <typename in1_storage_fx_t, typename in2_storage_fx_t>
#if DISTRIBUTED_ARCH
        void LAYERNORM_2ND_HALF(uint16_t input_addr, uint16_t gamma_addr, uint16_t beta_addr) {
#elif CENTRALIZED_ARCH
        void LAYERNORM_2ND_HALF(uint32_t input_addr, uint32_t gamma_addr, uint32_t beta_addr, uint16_t num_rows) {
#endif
            /* 2nd half of Layer normalization of input. This applies gamma and beta on each column. */
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start LAYERNORM_2ND_HALF!"); }
            compute_in_progress = true;

            compute_temp_fp_2 = comp_fx_t { static_cast<in2_storage_fx_t>(param_read(gamma_addr)) }; // Gamma
            compute_temp_fp_3 = comp_fx_t { static_cast<in2_storage_fx_t>(param_read(beta_addr)) }; // Beta

            // Scale and center
#if DISTRIBUTED_ARCH
            for (uint16_t i = 0; i < NUM_PATCHES+1; i++) { // Go through each row in given column
                compute_temp_fp_1 = comp_fx_t { static_cast<in1_storage_fx_t>(int_res_read(input_addr + DOUBLE_WIDTH*i)) };
                float result = static_cast<float>(compute_temp_fp_2 * compute_temp_fp_1 + compute_temp_fp_3);
                int_res_write(result, input_addr + DOUBLE_WIDTH*i, DOUBLE_WIDTH);
            }
#elif CENTRALIZED_ARCH
            for (uint16_t i = 0; i < num_rows; i++) { // Go through each row in given column
                compute_temp_fp_1 = comp_fx_t { static_cast<in1_storage_fx_t>(int_res_read(input_addr + i*EMB_DEPTH)) }; // Go down column
                float result = static_cast<float>(compute_temp_fp_2 * compute_temp_fp_1 + compute_temp_fp_3);
                int_res_write(result, input_addr + i*EMB_DEPTH, DOUBLE_WIDTH);
            }
#endif
        }

        template <typename storage_fx_t>
        void SOFTMAX(uint32_t input_addr, uint16_t len, DATA_WIDTH data_width) {
            /* Softmax of input (performed in-place). Input is in intermediate storage location. Note: All Softmax are done over a row of len length. */
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start SOFTMAX!"); }
            compute_in_progress = true;

            compute_temp_fp_1 = comp_fx_t { 0.0 };
            compute_temp_fp_2 = comp_fx_t { 0.0 };

#if DISTRIBUTED_ARCH
            uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;
#elif CENTRALIZED_ARCH
            uint16_t stride = 1;
#endif
            // Exponentiate all elements and sum
            for (uint16_t i = 0; i < stride*len; i += stride) {
                if (comp_fx_t { int_res_read(input_addr+i) } < comp_fx_t { 0.0 }) { _neg_exp_cnt++; }
                if (comp_fx_t { int_res_read(input_addr+i) } < _min_exp_input_arg) { _min_exp_input_arg = comp_fx_t { int_res_read(input_addr+i) }; }
                if (comp_fx_t { int_res_read(input_addr+i) } > _max_exp_input_arg) { _max_exp_input_arg = comp_fx_t { int_res_read(input_addr+i) }; }
                _total_exp_cnt++;
                compute_temp_fp_2 = EXP_APPROX(comp_fx_t { static_cast<storage_fx_t>(int_res_read(input_addr+i)) });
                softmax_exp_int_res[i] = static_cast<float> (compute_temp_fp_2);
                compute_temp_fp_1 += compute_temp_fp_2;
            }

            // Normalize
            for (uint32_t i = 0; i < stride*len; i += stride) {
                float result = static_cast<float>((comp_fx_t { static_cast<softmax_exp_fx_t>(softmax_exp_int_res[i]) } / compute_temp_fp_1));
                int_res_write(result, input_addr+i, data_width);
            }
        }

#if DISTRIBUTED_ARCH
        void ARGMAX(uint16_t input_addr, uint16_t len, DATA_WIDTH data_width) {
#elif CENTRALIZED_ARCH
        void ARGMAX(uint16_t input_addr, uint16_t len) {
#endif

            /* Index of maximum element of input. Input is in intermediate storage location. Note: All Max are done over a row of len length. */
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start MAX!"); }
            compute_in_progress = true;

            compute_temp_fp_3 = comp_fx_t { 0 }; // Index of maximum element
            compute_temp_fp_1 = comp_fx_t { int_res_read(input_addr) }; // Value of maximum element

#if DISTRIBUTED_ARCH
            uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;
#elif CENTRALIZED_ARCH
            uint16_t stride = 1;
#endif
            for (uint16_t i = 1; i < len; i++) {
                compute_temp_fp_2 = comp_fx_t { int_res_read(input_addr+stride*i) };
                if (compute_temp_fp_2 > compute_temp_fp_1) {
                    compute_temp_fp_3 = comp_fx_t { i };
                    compute_temp_fp_1 = compute_temp_fp_2;
                }
            }

            computation_result = static_cast<float>(compute_temp_fp_3);
        }

        comp_fx_t EXP_APPROX(comp_fx_t input) {
            /* Approximation of exp(x) as used in the ASIC
            Uses the identy exp(x) = 2^(x/ln(2)), float/int exponent splitting and Taylor approximation of the fractional part (first 4 terms).
            */

            comp_fx_t ln_2 = comp_fx_t{0.69314718056}; //ln(2)
            comp_fx_t input_mapped = input/ln_2;
            comp_fx_t input_mapped_fract = input/ln_2 - FLOOR(input/ln_2);

            // Taylor approximation of the fractional part
            comp_fx_t exp_approx_fract = 1;
            for (int i=0; i<NUM_TERMS_EXP_TAYLOR_APPROX-1; i++) {
                comp_fx_t factorial = 1;
                for (int j=1; j<=i+1; j++) { factorial *= j; }
                exp_approx_fract += POW(ln_2,i+1) / factorial * POW(input_mapped_fract,i+1);
            }

            // Integer part
            comp_fx_t exp_approx_int = POW(comp_fx_t{2}, static_cast<int>(FLOOR(input_mapped)));

            return exp_approx_int * exp_approx_fract;
        }

        comp_fx_t SQRT(comp_fx_t input) {
            // Approximate sqrt. Convert to float, use standard sqrt() function and convert back to fixed-point.
            if (input <= 0) { return comp_fx_t{0}; }

            return comp_fx_t { sqrt(static_cast<float>(input)) };
        }

        comp_fx_t POW(comp_fx_t base, int exp) {
            // Implements base^exp, where exp is an integer.
            if (exp == 0) { return comp_fx_t{1}; }

            comp_fx_t temp = base;
            for (int i=1; i<std::abs(exp); i++){
                temp *= base;
            }

            if (exp > 0) { return temp; }
            else { return comp_fx_t{1}/temp; }
        }

        comp_fx_t FLOOR(comp_fx_t input) {
            // Performs floor() on fixed-point input
            return comp_fx_t { floor(static_cast<float>(input)) };
        }

        uint16_t find_softmax_argmax(uint32_t base_addr) {
            uint16_t argmax_index = 0;
            float softmax_max = 0;
            cout << "Softmax (not averaged): ";
            for (uint32_t i=0; i<NUM_SLEEP_STAGES; i++) {
                cout << int_res_read(base_addr + i) << " ";
                if (int_res_read(base_addr + i) > softmax_max) {
                    softmax_max = int_res_read(base_addr + i);
                    argmax_index = i;
                }
            }
            cout << "--> Argmax index: " << argmax_index << endl;
            return argmax_index;
        }

        void update_compute_process_cnt() {
            if (compute_in_progress == true) { _compute_process_cnt++; }
            if (_compute_process_cnt == COMPUTE_CNT_THRESHOLD) {
                _compute_process_cnt = 0;
                _num_compute_done++;
                compute_in_progress = false;
                compute_done = true;
            }
        }

        void reset_compute_done() {
            compute_done = false;
        }
};

#endif //CIM_COMPUTE_H
