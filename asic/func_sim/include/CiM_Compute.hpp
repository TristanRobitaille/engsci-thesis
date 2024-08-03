#ifndef CIM_COMPUTE_H
#define CIM_COMPUTE_H

#include <Misc.hpp>
#include <Compute_Verification.hpp>

using namespace std;

/*----- DEFINE -----*/
#define CIM_PARAMS_STORAGE_SIZE_NUM_ELEM 528
#define CIM_INT_RES_SIZE_NUM_ELEM 886

/*----- CLASS -----*/
class CiM_Compute {
    private:
        comp_fx_t compute_temp_fp_1;
        comp_fx_t compute_temp_fp_2;
        comp_fx_t compute_temp_fp_3;
        float softmax_exp_int_res[PATCH_LEN];

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

        /*----- VARIABLES ----*/
        float computation_result;
        bool compute_in_progress;
        float params[CIM_PARAMS_STORAGE_SIZE_NUM_ELEM];
        float int_res[CIM_INT_RES_SIZE_NUM_ELEM];

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
            compute_in_progress = false;
        }

        template <typename in1_storage_fx_t, typename in2_storage_fx_t>
        void MAC(uint16_t in1_start_addr, uint16_t in2_start_addr, uint16_t len, uint16_t bias_addr, INPUT_TYPE param_type, ACTIVATION activation, DATA_WIDTH data_width) {
            /* Dot-product between two vectors with selectable activation function. data_width only applies to intermediate results. */
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start MAC!"); }
            if (param_type != INTERMEDIATE_RES && param_type != MODEL_PARAM) { throw runtime_error("Invalid parameter type for MAC!"); }

            compute_in_progress = true;

            compute_temp_fp_1 = comp_fx_t { 0 }; // Accumulator
            compute_temp_fp_2 = comp_fx_t { static_cast<in2_storage_fx_t>(params[bias_addr]) };

            uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;

            // MAC
            uint16_t params_cnt = 0;
            for (uint16_t i = 0; i < stride*len; i += stride) {
                if (param_type == INTERMEDIATE_RES) { compute_temp_fp_1 += comp_fx_t { static_cast<in2_storage_fx_t>(int_res[in2_start_addr+i]) } * comp_fx_t { static_cast<in1_storage_fx_t>(int_res[in1_start_addr+i]) }; }
                else if (param_type == MODEL_PARAM) { compute_temp_fp_1 += comp_fx_t { static_cast<in2_storage_fx_t>(params[in2_start_addr+params_cnt]) } * comp_fx_t { static_cast<in1_storage_fx_t>(int_res[in1_start_addr+i]) }; }
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

            if (in2_type == MODEL_PARAM) { compute_temp_fp_1 = comp_fx_t { int_res[num_addr] } / comp_fx_t { params[in2] }; } // Second input is a model parameter
            else if (in2_type == IMMEDIATE_VAL) { compute_temp_fp_1 = comp_fx_t { int_res[num_addr] } / comp_fx_t { in2 }; } // Second input is an immediate value
            else if (in2_type == ADC_INPUT) { compute_temp_fp_1 = comp_fx_t { int_res[num_addr] / in2 }; } // Second input is a ADC input (16b)
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
            compute_temp_fp_1 = comp_fx_t { static_cast<in1_storage_fx_t>(int_res[in1_addr]) };

            if (in2_type == INTERMEDIATE_RES) { compute_temp_fp_2 = comp_fx_t { static_cast<in2_storage_fx_t>(int_res[in2_addr]) }; }
            else if (in2_type == MODEL_PARAM) { compute_temp_fp_2 = comp_fx_t { static_cast<in2_storage_fx_t>(params[in2_addr]) }; }
            else {
                cout << "Received unknown parameter type " << in2_type << endl;
                exit(-1);
            }
            compute_temp_fp_1 += compute_temp_fp_2;
            computation_result = static_cast<float>(compute_temp_fp_1);
        }

        template <typename storage_fx_t>
        void LAYERNORM_1ST_HALF(uint8_t id, uint16_t input_addr, DATA_WIDTH data_width) {
            /* 1st half of Layer normalization of input. Input is in intermediate storage location. Note: All LayerNorms are done over a row of EMB_DEPTH length. */
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start LAYERNORM_1ST_HALF!"); }
            compute_in_progress = true; // Using this compute_in_progress signal as it will be used in the CiM (in ASIC, this compute is multi-cycle, so leaving this here for visibility)

            compute_temp_fp_1 = comp_fx_t { 0.0f }; // Mean
            compute_temp_fp_2 = comp_fx_t { 0.0f }; // Variance
            compute_temp_fp_3 = comp_fx_t { 0.0f }; // Variance

            uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;

            // Mean
            for (uint16_t i = 0; i < stride*EMB_DEPTH; i += stride) { compute_temp_fp_1 += comp_fx_t { static_cast<storage_fx_t>(int_res[input_addr+i]) }; }
            compute_temp_fp_1 /= EMB_DEPTH;
            verify_result(MEAN, static_cast<float>(compute_temp_fp_1), int_res, input_addr, EMB_DEPTH, id, DOUBLE_WIDTH);

            // Variance
            for (uint16_t i = 0; i < stride*EMB_DEPTH; i += stride) {
                compute_temp_fp_3 = comp_fx_t { static_cast<storage_fx_t>(int_res[input_addr+i]) } - compute_temp_fp_1; // Subtract
                compute_temp_fp_3 *= compute_temp_fp_3; // Square
                compute_temp_fp_2 += compute_temp_fp_3; // Sum
            }

            compute_temp_fp_2 /= EMB_DEPTH;
            verify_result(VARIANCE, static_cast<float> (compute_temp_fp_2), int_res, input_addr, EMB_DEPTH, id, DOUBLE_WIDTH);
            compute_temp_fp_2 = SQRT(compute_temp_fp_2); // Standard deviation
            if (compute_temp_fp_2 == comp_fx_t {0.0f}) { compute_temp_fp_3 = POW(comp_fx_t{2}, N_COMP); } // Avoid division by zero by saturating
            else { compute_temp_fp_3 = comp_fx_t {1.0f} / compute_temp_fp_2; } // Inverse standard deviation (so we can simply multiply instead of divide in the next step)

            // Partial normalization (excludes gamma and beta, which are applied in LAYERNORM_2ND_HALF() since they need to be applied column-wise)
            for (uint16_t i = 0; i < stride*EMB_DEPTH; i += stride) {
                float result = static_cast<float>((comp_fx_t { static_cast<storage_fx_t>(int_res[input_addr+i]) } - compute_temp_fp_1) * compute_temp_fp_3);
                int_res_write(result, input_addr+i, data_width);
            }
        }

        template <typename in1_storage_fx_t, typename in2_storage_fx_t>
        void LAYERNORM_2ND_HALF(uint16_t input_addr, uint16_t gamma_addr, uint16_t beta_addr, DATA_WIDTH data_width) {
            /* 2nd half of Layer normalization of input. This applies gamma and beta on each column. */
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start LAYERNORM_2ND_HALF!"); }
            compute_in_progress = true;

            compute_temp_fp_2 = comp_fx_t { static_cast<in2_storage_fx_t>(params[gamma_addr]) }; // Gamma
            compute_temp_fp_3 = comp_fx_t { static_cast<in2_storage_fx_t>(params[beta_addr]) }; // Beta

            uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;

            // Normalize
            for (uint16_t i = 0; i < stride*(NUM_PATCHES+1); i += stride) {
                compute_temp_fp_1 = comp_fx_t { static_cast<in1_storage_fx_t>(int_res[input_addr+i]) };
                float result = static_cast<float>(compute_temp_fp_2 * compute_temp_fp_1 + compute_temp_fp_3);
                int_res_write(result, input_addr+i, data_width);
            }
        }

        template <typename storage_fx_t>
        void SOFTMAX(uint16_t input_addr, uint16_t len, DATA_WIDTH data_width) {
            /* Softmax of input (performed in-place). Input is in intermediate storage location. Note: All Softmax are done over a row of len length. */
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start SOFTMAX!"); }
            compute_in_progress = true;

            compute_temp_fp_1 = comp_fx_t { 0.0 };
            compute_temp_fp_2 = comp_fx_t { 0.0 };

            uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;

            // Exponentiate all elements and sum
            for (uint16_t i = 0; i < stride*len; i += stride) {
                if (comp_fx_t { int_res[input_addr+i] } < comp_fx_t { 0.0 }) { _neg_exp_cnt++; }
                if (comp_fx_t { int_res[input_addr+i] } < _min_exp_input_arg) { _min_exp_input_arg = comp_fx_t { int_res[input_addr+i] }; }
                if (comp_fx_t { int_res[input_addr+i] } > _max_exp_input_arg) { _max_exp_input_arg = comp_fx_t { int_res[input_addr+i] }; }
                _total_exp_cnt++;
                compute_temp_fp_2 = EXP_APPROX(comp_fx_t { static_cast<storage_fx_t>(int_res[input_addr+i]) });
                softmax_exp_int_res[i] = static_cast<float> (compute_temp_fp_2);
                compute_temp_fp_1 += compute_temp_fp_2;
            }

            // Normalize
            for (uint16_t i = 0; i < stride*len; i += stride) {
                int_res[input_addr+i] = static_cast<float>((comp_fx_t { static_cast<softmax_exp_fx_t>(softmax_exp_int_res[i]) } / compute_temp_fp_1));
            }
        }

        void ARGMAX(uint16_t input_addr, uint16_t len, DATA_WIDTH data_width) {
            /* Index of maximum element of input. Input is in intermediate storage location. Note: All Max are done over a row of len length. */
            if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start MAX!"); }
            compute_in_progress = true;

            compute_temp_fp_3 = comp_fx_t { 0 }; // Index of maximum element
            compute_temp_fp_1 = comp_fx_t { int_res[input_addr] }; // Value of maximum element

            uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;

            for (uint16_t i = 1; i < stride*len; i += stride) {
                compute_temp_fp_2 = comp_fx_t { int_res[input_addr+i] };
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
            /* Performs floor() on fixed-point input */
            return comp_fx_t { floor(static_cast<float>(input)) };
        }

        void int_res_write(float data, uint16_t index, DATA_WIDTH data_width) {
            int_res[index] = data;
            if (data_width == DOUBLE_WIDTH) {
                int_res[index + 1] = data;
            }
        }
};

#endif //CIM_COMPUTE_H
