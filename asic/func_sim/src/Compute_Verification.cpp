#include <Compute_Verification.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- GLOBAL -----*/
float min_val = 0.0f;
float max_val = 0.0f;

/*----- DECLARATION -----*/
bool are_equal(float a, float b, uint16_t index, uint8_t id) {
    float diff = abs(a - b);
    min_val = (b < min_val) ? b : min_val;
    max_val = (b > max_val) ? b : max_val;

    if ((diff > REL_TOLERANCE*abs(b)) && (diff > ABS_TOLERANCE)) {
        std::cout << "Mismatch for CiM #" << (uint16_t) id << " at index " << index << ": Expected: " << a << ", got " << b << " (error: " << 100*(a-b)/a << "%)" << std::endl;
        return false;
    }
    return true;
}

void verify_computation(COMPUTE_VERIFICATION_STEP cim_state, uint8_t id, float* data, uint16_t starting_addr) {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }

    if ((cim_state == ENC_MHSA_DENSE_QK_T_VERIF) || (cim_state == ENC_SOFTMAX_VERIF)) {
        for (int head = 0; head < NUM_HEADS; head++) {
            std::string filename = step_verif_info[cim_state].csv_fp + std::to_string(head) + ".csv";
            rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));
            std::vector<float> ref_data;
            if (cim_state == ENC_MHSA_DENSE_QK_T_VERIF) { ref_data = csv.GetColumn<float>(id); }
            else if (cim_state == ENC_SOFTMAX_VERIF) { ref_data = csv.GetRow<float>(id); }
            for (int i = 0; i < ref_data.size(); i++) { are_equal(ref_data[i], data[i+starting_addr+head*(NUM_PATCHES+1)], i, id); }
        }
    } else if (cim_state == POST_SOFTMAX_AVG_VERIF) {
        std::vector<float> ref_accumulator;
        for (int i = 0; i < NUM_SLEEP_STAGES; i++) { // Softmax for current epoch
            ref_accumulator.push_back(data[MLP_DIM+i]); // Softmax for current epoch
        }
        for (int i = 0; i < (NUM_SAMPLES_OUT_AVG-1); i++) { // Softmax for previous dummy epochs
            std::string filename = step_verif_info[cim_state].csv_fp + std::to_string(i) + ".csv";
            rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));
            std::vector<float> dummy_softmax = csv.GetRow<float>(id);
            for (int j = 0; j < NUM_SLEEP_STAGES; j++) { ref_accumulator[j] += dummy_softmax[j] / NUM_SAMPLES_OUT_AVG; }
        }
        for (int i = 0; i < NUM_SLEEP_STAGES; i++) { are_equal(ref_accumulator[i], data[starting_addr+i], i, id); }
    } else {
        rapidcsv::Document csv(step_verif_info[cim_state].csv_fp, rapidcsv::LabelParams(-1, -1));
        if (cim_state == ENC_OUT_VERIF) { 
            std::vector<float> col = csv.GetColumn<float>(id);
            are_equal(col[0], data[starting_addr], 0, id); // Only check the first row since we are not computing the rest
        } else {
            std::vector<float> ref_data;
            if (cim_state == MLP_HEAD_LAYERNORM_VERIF || cim_state == MLP_HEAD_DENSE_1_VERIF || cim_state == MLP_HEAD_SOFTMAX_VERIF) { ref_data = csv.GetRow<float>(id); }
            else { ref_data = csv.GetColumn<float>(id); }
            for (int i = 0; i < ref_data.size(); i++) { 
                if (cim_state == MLP_HEAD_SOFTMAX_DIV_VERIF) { are_equal(ref_data[i]/NUM_SAMPLES_OUT_AVG, data[i+starting_addr], i, id); }
                else { are_equal(ref_data[i], data[i+starting_addr], i, id); }
            }
        }
    }
}

void verify_result(RESULT_TYPE type, float result, float* input_data, uint16_t starting_addr, uint16_t len, uint8_t id) {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }

    float data[len];
    float reference_result = 0.0f;
    for (int i = 0; i < len; i++) { data[i] = input_data[i+starting_addr]; }
    arma::fvec arma_data(data, len);

    if (type == MEAN) { reference_result = arma::mean(arma_data); }
    else if (type == VARIANCE) { reference_result = arma::var(arma_data, 1 /*norm_type of 1 to divide by N to match TF*/); }

    are_equal(reference_result, result, -1, id);
}

void verify_softmax_storage(float* intermediate_res, float* prev_softmax_storage) {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }

    for (int i = 0; i < (NUM_SAMPLES_OUT_AVG-2); i++) {
        for (int j = 0; j < NUM_SLEEP_STAGES; j++) {
            // Check that the previous sleep epochs have been shifted
            std::string filename = step_verif_info[POST_SOFTMAX_AVG_VERIF].csv_fp + std::to_string(i) + ".csv";
            rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));
            std::vector<float> dummy_softmax = csv.GetRow<float>(0);
            are_equal(dummy_softmax[j] / NUM_SAMPLES_OUT_AVG, prev_softmax_storage[j+(i+1)*NUM_SLEEP_STAGES], j, 0);
            
            // Check that the current sleep epoch's softmax got moved to the previous sleep epochs softmax storage
            rapidcsv::Document csv_current(step_verif_info[MLP_HEAD_SOFTMAX_VERIF].csv_fp, rapidcsv::LabelParams(-1, -1));
            std::vector<float> ref_softmax = csv_current.GetColumn<float>(0);
            are_equal(ref_softmax[j] / NUM_SAMPLES_OUT_AVG, prev_softmax_storage[j], j, 0);
        }
    }
}

void print_intermediate_value_stats() {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }
    std::cout << "Min. intermediate value: " << min_val << std::endl;
    std::cout << "Max. intermediate value: " << max_val << std::endl;
}