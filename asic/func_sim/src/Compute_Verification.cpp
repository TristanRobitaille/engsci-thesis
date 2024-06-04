#include <Compute_Verification.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- GLOBAL -----*/
float min_val = 0.0f;
float max_val = 0.0f;
float highest_abs_error = 0.0f;
float highest_rel_error = 0.0f;
vector<float> abs_errors;
vector<float> rel_errors;
uint32_t data_cnt = 0;
uint32_t data_over_threshold_cnt = 0;

/*----- DECLARATION -----*/
bool are_equal(float a, float b, uint16_t index, uint8_t id) {
    float diff = abs(a - b);
    min_val = (b < min_val) ? b : min_val;
    max_val = (b > max_val) ? b : max_val;
    highest_abs_error = (diff > highest_abs_error) ? diff : highest_abs_error;
    abs_errors.push_back(diff);

    if ((diff > REL_TOLERANCE*abs(b)) && (diff > ABS_TOLERANCE)) {
        highest_rel_error = (100*(a-b)/b > highest_rel_error) ? 100*(a-b)/b : highest_rel_error; // Instead if statement to avoid extremely high relative errors when the abolute error is very small
        rel_errors.push_back(100*(a-b)/b);
        std::cout << "Mismatch for CiM #" << (uint16_t) id << " at index " << index << ": Expected: " << a << ", got " << b << " (error: " << 100*(a-b)/b << "%)" << std::endl;
        data_over_threshold_cnt++;
        return false;
    }

    data_cnt++;
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
            if (cim_state == MLP_HEAD_LAYERNORM_VERIF || cim_state == MLP_HEAD_DENSE_1_VERIF) { ref_data = csv.GetRow<float>(id); }
            else { ref_data = csv.GetColumn<float>(id); }
            for (int i = 0; i < ref_data.size(); i++) { 
                if (cim_state == MLP_HEAD_SOFTMAX_DIV_VERIF) { are_equal(ref_data[i]/NUM_SAMPLES_OUT_AVG, data[i+starting_addr], i, id); }
                else { are_equal(ref_data[i], data[i+starting_addr], i, id); }
            }
        }
    }
}

void print_softmax_error(float* data, uint16_t starting_addr) {
    rapidcsv::Document csv(step_verif_info[MLP_HEAD_SOFTMAX_VERIF].csv_fp, rapidcsv::LabelParams(-1, -1));
    std::vector<float> ref_softmax = csv.GetColumn<float>(0);
    std::vector<float> softmax_err_rel, softmax_err_abs;

    cout << "Error on final softmax: ";
    for (int i = 0; i < NUM_SLEEP_STAGES; i++) { 
        softmax_err_rel.push_back(100*(ref_softmax[i] - data[starting_addr+i]) / ref_softmax[i]);
        softmax_err_abs.push_back(ref_softmax[i] - data[starting_addr+i]);
        cout << softmax_err_abs[i] << " (" << softmax_err_rel[i] << "%) ";
    }
    cout << endl;
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

void verify_softmax_storage(float* intermediate_res, uint16_t prev_softmax_base_addr) {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }

    for (int i = 0; i < (NUM_SAMPLES_OUT_AVG-2); i++) {
        for (int j = 0; j < NUM_SLEEP_STAGES; j++) {
            // Check that the previous sleep epochs have been shifted
            std::string filename = step_verif_info[POST_SOFTMAX_AVG_VERIF].csv_fp + std::to_string(i) + ".csv";
            rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));
            std::vector<float> dummy_softmax = csv.GetRow<float>(0);
            are_equal(dummy_softmax[j] / NUM_SAMPLES_OUT_AVG, intermediate_res[prev_softmax_base_addr + j+(i+1)*NUM_SLEEP_STAGES], j, 0);
            
            // Check that the current sleep epoch's softmax got moved to the previous sleep epochs softmax storage
            rapidcsv::Document csv_current(step_verif_info[MLP_HEAD_SOFTMAX_VERIF].csv_fp, rapidcsv::LabelParams(-1, -1));
            std::vector<float> ref_softmax = csv_current.GetColumn<float>(0);
            are_equal(ref_softmax[j] / NUM_SAMPLES_OUT_AVG, intermediate_res[prev_softmax_base_addr + j], j, 0);
        }
    }
}

void print_intermediate_value_stats() {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }
    arma::fvec arma_abs_errors(abs_errors.data(), abs_errors.size());
    arma::fvec arma_rel_errors(rel_errors.data(), rel_errors.size());
    
    std::cout << ">----- COMPUTE VERIFICATION STATS -----<" << std::endl;
    std::cout << "Min. intermediate value: " << min_val << std::endl;
    std::cout << "Max. intermediate value: " << max_val << std::endl;
    std::cout << "Highest valid absolute error: " << highest_abs_error << ". Avg: " << arma::mean(arma_abs_errors) << ". Std. dev.: " << arma::stddev(arma_abs_errors) << std::endl;
    if (rel_errors.size() == 0) { std::cout << "Highest valid relative error: " << highest_rel_error << ". Avg: " << 0 << ". Std. dev.: " << 0 << std::endl; }
    else { std::cout << "Highest valid relative error: " << highest_rel_error << ". Avg: " << arma::mean(arma_rel_errors) << ". Std. dev.: " << arma::stddev(arma_rel_errors) << std::endl; }
    std::cout << data_over_threshold_cnt << "/" << data_cnt << " over threshold (" << (100.0f*data_over_threshold_cnt/data_cnt) << "%)" << std::endl;
}

void reset_stats() {
    min_val = 0.0f;
    max_val = 0.0f;
    highest_abs_error = 0.0f;
    highest_rel_error = 0.0f;
    abs_errors.clear();
    rel_errors.clear();
    data_cnt = 0;
    data_over_threshold_cnt = 0;
}