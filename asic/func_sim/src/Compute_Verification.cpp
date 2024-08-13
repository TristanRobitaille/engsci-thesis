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
#if DISTRIBUTED_ARCH
bool are_equal(float a, float b, int32_t index, uint8_t id) {
#elif CENTRALIZED_ARCH
bool are_equal(float a, float b, int32_t index) {
#endif
    float diff = abs(a - b);
    min_val = (b < min_val) ? b : min_val;
    max_val = (b > max_val) ? b : max_val;
    highest_abs_error = (diff > highest_abs_error) ? diff : highest_abs_error;
    abs_errors.push_back(diff);

    if ((diff > REL_TOLERANCE*abs(b)) && (diff > ABS_TOLERANCE)) {
        highest_rel_error = (100*(a-b)/b > highest_rel_error) ? 100*(a-b)/b : highest_rel_error; // Instead if statement to avoid extremely high relative errors when the abolute error is very small
        rel_errors.push_back(100*(a-b)/b);
#if DISTRIBUTED_ARCH
        cout << "Mismatch for CiM #" << (uint16_t) id << " at index " << index << ": Expected: " << a << ", got " << b << " (error: " << 100*(a-b)/b << "%)" << endl;
#elif CENTRALIZED_ARCH
        cout << "Mismatch detected at index " << index << ": Expected: " << a << ", got " << b << " (error: " << 100*(a-b)/b << "%)" << endl;
#endif
        data_over_threshold_cnt++;
        return false;
    }

    data_cnt++;
    return true;
}

#if DISTRIBUTED_ARCH
void verify_layer_out(COMPUTE_VERIFICATION_STEP cim_step, uint8_t id, float* data, uint16_t starting_addr, DATA_WIDTH data_width) {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }

    uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;
    vector<float> ref_data;

    if ((cim_step == ENC_MHSA_DENSE_QK_T_VERIF) || (cim_step == ENC_SOFTMAX_VERIF)) {
        for (int head = 0; head < NUM_HEADS; head++) {
            string filename = step_verif_info[cim_step].csv_fp + to_string(head) + ".csv";
            rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));
            if (cim_step == ENC_MHSA_DENSE_QK_T_VERIF) { ref_data = csv.GetColumn<float>(id); }
            else if (cim_step == ENC_SOFTMAX_VERIF) { ref_data = csv.GetRow<float>(id); }
            for (int i = 0; i < ref_data.size(); i++) { are_equal(ref_data[i], data[stride*i + starting_addr + head*(NUM_PATCHES+1)], stride*i + starting_addr + head*(NUM_PATCHES+1), id); }
        }
    } else if (cim_step == POST_SOFTMAX_AVG_VERIF) {
        rapidcsv::Document csv(step_verif_info[MLP_HEAD_SOFTMAX_DIV_VERIF].csv_fp, rapidcsv::LabelParams(-1, -1));
        ref_data = csv.GetColumn<float>(0);
        for (int i = 0; i < ref_data.size(); i++) { ref_data[i] = ref_data[i] / NUM_SAMPLES_OUT_AVG; } // Divide this epoch's reference softmax

        for (int i = 0; i < (NUM_SAMPLES_OUT_AVG-1); i++) { // Softmax for previous dummy epochs
            string filename = step_verif_info[cim_step].csv_fp + to_string(i) + ".csv";
            rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));
            vector<float> dummy_softmax = csv.GetRow<float>(0);
            for (int j = 0; j < NUM_SLEEP_STAGES; j++) { ref_data[j] += dummy_softmax[j] / NUM_SAMPLES_OUT_AVG; }
        }
        for (int i = 0; i < NUM_SLEEP_STAGES; i++) { are_equal(ref_data[i], data[starting_addr + stride*i], starting_addr + stride*i, id); }
    } else {
        rapidcsv::Document csv(step_verif_info[cim_step].csv_fp, rapidcsv::LabelParams(-1, -1));
        if (cim_step == ENC_OUT_VERIF) { 
            vector<float> col = csv.GetColumn<float>(id);
            are_equal(col[0], data[starting_addr], starting_addr, id); // Only check the first row since we are not computing the rest
        } else {
            if (cim_step == MLP_HEAD_LAYERNORM_VERIF || cim_step == MLP_HEAD_DENSE_1_VERIF) { ref_data = csv.GetRow<float>(id); }
            else { ref_data = csv.GetColumn<float>(id); }
            for (int i = 0; i < ref_data.size(); i++) { 
                if (cim_step == MLP_HEAD_SOFTMAX_DIV_VERIF) { are_equal(ref_data[i]/NUM_SAMPLES_OUT_AVG, data[stride*i + starting_addr], stride*i + starting_addr, id); }
                else { are_equal(ref_data[i], data[stride*i + starting_addr], stride*i + starting_addr, id); }
            }
        }
    }
}
#elif CENTRALIZED_ARCH
void verify_layer_out(COMPUTE_VERIFICATION_STEP cim_step, float* data, uint16_t starting_addr, uint16_t outside_dim_len, DATA_WIDTH data_width) {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }
    uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;

    if (cim_step == POST_SOFTMAX_AVG_VERIF) {
        rapidcsv::Document csv(step_verif_info[MLP_HEAD_SOFTMAX_DIV_VERIF].csv_fp, rapidcsv::LabelParams(-1, -1));
        vector<float> ref_data = csv.GetColumn<float>(0);
        for (int i = 0; i < ref_data.size(); i++) { ref_data[i] = ref_data[i] / NUM_SAMPLES_OUT_AVG; } // Divide this epoch's reference softmax

        for (int i = 0; i < (NUM_SAMPLES_OUT_AVG-1); i++) { // Softmax for previous dummy epochs
            string filename = step_verif_info[cim_step].csv_fp + to_string(i) + ".csv";
            rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));
            vector<float> dummy_softmax = csv.GetRow<float>(0);
            for (int j = 0; j < NUM_SLEEP_STAGES; j++) { ref_data[j] += dummy_softmax[j] / NUM_SAMPLES_OUT_AVG; }
        }
        for (int i = 0; i < NUM_SLEEP_STAGES; i++) { are_equal(ref_data[i], data[starting_addr+stride*i], starting_addr+stride*i); }
    } else {
        uint8_t num_repeats;
        if (cim_step == ENC_MHSA_DENSE_QK_T_VERIF || cim_step == ENC_SOFTMAX_VERIF) { num_repeats = NUM_HEADS; }
        else { num_repeats = 1; }

        for (int repeat=0; repeat<num_repeats; repeat++) {
            string filename = step_verif_info[cim_step].csv_fp;
            if (num_repeats == NUM_HEADS) { filename = filename + to_string(repeat) + ".csv"; }        
            rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));

            uint16_t num_rows;
            if (cim_step == ENC_OUT_VERIF) { num_rows = 1; } // Only compute one row so don't check other
            else { num_rows = csv.GetRowCount(); }

            for (int i = 0; i < num_rows; i++) { // Row
                vector<float> ref_data = csv.GetRow<float>(i);
                for (int j = 0; j < csv.GetColumnCount(); j++) { // Colum
                    int32_t addr = starting_addr + stride*(j + i*outside_dim_len + repeat*(NUM_PATCHES+1)*(NUM_PATCHES+1));
                    if (cim_step == MLP_HEAD_SOFTMAX_DIV_VERIF) { are_equal(ref_data[j]/NUM_SAMPLES_OUT_AVG, data[addr], addr); }
                    else { are_equal(ref_data[j], data[addr], addr); }
                }
            }
        }
    }
}

#endif

void print_softmax_error(float* data, uint16_t starting_addr, DATA_WIDTH data_width) {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }
    uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;

    rapidcsv::Document csv(step_verif_info[MLP_HEAD_SOFTMAX_VERIF].csv_fp, rapidcsv::LabelParams(-1, -1));
    vector<float> ref_softmax = csv.GetColumn<float>(0);
    vector<float> softmax_err_rel, softmax_err_abs;

    cout << "Error on final softmax: ";
    for (int i = 0; i < NUM_SLEEP_STAGES; i++) { 
        softmax_err_rel.push_back(100*(ref_softmax[i] - data[starting_addr+stride*i]) / ref_softmax[i]);
        softmax_err_abs.push_back(ref_softmax[i] - data[starting_addr+i]);
        cout << softmax_err_abs[i] << " (" << softmax_err_rel[i] << "%) ";
    }
    cout << endl;
}

#if DISTRIBUTED_ARCH
void verify_result(RESULT_TYPE type, float result, float* input_data, uint16_t starting_addr, uint16_t len, uint8_t id, DATA_WIDTH data_width) {
#elif CENTRALIZED_ARCH
void verify_result(RESULT_TYPE type, float result, float* input_data, uint16_t starting_addr, uint16_t len, DATA_WIDTH data_width) {
#endif
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }

    float data[len];
    float reference_result = 0.0f;
    uint16_t stride = (data_width == SINGLE_WIDTH) ? 1 : 2;
    for (int i = 0; i < len; i++) { data[i] = input_data[starting_addr+stride*i]; }
    arma::fvec arma_data(data, len);

    if (type == MEAN) { reference_result = arma::mean(arma_data); }
    else if (type == VARIANCE) { reference_result = arma::var(arma_data, 1 /*norm_type of 1 to divide by N to match TF*/); }
#if DISTRIBUTED_ARCH
    are_equal(reference_result, result, -1, id);
#elif CENTRALIZED_ARCH
    are_equal(reference_result, result, -1);
#endif
}

void verify_softmax_storage(float* intermediate_res, uint16_t prev_softmax_base_addr) {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }

    for (int i = 0; i < (NUM_SAMPLES_OUT_AVG-2); i++) {
        for (int j = 0; j < NUM_SLEEP_STAGES; j++) {
            // Check that the previous sleep epochs have been shifted
            string filename = step_verif_info[POST_SOFTMAX_AVG_VERIF].csv_fp + to_string(i) + ".csv";
            rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));
            vector<float> dummy_softmax = csv.GetRow<float>(0);
            
            // Check that the current sleep epoch's softmax got moved to the previous sleep epochs softmax storage
            rapidcsv::Document csv_current(step_verif_info[MLP_HEAD_SOFTMAX_VERIF].csv_fp, rapidcsv::LabelParams(-1, -1));
            vector<float> ref_softmax = csv_current.GetColumn<float>(0);

            int32_t addr = prev_softmax_base_addr + DOUBLE_WIDTH*(j +(i+1)*NUM_SLEEP_STAGES);
#if DISTRIBUTED_ARCH
            are_equal(dummy_softmax[j]/NUM_SAMPLES_OUT_AVG, intermediate_res[addr], addr, 0);
            addr = prev_softmax_base_addr + DOUBLE_WIDTH*j;
            are_equal(ref_softmax[j]/NUM_SAMPLES_OUT_AVG, intermediate_res[addr], addr, 0);
#elif CENTRALIZED_ARCH
            are_equal(dummy_softmax[j]/NUM_SAMPLES_OUT_AVG, intermediate_res[addr], addr);
            addr = prev_softmax_base_addr + DOUBLE_WIDTH*j;
            are_equal(ref_softmax[j]/NUM_SAMPLES_OUT_AVG, intermediate_res[addr], addr);
#endif
        }
    }
}

void print_intermediate_value_stats() {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }
    arma::fvec arma_abs_errors(abs_errors.data(), abs_errors.size());
    arma::fvec arma_rel_errors(rel_errors.data(), rel_errors.size());
    
    cout << ">----- COMPUTE VERIFICATION STATS -----<" << endl;
    cout << "Min. intermediate value: " << min_val << endl;
    cout << "Max. intermediate value: " << max_val << endl;
    cout << "Highest valid absolute error: " << highest_abs_error << ". Avg: " << arma::mean(arma_abs_errors) << ". Std. dev.: " << arma::stddev(arma_abs_errors) << endl;
    if (rel_errors.size() == 0) { cout << "Highest valid relative error: " << highest_rel_error << ". Avg: " << 0 << ". Std. dev.: " << 0 << endl; }
    else { cout << "Highest valid relative error: " << highest_rel_error << ". Avg: " << arma::mean(arma_rel_errors) << ". Std. dev.: " << arma::stddev(arma_rel_errors) << endl; }
    cout << data_over_threshold_cnt << "/" << data_cnt << " over threshold (" << (100.0f*data_over_threshold_cnt/data_cnt) << "%)" << endl;
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