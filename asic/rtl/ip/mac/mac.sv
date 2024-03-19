/* Note:
- Fixed-point MAC
- Uses external multiplier and adder modules to be shared with other modules in the CiM.
- Implements three different activations: No activation (no bias), linear and SWISH
*/
module mac #(
    parameter N_STORAGE = 16, // 16b total
    parameter N_COMP = 22, // 22b total
    parameter TEMP_RES_STORAGE_SIZE_CIM = 848, // # of words in the intermediate result storage
    parameter PARAMS_STORAGE_SIZE_CIM = 528, // # of words in the model parameters storage
    parameter MAX_LEN = 64 // Maximum length of the vectors to MAC
)(
    input wire clk,
    input wire rst_n,

    // Control signals
    input wire start, param_type,
    input wire [$clog2(MAX_LEN+1)-1:0] len,
    input wire [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] start_addr1,
    input wire [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] start_addr2, // Either intermediate res or params
    input wire [1:0] activation,
    output logic busy, done,

    // Memory access signals
    input wire signed [N_STORAGE-1:0] param_data,
    input wire signed [N_STORAGE-1:0] intermediate_res_data,
    output logic int_res_mem_access_req, params_mem_access_req,
    output logic [$clog2(PARAMS_STORAGE_SIZE_CIM)-1:0] param_addr,
    output logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] intermediate_res_addr,

    // Computation signals
    output logic signed [N_COMP-1:0] computation_result,

    // Adder signals
    input wire signed [N_COMP-1:0] add_output_q,
    output logic signed [N_COMP-1:0] add_input_q_1,
    output logic signed [N_COMP-1:0] add_input_q_2,
    output logic add_refresh,

    // Multiplier signals
    input wire signed [N_COMP-1:0] mult_output_q,
    output logic signed [N_COMP-1:0] mult_input_q_1,
    output logic signed [N_COMP-1:0] mult_input_q_2,
    output logic mult_refresh
);
    /*----- ENUM -----*/
    enum logic {MODEL_PARAM = 1'b0, INTERMEDIATE_RES = 1'b1} PARAM_TYPE_E;
    enum logic [1:0] {NO_ACTIVATION = 2'd0, LINEAR_ACTIVATION = 2'd1, SWISH_ACTIVATION = 2'd2} ACTIVATION_TYPE_E;
    enum logic [2:0] {IDLE, COMPUTE_MUL_IN1, COMPUTE_MUL_IN2, COMPUTE_MUL_OUT, COMPUTE_ADD, MAC_DONE} state;

    /*----- LOGIC -----*/
    logic signed [N_COMP-1:0] compute_temp; // TODO: Consider if it should be shared with other modules in CiM
    logic signed [N_COMP-1:0] compute_temp_2; // TODO: Consider if it should be shared with other modules in CiM

    logic [$clog2(MAX_LEN+1)-1:0] index;
    always_ff @ (posedge clk) begin : mac_fsm
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            unique case (state)
                IDLE: begin
                    if (start) begin
                        state <= COMPUTE_MUL_IN1;
                        busy <= 1'b1;
                        done <= 1'b0;
                        index <= 'd0;
                        mult_refresh <= 1'b0;
                        add_refresh <= 1'b0;
                        intermediate_res_addr <= start_addr1;
                        int_res_mem_access_req <= 1'b1;
                        compute_temp <= 'd0;
                    end
                end 
                COMPUTE_MUL_IN1: begin
                    index <= index + 1;
                    intermediate_res_addr   <= (param_type == INTERMEDIATE_RES) ? (start_addr2 + {3'd0, index}) : (intermediate_res_addr);
                    param_addr              <= (param_type == MODEL_PARAM) ? (start_addr2 + {3'd0, index}) : (param_addr);
                    int_res_mem_access_req  <= (param_type == INTERMEDIATE_RES);
                    params_mem_access_req   <= (param_type == MODEL_PARAM);
                    state <= COMPUTE_MUL_IN2;
                end
                COMPUTE_MUL_IN2: begin
                    compute_temp <= (index > 1) ? add_output_q : compute_temp; // Grab data from previous iteration (unless it's the first iteration)
                    mult_input_q_1 <= {{(N_COMP-N_STORAGE){intermediate_res_data[N_STORAGE-1]}}, intermediate_res_data}; // Sign extend
                    int_res_mem_access_req <= 1'b0;
                    params_mem_access_req <= 1'b0;
                    state <= COMPUTE_MUL_OUT;
                    add_refresh <= 1'b0;
                end
                COMPUTE_MUL_OUT: begin
                    mult_input_q_2 <= (param_type == INTERMEDIATE_RES) ? {{(N_COMP-N_STORAGE){intermediate_res_data[N_STORAGE-1]}}, intermediate_res_data} : {{(N_COMP-N_STORAGE){param_data[N_STORAGE-1]}}, param_data};
                    mult_refresh <= 1'b1;
                    state <= (mult_refresh) ? COMPUTE_ADD : COMPUTE_MUL_OUT;
                end
                COMPUTE_ADD: begin
                    add_input_q_1 <= mult_output_q;
                    add_input_q_2 <= compute_temp;
                    mult_refresh <= 1'b0;
                    add_refresh <= 1'b1;
                    if (add_refresh) begin
                        state <= (index == len) ? MAC_DONE : COMPUTE_MUL_IN1;
                    end
                    // In case we need to go back to COMPUTE_MUL_IN1
                    intermediate_res_addr <= start_addr1 + {3'd0, index};
                    int_res_mem_access_req <= (index != len);
                end
                MAC_DONE: begin
                    if (activation == NO_ACTIVATION) begin
                        add_refresh <= 1'b0;
                        computation_result <= add_output_q;
                        done <= 1'b1;
                        state <= IDLE;
                    end else if (activation == LINEAR_ACTIVATION) begin
                        // TODO
                    end else if (activation == SWISH_ACTIVATION) begin
                        // TODO
                        $fatal("SWISH activation not implementedin MAC unit");
                    end
                end
                default: begin
                    $fatal("Invalid state in MAC FSM");
                end
            endcase
        end
    end

endmodule
