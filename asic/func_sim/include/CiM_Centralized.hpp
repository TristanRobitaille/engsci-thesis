#ifndef CIM_CENTRALIZED_H
#define CIM_CENTRALIZED_H

#include <CiM_Compute.hpp>

/*----- CLASS -----*/
class CiM_Centralized : public CiM_Compute {
    private:
        /*----- ENUM -----*/
        enum STATE {
            SLEEP_CIM, // Parameter memory unpowered
            IDLE_CIM,
            INVALID_CIM
        };

        enum INFERENCE_STEP {
            PATCH_DENSE_STEP,
        };

        /*----- PRIVATE VARIABLES -----*/
        STATE cim_state = INVALID_CIM;
        SYSTEM_STATE system_state;
        INFERENCE_STEP current_inf_step;

        void load_previous_softmax();

    public:
        CiM_Centralized();
        int reset();
        SYSTEM_STATE run(struct ext_signals* ext_sigs);
};

#endif //CIM_CENTRALIZED_H