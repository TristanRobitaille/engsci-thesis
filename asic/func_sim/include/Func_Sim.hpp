#ifndef FUNC_SIM_H
#define FUNC_SIM_H

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <map>
#include <variant>

#include <Misc.hpp>
#if DISTRIBUTED_ARCH
#include <CiM.hpp>
#include <Master_Ctrl.hpp>
#elif CENTRALIZED_ARCH
#include <CiM_Centralized.hpp>
#endif
#include <Compute_Verification.hpp>

typedef void (*FcnPtr)(struct ext_signals*);

/*----- DEFINES -----*/
#define FIXED_POINT_ACCURACY_STUDY_START_N_STO 2 

/*----- FUNCTION -----*/
void master_nrst(struct ext_signals* ext_sig) { ext_sig->master_nrst = false; }
void master_nrst_reset(struct ext_signals* ext_sig) { ext_sig->master_nrst = true; }
void param_load(struct ext_signals* ext_sig) { ext_sig->start_param_load = true; }
void param_load_reset(struct ext_signals* ext_sig) { ext_sig->start_param_load = false; }
void epoch_start(struct ext_signals* ext_sig){ ext_sig->new_sleep_epoch = true; }
void epoch_start_reset(struct ext_signals* ext_sig){ ext_sig->new_sleep_epoch = false; }
void run_sim(uint32_t clip_num, std::string results_csv_fp);
void update_results_csv(uint32_t inferred_sleep_stage, uint32_t clip_num, std::string results_csv_fp);

#endif //FUNC_SIM_H
