#ifndef FUNC_SIM_H
#define FUNC_SIM_H

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <map>
#include <variant>

#include <CiM.hpp>
#include <Master_Ctrl.hpp>
#include <Misc.hpp>

typedef void (*FcnPtr)(struct ext_signals*);

/*----- FUNCTION -----*/
void master_nrst(struct ext_signals* ext_sig) { ext_sig->master_nrst = false; }
void master_nrst_reset(struct ext_signals* ext_sig) { ext_sig->master_nrst = true; }
void master_param_load(struct ext_signals* ext_sig) { ext_sig->start_param_load = true; }
void master_param_load_reset(struct ext_signals* ext_sig) { ext_sig->start_param_load = false; }
void epoch_start(struct ext_signals* ext_sig){ ext_sig->new_sleep_epoch = true; }
void epoch_start_reset(struct ext_signals* ext_sig){ ext_sig->new_sleep_epoch = false; }

#endif //FUNC_SIM_H