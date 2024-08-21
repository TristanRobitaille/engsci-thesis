// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "verilated.h"
#include "verilated_dpi.h"

#include "Vtop__Syms.h"
#include "Vtop_Defines.h"
#include "Vtop__Syms.h"

// Parameter definitions for Vtop_Defines
constexpr IData/*31:0*/ Vtop_Defines::CIM_PARAMS_BANK_SIZE_NUM_WORD;
constexpr IData/*31:0*/ Vtop_Defines::CIM_INT_RES_BANK_SIZE_NUM_WORD;
constexpr IData/*31:0*/ Vtop_Defines::CIM_PARAMS_NUM_BANKS;
constexpr IData/*31:0*/ Vtop_Defines::CIM_INT_RES_NUM_BANKS;
constexpr IData/*31:0*/ Vtop_Defines::N_STO_INT_RES;
constexpr IData/*31:0*/ Vtop_Defines::N_STO_PARAMS;


void Vtop_Defines___ctor_var_reset(Vtop_Defines* vlSelf);

Vtop_Defines::Vtop_Defines(Vtop__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtop_Defines___ctor_var_reset(this);
}

void Vtop_Defines::__Vconfigure(bool first) {
    if (false && first) {}  // Prevent unused
}

Vtop_Defines::~Vtop_Defines() {
}
