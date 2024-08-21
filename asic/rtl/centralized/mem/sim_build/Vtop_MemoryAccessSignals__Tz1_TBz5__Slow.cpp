// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "verilated.h"
#include "verilated_dpi.h"

#include "Vtop__Syms.h"
#include "Vtop_MemoryAccessSignals__Tz1_TBz5.h"
#include "Vtop__Syms.h"

void Vtop_MemoryAccessSignals__Tz1_TBz5___ctor_var_reset(Vtop_MemoryAccessSignals__Tz1_TBz5* vlSelf);

Vtop_MemoryAccessSignals__Tz1_TBz5::Vtop_MemoryAccessSignals__Tz1_TBz5(Vtop__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtop_MemoryAccessSignals__Tz1_TBz5___ctor_var_reset(this);
}

void Vtop_MemoryAccessSignals__Tz1_TBz5::__Vconfigure(bool first) {
    if (false && first) {}  // Prevent unused
}

Vtop_MemoryAccessSignals__Tz1_TBz5::~Vtop_MemoryAccessSignals__Tz1_TBz5() {
}
