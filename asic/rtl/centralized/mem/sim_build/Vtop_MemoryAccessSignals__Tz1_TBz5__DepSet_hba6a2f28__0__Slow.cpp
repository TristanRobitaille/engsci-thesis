// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "verilated.h"
#include "verilated_dpi.h"

#include "Vtop__Syms.h"
#include "Vtop_MemoryAccessSignals__Tz1_TBz5.h"

VL_ATTR_COLD void Vtop_MemoryAccessSignals__Tz1_TBz5___ctor_var_reset(Vtop_MemoryAccessSignals__Tz1_TBz5* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vtop_MemoryAccessSignals__Tz1_TBz5___ctor_var_reset\n"); );
    // Body
    vlSelf->en = VL_RAND_RESET_I(1);
    vlSelf->chip_en = VL_RAND_RESET_I(1);
    vlSelf->data_width = VL_RAND_RESET_I(1);
    vlSelf->data = VL_RAND_RESET_I(9);
    vlSelf->addr = VL_RAND_RESET_I(14);
}
