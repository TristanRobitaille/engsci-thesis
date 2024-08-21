// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "verilated.h"
#include "verilated_dpi.h"

#include "Vtop__Syms.h"
#include "Vtop___024root.h"

VL_ATTR_COLD void Vtop___024root___eval_static(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_static\n"); );
}

VL_ATTR_COLD void Vtop___024root___eval_initial(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_initial\n"); );
    // Body
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
}

VL_ATTR_COLD void Vtop___024root___eval_final(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_final\n"); );
}

VL_ATTR_COLD void Vtop___024root___eval_triggers__stl(Vtop___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__stl(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD void Vtop___024root___eval_stl(Vtop___024root* vlSelf);

VL_ATTR_COLD void Vtop___024root___eval_settle(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_settle\n"); );
    // Init
    CData/*0:0*/ __VstlContinue;
    // Body
    vlSelf->__VstlIterCount = 0U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        __VstlContinue = 0U;
        Vtop___024root___eval_triggers__stl(vlSelf);
        if (vlSelf->__VstlTriggered.any()) {
            __VstlContinue = 1U;
            if (VL_UNLIKELY((0x64U < vlSelf->__VstlIterCount))) {
#ifdef VL_DEBUG
                Vtop___024root___dump_triggers__stl(vlSelf);
#endif
                VL_FATAL_MT("/tmp/asic/rtl/centralized/cim_centralized_tb.sv", 1, "", "Settle region did not converge.");
            }
            vlSelf->__VstlIterCount = ((IData)(1U) 
                                       + vlSelf->__VstlIterCount);
            Vtop___024root___eval_stl(vlSelf);
        }
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__stl(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

void Vtop___024root___ico_sequent__TOP__0(Vtop___024root* vlSelf);

VL_ATTR_COLD void Vtop___024root___eval_stl(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vtop___024root___ico_sequent__TOP__0(vlSelf);
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__ico(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__ico\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VicoTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        VL_DBG_MSGF("         'ico' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__act(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__nba(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtop___024root___ctor_var_reset(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->clk = VL_RAND_RESET_I(1);
    vlSelf->rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_inc = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_cnt = VL_RAND_RESET_I(4);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_inc = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_cnt = VL_RAND_RESET_I(7);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_inc = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_cnt = VL_RAND_RESET_I(9);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__temp_res_addr = VL_RAND_RESET_I(15);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__param_addr = VL_RAND_RESET_I(16);
    for (int __Vi0 = 0; __Vi0 < 31648; ++__Vi0) {
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__params[__Vi0] = VL_RAND_RESET_I(9);
    }
    for (int __Vi0 = 0; __Vi0 < 57116; ++__Vi0) {
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__int_res[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__cnt = VL_RAND_RESET_I(4);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc_prev = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__cnt = VL_RAND_RESET_I(7);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc_prev = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc = VL_RAND_RESET_I(1);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__cnt = VL_RAND_RESET_I(9);
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc_prev = VL_RAND_RESET_I(1);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_RAND_RESET_I(1);
}
