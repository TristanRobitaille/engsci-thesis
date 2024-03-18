// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "Vtop__pch.h"
#include "Vtop___024root.h"

VL_ATTR_COLD void Vtop___024root___eval_static(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_static\n"); );
}

VL_ATTR_COLD void Vtop___024root___eval_initial(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_initial\n"); );
    // Body
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = vlSelf->rst_n;
}

VL_ATTR_COLD void Vtop___024root___eval_final(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_final\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__stl(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtop___024root___eval_phase__stl(Vtop___024root* vlSelf);

VL_ATTR_COLD void Vtop___024root___eval_settle(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_settle\n"); );
    // Init
    IData/*31:0*/ __VstlIterCount;
    CData/*0:0*/ __VstlContinue;
    // Body
    __VstlIterCount = 0U;
    vlSelf->__VstlFirstIteration = 1U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        if (VL_UNLIKELY((0x64U < __VstlIterCount))) {
#ifdef VL_DEBUG
            Vtop___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("cim.sv", 6, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vtop___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelf->__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__stl(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
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
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vtop___024root___ico_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD void Vtop___024root___eval_triggers__stl(Vtop___024root* vlSelf);

VL_ATTR_COLD bool Vtop___024root___eval_phase__stl(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_phase__stl\n"); );
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vtop___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelf->__VstlTriggered.any();
    if (__VstlExecute) {
        Vtop___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__ico(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
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
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clk or negedge rst_n)\n");
    }
    if ((2ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__nba(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clk or negedge rst_n)\n");
    }
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtop___024root___ctor_var_reset(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->clk = VL_RAND_RESET_I(1);
    vlSelf->rst_n = VL_RAND_RESET_I(1);
    vlSelf->is_ready = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__is_ready = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 528; ++__Vi0) {
        vlSelf->cim__DOT__params[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 848; ++__Vi0) {
        vlSelf->cim__DOT__intermediate_res[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->cim__DOT__compute_in_progress = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__gen_cnt_3b = VL_RAND_RESET_I(3);
    vlSelf->cim__DOT__sender_id = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__data_len = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__tx_addr = VL_RAND_RESET_I(10);
    vlSelf->cim__DOT__rx_addr = VL_RAND_RESET_I(10);
    vlSelf->cim__DOT__compute_temp = VL_RAND_RESET_I(22);
    vlSelf->cim__DOT__compute_temp_2 = VL_RAND_RESET_I(22);
    vlSelf->cim__DOT__compute_temp_3 = VL_RAND_RESET_I(22);
    vlSelf->cim__DOT__computation_result = VL_RAND_RESET_I(22);
    vlSelf->cim__DOT__gen_cnt_7b_rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__gen_cnt_7b_2_rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__word_rec_cnt_rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__word_snt_cnt_rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__gen_cnt_7b_inc = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__gen_cnt_7b_2_inc = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__word_rec_cnt_inc = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__word_snt_cnt_inc = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__gen_cnt_7b_cnt = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__gen_cnt_7b_2_cnt = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__word_rec_cnt = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__word_snt_cnt = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__cim_state = VL_RAND_RESET_I(3);
    vlSelf->cim__DOT__current_inf_step = VL_RAND_RESET_I(6);
    vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__inc = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__cnt = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__inc_prev = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__inc = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__cnt = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__inc_prev = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__word_rec_cnt_inst__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__word_rec_cnt_inst__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__word_rec_cnt_inst__DOT__inc = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__word_rec_cnt_inst__DOT__cnt = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__word_rec_cnt_inst__DOT__inc_prev = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__word_snt_cnt_inst__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__word_snt_cnt_inst__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->cim__DOT__word_snt_cnt_inst__DOT__inc = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__word_snt_cnt_inst__DOT__cnt = VL_RAND_RESET_I(7);
    vlSelf->cim__DOT__word_snt_cnt_inst__DOT__inc_prev = VL_RAND_RESET_I(7);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_RAND_RESET_I(1);
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = VL_RAND_RESET_I(1);
}
