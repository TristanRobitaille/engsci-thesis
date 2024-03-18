// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "Vtop__pch.h"
#include "Vtop___024root.h"

VL_INLINE_OPT void Vtop___024root___ico_sequent__TOP__0(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___ico_sequent__TOP__0\n"); );
    // Body
    vlSelf->cim__DOT__rst_n = vlSelf->rst_n;
    vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__rst_n = vlSelf->cim__DOT__gen_cnt_7b_rst_n;
    vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__inc = vlSelf->cim__DOT__gen_cnt_7b_inc;
    vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__rst_n 
        = vlSelf->cim__DOT__gen_cnt_7b_2_rst_n;
    vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__inc = vlSelf->cim__DOT__gen_cnt_7b_2_inc;
    vlSelf->cim__DOT__word_rec_cnt_inst__DOT__rst_n 
        = vlSelf->cim__DOT__word_rec_cnt_rst_n;
    vlSelf->cim__DOT__word_rec_cnt_inst__DOT__inc = vlSelf->cim__DOT__word_rec_cnt_inc;
    vlSelf->cim__DOT__word_snt_cnt_inst__DOT__rst_n 
        = vlSelf->cim__DOT__word_snt_cnt_rst_n;
    vlSelf->cim__DOT__word_snt_cnt_inst__DOT__inc = vlSelf->cim__DOT__word_snt_cnt_inc;
    vlSelf->is_ready = vlSelf->cim__DOT__is_ready;
    vlSelf->cim__DOT__gen_cnt_7b_cnt = vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__cnt;
    vlSelf->cim__DOT__gen_cnt_7b_2_cnt = vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__cnt;
    vlSelf->cim__DOT__word_rec_cnt = vlSelf->cim__DOT__word_rec_cnt_inst__DOT__cnt;
    vlSelf->cim__DOT__word_snt_cnt = vlSelf->cim__DOT__word_snt_cnt_inst__DOT__cnt;
    vlSelf->cim__DOT__clk = vlSelf->clk;
    vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__clk = vlSelf->cim__DOT__clk;
    vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__clk = vlSelf->cim__DOT__clk;
    vlSelf->cim__DOT__word_rec_cnt_inst__DOT__clk = vlSelf->cim__DOT__clk;
    vlSelf->cim__DOT__word_snt_cnt_inst__DOT__clk = vlSelf->cim__DOT__clk;
}

void Vtop___024root___eval_ico(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        Vtop___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void Vtop___024root___eval_triggers__ico(Vtop___024root* vlSelf);

bool Vtop___024root___eval_phase__ico(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_phase__ico\n"); );
    // Init
    CData/*0:0*/ __VicoExecute;
    // Body
    Vtop___024root___eval_triggers__ico(vlSelf);
    __VicoExecute = vlSelf->__VicoTriggered.any();
    if (__VicoExecute) {
        Vtop___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vtop___024root___eval_act(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_act\n"); );
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__0(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__0\n"); );
    // Body
    vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__cnt = ((IData)(vlSelf->cim__DOT__gen_cnt_7b_rst_n)
                                                    ? 
                                                   (0x7fU 
                                                    & (((IData)(vlSelf->cim__DOT__gen_cnt_7b_inc) 
                                                        != (IData)(vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__inc_prev))
                                                        ? 
                                                       ((IData)(vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__cnt) 
                                                        + (IData)(vlSelf->cim__DOT__gen_cnt_7b_inc))
                                                        : (IData)(vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__cnt)))
                                                    : 0U);
    vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__cnt = 
        ((IData)(vlSelf->cim__DOT__gen_cnt_7b_2_rst_n)
          ? (0x7fU & (((IData)(vlSelf->cim__DOT__gen_cnt_7b_2_inc) 
                       != (IData)(vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__inc_prev))
                       ? ((IData)(vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__cnt) 
                          + (IData)(vlSelf->cim__DOT__gen_cnt_7b_2_inc))
                       : (IData)(vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__cnt)))
          : 0U);
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__1(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__1\n"); );
    // Body
    vlSelf->cim__DOT__word_rec_cnt_inst__DOT__cnt = 
        ((IData)(vlSelf->cim__DOT__word_rec_cnt_rst_n)
          ? (0x7fU & (((IData)(vlSelf->cim__DOT__word_rec_cnt_inc) 
                       != (IData)(vlSelf->cim__DOT__word_rec_cnt_inst__DOT__inc_prev))
                       ? ((IData)(vlSelf->cim__DOT__word_rec_cnt_inst__DOT__cnt) 
                          + (IData)(vlSelf->cim__DOT__word_rec_cnt_inc))
                       : (IData)(vlSelf->cim__DOT__word_rec_cnt_inst__DOT__cnt)))
          : 0U);
    vlSelf->cim__DOT__word_snt_cnt_inst__DOT__cnt = 
        ((IData)(vlSelf->cim__DOT__word_snt_cnt_rst_n)
          ? (0x7fU & (((IData)(vlSelf->cim__DOT__word_snt_cnt_inc) 
                       != (IData)(vlSelf->cim__DOT__word_snt_cnt_inst__DOT__inc_prev))
                       ? ((IData)(vlSelf->cim__DOT__word_snt_cnt_inst__DOT__cnt) 
                          + (IData)(vlSelf->cim__DOT__word_snt_cnt_inc))
                       : (IData)(vlSelf->cim__DOT__word_snt_cnt_inst__DOT__cnt)))
          : 0U);
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__3(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__3\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->rst_n)))) {
        vlSelf->cim__DOT__cim_state = 0U;
    }
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__4(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__4\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->rst_n)))) {
        vlSelf->cim__DOT__current_inf_step = 0U;
    }
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__5(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__5\n"); );
    // Body
    vlSelf->cim__DOT__gen_cnt_7b_cnt = vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__cnt;
    vlSelf->cim__DOT__gen_cnt_7b_2_cnt = vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__cnt;
    vlSelf->cim__DOT__word_rec_cnt = vlSelf->cim__DOT__word_rec_cnt_inst__DOT__cnt;
    vlSelf->cim__DOT__word_snt_cnt = vlSelf->cim__DOT__word_snt_cnt_inst__DOT__cnt;
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__6(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__6\n"); );
    // Body
    if (vlSelf->cim__DOT__gen_cnt_7b_rst_n) {
        vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__inc_prev 
            = vlSelf->cim__DOT__gen_cnt_7b_inc;
    }
    if (vlSelf->cim__DOT__gen_cnt_7b_2_rst_n) {
        vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__inc_prev 
            = vlSelf->cim__DOT__gen_cnt_7b_2_inc;
    }
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__7(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__7\n"); );
    // Body
    if (vlSelf->cim__DOT__word_rec_cnt_rst_n) {
        vlSelf->cim__DOT__word_rec_cnt_inst__DOT__inc_prev 
            = vlSelf->cim__DOT__word_rec_cnt_inc;
    }
    if (vlSelf->cim__DOT__word_snt_cnt_rst_n) {
        vlSelf->cim__DOT__word_snt_cnt_inst__DOT__inc_prev 
            = vlSelf->cim__DOT__word_snt_cnt_inc;
    }
}

void Vtop___024root___eval_triggers__act(Vtop___024root* vlSelf);

bool Vtop___024root___eval_phase__act(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<2> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vtop___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vtop___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

void Vtop___024root___eval_nba(Vtop___024root* vlSelf);

bool Vtop___024root___eval_phase__nba(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vtop___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__ico(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__nba(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__act(Vtop___024root* vlSelf);
#endif  // VL_DEBUG

void Vtop___024root___eval(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval\n"); );
    // Init
    IData/*31:0*/ __VicoIterCount;
    CData/*0:0*/ __VicoContinue;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VicoIterCount = 0U;
    vlSelf->__VicoFirstIteration = 1U;
    __VicoContinue = 1U;
    while (__VicoContinue) {
        if (VL_UNLIKELY((0x64U < __VicoIterCount))) {
#ifdef VL_DEBUG
            Vtop___024root___dump_triggers__ico(vlSelf);
#endif
            VL_FATAL_MT("cim.sv", 6, "", "Input combinational region did not converge.");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        __VicoContinue = 0U;
        if (Vtop___024root___eval_phase__ico(vlSelf)) {
            __VicoContinue = 1U;
        }
        vlSelf->__VicoFirstIteration = 0U;
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vtop___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("cim.sv", 6, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vtop___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("cim.sv", 6, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vtop___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vtop___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vtop___024root___eval_debug_assertions(Vtop___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((vlSelf->rst_n & 0xfeU))) {
        Verilated::overWidthError("rst_n");}
}
#endif  // VL_DEBUG
