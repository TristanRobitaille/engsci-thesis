// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "verilated.h"
#include "verilated_dpi.h"

#include "Vtop__Syms.h"
#include "Vtop___024root.h"

VL_INLINE_OPT void Vtop___024root___ico_sequent__TOP__0(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___ico_sequent__TOP__0\n"); );
    // Body
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__rst_n 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_rst_n;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_inc;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__rst_n 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_rst_n;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_inc;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__rst_n 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_rst_n;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_inc;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_cnt 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__cnt;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_cnt 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__cnt;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_cnt 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__cnt;
    vlSelf->cim_centralized_tb__DOT__rst_n = vlSelf->rst_n;
    vlSelf->cim_centralized_tb__DOT__clk = vlSelf->clk;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__rst_n 
        = vlSelf->cim_centralized_tb__DOT__rst_n;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__clk 
        = vlSelf->cim_centralized_tb__DOT__clk;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__clk 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__clk;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__clk 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__clk;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__clk 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__clk;
}

void Vtop___024root___eval_ico(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        Vtop___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void Vtop___024root___eval_act(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_act\n"); );
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__0(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__0\n"); );
    // Body
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_inc = 0U;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_inc = 0U;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_inc = 0U;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_rst_n 
        = vlSelf->rst_n;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_rst_n 
        = vlSelf->rst_n;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_rst_n 
        = vlSelf->rst_n;
    if (vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_rst_n) {
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__cnt 
            = (0x1ffU & (((IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_inc) 
                          != (IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc_prev))
                          ? ((IData)(1U) + (IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__cnt))
                          : (IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__cnt)));
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc_prev 
            = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_inc;
    } else {
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__cnt = 0U;
    }
    if (vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_rst_n) {
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__cnt 
            = (0x7fU & (((IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_inc) 
                         != (IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc_prev))
                         ? ((IData)(1U) + (IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__cnt))
                         : (IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__cnt)));
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc_prev 
            = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_inc;
    } else {
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__cnt = 0U;
    }
    if (vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_rst_n) {
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__cnt 
            = (0xfU & (((IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_inc) 
                        != (IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc_prev))
                        ? ((IData)(1U) + (IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__cnt))
                        : (IData)(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__cnt)));
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc_prev 
            = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_inc;
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc 
            = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_inc;
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc 
            = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_inc;
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc 
            = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_inc;
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__rst_n = 1U;
    } else {
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__cnt = 0U;
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc 
            = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_inc;
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc 
            = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_inc;
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc 
            = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_inc;
        vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__rst_n = 0U;
    }
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__rst_n 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_rst_n;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__rst_n 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_rst_n;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_cnt 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__cnt;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_cnt 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__cnt;
    vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_cnt 
        = vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__cnt;
}

void Vtop___024root___eval_nba(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_nba\n"); );
    // Body
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vtop___024root___nba_sequent__TOP__0(vlSelf);
    }
}

void Vtop___024root___eval_triggers__ico(Vtop___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__ico(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
void Vtop___024root___eval_triggers__act(Vtop___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__act(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__nba(Vtop___024root* vlSelf);
#endif  // VL_DEBUG

void Vtop___024root___eval(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval\n"); );
    // Init
    CData/*0:0*/ __VicoContinue;
    VlTriggerVec<1> __VpreTriggered;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    vlSelf->__VicoIterCount = 0U;
    __VicoContinue = 1U;
    while (__VicoContinue) {
        __VicoContinue = 0U;
        Vtop___024root___eval_triggers__ico(vlSelf);
        if (vlSelf->__VicoTriggered.any()) {
            __VicoContinue = 1U;
            if (VL_UNLIKELY((0x64U < vlSelf->__VicoIterCount))) {
#ifdef VL_DEBUG
                Vtop___024root___dump_triggers__ico(vlSelf);
#endif
                VL_FATAL_MT("/tmp/asic/rtl/centralized/cim_centralized_tb.sv", 1, "", "Input combinational region did not converge.");
            }
            vlSelf->__VicoIterCount = ((IData)(1U) 
                                       + vlSelf->__VicoIterCount);
            Vtop___024root___eval_ico(vlSelf);
        }
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        __VnbaContinue = 0U;
        vlSelf->__VnbaTriggered.clear();
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            vlSelf->__VactContinue = 0U;
            Vtop___024root___eval_triggers__act(vlSelf);
            if (vlSelf->__VactTriggered.any()) {
                vlSelf->__VactContinue = 1U;
                if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                    Vtop___024root___dump_triggers__act(vlSelf);
#endif
                    VL_FATAL_MT("/tmp/asic/rtl/centralized/cim_centralized_tb.sv", 1, "", "Active region did not converge.");
                }
                vlSelf->__VactIterCount = ((IData)(1U) 
                                           + vlSelf->__VactIterCount);
                __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
                vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
                Vtop___024root___eval_act(vlSelf);
            }
        }
        if (vlSelf->__VnbaTriggered.any()) {
            __VnbaContinue = 1U;
            if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
                Vtop___024root___dump_triggers__nba(vlSelf);
#endif
                VL_FATAL_MT("/tmp/asic/rtl/centralized/cim_centralized_tb.sv", 1, "", "NBA region did not converge.");
            }
            __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
            Vtop___024root___eval_nba(vlSelf);
        }
    }
}

#ifdef VL_DEBUG
void Vtop___024root___eval_debug_assertions(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((vlSelf->rst_n & 0xfeU))) {
        Verilated::overWidthError("rst_n");}
}
#endif  // VL_DEBUG
