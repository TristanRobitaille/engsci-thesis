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
                VL_FATAL_MT("/tmp/asic/rtl/centralized/mem/mem_tb.sv", 3, "", "Settle region did not converge.");
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
    vlSelf->param_read_en = VL_RAND_RESET_I(1);
    vlSelf->param_write_en = VL_RAND_RESET_I(1);
    vlSelf->param_chip_en = VL_RAND_RESET_I(1);
    vlSelf->param_read_data_width = VL_RAND_RESET_I(1);
    vlSelf->param_write_data_width = VL_RAND_RESET_I(1);
    vlSelf->param_read_addr = VL_RAND_RESET_I(15);
    vlSelf->param_write_addr = VL_RAND_RESET_I(15);
    vlSelf->param_write_data = VL_RAND_RESET_I(9);
    vlSelf->param_read_data = VL_RAND_RESET_I(9);
    vlSelf->int_res_read_en = VL_RAND_RESET_I(1);
    vlSelf->int_res_write_en = VL_RAND_RESET_I(1);
    vlSelf->int_res_chip_en = VL_RAND_RESET_I(1);
    vlSelf->int_res_read_data_width = VL_RAND_RESET_I(1);
    vlSelf->int_res_write_data_width = VL_RAND_RESET_I(1);
    vlSelf->int_res_read_addr = VL_RAND_RESET_I(16);
    vlSelf->int_res_write_addr = VL_RAND_RESET_I(16);
    vlSelf->int_res_write_data = VL_RAND_RESET_I(18);
    vlSelf->int_res_read_data = VL_RAND_RESET_I(18);
    vlSelf->mem_tb__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__param_read_en = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__param_write_en = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__param_chip_en = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__param_read_data_width = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__param_write_data_width = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__param_read_addr = VL_RAND_RESET_I(15);
    vlSelf->mem_tb__DOT__param_write_addr = VL_RAND_RESET_I(15);
    vlSelf->mem_tb__DOT__param_write_data = VL_RAND_RESET_I(9);
    vlSelf->mem_tb__DOT__param_read_data = VL_RAND_RESET_I(9);
    vlSelf->mem_tb__DOT__int_res_read_en = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res_write_en = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res_chip_en = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res_read_data_width = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res_write_data_width = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res_read_addr = VL_RAND_RESET_I(16);
    vlSelf->mem_tb__DOT__int_res_write_addr = VL_RAND_RESET_I(16);
    vlSelf->mem_tb__DOT__int_res_write_data = VL_RAND_RESET_I(18);
    vlSelf->mem_tb__DOT__int_res_read_data = VL_RAND_RESET_I(18);
    vlSelf->mem_tb__DOT__params__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__params__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__params__DOT__params_0_read_en_prev = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__params__DOT__params_1_read_en_prev = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__WEN = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__CEN = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 15872; ++__Vi0) {
        vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__memory[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->mem_tb__DOT__params__DOT__params_0__DOT____Vlvbound_h4cea1195__0 = VL_RAND_RESET_I(9);
    vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__WEN = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__CEN = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 15872; ++__Vi0) {
        vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__memory[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->mem_tb__DOT__params__DOT__params_1__DOT____Vlvbound_h4cea1195__0 = VL_RAND_RESET_I(9);
    vlSelf->mem_tb__DOT__int_res__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current = VL_RAND_RESET_I(2);
    vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev = VL_RAND_RESET_I(2);
    vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current = VL_RAND_RESET_I(2);
    vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr = VL_RAND_RESET_I(14);
    vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr = VL_RAND_RESET_I(14);
    vlSelf->mem_tb__DOT__int_res__DOT__read_data_width_prev = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__WEN = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__CEN = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 14336; ++__Vi0) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT____Vlvbound_h203f7518__0 = VL_RAND_RESET_I(9);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__WEN = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__CEN = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 14336; ++__Vi0) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT____Vlvbound_h203f7518__0 = VL_RAND_RESET_I(9);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__WEN = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__CEN = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 14336; ++__Vi0) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT____Vlvbound_h203f7518__0 = VL_RAND_RESET_I(9);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__WEN = VL_RAND_RESET_I(1);
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__CEN = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 14336; ++__Vi0) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT____Vlvbound_h203f7518__0 = VL_RAND_RESET_I(9);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_RAND_RESET_I(1);
}
