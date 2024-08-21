// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_fst_c.h"
#include "Vtop__Syms.h"


void Vtop___024root__trace_chg_sub_0(Vtop___024root* vlSelf, VerilatedFst::Buffer* bufp);

void Vtop___024root__trace_chg_top_0(void* voidSelf, VerilatedFst::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_chg_top_0\n"); );
    // Init
    Vtop___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop___024root*>(voidSelf);
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vtop___024root__trace_chg_sub_0((&vlSymsp->TOP), bufp);
}

void Vtop___024root__trace_chg_sub_0(Vtop___024root* vlSelf, VerilatedFst::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_chg_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    bufp->chgBit(oldp+0,(vlSelf->clk));
    bufp->chgBit(oldp+1,(vlSelf->rst_n));
    bufp->chgBit(oldp+2,(vlSelf->cim_centralized_tb__DOT__clk));
    bufp->chgBit(oldp+3,(vlSelf->cim_centralized_tb__DOT__rst_n));
    bufp->chgBit(oldp+4,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__clk));
    bufp->chgBit(oldp+5,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__rst_n));
    bufp->chgBit(oldp+6,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_rst_n));
    bufp->chgBit(oldp+7,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_inc));
    bufp->chgCData(oldp+8,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_cnt),4);
    bufp->chgBit(oldp+9,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_rst_n));
    bufp->chgBit(oldp+10,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_inc));
    bufp->chgCData(oldp+11,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_cnt),7);
    bufp->chgBit(oldp+12,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_rst_n));
    bufp->chgBit(oldp+13,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_inc));
    bufp->chgSData(oldp+14,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_cnt),9);
    bufp->chgSData(oldp+15,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__temp_res_addr),15);
    bufp->chgSData(oldp+16,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__param_addr),16);
    bufp->chgBit(oldp+17,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__clk));
    bufp->chgBit(oldp+18,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__rst_n));
    bufp->chgBit(oldp+19,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc));
    bufp->chgCData(oldp+20,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__cnt),4);
    bufp->chgBit(oldp+21,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc_prev));
    bufp->chgBit(oldp+22,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__clk));
    bufp->chgBit(oldp+23,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__rst_n));
    bufp->chgBit(oldp+24,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc));
    bufp->chgCData(oldp+25,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__cnt),7);
    bufp->chgBit(oldp+26,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc_prev));
    bufp->chgBit(oldp+27,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__clk));
    bufp->chgBit(oldp+28,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__rst_n));
    bufp->chgBit(oldp+29,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc));
    bufp->chgSData(oldp+30,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__cnt),9);
    bufp->chgBit(oldp+31,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc_prev));
}

void Vtop___024root__trace_cleanup(void* voidSelf, VerilatedFst* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_cleanup\n"); );
    // Init
    Vtop___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop___024root*>(voidSelf);
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VlUnpacked<CData/*0:0*/, 1> __Vm_traceActivity;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        __Vm_traceActivity[__Vi0] = 0;
    }
    // Body
    vlSymsp->__Vm_activity = false;
    __Vm_traceActivity[0U] = 0U;
}
