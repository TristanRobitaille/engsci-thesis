// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtop__Syms.h"


void Vtop___024root__trace_chg_0_sub_0(Vtop___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vtop___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_chg_0\n"); );
    // Init
    Vtop___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop___024root*>(voidSelf);
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vtop___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vtop___024root__trace_chg_0_sub_0(Vtop___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_chg_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    bufp->chgBit(oldp+0,(vlSelf->clk));
    bufp->chgBit(oldp+1,(vlSelf->rst_n));
    bufp->chgBit(oldp+2,(vlSelf->is_ready));
    bufp->chgBit(oldp+3,(vlSelf->cim__DOT__clk));
    bufp->chgBit(oldp+4,(vlSelf->cim__DOT__rst_n));
    bufp->chgBit(oldp+5,(vlSelf->cim__DOT__is_ready));
    bufp->chgBit(oldp+6,(vlSelf->cim__DOT__compute_in_progress));
    bufp->chgCData(oldp+7,(vlSelf->cim__DOT__gen_cnt_3b),3);
    bufp->chgCData(oldp+8,(vlSelf->cim__DOT__sender_id),7);
    bufp->chgCData(oldp+9,(vlSelf->cim__DOT__data_len),7);
    bufp->chgSData(oldp+10,(vlSelf->cim__DOT__tx_addr),10);
    bufp->chgSData(oldp+11,(vlSelf->cim__DOT__rx_addr),10);
    bufp->chgIData(oldp+12,(vlSelf->cim__DOT__compute_temp),22);
    bufp->chgIData(oldp+13,(vlSelf->cim__DOT__compute_temp_2),22);
    bufp->chgIData(oldp+14,(vlSelf->cim__DOT__compute_temp_3),22);
    bufp->chgIData(oldp+15,(vlSelf->cim__DOT__computation_result),22);
    bufp->chgBit(oldp+16,(vlSelf->cim__DOT__gen_cnt_7b_rst_n));
    bufp->chgBit(oldp+17,(vlSelf->cim__DOT__gen_cnt_7b_2_rst_n));
    bufp->chgBit(oldp+18,(vlSelf->cim__DOT__word_rec_cnt_rst_n));
    bufp->chgBit(oldp+19,(vlSelf->cim__DOT__word_snt_cnt_rst_n));
    bufp->chgCData(oldp+20,(vlSelf->cim__DOT__gen_cnt_7b_inc),7);
    bufp->chgCData(oldp+21,(vlSelf->cim__DOT__gen_cnt_7b_2_inc),7);
    bufp->chgCData(oldp+22,(vlSelf->cim__DOT__word_rec_cnt_inc),7);
    bufp->chgCData(oldp+23,(vlSelf->cim__DOT__word_snt_cnt_inc),7);
    bufp->chgCData(oldp+24,(vlSelf->cim__DOT__gen_cnt_7b_cnt),7);
}

void Vtop___024root__trace_chg_1_sub_0(Vtop___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vtop___024root__trace_chg_1(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_chg_1\n"); );
    // Init
    Vtop___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop___024root*>(voidSelf);
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vtop___024root__trace_chg_1_sub_0((&vlSymsp->TOP), bufp);
}

void Vtop___024root__trace_chg_1_sub_0(Vtop___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_chg_1_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 26);
    // Body
    bufp->chgCData(oldp+0,(vlSelf->cim__DOT__gen_cnt_7b_2_cnt),7);
    bufp->chgCData(oldp+1,(vlSelf->cim__DOT__word_rec_cnt),7);
    bufp->chgCData(oldp+2,(vlSelf->cim__DOT__word_snt_cnt),7);
    bufp->chgCData(oldp+3,(vlSelf->cim__DOT__cim_state),3);
    bufp->chgCData(oldp+4,(vlSelf->cim__DOT__current_inf_step),6);
    bufp->chgBit(oldp+5,(vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__clk));
    bufp->chgBit(oldp+6,(vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__rst_n));
    bufp->chgCData(oldp+7,(vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__inc),7);
    bufp->chgCData(oldp+8,(vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__cnt),7);
    bufp->chgCData(oldp+9,(vlSelf->cim__DOT__gen_cnt_7b_2_inst__DOT__inc_prev),7);
    bufp->chgBit(oldp+10,(vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__clk));
    bufp->chgBit(oldp+11,(vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__rst_n));
    bufp->chgCData(oldp+12,(vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__inc),7);
    bufp->chgCData(oldp+13,(vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__cnt),7);
    bufp->chgCData(oldp+14,(vlSelf->cim__DOT__gen_cnt_7b_inst__DOT__inc_prev),7);
    bufp->chgBit(oldp+15,(vlSelf->cim__DOT__word_rec_cnt_inst__DOT__clk));
    bufp->chgBit(oldp+16,(vlSelf->cim__DOT__word_rec_cnt_inst__DOT__rst_n));
    bufp->chgCData(oldp+17,(vlSelf->cim__DOT__word_rec_cnt_inst__DOT__inc),7);
    bufp->chgCData(oldp+18,(vlSelf->cim__DOT__word_rec_cnt_inst__DOT__cnt),7);
    bufp->chgCData(oldp+19,(vlSelf->cim__DOT__word_rec_cnt_inst__DOT__inc_prev),7);
    bufp->chgBit(oldp+20,(vlSelf->cim__DOT__word_snt_cnt_inst__DOT__clk));
    bufp->chgBit(oldp+21,(vlSelf->cim__DOT__word_snt_cnt_inst__DOT__rst_n));
    bufp->chgCData(oldp+22,(vlSelf->cim__DOT__word_snt_cnt_inst__DOT__inc),7);
    bufp->chgCData(oldp+23,(vlSelf->cim__DOT__word_snt_cnt_inst__DOT__cnt),7);
    bufp->chgCData(oldp+24,(vlSelf->cim__DOT__word_snt_cnt_inst__DOT__inc_prev),7);
}

void Vtop___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
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
