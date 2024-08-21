// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_fst_c.h"
#include "Vtop__Syms.h"


VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+1,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+2,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->pushNamePrefix("cim_centralized_tb ");
    tracep->declBit(c+3,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+4,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->pushNamePrefix("cim_centralized ");
    tracep->declBit(c+5,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+6,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+7,"cnt_4b_rst_n",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+8,"cnt_4b_inc",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+9,"cnt_4b_cnt",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 3,0);
    tracep->declBit(c+10,"cnt_7b_rst_n",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+11,"cnt_7b_inc",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+12,"cnt_7b_cnt",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 6,0);
    tracep->declBit(c+13,"cnt_9b_rst_n",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+14,"cnt_9b_inc",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+15,"cnt_9b_cnt",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+16,"temp_res_addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 14,0);
    tracep->declBus(c+17,"param_addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 15,0);
    tracep->pushNamePrefix("cnt_4b ");
    tracep->declBus(c+33,"WIDTH",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBus(c+34,"MODE",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBit(c+18,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+19,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+20,"inc",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBus(c+21,"cnt",-1,FST_VD_OUTPUT,FST_VT_VCD_WIRE, false,-1, 3,0);
    tracep->declBit(c+22,"inc_prev",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("cnt_7b ");
    tracep->declBus(c+35,"WIDTH",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBus(c+34,"MODE",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBit(c+23,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+24,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+25,"inc",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBus(c+26,"cnt",-1,FST_VD_OUTPUT,FST_VT_VCD_WIRE, false,-1, 6,0);
    tracep->declBit(c+27,"inc_prev",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("cnt_9b ");
    tracep->declBus(c+36,"WIDTH",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBus(c+34,"MODE",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBit(c+28,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+29,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+30,"inc",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBus(c+31,"cnt",-1,FST_VD_OUTPUT,FST_VT_VCD_WIRE, false,-1, 8,0);
    tracep->declBit(c+32,"inc_prev",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->popNamePrefix(3);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__Defines__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__Defines__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBus(c+37,"CIM_PARAMS_STORAGE_SIZE_NUM_ELEM",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBus(c+38,"CIM_INT_RES_SIZE_NUM_ELEM",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBus(c+36,"N_STO_INT_RES",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBus(c+36,"N_STO_PARAMS",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_top(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_top\n"); );
    // Body
    Vtop___024root__trace_init_sub__TOP__0(vlSelf, tracep);
    tracep->pushNamePrefix("Defines ");
    Vtop___024root__trace_init_sub__TOP__Defines__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
}

VL_ATTR_COLD void Vtop___024root__trace_full_top_0(void* voidSelf, VerilatedFst::Buffer* bufp);
void Vtop___024root__trace_chg_top_0(void* voidSelf, VerilatedFst::Buffer* bufp);
void Vtop___024root__trace_cleanup(void* voidSelf, VerilatedFst* /*unused*/);

VL_ATTR_COLD void Vtop___024root__trace_register(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_register\n"); );
    // Body
    tracep->addFullCb(&Vtop___024root__trace_full_top_0, vlSelf);
    tracep->addChgCb(&Vtop___024root__trace_chg_top_0, vlSelf);
    tracep->addCleanupCb(&Vtop___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vtop___024root__trace_full_sub_0(Vtop___024root* vlSelf, VerilatedFst::Buffer* bufp);

VL_ATTR_COLD void Vtop___024root__trace_full_top_0(void* voidSelf, VerilatedFst::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_full_top_0\n"); );
    // Init
    Vtop___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop___024root*>(voidSelf);
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vtop___024root__trace_full_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vtop___024root__trace_full_sub_0(Vtop___024root* vlSelf, VerilatedFst::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_full_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullBit(oldp+1,(vlSelf->clk));
    bufp->fullBit(oldp+2,(vlSelf->rst_n));
    bufp->fullBit(oldp+3,(vlSelf->cim_centralized_tb__DOT__clk));
    bufp->fullBit(oldp+4,(vlSelf->cim_centralized_tb__DOT__rst_n));
    bufp->fullBit(oldp+5,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__clk));
    bufp->fullBit(oldp+6,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__rst_n));
    bufp->fullBit(oldp+7,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_rst_n));
    bufp->fullBit(oldp+8,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_inc));
    bufp->fullCData(oldp+9,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_cnt),4);
    bufp->fullBit(oldp+10,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_rst_n));
    bufp->fullBit(oldp+11,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_inc));
    bufp->fullCData(oldp+12,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_cnt),7);
    bufp->fullBit(oldp+13,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_rst_n));
    bufp->fullBit(oldp+14,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_inc));
    bufp->fullSData(oldp+15,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_cnt),9);
    bufp->fullSData(oldp+16,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__temp_res_addr),15);
    bufp->fullSData(oldp+17,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__param_addr),16);
    bufp->fullBit(oldp+18,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__clk));
    bufp->fullBit(oldp+19,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__rst_n));
    bufp->fullBit(oldp+20,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc));
    bufp->fullCData(oldp+21,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__cnt),4);
    bufp->fullBit(oldp+22,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc_prev));
    bufp->fullBit(oldp+23,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__clk));
    bufp->fullBit(oldp+24,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__rst_n));
    bufp->fullBit(oldp+25,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc));
    bufp->fullCData(oldp+26,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__cnt),7);
    bufp->fullBit(oldp+27,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc_prev));
    bufp->fullBit(oldp+28,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__clk));
    bufp->fullBit(oldp+29,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__rst_n));
    bufp->fullBit(oldp+30,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc));
    bufp->fullSData(oldp+31,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__cnt),9);
    bufp->fullBit(oldp+32,(vlSelf->cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc_prev));
    bufp->fullIData(oldp+33,(4U),32);
    bufp->fullIData(oldp+34,(0U),32);
    bufp->fullIData(oldp+35,(7U),32);
    bufp->fullIData(oldp+36,(9U),32);
    bufp->fullIData(oldp+37,(0x7ba0U),32);
    bufp->fullIData(oldp+38,(0xdf1cU),32);
}
