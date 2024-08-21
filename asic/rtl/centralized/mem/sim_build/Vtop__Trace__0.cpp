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
    bufp->chgBit(oldp+2,(vlSelf->param_read_en));
    bufp->chgBit(oldp+3,(vlSelf->param_write_en));
    bufp->chgBit(oldp+4,(vlSelf->param_chip_en));
    bufp->chgBit(oldp+5,(vlSelf->param_read_data_width));
    bufp->chgBit(oldp+6,(vlSelf->param_write_data_width));
    bufp->chgSData(oldp+7,(vlSelf->param_read_addr),15);
    bufp->chgSData(oldp+8,(vlSelf->param_write_addr),15);
    bufp->chgSData(oldp+9,(vlSelf->param_write_data),9);
    bufp->chgSData(oldp+10,(vlSelf->param_read_data),9);
    bufp->chgBit(oldp+11,(vlSelf->int_res_read_en));
    bufp->chgBit(oldp+12,(vlSelf->int_res_write_en));
    bufp->chgBit(oldp+13,(vlSelf->int_res_chip_en));
    bufp->chgBit(oldp+14,(vlSelf->int_res_read_data_width));
    bufp->chgBit(oldp+15,(vlSelf->int_res_write_data_width));
    bufp->chgSData(oldp+16,(vlSelf->int_res_read_addr),16);
    bufp->chgSData(oldp+17,(vlSelf->int_res_write_addr),16);
    bufp->chgIData(oldp+18,(vlSelf->int_res_write_data),18);
    bufp->chgIData(oldp+19,(vlSelf->int_res_read_data),18);
    bufp->chgBit(oldp+20,(vlSelf->mem_tb__DOT__clk));
    bufp->chgBit(oldp+21,(vlSelf->mem_tb__DOT__rst_n));
    bufp->chgBit(oldp+22,(vlSelf->mem_tb__DOT__param_read_en));
    bufp->chgBit(oldp+23,(vlSelf->mem_tb__DOT__param_write_en));
    bufp->chgBit(oldp+24,(vlSelf->mem_tb__DOT__param_chip_en));
    bufp->chgBit(oldp+25,(vlSelf->mem_tb__DOT__param_read_data_width));
    bufp->chgBit(oldp+26,(vlSelf->mem_tb__DOT__param_write_data_width));
    bufp->chgSData(oldp+27,(vlSelf->mem_tb__DOT__param_read_addr),15);
    bufp->chgSData(oldp+28,(vlSelf->mem_tb__DOT__param_write_addr),15);
    bufp->chgSData(oldp+29,(vlSelf->mem_tb__DOT__param_write_data),9);
    bufp->chgSData(oldp+30,(vlSelf->mem_tb__DOT__param_read_data),9);
    bufp->chgBit(oldp+31,(vlSelf->mem_tb__DOT__int_res_read_en));
    bufp->chgBit(oldp+32,(vlSelf->mem_tb__DOT__int_res_write_en));
    bufp->chgBit(oldp+33,(vlSelf->mem_tb__DOT__int_res_chip_en));
    bufp->chgBit(oldp+34,(vlSelf->mem_tb__DOT__int_res_read_data_width));
    bufp->chgBit(oldp+35,(vlSelf->mem_tb__DOT__int_res_write_data_width));
    bufp->chgSData(oldp+36,(vlSelf->mem_tb__DOT__int_res_read_addr),16);
    bufp->chgSData(oldp+37,(vlSelf->mem_tb__DOT__int_res_write_addr),16);
    bufp->chgIData(oldp+38,(vlSelf->mem_tb__DOT__int_res_write_data),18);
    bufp->chgIData(oldp+39,(vlSelf->mem_tb__DOT__int_res_read_data),18);
    bufp->chgBit(oldp+40,(vlSelf->mem_tb__DOT__int_res__DOT__clk));
    bufp->chgBit(oldp+41,(vlSelf->mem_tb__DOT__int_res__DOT__rst_n));
    bufp->chgCData(oldp+42,(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current),2);
    bufp->chgCData(oldp+43,(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev),2);
    bufp->chgCData(oldp+44,(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current),2);
    bufp->chgSData(oldp+45,(vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr),14);
    bufp->chgSData(oldp+46,(vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr),14);
    bufp->chgBit(oldp+47,(vlSelf->mem_tb__DOT__int_res__DOT__read_data_width_prev));
    bufp->chgBit(oldp+48,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__clk));
    bufp->chgBit(oldp+49,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__rst_n));
    bufp->chgBit(oldp+50,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__WEN));
    bufp->chgBit(oldp+51,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__CEN));
    bufp->chgBit(oldp+52,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__clk));
    bufp->chgBit(oldp+53,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__rst_n));
    bufp->chgBit(oldp+54,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__WEN));
    bufp->chgBit(oldp+55,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__CEN));
    bufp->chgBit(oldp+56,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__clk));
    bufp->chgBit(oldp+57,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__rst_n));
    bufp->chgBit(oldp+58,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__WEN));
    bufp->chgBit(oldp+59,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__CEN));
    bufp->chgBit(oldp+60,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__clk));
    bufp->chgBit(oldp+61,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__rst_n));
    bufp->chgBit(oldp+62,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__WEN));
    bufp->chgBit(oldp+63,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__CEN));
    bufp->chgBit(oldp+64,(vlSelf->mem_tb__DOT__params__DOT__clk));
    bufp->chgBit(oldp+65,(vlSelf->mem_tb__DOT__params__DOT__rst_n));
    bufp->chgBit(oldp+66,(vlSelf->mem_tb__DOT__params__DOT__params_0_read_en_prev));
    bufp->chgBit(oldp+67,(vlSelf->mem_tb__DOT__params__DOT__params_1_read_en_prev));
    bufp->chgBit(oldp+68,(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__clk));
    bufp->chgBit(oldp+69,(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__rst_n));
    bufp->chgBit(oldp+70,(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__WEN));
    bufp->chgBit(oldp+71,(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__CEN));
    bufp->chgBit(oldp+72,(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__clk));
    bufp->chgBit(oldp+73,(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__rst_n));
    bufp->chgBit(oldp+74,(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__WEN));
    bufp->chgBit(oldp+75,(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__CEN));
    bufp->chgBit(oldp+76,(vlSymsp->TOP__mem_tb__DOT__param_read_sig.en));
    bufp->chgBit(oldp+77,(vlSymsp->TOP__mem_tb__DOT__param_read_sig.chip_en));
    bufp->chgBit(oldp+78,(vlSymsp->TOP__mem_tb__DOT__param_read_sig.data_width));
    bufp->chgSData(oldp+79,(vlSymsp->TOP__mem_tb__DOT__param_read_sig.data),9);
    bufp->chgSData(oldp+80,(vlSymsp->TOP__mem_tb__DOT__param_read_sig.addr),15);
    bufp->chgBit(oldp+81,(vlSymsp->TOP__mem_tb__DOT__param_write_sig.en));
    bufp->chgBit(oldp+82,(vlSymsp->TOP__mem_tb__DOT__param_write_sig.chip_en));
    bufp->chgBit(oldp+83,(vlSymsp->TOP__mem_tb__DOT__param_write_sig.data_width));
    bufp->chgSData(oldp+84,(vlSymsp->TOP__mem_tb__DOT__param_write_sig.data),9);
    bufp->chgSData(oldp+85,(vlSymsp->TOP__mem_tb__DOT__param_write_sig.addr),15);
    bufp->chgBit(oldp+86,(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.en));
    bufp->chgBit(oldp+87,(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.chip_en));
    bufp->chgBit(oldp+88,(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.data_width));
    bufp->chgIData(oldp+89,(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.data),18);
    bufp->chgSData(oldp+90,(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.addr),16);
    bufp->chgBit(oldp+91,(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.en));
    bufp->chgBit(oldp+92,(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.chip_en));
    bufp->chgBit(oldp+93,(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data_width));
    bufp->chgIData(oldp+94,(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data),18);
    bufp->chgSData(oldp+95,(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr),16);
    bufp->chgBit(oldp+96,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.en));
    bufp->chgBit(oldp+97,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.chip_en));
    bufp->chgBit(oldp+98,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.data_width));
    bufp->chgSData(oldp+99,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.data),9);
    bufp->chgSData(oldp+100,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.addr),14);
    bufp->chgBit(oldp+101,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.en));
    bufp->chgBit(oldp+102,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.chip_en));
    bufp->chgBit(oldp+103,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.data_width));
    bufp->chgSData(oldp+104,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.data),9);
    bufp->chgSData(oldp+105,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.addr),14);
    bufp->chgBit(oldp+106,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.en));
    bufp->chgBit(oldp+107,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.chip_en));
    bufp->chgBit(oldp+108,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.data_width));
    bufp->chgSData(oldp+109,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.data),9);
    bufp->chgSData(oldp+110,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.addr),14);
    bufp->chgBit(oldp+111,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.en));
    bufp->chgBit(oldp+112,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.chip_en));
    bufp->chgBit(oldp+113,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.data_width));
    bufp->chgSData(oldp+114,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.data),9);
    bufp->chgSData(oldp+115,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.addr),14);
    bufp->chgBit(oldp+116,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.en));
    bufp->chgBit(oldp+117,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.chip_en));
    bufp->chgBit(oldp+118,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data_width));
    bufp->chgSData(oldp+119,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data),9);
    bufp->chgSData(oldp+120,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.addr),14);
    bufp->chgBit(oldp+121,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.en));
    bufp->chgBit(oldp+122,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.chip_en));
    bufp->chgBit(oldp+123,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.data_width));
    bufp->chgSData(oldp+124,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.data),9);
    bufp->chgSData(oldp+125,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.addr),14);
    bufp->chgBit(oldp+126,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.en));
    bufp->chgBit(oldp+127,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.chip_en));
    bufp->chgBit(oldp+128,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data_width));
    bufp->chgSData(oldp+129,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data),9);
    bufp->chgSData(oldp+130,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.addr),14);
    bufp->chgBit(oldp+131,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.en));
    bufp->chgBit(oldp+132,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.chip_en));
    bufp->chgBit(oldp+133,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data_width));
    bufp->chgSData(oldp+134,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data),9);
    bufp->chgSData(oldp+135,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.addr),14);
    bufp->chgBit(oldp+136,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.en));
    bufp->chgBit(oldp+137,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.chip_en));
    bufp->chgBit(oldp+138,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data_width));
    bufp->chgSData(oldp+139,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data),9);
    bufp->chgSData(oldp+140,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.addr),14);
    bufp->chgBit(oldp+141,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.en));
    bufp->chgBit(oldp+142,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.chip_en));
    bufp->chgBit(oldp+143,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.data_width));
    bufp->chgSData(oldp+144,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.data),9);
    bufp->chgSData(oldp+145,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.addr),14);
    bufp->chgBit(oldp+146,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.en));
    bufp->chgBit(oldp+147,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.chip_en));
    bufp->chgBit(oldp+148,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data_width));
    bufp->chgSData(oldp+149,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data),9);
    bufp->chgSData(oldp+150,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.addr),14);
    bufp->chgBit(oldp+151,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.en));
    bufp->chgBit(oldp+152,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.chip_en));
    bufp->chgBit(oldp+153,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data_width));
    bufp->chgSData(oldp+154,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data),9);
    bufp->chgSData(oldp+155,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.addr),14);
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
