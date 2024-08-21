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
    tracep->declBit(c+3,"param_read_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+4,"param_write_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+5,"param_chip_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    {
        const char* __VenumItemNames[]
        = {"SINGLE_WIDTH", "DOUBLE_WIDTH"};
        const char* __VenumItemValues[]
        = {"0", "1"};
        tracep->declDTypeEnum(1, "Defines::DataWidth_t", 2, 1, __VenumItemNames, __VenumItemValues);
    }
    tracep->declBit(c+6,"param_read_data_width",1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+7,"param_write_data_width",1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBus(c+8,"param_read_addr",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 14,0);
    tracep->declBus(c+9,"param_write_addr",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 14,0);
    tracep->declBus(c+10,"param_write_data",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 8,0);
    tracep->declBus(c+11,"param_read_data",-1,FST_VD_OUTPUT,FST_VT_VCD_WIRE, false,-1, 8,0);
    tracep->declBit(c+12,"int_res_read_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+13,"int_res_write_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+14,"int_res_chip_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+15,"int_res_read_data_width",1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+16,"int_res_write_data_width",1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBus(c+17,"int_res_read_addr",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 15,0);
    tracep->declBus(c+18,"int_res_write_addr",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 15,0);
    tracep->declBus(c+19,"int_res_write_data",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 17,0);
    tracep->declBus(c+20,"int_res_read_data",-1,FST_VD_OUTPUT,FST_VT_VCD_WIRE, false,-1, 17,0);
    tracep->pushNamePrefix("mem_tb ");
    tracep->declBit(c+21,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+22,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+23,"param_read_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+24,"param_write_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+25,"param_chip_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+26,"param_read_data_width",1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+27,"param_write_data_width",1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBus(c+28,"param_read_addr",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 14,0);
    tracep->declBus(c+29,"param_write_addr",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 14,0);
    tracep->declBus(c+30,"param_write_data",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 8,0);
    tracep->declBus(c+31,"param_read_data",-1,FST_VD_OUTPUT,FST_VT_VCD_WIRE, false,-1, 8,0);
    tracep->declBit(c+32,"int_res_read_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+33,"int_res_write_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+34,"int_res_chip_en",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+35,"int_res_read_data_width",1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+36,"int_res_write_data_width",1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBus(c+37,"int_res_read_addr",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 15,0);
    tracep->declBus(c+38,"int_res_write_addr",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 15,0);
    tracep->declBus(c+39,"int_res_write_data",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1, 17,0);
    tracep->declBus(c+40,"int_res_read_data",-1,FST_VD_OUTPUT,FST_VT_VCD_WIRE, false,-1, 17,0);
    tracep->pushNamePrefix("int_res ");
    tracep->declBit(c+41,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+42,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBus(c+43,"bank_read_current",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 1,0);
    tracep->declBus(c+44,"bank_read_prev",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 1,0);
    tracep->declBus(c+45,"bank_write_current",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 1,0);
    tracep->declBus(c+46,"read_base_addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
    tracep->declBus(c+47,"write_base_addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
    tracep->declBit(c+48,"read_data_width_prev",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->pushNamePrefix("int_res_0 ");
    tracep->declBus(c+157,"DEPTH",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBit(c+49,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+50,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+51,"WEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+52,"CEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("int_res_1 ");
    tracep->declBus(c+157,"DEPTH",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBit(c+53,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+54,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+55,"WEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+56,"CEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("int_res_2 ");
    tracep->declBus(c+157,"DEPTH",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBit(c+57,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+58,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+59,"WEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+60,"CEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("int_res_3 ");
    tracep->declBus(c+157,"DEPTH",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBit(c+61,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+62,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+63,"WEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+64,"CEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->popNamePrefix(2);
    tracep->pushNamePrefix("params ");
    tracep->declBit(c+65,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+66,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+67,"params_0_read_en_prev",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+68,"params_1_read_en_prev",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->pushNamePrefix("params_0 ");
    tracep->declBus(c+158,"DEPTH",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBit(c+69,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+70,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+71,"WEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+72,"CEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("params_1 ");
    tracep->declBus(c+158,"DEPTH",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBit(c+73,"clk",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+74,"rst_n",-1,FST_VD_INPUT,FST_VT_VCD_WIRE, false,-1);
    tracep->declBit(c+75,"WEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+76,"CEN",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->popNamePrefix(3);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__Defines__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__Defines__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBus(c+158,"CIM_PARAMS_BANK_SIZE_NUM_WORD",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBus(c+157,"CIM_INT_RES_BANK_SIZE_NUM_WORD",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBus(c+159,"CIM_PARAMS_NUM_BANKS",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBus(c+160,"CIM_INT_RES_NUM_BANKS",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBus(c+161,"N_STO_INT_RES",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
    tracep->declBus(c+161,"N_STO_PARAMS",-1, FST_VD_IMPLICIT,FST_VT_VCD_PARAMETER, false,-1, 31,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__param_read_sig__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__param_read_sig__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+77,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+78,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+79,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+80,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+81,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 14,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__param_write_sig__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__param_write_sig__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+82,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+83,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+84,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+85,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+86,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 14,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res_read_sig__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res_read_sig__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+87,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+88,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+89,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+90,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 17,0);
    tracep->declBus(c+91,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 15,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res_write_sig__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res_write_sig__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+92,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+93,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+94,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+95,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 17,0);
    tracep->declBus(c+96,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 15,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_0_read__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_0_read__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+97,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+98,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+99,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+100,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+101,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_0_write__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_0_write__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+102,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+103,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+104,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+105,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+106,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_1_read__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_1_read__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+107,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+108,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+109,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+110,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+111,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_1_write__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_1_write__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+112,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+113,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+114,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+115,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+116,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_0_read__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_0_read__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+117,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+118,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+119,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+120,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+121,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_0_write__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_0_write__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+122,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+123,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+124,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+125,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+126,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_1_read__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_1_read__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+127,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+128,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+129,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+130,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+131,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_1_write__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_1_write__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+132,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+133,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+134,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+135,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+136,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_2_read__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_2_read__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+137,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+138,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+139,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+140,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+141,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_2_write__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_2_write__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+142,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+143,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+144,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+145,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+146,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_3_read__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_3_read__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+147,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+148,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+149,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+150,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+151,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
}

VL_ATTR_COLD void Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_3_write__0(Vtop___024root* vlSelf, VerilatedFst* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_3_write__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+152,"en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+153,"chip_en",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBit(c+154,"data_width",1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1);
    tracep->declBus(c+155,"data",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 8,0);
    tracep->declBus(c+156,"addr",-1, FST_VD_IMPLICIT,FST_VT_SV_LOGIC, false,-1, 13,0);
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
    tracep->pushNamePrefix("mem_tb ");
    tracep->pushNamePrefix("int_res ");
    tracep->pushNamePrefix("int_res_0 ");
    tracep->pushNamePrefix("read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_0_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_0_write__0(vlSelf, tracep);
    tracep->popNamePrefix(2);
    tracep->pushNamePrefix("int_res_0_read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_0_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("int_res_0_write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_0_write__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("int_res_1 ");
    tracep->pushNamePrefix("read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_1_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_1_write__0(vlSelf, tracep);
    tracep->popNamePrefix(2);
    tracep->pushNamePrefix("int_res_1_read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_1_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("int_res_1_write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_1_write__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("int_res_2 ");
    tracep->pushNamePrefix("read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_2_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_2_write__0(vlSelf, tracep);
    tracep->popNamePrefix(2);
    tracep->pushNamePrefix("int_res_2_read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_2_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("int_res_2_write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_2_write__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("int_res_3 ");
    tracep->pushNamePrefix("read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_3_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_3_write__0(vlSelf, tracep);
    tracep->popNamePrefix(2);
    tracep->pushNamePrefix("int_res_3_read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_3_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("int_res_3_write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res__DOT__int_res_3_write__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res_read_sig__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res_write_sig__0(vlSelf, tracep);
    tracep->popNamePrefix(2);
    tracep->pushNamePrefix("int_res_read_sig\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res_read_sig__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("int_res_write_sig\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__int_res_write_sig__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("param_read_sig\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__param_read_sig__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("param_write_sig\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__param_write_sig__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("params ");
    tracep->pushNamePrefix("params_0 ");
    tracep->pushNamePrefix("read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_0_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_0_write__0(vlSelf, tracep);
    tracep->popNamePrefix(2);
    tracep->pushNamePrefix("params_0_read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_0_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("params_0_write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_0_write__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("params_1 ");
    tracep->pushNamePrefix("read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_1_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_1_write__0(vlSelf, tracep);
    tracep->popNamePrefix(2);
    tracep->pushNamePrefix("params_1_read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_1_read__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("params_1_write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__params__DOT__params_1_write__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("read\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__param_read_sig__0(vlSelf, tracep);
    tracep->popNamePrefix(1);
    tracep->pushNamePrefix("write\211 ");
    Vtop___024root__trace_init_sub__TOP__mem_tb__DOT__param_write_sig__0(vlSelf, tracep);
    tracep->popNamePrefix(3);
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
    bufp->fullBit(oldp+3,(vlSelf->param_read_en));
    bufp->fullBit(oldp+4,(vlSelf->param_write_en));
    bufp->fullBit(oldp+5,(vlSelf->param_chip_en));
    bufp->fullBit(oldp+6,(vlSelf->param_read_data_width));
    bufp->fullBit(oldp+7,(vlSelf->param_write_data_width));
    bufp->fullSData(oldp+8,(vlSelf->param_read_addr),15);
    bufp->fullSData(oldp+9,(vlSelf->param_write_addr),15);
    bufp->fullSData(oldp+10,(vlSelf->param_write_data),9);
    bufp->fullSData(oldp+11,(vlSelf->param_read_data),9);
    bufp->fullBit(oldp+12,(vlSelf->int_res_read_en));
    bufp->fullBit(oldp+13,(vlSelf->int_res_write_en));
    bufp->fullBit(oldp+14,(vlSelf->int_res_chip_en));
    bufp->fullBit(oldp+15,(vlSelf->int_res_read_data_width));
    bufp->fullBit(oldp+16,(vlSelf->int_res_write_data_width));
    bufp->fullSData(oldp+17,(vlSelf->int_res_read_addr),16);
    bufp->fullSData(oldp+18,(vlSelf->int_res_write_addr),16);
    bufp->fullIData(oldp+19,(vlSelf->int_res_write_data),18);
    bufp->fullIData(oldp+20,(vlSelf->int_res_read_data),18);
    bufp->fullBit(oldp+21,(vlSelf->mem_tb__DOT__clk));
    bufp->fullBit(oldp+22,(vlSelf->mem_tb__DOT__rst_n));
    bufp->fullBit(oldp+23,(vlSelf->mem_tb__DOT__param_read_en));
    bufp->fullBit(oldp+24,(vlSelf->mem_tb__DOT__param_write_en));
    bufp->fullBit(oldp+25,(vlSelf->mem_tb__DOT__param_chip_en));
    bufp->fullBit(oldp+26,(vlSelf->mem_tb__DOT__param_read_data_width));
    bufp->fullBit(oldp+27,(vlSelf->mem_tb__DOT__param_write_data_width));
    bufp->fullSData(oldp+28,(vlSelf->mem_tb__DOT__param_read_addr),15);
    bufp->fullSData(oldp+29,(vlSelf->mem_tb__DOT__param_write_addr),15);
    bufp->fullSData(oldp+30,(vlSelf->mem_tb__DOT__param_write_data),9);
    bufp->fullSData(oldp+31,(vlSelf->mem_tb__DOT__param_read_data),9);
    bufp->fullBit(oldp+32,(vlSelf->mem_tb__DOT__int_res_read_en));
    bufp->fullBit(oldp+33,(vlSelf->mem_tb__DOT__int_res_write_en));
    bufp->fullBit(oldp+34,(vlSelf->mem_tb__DOT__int_res_chip_en));
    bufp->fullBit(oldp+35,(vlSelf->mem_tb__DOT__int_res_read_data_width));
    bufp->fullBit(oldp+36,(vlSelf->mem_tb__DOT__int_res_write_data_width));
    bufp->fullSData(oldp+37,(vlSelf->mem_tb__DOT__int_res_read_addr),16);
    bufp->fullSData(oldp+38,(vlSelf->mem_tb__DOT__int_res_write_addr),16);
    bufp->fullIData(oldp+39,(vlSelf->mem_tb__DOT__int_res_write_data),18);
    bufp->fullIData(oldp+40,(vlSelf->mem_tb__DOT__int_res_read_data),18);
    bufp->fullBit(oldp+41,(vlSelf->mem_tb__DOT__int_res__DOT__clk));
    bufp->fullBit(oldp+42,(vlSelf->mem_tb__DOT__int_res__DOT__rst_n));
    bufp->fullCData(oldp+43,(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current),2);
    bufp->fullCData(oldp+44,(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev),2);
    bufp->fullCData(oldp+45,(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current),2);
    bufp->fullSData(oldp+46,(vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr),14);
    bufp->fullSData(oldp+47,(vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr),14);
    bufp->fullBit(oldp+48,(vlSelf->mem_tb__DOT__int_res__DOT__read_data_width_prev));
    bufp->fullBit(oldp+49,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__clk));
    bufp->fullBit(oldp+50,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__rst_n));
    bufp->fullBit(oldp+51,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__WEN));
    bufp->fullBit(oldp+52,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__CEN));
    bufp->fullBit(oldp+53,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__clk));
    bufp->fullBit(oldp+54,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__rst_n));
    bufp->fullBit(oldp+55,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__WEN));
    bufp->fullBit(oldp+56,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__CEN));
    bufp->fullBit(oldp+57,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__clk));
    bufp->fullBit(oldp+58,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__rst_n));
    bufp->fullBit(oldp+59,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__WEN));
    bufp->fullBit(oldp+60,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__CEN));
    bufp->fullBit(oldp+61,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__clk));
    bufp->fullBit(oldp+62,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__rst_n));
    bufp->fullBit(oldp+63,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__WEN));
    bufp->fullBit(oldp+64,(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__CEN));
    bufp->fullBit(oldp+65,(vlSelf->mem_tb__DOT__params__DOT__clk));
    bufp->fullBit(oldp+66,(vlSelf->mem_tb__DOT__params__DOT__rst_n));
    bufp->fullBit(oldp+67,(vlSelf->mem_tb__DOT__params__DOT__params_0_read_en_prev));
    bufp->fullBit(oldp+68,(vlSelf->mem_tb__DOT__params__DOT__params_1_read_en_prev));
    bufp->fullBit(oldp+69,(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__clk));
    bufp->fullBit(oldp+70,(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__rst_n));
    bufp->fullBit(oldp+71,(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__WEN));
    bufp->fullBit(oldp+72,(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__CEN));
    bufp->fullBit(oldp+73,(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__clk));
    bufp->fullBit(oldp+74,(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__rst_n));
    bufp->fullBit(oldp+75,(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__WEN));
    bufp->fullBit(oldp+76,(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__CEN));
    bufp->fullBit(oldp+77,(vlSymsp->TOP__mem_tb__DOT__param_read_sig.en));
    bufp->fullBit(oldp+78,(vlSymsp->TOP__mem_tb__DOT__param_read_sig.chip_en));
    bufp->fullBit(oldp+79,(vlSymsp->TOP__mem_tb__DOT__param_read_sig.data_width));
    bufp->fullSData(oldp+80,(vlSymsp->TOP__mem_tb__DOT__param_read_sig.data),9);
    bufp->fullSData(oldp+81,(vlSymsp->TOP__mem_tb__DOT__param_read_sig.addr),15);
    bufp->fullBit(oldp+82,(vlSymsp->TOP__mem_tb__DOT__param_write_sig.en));
    bufp->fullBit(oldp+83,(vlSymsp->TOP__mem_tb__DOT__param_write_sig.chip_en));
    bufp->fullBit(oldp+84,(vlSymsp->TOP__mem_tb__DOT__param_write_sig.data_width));
    bufp->fullSData(oldp+85,(vlSymsp->TOP__mem_tb__DOT__param_write_sig.data),9);
    bufp->fullSData(oldp+86,(vlSymsp->TOP__mem_tb__DOT__param_write_sig.addr),15);
    bufp->fullBit(oldp+87,(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.en));
    bufp->fullBit(oldp+88,(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.chip_en));
    bufp->fullBit(oldp+89,(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.data_width));
    bufp->fullIData(oldp+90,(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.data),18);
    bufp->fullSData(oldp+91,(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.addr),16);
    bufp->fullBit(oldp+92,(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.en));
    bufp->fullBit(oldp+93,(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.chip_en));
    bufp->fullBit(oldp+94,(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data_width));
    bufp->fullIData(oldp+95,(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data),18);
    bufp->fullSData(oldp+96,(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr),16);
    bufp->fullBit(oldp+97,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.en));
    bufp->fullBit(oldp+98,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.chip_en));
    bufp->fullBit(oldp+99,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.data_width));
    bufp->fullSData(oldp+100,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.data),9);
    bufp->fullSData(oldp+101,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.addr),14);
    bufp->fullBit(oldp+102,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.en));
    bufp->fullBit(oldp+103,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.chip_en));
    bufp->fullBit(oldp+104,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.data_width));
    bufp->fullSData(oldp+105,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.data),9);
    bufp->fullSData(oldp+106,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.addr),14);
    bufp->fullBit(oldp+107,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.en));
    bufp->fullBit(oldp+108,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.chip_en));
    bufp->fullBit(oldp+109,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.data_width));
    bufp->fullSData(oldp+110,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.data),9);
    bufp->fullSData(oldp+111,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.addr),14);
    bufp->fullBit(oldp+112,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.en));
    bufp->fullBit(oldp+113,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.chip_en));
    bufp->fullBit(oldp+114,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.data_width));
    bufp->fullSData(oldp+115,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.data),9);
    bufp->fullSData(oldp+116,(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.addr),14);
    bufp->fullBit(oldp+117,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.en));
    bufp->fullBit(oldp+118,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.chip_en));
    bufp->fullBit(oldp+119,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data_width));
    bufp->fullSData(oldp+120,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data),9);
    bufp->fullSData(oldp+121,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.addr),14);
    bufp->fullBit(oldp+122,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.en));
    bufp->fullBit(oldp+123,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.chip_en));
    bufp->fullBit(oldp+124,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.data_width));
    bufp->fullSData(oldp+125,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.data),9);
    bufp->fullSData(oldp+126,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.addr),14);
    bufp->fullBit(oldp+127,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.en));
    bufp->fullBit(oldp+128,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.chip_en));
    bufp->fullBit(oldp+129,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data_width));
    bufp->fullSData(oldp+130,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data),9);
    bufp->fullSData(oldp+131,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.addr),14);
    bufp->fullBit(oldp+132,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.en));
    bufp->fullBit(oldp+133,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.chip_en));
    bufp->fullBit(oldp+134,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data_width));
    bufp->fullSData(oldp+135,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data),9);
    bufp->fullSData(oldp+136,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.addr),14);
    bufp->fullBit(oldp+137,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.en));
    bufp->fullBit(oldp+138,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.chip_en));
    bufp->fullBit(oldp+139,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data_width));
    bufp->fullSData(oldp+140,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data),9);
    bufp->fullSData(oldp+141,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.addr),14);
    bufp->fullBit(oldp+142,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.en));
    bufp->fullBit(oldp+143,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.chip_en));
    bufp->fullBit(oldp+144,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.data_width));
    bufp->fullSData(oldp+145,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.data),9);
    bufp->fullSData(oldp+146,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.addr),14);
    bufp->fullBit(oldp+147,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.en));
    bufp->fullBit(oldp+148,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.chip_en));
    bufp->fullBit(oldp+149,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data_width));
    bufp->fullSData(oldp+150,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data),9);
    bufp->fullSData(oldp+151,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.addr),14);
    bufp->fullBit(oldp+152,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.en));
    bufp->fullBit(oldp+153,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.chip_en));
    bufp->fullBit(oldp+154,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data_width));
    bufp->fullSData(oldp+155,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data),9);
    bufp->fullSData(oldp+156,(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.addr),14);
    bufp->fullIData(oldp+157,(0x3800U),32);
    bufp->fullIData(oldp+158,(0x3e00U),32);
    bufp->fullIData(oldp+159,(2U),32);
    bufp->fullIData(oldp+160,(4U),32);
    bufp->fullIData(oldp+161,(9U),32);
}
