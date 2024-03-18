// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "Vtop__pch.h"
#include "Vtop__Syms.h"
#include "Vtop___024root.h"

// Parameter definitions for Vtop___024root
constexpr IData/*31:0*/ Vtop___024root::cim__DOT__gen_cnt_7b_inst__DOT__WIDTH;
constexpr IData/*31:0*/ Vtop___024root::cim__DOT__gen_cnt_7b_inst__DOT__MODE;
constexpr IData/*31:0*/ Vtop___024root::cim__DOT__gen_cnt_7b_2_inst__DOT__WIDTH;
constexpr IData/*31:0*/ Vtop___024root::cim__DOT__gen_cnt_7b_2_inst__DOT__MODE;
constexpr IData/*31:0*/ Vtop___024root::cim__DOT__word_rec_cnt_inst__DOT__WIDTH;
constexpr IData/*31:0*/ Vtop___024root::cim__DOT__word_rec_cnt_inst__DOT__MODE;
constexpr IData/*31:0*/ Vtop___024root::cim__DOT__word_snt_cnt_inst__DOT__WIDTH;
constexpr IData/*31:0*/ Vtop___024root::cim__DOT__word_snt_cnt_inst__DOT__MODE;


void Vtop___024root___ctor_var_reset(Vtop___024root* vlSelf);

Vtop___024root::Vtop___024root(Vtop__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , __Vm_mtaskstate_6(1U)
    , __Vm_mtaskstate_final__nba(2U)
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtop___024root___ctor_var_reset(this);
}

void Vtop___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vtop___024root::~Vtop___024root() {
}
