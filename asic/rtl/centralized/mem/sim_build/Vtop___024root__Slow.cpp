// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "verilated.h"
#include "verilated_dpi.h"

#include "Vtop__Syms.h"
#include "Vtop__Syms.h"
#include "Vtop___024root.h"

// Parameter definitions for Vtop___024root
constexpr IData/*31:0*/ Vtop___024root::mem_tb__DOT__params__DOT__params_0__DOT__DEPTH;
constexpr IData/*31:0*/ Vtop___024root::mem_tb__DOT__params__DOT__params_1__DOT__DEPTH;
constexpr IData/*31:0*/ Vtop___024root::mem_tb__DOT__int_res__DOT__int_res_0__DOT__DEPTH;
constexpr IData/*31:0*/ Vtop___024root::mem_tb__DOT__int_res__DOT__int_res_1__DOT__DEPTH;
constexpr IData/*31:0*/ Vtop___024root::mem_tb__DOT__int_res__DOT__int_res_2__DOT__DEPTH;
constexpr IData/*31:0*/ Vtop___024root::mem_tb__DOT__int_res__DOT__int_res_3__DOT__DEPTH;


void Vtop___024root___ctor_var_reset(Vtop___024root* vlSelf);

Vtop___024root::Vtop___024root(Vtop__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtop___024root___ctor_var_reset(this);
}

void Vtop___024root::__Vconfigure(bool first) {
    if (false && first) {}  // Prevent unused
}

Vtop___024root::~Vtop___024root() {
}
