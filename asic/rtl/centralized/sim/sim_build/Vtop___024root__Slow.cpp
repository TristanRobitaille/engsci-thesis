// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "verilated.h"
#include "verilated_dpi.h"

#include "Vtop__Syms.h"
#include "Vtop__Syms.h"
#include "Vtop___024root.h"

// Parameter definitions for Vtop___024root
constexpr IData/*31:0*/ Vtop___024root::cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__WIDTH;
constexpr IData/*31:0*/ Vtop___024root::cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__MODE;
constexpr IData/*31:0*/ Vtop___024root::cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__WIDTH;
constexpr IData/*31:0*/ Vtop___024root::cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__MODE;
constexpr IData/*31:0*/ Vtop___024root::cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__WIDTH;
constexpr IData/*31:0*/ Vtop___024root::cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__MODE;


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
