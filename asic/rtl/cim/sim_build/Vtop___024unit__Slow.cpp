// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "Vtop__pch.h"
#include "Vtop__Syms.h"
#include "Vtop___024unit.h"

// Parameter definitions for Vtop___024unit
constexpr IData/*31:0*/ Vtop___024unit::N_STORAGE;
constexpr IData/*31:0*/ Vtop___024unit::N_COMP;
constexpr IData/*31:0*/ Vtop___024unit::Q;
constexpr IData/*31:0*/ Vtop___024unit::BUS_OP_WIDTH;
constexpr IData/*31:0*/ Vtop___024unit::NUM_CIMS;
constexpr IData/*31:0*/ Vtop___024unit::NUM_PATCHES;
constexpr IData/*31:0*/ Vtop___024unit::PATCH_LEN;
constexpr IData/*31:0*/ Vtop___024unit::EMB_DEPTH;
constexpr IData/*31:0*/ Vtop___024unit::MLP_DIM;
constexpr IData/*31:0*/ Vtop___024unit::NUM_SLEEP_STAGES;
constexpr IData/*31:0*/ Vtop___024unit::NUM_HEADS;
constexpr IData/*31:0*/ Vtop___024unit::NUM_PARAMS;
constexpr IData/*31:0*/ Vtop___024unit::PARAMS_STORAGE_SIZE_CIM;
constexpr IData/*31:0*/ Vtop___024unit::TEMP_RES_STORAGE_SIZE_CIM;
constexpr IData/*31:0*/ Vtop___024unit::EEG_SAMPLE_DEPTH;


void Vtop___024unit___ctor_var_reset(Vtop___024unit* vlSelf);

Vtop___024unit::Vtop___024unit(Vtop__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtop___024unit___ctor_var_reset(this);
}

void Vtop___024unit::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vtop___024unit::~Vtop___024unit() {
}
