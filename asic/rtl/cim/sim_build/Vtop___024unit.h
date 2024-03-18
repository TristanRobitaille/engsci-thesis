// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtop.h for the primary calling header

#ifndef VERILATED_VTOP___024UNIT_H_
#define VERILATED_VTOP___024UNIT_H_  // guard

#include "verilated.h"
#include "verilated_threads.h"


class Vtop__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtop___024unit final : public VerilatedModule {
  public:

    // INTERNAL VARIABLES
    Vtop__Syms* const vlSymsp;

    // PARAMETERS
    static constexpr IData/*31:0*/ N_STORAGE = 0x00000010U;
    static constexpr IData/*31:0*/ N_COMP = 0x00000016U;
    static constexpr IData/*31:0*/ Q = 0x0000000aU;
    static constexpr IData/*31:0*/ BUS_OP_WIDTH = 4U;
    static constexpr IData/*31:0*/ NUM_CIMS = 0x00000040U;
    static constexpr IData/*31:0*/ NUM_PATCHES = 0x0000003cU;
    static constexpr IData/*31:0*/ PATCH_LEN = 0x00000040U;
    static constexpr IData/*31:0*/ EMB_DEPTH = 0x00000040U;
    static constexpr IData/*31:0*/ MLP_DIM = 0x00000020U;
    static constexpr IData/*31:0*/ NUM_SLEEP_STAGES = 5U;
    static constexpr IData/*31:0*/ NUM_HEADS = 8U;
    static constexpr IData/*31:0*/ NUM_PARAMS = 0x00007b65U;
    static constexpr IData/*31:0*/ PARAMS_STORAGE_SIZE_CIM = 0x00000210U;
    static constexpr IData/*31:0*/ TEMP_RES_STORAGE_SIZE_CIM = 0x00000350U;
    static constexpr IData/*31:0*/ EEG_SAMPLE_DEPTH = 0x00000010U;

    // CONSTRUCTORS
    Vtop___024unit(Vtop__Syms* symsp, const char* v__name);
    ~Vtop___024unit();
    VL_UNCOPYABLE(Vtop___024unit);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
