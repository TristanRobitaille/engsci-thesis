// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtop.h for the primary calling header

#ifndef VERILATED_VTOP_DEFINES_H_
#define VERILATED_VTOP_DEFINES_H_  // guard

#include "verilated.h"


class Vtop__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtop_Defines final : public VerilatedModule {
  public:

    // INTERNAL VARIABLES
    Vtop__Syms* const vlSymsp;

    // PARAMETERS
    static constexpr IData/*31:0*/ CIM_PARAMS_STORAGE_SIZE_NUM_ELEM = 0x00007ba0U;
    static constexpr IData/*31:0*/ CIM_INT_RES_SIZE_NUM_ELEM = 0x0000df1cU;
    static constexpr IData/*31:0*/ N_STO_INT_RES = 9U;
    static constexpr IData/*31:0*/ N_STO_PARAMS = 9U;

    // CONSTRUCTORS
    Vtop_Defines(Vtop__Syms* symsp, const char* v__name);
    ~Vtop_Defines();
    VL_UNCOPYABLE(Vtop_Defines);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
