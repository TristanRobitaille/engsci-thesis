// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtop.h for the primary calling header

#ifndef VERILATED_VTOP_MEMORYACCESSSIGNALS__TZ3_TBZ4_H_
#define VERILATED_VTOP_MEMORYACCESSSIGNALS__TZ3_TBZ4_H_  // guard

#include "verilated.h"


class Vtop__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtop_MemoryAccessSignals__Tz3_TBz4 final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ en;
    CData/*0:0*/ chip_en;
    CData/*0:0*/ data_width;
    SData/*15:0*/ addr;
    IData/*17:0*/ data;

    // INTERNAL VARIABLES
    Vtop__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vtop_MemoryAccessSignals__Tz3_TBz4(Vtop__Syms* symsp, const char* v__name);
    ~Vtop_MemoryAccessSignals__Tz3_TBz4();
    VL_UNCOPYABLE(Vtop_MemoryAccessSignals__Tz3_TBz4);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};

std::string VL_TO_STRING(const Vtop_MemoryAccessSignals__Tz3_TBz4* obj);

#endif  // guard
