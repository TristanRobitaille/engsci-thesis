// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtop.h for the primary calling header

#ifndef VERILATED_VTOP___024ROOT_H_
#define VERILATED_VTOP___024ROOT_H_  // guard

#include "verilated.h"
class Vtop_Defines;


class Vtop__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtop___024root final : public VerilatedModule {
  public:
    // CELLS
    Vtop_Defines* __PVT__Defines;

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    CData/*0:0*/ cim_centralized_tb__DOT__clk;
    CData/*0:0*/ cim_centralized_tb__DOT__rst_n;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__clk;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__rst_n;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_rst_n;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_inc;
    CData/*3:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_cnt;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_rst_n;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_inc;
    CData/*6:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_cnt;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_rst_n;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_inc;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__clk;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__rst_n;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc;
    CData/*3:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__cnt;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc_prev;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__clk;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__rst_n;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc;
    CData/*6:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__cnt;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc_prev;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__clk;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__rst_n;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc;
    CData/*0:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc_prev;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __VactContinue;
    SData/*8:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_cnt;
    SData/*14:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__temp_res_addr;
    SData/*15:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__param_addr;
    SData/*8:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__cnt;
    IData/*31:0*/ __VstlIterCount;
    IData/*31:0*/ __VicoIterCount;
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<SData/*8:0*/, 31648> cim_centralized_tb__DOT__cim_centralized__DOT__params;
    VlUnpacked<SData/*8:0*/, 57116> cim_centralized_tb__DOT__cim_centralized__DOT__int_res;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<1> __VactTriggered;
    VlTriggerVec<1> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vtop__Syms* const vlSymsp;

    // PARAMETERS
    static constexpr IData/*31:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__WIDTH = 4U;
    static constexpr IData/*31:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__MODE = 0U;
    static constexpr IData/*31:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__WIDTH = 7U;
    static constexpr IData/*31:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__MODE = 0U;
    static constexpr IData/*31:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__WIDTH = 9U;
    static constexpr IData/*31:0*/ cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__MODE = 0U;

    // CONSTRUCTORS
    Vtop___024root(Vtop__Syms* symsp, const char* v__name);
    ~Vtop___024root();
    VL_UNCOPYABLE(Vtop___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
