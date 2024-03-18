// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtop.h for the primary calling header

#ifndef VERILATED_VTOP___024ROOT_H_
#define VERILATED_VTOP___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_threads.h"
class Vtop___024unit;


class Vtop__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtop___024root final : public VerilatedModule {
  public:
    // CELLS
    Vtop___024unit* __PVT____024unit;

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ cim__DOT__gen_cnt_7b_rst_n;
    CData/*0:0*/ cim__DOT__gen_cnt_7b_2_rst_n;
    CData/*6:0*/ cim__DOT__gen_cnt_7b_inc;
    CData/*6:0*/ cim__DOT__gen_cnt_7b_2_inc;
    CData/*6:0*/ cim__DOT__gen_cnt_7b_inst__DOT__inc_prev;
    CData/*6:0*/ cim__DOT__gen_cnt_7b_2_inst__DOT__inc_prev;
    CData/*6:0*/ cim__DOT__gen_cnt_7b_inst__DOT__cnt;
    CData/*6:0*/ cim__DOT__gen_cnt_7b_2_inst__DOT__cnt;
    CData/*6:0*/ cim__DOT__gen_cnt_7b_cnt;
    CData/*6:0*/ cim__DOT__gen_cnt_7b_2_cnt;
    CData/*6:0*/ cim__DOT__word_rec_cnt;
    CData/*6:0*/ cim__DOT__word_snt_cnt;
    CData/*6:0*/ cim__DOT__word_rec_cnt_inst__DOT__cnt;
    CData/*6:0*/ cim__DOT__word_snt_cnt_inst__DOT__cnt;
    CData/*6:0*/ cim__DOT__word_rec_cnt_inst__DOT__inc_prev;
    CData/*6:0*/ cim__DOT__word_snt_cnt_inst__DOT__inc_prev;
    CData/*0:0*/ cim__DOT__word_rec_cnt_rst_n;
    CData/*0:0*/ cim__DOT__word_snt_cnt_rst_n;
    CData/*6:0*/ cim__DOT__word_rec_cnt_inc;
    CData/*6:0*/ cim__DOT__word_snt_cnt_inc;
    VL_OUT8(is_ready,0,0);
    CData/*0:0*/ cim__DOT__rst_n;
    CData/*0:0*/ cim__DOT__is_ready;
    CData/*5:0*/ cim__DOT__current_inf_step;
    CData/*0:0*/ cim__DOT__gen_cnt_7b_inst__DOT__rst_n;
    CData/*6:0*/ cim__DOT__gen_cnt_7b_inst__DOT__inc;
    CData/*0:0*/ cim__DOT__gen_cnt_7b_2_inst__DOT__rst_n;
    CData/*6:0*/ cim__DOT__gen_cnt_7b_2_inst__DOT__inc;
    CData/*0:0*/ cim__DOT__word_rec_cnt_inst__DOT__rst_n;
    CData/*6:0*/ cim__DOT__word_rec_cnt_inst__DOT__inc;
    CData/*0:0*/ cim__DOT__word_snt_cnt_inst__DOT__rst_n;
    CData/*6:0*/ cim__DOT__word_snt_cnt_inst__DOT__inc;
    VL_IN8(rst_n,0,0);
    VL_IN8(clk,0,0);
    CData/*0:0*/ cim__DOT__clk;
    CData/*2:0*/ cim__DOT__cim_state;
    CData/*0:0*/ cim__DOT__gen_cnt_7b_inst__DOT__clk;
    CData/*0:0*/ cim__DOT__gen_cnt_7b_2_inst__DOT__clk;
    CData/*0:0*/ cim__DOT__word_rec_cnt_inst__DOT__clk;
    CData/*0:0*/ cim__DOT__word_snt_cnt_inst__DOT__clk;
    CData/*0:0*/ cim__DOT__compute_in_progress;
    CData/*2:0*/ cim__DOT__gen_cnt_3b;
    CData/*6:0*/ cim__DOT__sender_id;
    CData/*6:0*/ cim__DOT__data_len;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__rst_n__0;
    CData/*0:0*/ __VactContinue;
    SData/*9:0*/ cim__DOT__tx_addr;
    SData/*9:0*/ cim__DOT__rx_addr;
    IData/*21:0*/ cim__DOT__compute_temp;
    IData/*21:0*/ cim__DOT__compute_temp_2;
    IData/*21:0*/ cim__DOT__compute_temp_3;
    IData/*21:0*/ cim__DOT__computation_result;
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<SData/*15:0*/, 528> cim__DOT__params;
    VlUnpacked<SData/*15:0*/, 848> cim__DOT__intermediate_res;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<2> __VactTriggered;
    VlTriggerVec<2> __VnbaTriggered;
    VlMTaskVertex __Vm_mtaskstate_6;
    VlMTaskVertex __Vm_mtaskstate_final__nba;

    // INTERNAL VARIABLES
    Vtop__Syms* const vlSymsp;

    // PARAMETERS
    static constexpr IData/*31:0*/ cim__DOT__gen_cnt_7b_inst__DOT__WIDTH = 7U;
    static constexpr IData/*31:0*/ cim__DOT__gen_cnt_7b_inst__DOT__MODE = 0U;
    static constexpr IData/*31:0*/ cim__DOT__gen_cnt_7b_2_inst__DOT__WIDTH = 7U;
    static constexpr IData/*31:0*/ cim__DOT__gen_cnt_7b_2_inst__DOT__MODE = 0U;
    static constexpr IData/*31:0*/ cim__DOT__word_rec_cnt_inst__DOT__WIDTH = 7U;
    static constexpr IData/*31:0*/ cim__DOT__word_rec_cnt_inst__DOT__MODE = 0U;
    static constexpr IData/*31:0*/ cim__DOT__word_snt_cnt_inst__DOT__WIDTH = 7U;
    static constexpr IData/*31:0*/ cim__DOT__word_snt_cnt_inst__DOT__MODE = 0U;

    // CONSTRUCTORS
    Vtop___024root(Vtop__Syms* symsp, const char* v__name);
    ~Vtop___024root();
    VL_UNCOPYABLE(Vtop___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
