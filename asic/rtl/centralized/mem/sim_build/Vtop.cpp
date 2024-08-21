// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vtop.h"
#include "Vtop__Syms.h"
#include "verilated_fst_c.h"
#include "verilated_dpi.h"

//============================================================
// Constructors

Vtop::Vtop(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vtop__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , rst_n{vlSymsp->TOP.rst_n}
    , param_read_en{vlSymsp->TOP.param_read_en}
    , param_write_en{vlSymsp->TOP.param_write_en}
    , param_chip_en{vlSymsp->TOP.param_chip_en}
    , param_read_data_width{vlSymsp->TOP.param_read_data_width}
    , param_write_data_width{vlSymsp->TOP.param_write_data_width}
    , int_res_read_en{vlSymsp->TOP.int_res_read_en}
    , int_res_write_en{vlSymsp->TOP.int_res_write_en}
    , int_res_chip_en{vlSymsp->TOP.int_res_chip_en}
    , int_res_read_data_width{vlSymsp->TOP.int_res_read_data_width}
    , int_res_write_data_width{vlSymsp->TOP.int_res_write_data_width}
    , param_read_addr{vlSymsp->TOP.param_read_addr}
    , param_write_addr{vlSymsp->TOP.param_write_addr}
    , param_write_data{vlSymsp->TOP.param_write_data}
    , param_read_data{vlSymsp->TOP.param_read_data}
    , int_res_read_addr{vlSymsp->TOP.int_res_read_addr}
    , int_res_write_addr{vlSymsp->TOP.int_res_write_addr}
    , int_res_write_data{vlSymsp->TOP.int_res_write_data}
    , int_res_read_data{vlSymsp->TOP.int_res_read_data}
    , __PVT__Defines{vlSymsp->TOP.__PVT__Defines}
    , __PVT__mem_tb__DOT__param_read_sig{vlSymsp->TOP.__PVT__mem_tb__DOT__param_read_sig}
    , __PVT__mem_tb__DOT__param_write_sig{vlSymsp->TOP.__PVT__mem_tb__DOT__param_write_sig}
    , __PVT__mem_tb__DOT__int_res_read_sig{vlSymsp->TOP.__PVT__mem_tb__DOT__int_res_read_sig}
    , __PVT__mem_tb__DOT__int_res_write_sig{vlSymsp->TOP.__PVT__mem_tb__DOT__int_res_write_sig}
    , __PVT__mem_tb__DOT__params__DOT__params_0_read{vlSymsp->TOP.__PVT__mem_tb__DOT__params__DOT__params_0_read}
    , __PVT__mem_tb__DOT__params__DOT__params_0_write{vlSymsp->TOP.__PVT__mem_tb__DOT__params__DOT__params_0_write}
    , __PVT__mem_tb__DOT__params__DOT__params_1_read{vlSymsp->TOP.__PVT__mem_tb__DOT__params__DOT__params_1_read}
    , __PVT__mem_tb__DOT__params__DOT__params_1_write{vlSymsp->TOP.__PVT__mem_tb__DOT__params__DOT__params_1_write}
    , __PVT__mem_tb__DOT__int_res__DOT__int_res_0_read{vlSymsp->TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_0_read}
    , __PVT__mem_tb__DOT__int_res__DOT__int_res_0_write{vlSymsp->TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_0_write}
    , __PVT__mem_tb__DOT__int_res__DOT__int_res_1_read{vlSymsp->TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_1_read}
    , __PVT__mem_tb__DOT__int_res__DOT__int_res_1_write{vlSymsp->TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_1_write}
    , __PVT__mem_tb__DOT__int_res__DOT__int_res_2_read{vlSymsp->TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_2_read}
    , __PVT__mem_tb__DOT__int_res__DOT__int_res_2_write{vlSymsp->TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_2_write}
    , __PVT__mem_tb__DOT__int_res__DOT__int_res_3_read{vlSymsp->TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_3_read}
    , __PVT__mem_tb__DOT__int_res__DOT__int_res_3_write{vlSymsp->TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_3_write}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vtop::Vtop(const char* _vcname__)
    : Vtop(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vtop::~Vtop() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vtop___024root___eval_debug_assertions(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
void Vtop___024root___eval_static(Vtop___024root* vlSelf);
void Vtop___024root___eval_initial(Vtop___024root* vlSelf);
void Vtop___024root___eval_settle(Vtop___024root* vlSelf);
void Vtop___024root___eval(Vtop___024root* vlSelf);

void Vtop::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vtop::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vtop___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_activity = true;
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vtop___024root___eval_static(&(vlSymsp->TOP));
        Vtop___024root___eval_initial(&(vlSymsp->TOP));
        Vtop___024root___eval_settle(&(vlSymsp->TOP));
    }
    // MTask 0 start
    VL_DEBUG_IF(VL_DBG_MSGF("MTask0 starting\n"););
    Verilated::mtaskId(0);
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vtop___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfThreadMTask(vlSymsp->__Vm_evalMsgQp);
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vtop::eventsPending() { return false; }

uint64_t Vtop::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "%Error: No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vtop::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vtop___024root___eval_final(Vtop___024root* vlSelf);

VL_ATTR_COLD void Vtop::final() {
    Vtop___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vtop::hierName() const { return vlSymsp->name(); }
const char* Vtop::modelName() const { return "Vtop"; }
unsigned Vtop::threads() const { return 1; }
std::unique_ptr<VerilatedTraceConfig> Vtop::traceConfig() const {
    return std::unique_ptr<VerilatedTraceConfig>{new VerilatedTraceConfig{false, false, false}};
};

//============================================================
// Trace configuration

void Vtop___024root__trace_init_top(Vtop___024root* vlSelf, VerilatedFst* tracep);

VL_ATTR_COLD static void trace_init(void* voidSelf, VerilatedFst* tracep, uint32_t code) {
    // Callback from tracep->open()
    Vtop___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop___024root*>(voidSelf);
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (!vlSymsp->_vm_contextp__->calcUnusedSigs()) {
        VL_FATAL_MT(__FILE__, __LINE__, __FILE__,
            "Turning on wave traces requires Verilated::traceEverOn(true) call before time 0.");
    }
    vlSymsp->__Vm_baseCode = code;
    tracep->scopeEscape(' ');
    tracep->pushNamePrefix(std::string{vlSymsp->name()} + ' ');
    Vtop___024root__trace_init_top(vlSelf, tracep);
    tracep->popNamePrefix();
    tracep->scopeEscape('.');
}

VL_ATTR_COLD void Vtop___024root__trace_register(Vtop___024root* vlSelf, VerilatedFst* tracep);

VL_ATTR_COLD void Vtop::trace(VerilatedFstC* tfp, int levels, int options) {
    if (tfp->isOpen()) {
        vl_fatal(__FILE__, __LINE__, __FILE__,"'Vtop::trace()' shall not be called after 'VerilatedFstC::open()'.");
    }
    if (false && levels && options) {}  // Prevent unused
    tfp->spTrace()->addModel(this);
    tfp->spTrace()->addInitCb(&trace_init, &(vlSymsp->TOP));
    Vtop___024root__trace_register(&(vlSymsp->TOP), tfp->spTrace());
}
