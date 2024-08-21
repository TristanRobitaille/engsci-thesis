// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary model header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef VERILATED_VTOP_H_
#define VERILATED_VTOP_H_  // guard

#include "verilated.h"
#include "svdpi.h"

class Vtop__Syms;
class Vtop___024root;
class VerilatedFstC;
class Vtop_Defines;
class Vtop_MemoryAccessSignals__Tz1_TBz2;
class Vtop_MemoryAccessSignals__Tz3_TBz4;
class Vtop_MemoryAccessSignals__Tz1_TBz9;


// This class is the main interface to the Verilated model
class alignas(VL_CACHE_LINE_BYTES) Vtop VL_NOT_FINAL : public VerilatedModel {
  private:
    // Symbol table holding complete model state (owned by this class)
    Vtop__Syms* const vlSymsp;

  public:

    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    VL_IN8(&clk,0,0);
    VL_IN8(&rst_n,0,0);
    VL_IN8(&param_read_en,0,0);
    VL_IN8(&param_write_en,0,0);
    VL_IN8(&param_chip_en,0,0);
    VL_IN8(&param_read_data_width,0,0);
    VL_IN8(&param_write_data_width,0,0);
    VL_IN8(&int_res_read_en,0,0);
    VL_IN8(&int_res_write_en,0,0);
    VL_IN8(&int_res_chip_en,0,0);
    VL_IN8(&int_res_read_data_width,0,0);
    VL_IN8(&int_res_write_data_width,0,0);
    VL_IN16(&param_read_addr,14,0);
    VL_IN16(&param_write_addr,14,0);
    VL_IN16(&param_write_data,8,0);
    VL_OUT16(&param_read_data,8,0);
    VL_IN16(&int_res_read_addr,15,0);
    VL_IN16(&int_res_write_addr,15,0);
    VL_IN(&int_res_write_data,17,0);
    VL_OUT(&int_res_read_data,17,0);

    // CELLS
    // Public to allow access to /* verilator public */ items.
    // Otherwise the application code can consider these internals.
    Vtop_Defines* const __PVT__Defines;
    Vtop_MemoryAccessSignals__Tz1_TBz2* const __PVT__mem_tb__DOT__param_read_sig;
    Vtop_MemoryAccessSignals__Tz1_TBz2* const __PVT__mem_tb__DOT__param_write_sig;
    Vtop_MemoryAccessSignals__Tz3_TBz4* const __PVT__mem_tb__DOT__int_res_read_sig;
    Vtop_MemoryAccessSignals__Tz3_TBz4* const __PVT__mem_tb__DOT__int_res_write_sig;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__params__DOT__params_0_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__params__DOT__params_0_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__params__DOT__params_1_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__params__DOT__params_1_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__int_res__DOT__int_res_0_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__int_res__DOT__int_res_0_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__int_res__DOT__int_res_1_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__int_res__DOT__int_res_1_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__int_res__DOT__int_res_2_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__int_res__DOT__int_res_2_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__int_res__DOT__int_res_3_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* const __PVT__mem_tb__DOT__int_res__DOT__int_res_3_write;

    // Root instance pointer to allow access to model internals,
    // including inlined /* verilator public_flat_* */ items.
    Vtop___024root* const rootp;

    // CONSTRUCTORS
    /// Construct the model; called by application code
    /// If contextp is null, then the model will use the default global context
    /// If name is "", then makes a wrapper with a
    /// single model invisible with respect to DPI scope names.
    explicit Vtop(VerilatedContext* contextp, const char* name = "TOP");
    explicit Vtop(const char* name = "TOP");
    /// Destroy the model; called (often implicitly) by application code
    virtual ~Vtop();
  private:
    VL_UNCOPYABLE(Vtop);  ///< Copying not allowed

  public:
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval() { eval_step(); }
    /// Evaluate when calling multiple units/models per time step.
    void eval_step();
    /// Evaluate at end of a timestep for tracing, when using eval_step().
    /// Application must call after all eval() and before time changes.
    void eval_end_step() {}
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    /// Are there scheduled events to handle?
    bool eventsPending();
    /// Returns time at next time slot. Aborts if !eventsPending()
    uint64_t nextTimeSlot();
    /// Trace signals in the model; called by application code
    void trace(VerilatedFstC* tfp, int levels, int options = 0);
    /// Retrieve name of this model instance (as passed to constructor).
    const char* name() const;

    // Abstract methods from VerilatedModel
    const char* hierName() const override final;
    const char* modelName() const override final;
    unsigned threads() const override final;
    std::unique_ptr<VerilatedTraceConfig> traceConfig() const override final;
};

#endif  // guard
