// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "verilated.h"
#include "verilated_dpi.h"

#include "Vtop__Syms.h"
#include "Vtop__Syms.h"
#include "Vtop___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__ico(Vtop___024root* vlSelf);
#endif  // VL_DEBUG

void Vtop___024root___eval_triggers__ico(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_triggers__ico\n"); );
    // Body
    vlSelf->__VicoTriggered.set(0U, (0U == vlSelf->__VicoIterCount));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtop___024root___dump_triggers__ico(vlSelf);
    }
#endif
}

VL_INLINE_OPT void Vtop___024root___ico_sequent__TOP__0(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___ico_sequent__TOP__0\n"); );
    // Body
    vlSelf->mem_tb__DOT__param_read_en = vlSelf->param_read_en;
    vlSelf->mem_tb__DOT__param_write_en = vlSelf->param_write_en;
    vlSelf->mem_tb__DOT__param_chip_en = vlSelf->param_chip_en;
    vlSelf->mem_tb__DOT__param_read_data_width = vlSelf->param_read_data_width;
    vlSelf->mem_tb__DOT__param_write_data_width = vlSelf->param_write_data_width;
    vlSelf->mem_tb__DOT__param_read_addr = vlSelf->param_read_addr;
    vlSelf->mem_tb__DOT__param_write_addr = vlSelf->param_write_addr;
    vlSelf->mem_tb__DOT__param_write_data = vlSelf->param_write_data;
    vlSelf->mem_tb__DOT__int_res_read_en = vlSelf->int_res_read_en;
    vlSelf->mem_tb__DOT__int_res_write_en = vlSelf->int_res_write_en;
    vlSelf->mem_tb__DOT__int_res_chip_en = vlSelf->int_res_chip_en;
    vlSelf->mem_tb__DOT__int_res_read_data_width = vlSelf->int_res_read_data_width;
    vlSelf->mem_tb__DOT__int_res_write_data_width = vlSelf->int_res_write_data_width;
    vlSelf->mem_tb__DOT__int_res_read_addr = vlSelf->int_res_read_addr;
    vlSelf->mem_tb__DOT__int_res_write_addr = vlSelf->int_res_write_addr;
    vlSelf->mem_tb__DOT__int_res_write_data = vlSelf->int_res_write_data;
    vlSymsp->TOP__mem_tb__DOT__param_read_sig.data_width 
        = vlSelf->param_read_data_width;
    vlSymsp->TOP__mem_tb__DOT__param_write_sig.data_width 
        = vlSelf->param_write_data_width;
    vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.chip_en 
        = vlSelf->int_res_chip_en;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__CEN 
        = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.chip_en;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__CEN 
        = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.chip_en;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__CEN 
        = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.chip_en;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__CEN 
        = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.chip_en;
    vlSymsp->TOP__mem_tb__DOT__param_read_sig.data 
        = ((IData)(vlSelf->mem_tb__DOT__params__DOT__params_0_read_en_prev)
            ? (IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.data)
            : ((IData)(vlSelf->mem_tb__DOT__params__DOT__params_1_read_en_prev)
                ? (IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.data)
                : 0U));
    vlSymsp->TOP__mem_tb__DOT__param_write_sig.data 
        = vlSelf->param_write_data;
    vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.data 
        = ((IData)(vlSelf->mem_tb__DOT__int_res__DOT__read_data_width_prev)
            ? (((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev)) 
                | (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev)))
                ? (((IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data) 
                    << 9U) | (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data))
                : (((IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data) 
                    << 9U) | (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data)))
            : ((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev))
                ? (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data)
                : ((1U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev))
                    ? (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data)
                    : ((2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev))
                        ? (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data)
                        : (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data)))));
    vlSymsp->TOP__mem_tb__DOT__param_write_sig.chip_en 
        = vlSelf->param_chip_en;
    vlSelf->mem_tb__DOT__clk = vlSelf->clk;
    vlSelf->mem_tb__DOT__rst_n = vlSelf->rst_n;
    vlSymsp->TOP__mem_tb__DOT__param_read_sig.en = vlSelf->param_read_en;
    vlSymsp->TOP__mem_tb__DOT__param_read_sig.addr 
        = vlSelf->param_read_addr;
    vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data 
        = vlSelf->int_res_write_data;
    vlSymsp->TOP__mem_tb__DOT__param_write_sig.en = vlSelf->param_write_en;
    vlSymsp->TOP__mem_tb__DOT__param_write_sig.addr 
        = vlSelf->param_write_addr;
    vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.en 
        = vlSelf->int_res_read_en;
    vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.data_width 
        = vlSelf->int_res_read_data_width;
    vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.en 
        = vlSelf->int_res_write_en;
    vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data_width 
        = vlSelf->int_res_write_data_width;
    vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.addr 
        = vlSelf->int_res_read_addr;
    vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr 
        = vlSelf->int_res_write_addr;
    vlSelf->mem_tb__DOT__param_read_data = vlSymsp->TOP__mem_tb__DOT__param_read_sig.data;
    vlSelf->mem_tb__DOT__int_res_read_data = vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.data;
    vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.chip_en 
        = vlSymsp->TOP__mem_tb__DOT__param_write_sig.chip_en;
    vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.chip_en 
        = vlSymsp->TOP__mem_tb__DOT__param_write_sig.chip_en;
    vlSelf->mem_tb__DOT__params__DOT__clk = vlSelf->mem_tb__DOT__clk;
    vlSelf->mem_tb__DOT__int_res__DOT__clk = vlSelf->mem_tb__DOT__clk;
    vlSelf->mem_tb__DOT__params__DOT__rst_n = vlSelf->mem_tb__DOT__rst_n;
    vlSelf->mem_tb__DOT__int_res__DOT__rst_n = vlSelf->mem_tb__DOT__rst_n;
    if (vlSymsp->TOP__mem_tb__DOT__param_read_sig.en) {
        if ((0x3e00U > (IData)(vlSymsp->TOP__mem_tb__DOT__param_read_sig.addr))) {
            vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.addr 
                = (0x3fffU & (IData)(vlSymsp->TOP__mem_tb__DOT__param_read_sig.addr));
        }
        if ((0x3e00U <= (IData)(vlSymsp->TOP__mem_tb__DOT__param_read_sig.addr))) {
            vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.addr 
                = (0x3fffU & ((IData)(vlSymsp->TOP__mem_tb__DOT__param_read_sig.addr) 
                              - (IData)(0x3e00U)));
        }
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.en 
            = (0x3e00U > (IData)(vlSymsp->TOP__mem_tb__DOT__param_read_sig.addr));
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.en 
            = (0x3e00U <= (IData)(vlSymsp->TOP__mem_tb__DOT__param_read_sig.addr));
    } else {
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.en = 0U;
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.en = 0U;
    }
    if (vlSymsp->TOP__mem_tb__DOT__param_write_sig.en) {
        if ((0x3e00U <= (IData)(vlSymsp->TOP__mem_tb__DOT__param_write_sig.addr))) {
            vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.addr 
                = (0x3fffU & ((IData)(vlSymsp->TOP__mem_tb__DOT__param_write_sig.addr) 
                              - (IData)(0x3e00U)));
            vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.data 
                = vlSymsp->TOP__mem_tb__DOT__param_write_sig.data;
        }
        if ((0x3e00U > (IData)(vlSymsp->TOP__mem_tb__DOT__param_write_sig.addr))) {
            vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.addr 
                = (0x3fffU & (IData)(vlSymsp->TOP__mem_tb__DOT__param_write_sig.addr));
            vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.data 
                = vlSymsp->TOP__mem_tb__DOT__param_write_sig.data;
            vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.en = 1U;
        } else {
            vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.en = 0U;
        }
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.en 
            = (0x3e00U <= (IData)(vlSymsp->TOP__mem_tb__DOT__param_write_sig.addr));
    } else {
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.data = 0U;
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.data = 0U;
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.en = 0U;
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.en = 0U;
    }
    if ((0x3800U > (IData)(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.addr))) {
        vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr 
            = (0x3fffU & (IData)(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.addr));
        vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current = 0U;
    } else if ((0x7000U > (IData)(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.addr))) {
        vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr 
            = (0x3fffU & ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.addr) 
                          - (IData)(0x3800U)));
        vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current = 1U;
    } else if ((0xa800U > (IData)(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.addr))) {
        vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr 
            = (0x3fffU & ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.addr) 
                          - (IData)(0x3000U)));
        vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current = 2U;
    } else if ((0xe000U > (IData)(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.addr))) {
        vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr 
            = (0x3fffU & ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.addr) 
                          - (IData)(0x2800U)));
        vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current = 3U;
    } else {
        vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr 
            = (0x3fffU & 0U);
        vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current = 0U;
    }
    if ((0x3800U > (IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr))) {
        vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr 
            = (0x3fffU & (IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr));
        vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current = 0U;
    } else if ((0x7000U > (IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr))) {
        vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr 
            = (0x3fffU & ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr) 
                          - (IData)(0x3800U)));
        vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current = 1U;
    } else if ((0xa800U > (IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr))) {
        vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr 
            = (0x3fffU & ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr) 
                          - (IData)(0x3000U)));
        vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current = 2U;
    } else if ((0xe000U > (IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr))) {
        vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr 
            = (0x3fffU & ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr) 
                          - (IData)(0x2800U)));
        vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current = 3U;
    } else {
        vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr 
            = (0x3fffU & 0U);
        vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current = 0U;
    }
    vlSelf->param_read_data = vlSelf->mem_tb__DOT__param_read_data;
    vlSelf->int_res_read_data = vlSelf->mem_tb__DOT__int_res_read_data;
    vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__CEN 
        = vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.chip_en;
    vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__CEN 
        = vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.chip_en;
    vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__clk 
        = vlSelf->mem_tb__DOT__params__DOT__clk;
    vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__clk 
        = vlSelf->mem_tb__DOT__params__DOT__clk;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__clk 
        = vlSelf->mem_tb__DOT__int_res__DOT__clk;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__clk 
        = vlSelf->mem_tb__DOT__int_res__DOT__clk;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__clk 
        = vlSelf->mem_tb__DOT__int_res__DOT__clk;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__clk 
        = vlSelf->mem_tb__DOT__int_res__DOT__clk;
    vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__rst_n 
        = vlSelf->mem_tb__DOT__params__DOT__rst_n;
    vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__rst_n 
        = vlSelf->mem_tb__DOT__params__DOT__rst_n;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__rst_n 
        = vlSelf->mem_tb__DOT__int_res__DOT__rst_n;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__rst_n 
        = vlSelf->mem_tb__DOT__int_res__DOT__rst_n;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__rst_n 
        = vlSelf->mem_tb__DOT__int_res__DOT__rst_n;
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__rst_n 
        = vlSelf->mem_tb__DOT__int_res__DOT__rst_n;
    vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__WEN 
        = ((IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.en) 
           & (~ (IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.en)));
    vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__WEN 
        = ((IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.en) 
           & (~ (IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.en)));
    if (vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.en) {
        if (vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.data_width) {
            if (((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current)) 
                 | (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current)))) {
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.addr = 0U;
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.addr 
                    = vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr;
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.addr 
                    = vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr;
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.addr = 0U;
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.en = 1U;
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.en = 1U;
            } else {
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.addr 
                    = vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr;
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.addr = 0U;
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.addr = 0U;
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.addr 
                    = vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr;
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.en = 0U;
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.en = 0U;
            }
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.en 
                = (1U & (~ ((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current)) 
                            | (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current)))));
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.en 
                = (1U & (~ ((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current)) 
                            | (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current)))));
        } else {
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.addr 
                = ((3U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current))
                    ? (IData)(vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr)
                    : 0U);
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.addr 
                = ((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current))
                    ? (IData)(vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr)
                    : 0U);
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.addr 
                = ((2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current))
                    ? (IData)(vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr)
                    : 0U);
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.addr 
                = ((1U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current))
                    ? (IData)(vlSelf->mem_tb__DOT__int_res__DOT__read_base_addr)
                    : 0U);
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.en 
                = (0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current));
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.en 
                = (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current));
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.en 
                = (1U & (1U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current)));
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.en 
                = (1U & (3U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current)));
        }
    } else {
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.en = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.en = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.en = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.en = 0U;
    }
    if (vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.en) {
        if (vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data_width) {
            if (((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current)) 
                 | (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current)))) {
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.data 
                    = (0x1ffU & (vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data 
                                 >> 9U));
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.data 
                    = (0x1ffU & vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data 
                    = (0x1ffU & 0U);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data 
                    = (0x1ffU & 0U);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.addr 
                    = (0x3fffU & 0U);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.addr 
                    = (0x3fffU & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr));
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.addr 
                    = (0x3fffU & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr));
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.addr 
                    = (0x3fffU & 0U);
            } else {
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.data 
                    = (0x1ffU & 0U);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.data 
                    = (0x1ffU & 0U);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data 
                    = (0x1ffU & (vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data 
                                 >> 9U));
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data 
                    = (0x1ffU & vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.addr 
                    = (0x3fffU & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr));
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.addr 
                    = (0x3fffU & 0U);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.addr 
                    = (0x3fffU & 0U);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.addr 
                    = (0x3fffU & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__write_base_addr));
            }
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.en 
                = (1U & (~ ((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current)) 
                            | (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current)))));
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.en 
                = (1U & (~ ((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current)) 
                            | (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current)))));
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.en 
                = ((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current)) 
                   | (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current)));
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.en 
                = ((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current)) 
                   | (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current)));
        } else {
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.data 
                = (0x1ffU & ((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current))
                              ? vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data
                              : 0U));
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.data 
                = (0x1ffU & ((2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current))
                              ? vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data
                              : 0U));
            if ((1U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current))) {
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data 
                    = (0x1ffU & vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.addr 
                    = (0x3fffU & ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr) 
                                  - (IData)(0x3800U)));
            } else {
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data 
                    = (0x1ffU & 0U);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.addr 
                    = (0x3fffU & 0U);
            }
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.addr 
                = (0x3fffU & ((2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current))
                               ? ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr) 
                                  - (IData)(0x3000U))
                               : 0U));
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.addr 
                = (0x3fffU & ((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current))
                               ? (IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr)
                               : 0U));
            if ((3U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current))) {
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data 
                    = (0x1ffU & vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.data);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.addr 
                    = (0x3fffU & ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res_write_sig.addr) 
                                  - (IData)(0x2800U)));
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.en = 1U;
            } else {
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data 
                    = (0x1ffU & 0U);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.addr 
                    = (0x3fffU & 0U);
                vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.en = 0U;
            }
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.en 
                = (1U & (1U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current)));
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.en 
                = (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current));
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.en 
                = (0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_write_current));
        }
    } else {
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.data = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.data = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.addr = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.en = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.en = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.en = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.en = 0U;
    }
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__WEN 
        = ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.en) 
           & (~ (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.en)));
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__WEN 
        = ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.en) 
           & (~ (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.en)));
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__WEN 
        = ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.en) 
           & (~ (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.en)));
    vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__WEN 
        = ((IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.en) 
           & (~ (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.en)));
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__act(Vtop___024root* vlSelf);
#endif  // VL_DEBUG

void Vtop___024root___eval_triggers__act(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_triggers__act\n"); );
    // Body
    vlSelf->__VactTriggered.set(0U, ((IData)(vlSelf->clk) 
                                     & (~ (IData)(vlSelf->__Vtrigprevexpr___TOP__clk__0))));
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtop___024root___dump_triggers__act(vlSelf);
    }
#endif
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__0(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__0\n"); );
    // Init
    SData/*13:0*/ __Vdlyvdim0__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0;
    __Vdlyvdim0__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0 = 0;
    SData/*8:0*/ __Vdlyvval__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0;
    __Vdlyvval__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0;
    __Vdlyvset__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0 = 0;
    SData/*13:0*/ __Vdlyvdim0__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0;
    __Vdlyvdim0__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0 = 0;
    SData/*8:0*/ __Vdlyvval__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0;
    __Vdlyvval__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0;
    __Vdlyvset__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0 = 0;
    SData/*13:0*/ __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0;
    __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0 = 0;
    SData/*8:0*/ __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0;
    __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0;
    __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0 = 0;
    SData/*13:0*/ __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0;
    __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0 = 0;
    SData/*8:0*/ __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0;
    __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0;
    __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0 = 0;
    SData/*13:0*/ __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0;
    __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0 = 0;
    SData/*8:0*/ __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0;
    __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0;
    __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0 = 0;
    SData/*13:0*/ __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0;
    __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0 = 0;
    SData/*8:0*/ __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0;
    __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0;
    __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0 = 0;
    // Body
    __Vdlyvset__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0 = 0U;
    __Vdlyvset__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0 = 0U;
    __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0 = 0U;
    __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0 = 0U;
    __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0 = 0U;
    __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0 = 0U;
    if ((((IData)(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__WEN) 
          & (IData)(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__CEN)) 
         & (IData)(vlSelf->rst_n))) {
        vlSelf->mem_tb__DOT__params__DOT__params_1__DOT____Vlvbound_h4cea1195__0 
            = vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.data;
        if ((0x3dffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.addr))) {
            __Vdlyvval__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0 
                = vlSelf->mem_tb__DOT__params__DOT__params_1__DOT____Vlvbound_h4cea1195__0;
            __Vdlyvset__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0 = 1U;
            __Vdlyvdim0__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0 
                = vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_write.addr;
        }
    }
    if ((((IData)(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__WEN) 
          & (IData)(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__CEN)) 
         & (IData)(vlSelf->rst_n))) {
        vlSelf->mem_tb__DOT__params__DOT__params_0__DOT____Vlvbound_h4cea1195__0 
            = vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.data;
        if ((0x3dffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.addr))) {
            __Vdlyvval__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0 
                = vlSelf->mem_tb__DOT__params__DOT__params_0__DOT____Vlvbound_h4cea1195__0;
            __Vdlyvset__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0 = 1U;
            __Vdlyvdim0__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0 
                = vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_write.addr;
        }
    }
    if ((((IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__WEN) 
          & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__CEN)) 
         & (IData)(vlSelf->rst_n))) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT____Vlvbound_h203f7518__0 
            = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data;
        if ((0x37ffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.addr))) {
            __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0 
                = vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT____Vlvbound_h203f7518__0;
            __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0 = 1U;
            __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0 
                = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.addr;
        }
    }
    if ((((IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__WEN) 
          & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__CEN)) 
         & (IData)(vlSelf->rst_n))) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT____Vlvbound_h203f7518__0 
            = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.data;
        if ((0x37ffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.addr))) {
            __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0 
                = vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT____Vlvbound_h203f7518__0;
            __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0 = 1U;
            __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0 
                = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.addr;
        }
    }
    if ((((IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__WEN) 
          & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__CEN)) 
         & (IData)(vlSelf->rst_n))) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT____Vlvbound_h203f7518__0 
            = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data;
        if ((0x37ffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.addr))) {
            __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0 
                = vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT____Vlvbound_h203f7518__0;
            __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0 = 1U;
            __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0 
                = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.addr;
        }
    }
    if ((((IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__WEN) 
          & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__CEN)) 
         & (IData)(vlSelf->rst_n))) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT____Vlvbound_h203f7518__0 
            = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.data;
        if ((0x37ffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.addr))) {
            __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0 
                = vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT____Vlvbound_h203f7518__0;
            __Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0 = 1U;
            __Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0 
                = vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.addr;
        }
    }
    vlSelf->mem_tb__DOT__params__DOT__params_1_read_en_prev 
        = vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.en;
    vlSelf->mem_tb__DOT__params__DOT__params_0_read_en_prev 
        = vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.en;
    if (vlSelf->rst_n) {
        if ((((~ (IData)(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__WEN)) 
              & (IData)(vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__CEN)) 
             & (IData)(vlSelf->rst_n))) {
            vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.data 
                = ((0x3dffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.addr))
                    ? vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__memory
                   [vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.addr]
                    : 0U);
        }
        if ((((~ (IData)(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__WEN)) 
              & (IData)(vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__CEN)) 
             & (IData)(vlSelf->rst_n))) {
            vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.data 
                = ((0x3dffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.addr))
                    ? vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__memory
                   [vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.addr]
                    : 0U);
        }
        if ((((~ (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__WEN)) 
              & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__CEN)) 
             & (IData)(vlSelf->rst_n))) {
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data 
                = ((0x37ffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.addr))
                    ? vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory
                   [vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.addr]
                    : 0U);
        }
        if ((((~ (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__WEN)) 
              & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__CEN)) 
             & (IData)(vlSelf->rst_n))) {
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data 
                = ((0x37ffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.addr))
                    ? vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory
                   [vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.addr]
                    : 0U);
        }
        if ((((~ (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__WEN)) 
              & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__CEN)) 
             & (IData)(vlSelf->rst_n))) {
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data 
                = ((0x37ffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.addr))
                    ? vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory
                   [vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.addr]
                    : 0U);
        }
        if ((((~ (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__WEN)) 
              & (IData)(vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__CEN)) 
             & (IData)(vlSelf->rst_n))) {
            vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data 
                = ((0x37ffU >= (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.addr))
                    ? vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory
                   [vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.addr]
                    : 0U);
        }
    } else {
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.data = 0U;
        vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.data = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data = 0U;
        vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data = 0U;
    }
    vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev 
        = vlSelf->mem_tb__DOT__int_res__DOT__bank_read_current;
    vlSelf->mem_tb__DOT__int_res__DOT__read_data_width_prev 
        = vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.data_width;
    if (__Vdlyvset__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0) {
        vlSelf->mem_tb__DOT__params__DOT__params_1__DOT__memory[__Vdlyvdim0__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0] 
            = __Vdlyvval__mem_tb__DOT__params__DOT__params_1__DOT__memory__v0;
    }
    if (__Vdlyvset__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0) {
        vlSelf->mem_tb__DOT__params__DOT__params_0__DOT__memory[__Vdlyvdim0__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0] 
            = __Vdlyvval__mem_tb__DOT__params__DOT__params_0__DOT__memory__v0;
    }
    if (__Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory[__Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0] 
            = __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory__v0;
    }
    if (__Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory[__Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0] 
            = __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory__v0;
    }
    if (__Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory[__Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0] 
            = __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory__v0;
    }
    if (__Vdlyvset__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0) {
        vlSelf->mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory[__Vdlyvdim0__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0] 
            = __Vdlyvval__mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory__v0;
    }
    vlSymsp->TOP__mem_tb__DOT__param_read_sig.data 
        = ((IData)(vlSelf->mem_tb__DOT__params__DOT__params_0_read_en_prev)
            ? (IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_0_read.data)
            : ((IData)(vlSelf->mem_tb__DOT__params__DOT__params_1_read_en_prev)
                ? (IData)(vlSymsp->TOP__mem_tb__DOT__params__DOT__params_1_read.data)
                : 0U));
    vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.data 
        = ((IData)(vlSelf->mem_tb__DOT__int_res__DOT__read_data_width_prev)
            ? (((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev)) 
                | (2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev)))
                ? (((IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data) 
                    << 9U) | (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data))
                : (((IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data) 
                    << 9U) | (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data)))
            : ((0U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev))
                ? (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data)
                : ((1U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev))
                    ? (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data)
                    : ((2U == (IData)(vlSelf->mem_tb__DOT__int_res__DOT__bank_read_prev))
                        ? (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data)
                        : (IData)(vlSymsp->TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data)))));
    vlSelf->mem_tb__DOT__param_read_data = vlSymsp->TOP__mem_tb__DOT__param_read_sig.data;
    vlSelf->mem_tb__DOT__int_res_read_data = vlSymsp->TOP__mem_tb__DOT__int_res_read_sig.data;
    vlSelf->param_read_data = vlSelf->mem_tb__DOT__param_read_data;
    vlSelf->int_res_read_data = vlSelf->mem_tb__DOT__int_res_read_data;
}
