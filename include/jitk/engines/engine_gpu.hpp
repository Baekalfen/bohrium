/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once

#include "engine.hpp"

#include <bh_config_parser.hpp>
#include <jitk/statistics.hpp>
#include <jitk/compiler.hpp>
#include <jitk/apply_fusion.hpp>
#include <jitk/iterator.hpp>

#include <bh_view.hpp>
#include <bh_component.hpp>
#include <bh_instruction.hpp>
#include <bh_metasweep.hpp>
#include <bh_main_memory.hpp>

namespace bohrium {
namespace jitk {

class EngineGPU : public Engine {
public:
    // OpenCL compile flags
    const std::string compile_flg;
    // Default device type
    const std::string default_device_type;
    // Default device number
    const int default_device_number;
    // Default platform number
    const int platform_no;
    // Record profiling statistics
    const bool prof;
    // Maximum number of thread to use
    const uint64_t num_threads;
    // When max threads is set, use round robin?
    const bool num_threads_round_robin;

    EngineGPU(component::ComponentVE &comp, Statistics &stat) :
            Engine(comp, stat),
            compile_flg(jitk::expand_compile_cmd(comp.config.defaultGet<std::string>("compiler_flg", ""), "", "",
                                                 comp.config.file_dir.string())),
            default_device_type(comp.config.defaultGet<std::string>("device_type", "auto")),
            default_device_number(comp.config.defaultGet<int>("device_number", 0)),
            platform_no(comp.config.defaultGet<int>("platform_no", -1)),
            prof(comp.config.defaultGet<bool>("prof", false)),
            num_threads(comp.config.defaultGet<uint64_t>("num_threads", 0)),
            num_threads_round_robin(comp.config.defaultGet<bool>("num_threads_round_robin", false)) {
    }

    ~EngineGPU() override = default;

    virtual void copyToHost(const std::set<bh_base *> &bases) = 0;

    virtual void copyAllBasesToHost() = 0;

    virtual void copyToDevice(const std::set<bh_base *> &base_list) = 0;

    virtual void delBuffer(bh_base *base) = 0;

    virtual void writeKernel(const LoopB &kernel,
                             const SymbolTable &symbols,
                             const std::vector<uint64_t> &thread_stack,
                             uint64_t codegen_hash,
                             uint64_t source_hash,
                             std::stringstream &ss,
                             const std::vector<bh_metasweep> sweep_info) = 0;

    virtual void execute(const SymbolTable &symbols,
                         const std::string &source,
                         uint64_t codegen_hash,
                         const std::vector<uint64_t> &thread_stack,
                         const std::vector<const bh_instruction *> &constants,
                         const std::vector<bh_metasweep> sweep_info) = 0;

    void handleExecution(BhIR *bhir, bool opencl_scalar_reduction=false) override {
        using namespace std;

        const auto texecution = chrono::steady_clock::now();

        map<string, bool> kernel_config = {
                {"strides_as_var", comp.config.defaultGet<bool>("strides_as_var", true)},
                {"index_as_var",   comp.config.defaultGet<bool>("index_as_var", true)},
                {"const_as_var",   comp.config.defaultGet<bool>("const_as_var", true)},
                {"use_volatile",   comp.config.defaultGet<bool>("volatile", false)}
        };

        // Some statistics
        stat.record(*bhir);

        // Let's start by cleanup the instructions from the 'bhir'
        set<bh_base *> frees;
        vector<bh_instruction *> instr_list = jitk::remove_non_computed_system_instr(bhir->instr_list, frees);

        // Let's free device buffers and array memory
        for (bh_base *base: frees) {
            delBuffer(base);
            bh_data_free(base);
        }

        // Set the constructor flag
        if (comp.config.defaultGet<bool>("array_contraction", true)) {
            setConstructorFlag(instr_list);
        } else {
            for (bh_instruction *instr: instr_list) {
                instr->constructor = false;
            }
        }

        // NB: 'avoid_rank0_sweep' is set to true since GPUs cannot reduce over the outermost block
/* <<<<<<< HEAD */
        for (const jitk::LoopB &kernel: get_kernel_list(instr_list, comp.config, fcache, stat, true, false)) {
/* ======= */
/*         const bool avoid_rank0_sweep = !opencl_scalar_reduction; */

/*         // Let's get the kernel list -- this converts the instruction list into a fused kernel structure ready for codegen */
/*         for (const jitk::LoopB &kernel: get_kernel_list(instr_list, comp.config, fcache, stat, avoid_rank0_sweep, false)) { */
/* >>>>>>> 8f1107ad4... Trying to parallelize only single dimensions to optimize stride access */
            // Let's create the symbol table for the kernel
            const jitk::SymbolTable symbols(
                    kernel,
                    kernel_config["use_volatile"],
                    kernel_config["strides_as_var"],
                    kernel_config["index_as_var"],
                    kernel_config["const_as_var"]
            );
            stat.record(symbols);

            // We can skip a lot of steps if the kernel does no computation
            const bool kernel_is_computing = not kernel.isSystemOnly();

            cout << "\n\nUNIQUEID handleExecution \n" << kernel << "UNIQUEID handleExecution: Done." << endl;

            // Find the parallel blocks
            std::vector<uint64_t> thread_stack;
            /* cout << "thread_stack" << endl; */
            if (kernel._block_list.size() == 1 and kernel_is_computing) {
                int64_t opt_access_pattern = comp.config.defaultGet<int64_t>("optimize_access_pattern", 0);
                uint64_t nranks = parallel_ranks(kernel._block_list[0].getLoop(), opt_access_pattern == 0 ? 3 : -1).first;
                if (num_threads > 0 and nranks > 0) {
                    /* cout << kernel.stride << ", "; */
                    uint64_t nthds = static_cast<uint64_t>(kernel.size);
                    if (nthds > num_threads) {
                        nthds = num_threads;
                    }
                    thread_stack.push_back(nthds);
                } else {
                    auto first_block_list = get_first_loop_blocks(kernel._block_list[0].getLoop());
                    for (uint64_t i = 0; i < nranks; ++i) {
                        thread_stack.push_back(first_block_list[i]->size);
                    }
                }
            }

            // We might have to offload the execution to the CPU
            if (thread_stack.empty()) {
                cout << "CPU OffLoading" << endl;
                cpuOffload(comp, bhir, kernel, symbols);
            } else {
                cout << "GPGPU OffLoading" << endl;
                // Let's execute the kernel
                if (kernel_is_computing) {
                    executeKernel(kernel, symbols, thread_stack);
                }

                // Let's copy sync'ed arrays back to the host
                copyToHost(bhir->getSyncs());

                // Let's free device buffers
                for (bh_base *base: kernel.getAllFrees()) {
                    delBuffer(base);
                    bh_data_free(base);
                }
            }
        }
        stat.time_total_execution += chrono::steady_clock::now() - texecution;
    }

    void handleExtmethod(BhIR *bhir) override {
        std::vector<bh_instruction> instr_list;

        for (bh_instruction &instr: bhir->instr_list) {
            auto ext = comp.extmethods.find(instr.opcode);
            auto childext = comp.child_extmethods.find(instr.opcode);

            if (ext != comp.extmethods.end() or childext != comp.child_extmethods.end()) {
                // Execute the instructions up until now
                BhIR b(std::move(instr_list), bhir->getSyncs());
                comp.execute(&b);
                instr_list.clear(); // Notice, it is legal to clear a moved vector.

                if (ext != comp.extmethods.end()) {
                    const auto texecution = std::chrono::steady_clock::now();
                    ext->second.execute(&instr, &*this); // Execute the extension method
                    stat.time_ext_method += std::chrono::steady_clock::now() - texecution;
                } else if (childext != comp.child_extmethods.end()) {
                    // We let the child component execute the instruction
                    std::set<bh_base *> ext_bases = instr.get_bases();

                    copyToHost(ext_bases);

                    std::vector<bh_instruction> child_instr_list;
                    child_instr_list.push_back(instr);
                    b.instr_list = child_instr_list;
                    comp.child.execute(&b);
                }
            } else {
                instr_list.push_back(instr);
            }
        }
        bhir->instr_list = instr_list;
    }

private:
    void cpuOffload(component::ComponentImpl &comp,
                    BhIR *bhir,
                    const LoopB &kernel,
                    const SymbolTable &symbols) {
        using namespace std;

        if (&(comp.child) == nullptr) {
            throw runtime_error("handleExecution(): thread_stack cannot be empty when child == NULL!");
        }

        auto toffload = chrono::steady_clock::now();

        // Let's copy all non-temporary to the host
        const vector<bh_base *> &v = symbols.getParams();
        copyToHost(set<bh_base *>(v.begin(), v.end()));

        // Let's free device buffers
        for (bh_base *base: kernel.getAllFrees()) {
            delBuffer(base);
        }

        // Let's send the kernel instructions to our child
        vector<bh_instruction> child_instr_list;
        for (const jitk::InstrPtr &instr: bohrium::jitk::iterator::allInstr(kernel)) {
            child_instr_list.push_back(*instr);
        }
        // Notice, we have to re-create free instructions
        for (const bh_base *base: kernel.getAllFrees()) {
            vector<bh_view> operands = {bh_view(const_cast<bh_base *>(base))};
            bh_instruction instr(BH_FREE, std::move(operands));
            child_instr_list.push_back(std::move(instr));
        }
        BhIR tmp_bhir(std::move(child_instr_list), bhir->getSyncs());
        comp.child.execute(&tmp_bhir);
        stat.time_offload += chrono::steady_clock::now() - toffload;
    }

    void find_sweep_info(const LoopB &kernel, std::vector<bh_metasweep> &out){
        std::vector<bh_metasweep> sweep_info = {};
        auto subBlocks = kernel.getLocalSubBlocks();
        std::cout << "Find sweep info in this: " << std::endl;

        /* auto sweeps = rank0->getSweeps(); */
        std::cout << "#############################\n";
        for (auto block: subBlocks) {
            for (std::shared_ptr<const bh_instruction> _sweep: block->getSweeps()) {
                const bh_instruction &sweep = *_sweep.get();
                std::cout << sweep.pprint() << "\n";
                out.push_back(bh_metasweep(block->rank, sweep));
            }
            find_sweep_info(*block, out);
        }
        std::cout << "-----------------------------\n";
        // Print out instructions. We don't need this right now...
        /* for (auto instr: kernel.getLocalInstr()) { */
        /*     std::cout << instr.get()->pprint() << "\n"; */
        /* } */
    }

    void executeKernel(const LoopB &kernel,
                       const SymbolTable &symbols,
                       const std::vector<uint64_t> &thread_stack) {
        using namespace std;
        // We need a memory buffer on the device for each non-temporary array in the kernel
        const vector<bh_base *> &v = symbols.getParams();
        copyToDevice(set<bh_base *>(v.begin(), v.end()));

        // Create the constant vector
        vector<const bh_instruction *> constants;
        constants.reserve(symbols.constIDs().size());
        for (const jitk::InstrPtr &instr: symbols.constIDs()) {
            constants.push_back(&(*instr));
        }

        // Determine if there is a vector-reduction kernel. Only handle a single sweep, which is a reduction
        std::vector<bh_metasweep> sweep_info = {};
        /* auto rank0 = kernel.getLocalSubBlocks().front(); */
        /* auto sweeps = rank0->getSweeps(); */
        /* if (sweeps.size() == 1) { // TODO: Support multiple sweeps in same rank */
        /*     for(std::shared_ptr<const bh_instruction> sweep: sweeps) { */
        /*         auto views = sweep->getViews(); */

        /*         // TODO: Fix this abomination */
        /*         bh_view r; */
        /*         bh_view l; */
        /*         int i = 0; */
        /*         for (const bh_view &view: views) { */
        /*             if (i==0){ */
        /*                 r = view; */
        /*             } */
        /*             else{ */
        /*                 l = view; */
        /*             } */
        /*             i++; */
        /*         } */

        /*         if (bh_opcode_is_accumulate(sweep->opcode) || */
        /*              (bh_opcode_is_reduction(sweep->opcode) && */
        /*               r.ndim == 1 && r.shape[0] == 1 && */
        /*               l.ndim == 1 && l.shape[0] > 1)) { */
        /*             // TODO: Add rank, make vector, sort rank in descending order (opposite push_back). */
        /*             sweep_info.insert(sweep_info.begin(), bh_metasweep(0, sweep->opcode, l, r, sweep->sweep_axis())); */
        /*         } */
        /*     } */
        /* } */

        /* std::vector<bh_metasweep> sweep_info2; */
        find_sweep_info(kernel, sweep_info);
        std::cout << "sweep_info:" << endl;
        for (const bh_metasweep &metasweep: sweep_info) {
            cout << metasweep.pprint(true) << "\n";
        }

        const auto lookup = codegen_cache.lookup(kernel, symbols);
        if (not lookup.first.empty()) {
            // In debug mode, we check that the cached source code is correct
            #ifndef NDEBUG
                stringstream ss;
                writeKernel(kernel, symbols, thread_stack, lookup.second, 0, ss, sweep_info);
                if (ss.str().compare(lookup.first) != 0) {
                    cout << "\nCached source code: \n" << lookup.first;
                    cout << "\nReal source code: \n" << ss.str();
                    assert(1 == 2);
                }
            #endif
            execute(symbols, lookup.first, lookup.second, thread_stack, constants, sweep_info);
        } else {
            const auto tcodegen = chrono::steady_clock::now();
            stringstream ss;
            writeKernel(kernel, symbols, thread_stack, lookup.second, 0, ss, sweep_info);
            string source = ss.str();
            stat.time_codegen += chrono::steady_clock::now() - tcodegen;
            execute(symbols, source, lookup.second, thread_stack, constants, sweep_info);
            codegen_cache.insert(std::move(source), kernel, symbols);
        }
    }
};

}
} // namespace
