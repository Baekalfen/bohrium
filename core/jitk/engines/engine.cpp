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

#include <jitk/engines/engine.hpp>
#include <bh_instruction.hpp>
#include <bh_metasweep.hpp>

#define INDENT util::spaces(out, 4 + b.rank() * 4 + indent_level*4)

using namespace std;

namespace bohrium {
namespace jitk {

void Engine::writeKernelFunctionArguments(const jitk::SymbolTable &symbols,
                                          std::stringstream &ss,
                                          const char *array_type_prefix,
                                          const std::vector<bh_metasweep> sweep_info) {
    // We create the comma separated list of args and saves it in `stmp`
    std::stringstream stmp;
    for (size_t i = 0; i < symbols.getParams().size(); ++i) {
        bh_base *b = symbols.getParams()[i];
        if (array_type_prefix != nullptr) {
            stmp << array_type_prefix << " ";
        }
        stmp << writeType(b->type) << "* __restrict__ a" << symbols.baseID(b) << ", ";
    }

    for (const bh_view *view: symbols.offsetStrideViews()) {
        stmp << writeType(bh_type::UINT64);
        stmp << " vo" << symbols.offsetStridesID(*view) << ", ";
        for (int i = 0; i < view->ndim; ++i) {
            stmp << writeType(bh_type::UINT64) << " vs" << symbols.offsetStridesID(*view) << "_" << i << ", ";
        }
    }

    if (not symbols.constIDs().empty()) {
        for (auto it = symbols.constIDs().begin(); it != symbols.constIDs().end(); ++it) {
            const InstrPtr &instr = *it;
            stmp << "const " << writeType(instr->constant.type) << " c" << symbols.constID(*instr) << ", ";
        }
    }

    if ((sweep_info.size() == 1) && sweep_info.back().is_scalar()){
        const string dtype = writeType(sweep_info.back().type());
        stmp << "__global volatile " << dtype << " *res, __local volatile " << dtype << " *a, ";
    }

    // And then we write `stmp` into `ss` excluding the last comma
    const std::string strtmp = stmp.str();
    if (strtmp.empty()) {
        ss << "()";
    } else {
        // Excluding the last comma
        ss << "(" << strtmp.substr(0, strtmp.size() - 2) << ")";
    }
}

vector<string> Engine::writeBlock(const SymbolTable &symbols,
                        const Scope *parent_scope,
                        const LoopB &kernel,
                        const std::vector<uint64_t> &thread_stack,
                        bool opencl,
                        std::stringstream &out,
                        std::vector<bh_metasweep> sweep_info,
                        const size_t parallelize_rank,
                        const vector<bh_view> lookups) {

    if (kernel.isSystemOnly()) {
        out << "// Removed loop with only system instructions\n";
        return {};
    }

    std::set<jitk::InstrPtr> sweeps_in_child;
    for (const jitk::Block &sub_block: kernel._block_list) {
        if (not sub_block.isInstr()) {
            sweeps_in_child.insert(sub_block.getLoop()._sweeps.begin(), sub_block.getLoop()._sweeps.end());
        }
    }

    // Order all sweep instructions by the viewID of their first operand.
    // This makes the source of the kernels more identical, which improve the code and compile caches.
    const vector <jitk::InstrPtr> ordered_block_sweeps = order_sweep_set(sweeps_in_child, symbols);

    // Let's find the local temporary arrays and the arrays to scalar replace
    const set<bh_base *> &local_tmps = kernel.getLocalTemps();

    // We always scalar replace reduction outputs that reduces over the innermost axis
    vector<const bh_view *> scalar_replaced_reduction_outputs;
    for (const jitk::InstrPtr &instr: ordered_block_sweeps) {
        if (bh_opcode_is_reduction(instr->opcode) and jitk::sweeping_innermost_axis(instr)) {
            if (local_tmps.find(instr->operand[0].base) == local_tmps.end()) {
                scalar_replaced_reduction_outputs.push_back(&instr->operand[0]);
            }
        }
    }

    // Let's scalar replace input-only arrays that are used multiple times
    vector<const bh_view *> srio = jitk::scalar_replaced_input_only(kernel, parent_scope, local_tmps);
    jitk::Scope scope(symbols, parent_scope, local_tmps, scalar_replaced_reduction_outputs, srio);

    // Write temporary and scalar replaced array declarations
    vector<const bh_view *> scalar_replaced_to_write_back;
    for (const jitk::Block &block: kernel._block_list) {
        if (block.isInstr()) {
            const jitk::InstrPtr instr = block.getInstr();
            for (const bh_view &view: instr->getViews()) {
                if (not scope.isDeclared(view)) {
                    if (scope.isTmp(view.base)) {
                        util::spaces(out, 8 + kernel.rank * 4);
                        scope.writeDeclaration(view, writeType(view.base->type), out);
                        out << "\n";
                    } else if (scope.isScalarReplaced(view)) {
                        util::spaces(out, 8 + kernel.rank * 4);
                        scope.writeDeclaration(view, writeType(view.base->type), out);

                        // d4_reduce_nd_transposed Trips over this?
                        /* if (opencl){ */
                        /*     out << "if (!redundant) {"; */
                        /* } */
                        out << " " << scope.getName(view) << " = a" << symbols.baseID(view.base);
                        write_array_subscription(scope, view, out);
                        out << ";";
                        /* if (opencl){ */
                        /*     out << "}"; */
                        /* } */
                        out << "\n";

                        // TODO: Also don't allocate the array
                        // We don't want a writeback to a temp array in global memory on sweeps
                        if (!bh_opcode_is_sweep(instr->opcode) && scope.isScalarReplaced_RW(view.base)) {
                            scalar_replaced_to_write_back.push_back(&view);
                        }
                    }
                }
            }
        }
    }

    //Let's declare indexes if we are not at the kernel level (rank == -1)
    if (kernel.rank >= 0) {
        for (const jitk::Block &block: kernel._block_list) {
            if (block.isInstr()) {
                const jitk::InstrPtr instr = block.getInstr();
                for (const bh_view &view: instr->getViews()) {
                    if (symbols.existIdxID(view) and scope.isArray(view)) {
                        if (not scope.isIdxDeclared(view)) {
                            util::spaces(out, 8 + kernel.rank * 4);
                            scope.writeIdxDeclaration(view, writeType(bh_type::UINT64), out);
                            out << "\n";
                        }
                    }
                }
            }
        }
    }

    // Write the for-loop body
    // The body in OpenCL and OpenMP are very similar but OpenMP might need to insert "#pragma omp atomic/critical"
    if (opencl) {
        for (const Block &b: kernel._block_list) {
            if (b.isInstr()) { // Finally, let's write the instruction
                if (b.getInstr() != nullptr and not bh_opcode_is_system(b.getInstr()->opcode)) {

                    // If we have a vector-to-scalar reduction
                    if (bh_opcode_is_sweep(b.getInstr()->opcode) && b.rank() == 1 && (sweep_info.size() == 1) && sweep_info.back().is_scalar()){
                        util::spaces(out, 4 + b.rank() * 4);
                        out << "element = ";
                        const bh_instruction instr = *b.getInstr();

                        const bh_view &view = instr.operand[1];

                        // TODO: The following is borrowed from void write_other_instr(const Scope &scope, const bh_instruction &instr, stringstream &out, bool opencl) and should be merged
                        if (view.isConstant()) {
                            const int64_t constID = scope.symbols.constID(instr);
                            if (constID >= 0) {
                                out << "c" << scope.symbols.constID(instr);
                            } else {
                                instr.constant.pprint(out, opencl);
                            }
                        } else {
                            scope.getName(view, out);
                            if (scope.isArray(view)) {
                                write_array_subscription(scope, view, out);
                            }
                        }
                        out << ";" << endl;
                    } else{
                        util::spaces(out, 4 + b.rank() * 4);
                        const bh_instruction instr = *b.getInstr();

                        // Comment out reduction inside segmented reduction
                        /* if (bh_opcode_is_reduction(instr.opcode) && */
                        /*         kernel.isInnermost() && */
                        /*         bh_metasweep(b.rank(), 0, instr).is_segment() && */
                        /*         sweep_info.size() > 0 && */
                        /*         sweep_info.front().left_operand.shape.begin()[thread_stack.size()-1] < 8192 */
                        /*         ){ */
                        /*     out << "// "; */
                        /*     write_instr(scope, *b.getInstr(), out, true); */
                        /* } */
                        /* else */
                        if(lookups.size() == 0) {
                            out << "if (!redundant) {";
                            write_instr(scope, *b.getInstr(), out, true);
                            out << "}";
                        }
                        else {
                            write_instr(scope, *b.getInstr(), out, true);
                        }
                        out << "\n";
                    }
                }
            } else {
                const LoopB &next_rank = b.getLoop();

                // Check if there exist a segmented reduction, and if we are in the right rank.
                size_t parallel_rank = thread_stack.size()-1;
                bool inject_seg_reduce = false;
                    /* (sweep_info.size() > 0) && */
                    /* (sweep_info.front().sweep_axis() == b.rank()) && */
                    /* sweep_info.front().is_segment() && */
                    /* next_rank.isInnermost() && */
                    /* sweep_info.front().left_operand.shape.begin()[parallel_rank] < 8192; // Hack to try avoid overcommiting local memory */


                // TODO: Detect if the seg_reduce doesn't use global memory. If it doesn't, skip the injection

                // NOTE: If we create a config for it, segmented reductions can be disabled in the following if-statement:
                if (inject_seg_reduce){
                    vector<bh_metasweep> sweeps = {}; // The unique sweeps to handle
                    vector<bh_metasweep> _sweeps = {}; // All sweeps in rank to handle write_back -- Calculated once, written to two different places.
                    size_t desired_size = sweep_info.front().left_operand.shape.back(); // Avoid merging 2 different ranks by mistake
                    for (bh_metasweep &s: sweep_info) {
                        if (s.rank == next_rank.rank && s.is_segment() && s.left_operand.shape.back() == desired_size){
                            int base_id = symbols.baseID(s.left_operand.base);
                            s.base_id = base_id; // To assure cache consistency
                            sweeps.push_back(s);
                            _sweeps.push_back(s);
                        }
                    }
                    std::sort (sweeps.begin(), sweeps.end()); // Assures cache consistency
                    auto last = std::unique(sweeps.begin(), sweeps.end()); // To remove redundancy/reallocation
                    sweeps.erase(last, sweeps.end());

                    size_t indent_level = 0;
                    INDENT; out << "{ // Segmented reduction injected.\n";
                    indent_level = 1;

                    // Max length is max segments pr. workgroup (not alot).
                    for (const bh_metasweep s: sweeps) {
                        INDENT; out << "__local volatile " << writeType(s.type()) << " write_back" << s.base_id << "[" << s.left_operand.shape.begin()[parallel_rank] << "];\n";
                        INDENT; out << "__local volatile " << writeType(s.type()) << " _a" << s.base_id << "[SCRATCHPAD_MEM];\n";
                    }

                    INDENT; out << "size_t lid = flat_local_id;\n";

                    INDENT; out << "size_t size = " << sweeps.front().left_operand.shape.back() << ";\n";
                    INDENT; out << "size_t segment_size = round_up_power2(size);\n";

                    INDENT; out << "const size_t increment_size = (segment_size < wavefront_size ? segment_size : wavefront_size);\n";
                    INDENT; out << "size_t sid = lid % increment_size; // Internal segment thread ID\n";

                    // With 128 work-group size, this is between 4 and 64. The following loop will run between 32 to 2 times respectively.
                    INDENT; out << "const size_t segments_per_workgroup = DIM1 / increment_size;\n"; // TODO: This is not right! Hangs at segment_size > 128

                    INDENT; out << "// For each segment\n";
                    INDENT; out << "for (size_t segment_id = (lid/increment_size); segment_id < " << sweeps.front().left_operand.shape.begin()[parallel_rank] << "; segment_id += segments_per_workgroup){\n";

                    for (const bh_metasweep s: sweeps) {
                        INDENT; out << "    " << writeType(s.type()) << " acc" << s.base_id << " = ";
                        jitk::sweep_identity(s.opcode, s.type()).pprint(out, true);
                        out << ";\n";
                    }

                    INDENT; out << "    // Read in data\n";
                    // Offset for each wavefront at a contigous, oversized segment. Has to work once, multiple times, and smaller than wavefront sizes.\n";
                    INDENT; out << "    for (int j=sid; j < size; j += increment_size) {\n";

                    size_t dims = sweeps.front().left_operand.ndim;
                    INDENT; out << "        const ulong i" << parallel_rank << " = segment_id;\n";
                    INDENT; out << "        const ulong i" << dims-1 << " = j;\n";

                    vector<bh_view> wanted_lookups = {};
                    for (const bh_metasweep s: sweeps) {
                        wanted_lookups.push_back(s.left_operand);
                    }
                    vector<string> returned_lookups = writeBlock(symbols, &scope, next_rank, thread_stack, opencl, out, sweep_info, parallelize_rank, wanted_lookups);
                    out << "//";
                    for (std::string name: returned_lookups) {
                        out << " " << name;
                    }
                    out << "\n";

                    for (size_t i=0; i<sweeps.size(); i++) {
                        const bh_metasweep s = sweeps[i];
                        string &var = returned_lookups[i];
                        std::string acc_id; { std::stringstream t; t << "acc" << s.base_id; acc_id = t.str(); }

                        INDENT; out << "        " << acc_id<< " = ";
                        s.write_op(out, acc_id, var);
                        out << ";\n";
                    }
                    INDENT; out << "    }\n";

                    for (int i=0; i<sweeps.size(); i++) {
                        const bh_metasweep s = sweeps[i];
                        string &var = returned_lookups[i];
                        std::string acc_id; { std::stringstream t; t << "acc" << s.base_id; acc_id = t.str(); }

                        INDENT; out << "    _a" << s.base_id << "[lid] = " << acc_id << ";\n";
                    }

                    INDENT; out << "    // Reduce segment\n";
                    INDENT; out << "    {\n";
                    INDENT; out << "    bool running = ((sid%2) == 0);\n";
                    INDENT; out << "    for (size_t i=1; i<=increment_size/2; i<<=1){\n";
                    INDENT; out << "        if (running){\n";
                    INDENT; out << "            running = (sid%(i<<2) == 0);\n";
                    for (int i=0; i<sweeps.size(); i++) {
                        const bh_metasweep s = sweeps[i];
                        std::string acc_id; { std::stringstream t; t << "acc" << s.base_id; acc_id = t.str(); }

                        INDENT; out << "            " << acc_id << " = ";
                        s.write_op(out, acc_id, "_a" + to_string(s.base_id) + "[lid+i]");
                        out << ";\n";
                        INDENT; out << "            _a" << s.base_id << "[lid] = " << acc_id << ";\n";
                    }
                    INDENT; out << "        }\n";
                    INDENT; out << "    }\n";

                    // Writeback to result array. Saves barriers at expense of some local memory, compared calling barrier now, and fetching across wavefronts.
                    INDENT; out << "    if (sid == 0){\n";
                    for (int i=0; i<sweeps.size(); i++) {
                        const bh_metasweep s = sweeps[i];
                        std::string acc_id; { std::stringstream t; t << "acc" << s.base_id; acc_id = t.str(); }

                        INDENT; out << "        write_back" << s.base_id << "[segment_id] = " << acc_id << ";\n";
                    }
                    INDENT; out << "    }\n";
                    INDENT; out << "    }\n";

                    INDENT; out << "}\n";
                    INDENT; out << "barrier(CLK_LOCAL_MEM_FENCE); // Synchronize before closing, so write_back access is valid;\n";
                    /* if (b.rank()-1 > (int)parallelize_rank || ((int)b.rank())-3 > 0){ */
                    /*     out << "if (redundant) {continue;}\n"; */
                    /* } */
                    /* else{ */
                    /*     out << "if (redundant) {return;}\n"; */
                    /* } */

                    // Handling write-back to Bohriums scalar replacement
                    for (int i=0; i<sweeps.size(); i++) {
                        const bh_metasweep s = sweeps[i];
                        string &var = returned_lookups[i];
                        std::string acc_id; { std::stringstream t; t << "acc" << s.base_id; acc_id = t.str(); }

                        for (const bh_metasweep _s: _sweeps){
                            if (_s.base_id == s.base_id){
                                const bh_view &view = _s.right_operand;
                                INDENT;
                                scope.getName(view, out);
                                if (scope.isArray(view)) {
                                    write_array_subscription(scope, view, out);
                                }
                                out << " = write_back" << s.base_id << "[lid];\n";
                            }
                        }
                    }

                    indent_level = 0;
                    INDENT; out << "}\n";
                }
                else{
                    util::spaces(out, 4 + b.rank() * 4);
                    loopHeadWriter(symbols, scope, b.getLoop(), thread_stack, out, parallelize_rank);
                    writeBlock(symbols, &scope, b.getLoop(), thread_stack, opencl, out, sweep_info, parallelize_rank);
                    util::spaces(out, 4 + b.rank() * 4);
                    out << "}\n";

                    out << "// " << b.rank() << " " << parallelize_rank << endl;
                    /* if (b.rank()-1 > (int)parallelize_rank || ((int)b.rank())-3 > 0){ */
                    /*     util::spaces(out, 4 + b.rank() * 4); */
                    /*     out << "if (redundant) {continue;}\n"; */
                    /* } */
                    /* else if (b.rank() > 0){ */
                    /*     util::spaces(out, 4 + b.rank() * 4); */
                    /*     out << "if (redundant) {return;}\n"; */
                    /* } */
                }
            }
        }
    } else {
        for (const Block &b: kernel._block_list) {
            if (b.isInstr()) { // Finally, let's write the instruction
                const InstrPtr instr = b.getInstr();
                if (not bh_opcode_is_system(instr->opcode)) {
                    if (instr->operand.size() > 0) {
                        if (scope.isOpenmpAtomic(instr)) {
                            util::spaces(out, 4 + b.rank() * 4);
                            out << "#pragma omp atomic\n";
                        } else if (scope.isOpenmpCritical(instr)) {
                            util::spaces(out, 4 + b.rank() * 4);
                            out << "#pragma omp critical\n";
                        }
                    }
                    util::spaces(out, 4 + b.rank() * 4);
                    write_instr(scope, *instr, out);
                    out << "\n";
                }
            } else {
                util::spaces(out, 4 + b.rank() * 4);
                loopHeadWriter(symbols, scope, b.getLoop(), thread_stack, out);
                writeBlock(symbols, &scope, b.getLoop(), thread_stack, opencl, out, sweep_info, parallelize_rank);
                util::spaces(out, 4 + b.rank() * 4);
                out << "}\n";
            }
        }
    }


    // Let's copy the scalar replaced reduction outputs back to the original array
    for (const bh_view *view: scalar_replaced_to_write_back) {
        util::spaces(out, 8 + kernel.rank * 4);
        if (opencl){
            out << "if (!redundant) ";
        }
        out << "a" << symbols.baseID(view->base);
        write_array_subscription(scope, *view, out, true);
        out << " = ";
        scope.getName(*view, out);
        out << ";\n";
    }


    vector<string> returned_lookups = {};
    for (const bh_view view: lookups) {
        stringstream ss;
        scope.getName(view, ss);
        if (scope.isArray(view)) {
            write_array_subscription(scope, view, ss);
        }
        returned_lookups.push_back(ss.str());
    }

    return returned_lookups;
}

void Engine::setConstructorFlag(std::vector<bh_instruction *> &instr_list, std::set<bh_base *> &constructed_arrays) {
    for (bh_instruction *instr: instr_list) {
        instr->constructor = false;
        for (size_t o = 0; o < instr->operand.size(); ++o) {
            const bh_view &v = instr->operand[o];
            if (not v.isConstant()) {
                if (o == 0 and not util::exist_nconst(constructed_arrays, v.base)) {
                    instr->constructor = true;
                }
                constructed_arrays.insert(v.base);
            }
        }
    }
}

}
} // namespace
