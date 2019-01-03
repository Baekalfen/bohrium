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

void Engine::writeBlock(const SymbolTable &symbols,
                        const Scope *parent_scope,
                        const LoopB &kernel,
                        const std::vector<uint64_t> &thread_stack,
                        bool opencl,
                        std::stringstream &out,
                        std::vector<bh_metasweep> sweep_info,
                        const size_t parallelize_rank) {

    if (kernel.isSystemOnly()) {
        out << "// Removed loop with only system instructions\n";
        return;
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
                        out << " " << scope.getName(view) << " = a" << symbols.baseID(view.base);
                        write_array_subscription(scope, view, out);
                        out << ";";
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
                        write_instr(scope, *b.getInstr(), out, true);
                    }
                }
            } else {
                auto next_rank = b.getLoop();

                // Check if there exist a segmented reduction, and if we are in the right rank.
                bool inject_seg_reduce =
                    (sweep_info.size() == 1) &&
                    (sweep_info.front().sweep_axis() == b.rank()) &&
                    sweep_info.front().is_segment() &&
                    next_rank.isInnermost();


                if (inject_seg_reduce){ // Check that the segmented reduction is in the very next rank and alone.
                    for (const Block &b: next_rank._block_list) {
                        if (!b.isInstr() || !(bh_metasweep(b.rank(), *b.getInstr()).is_segment()) ) {//!bh_opcode_is_reduction(b.getInstr()->opcode) || !(b.getInstr()->stride)){
                            inject_seg_reduce = false;
                            break;
                        }
                    }
                }

                // NOTE: If we create a config for it, segmented reductions can be disabled in the following if-statement:
                if (inject_seg_reduce){
                    const bh_metasweep sweep = sweep_info.front();

                    size_t indent_level = 0;
                    INDENT; out << "/*" << next_rank << "*/" << endl;

                    INDENT; out << "{ // Segmented reduction injected.\n";
                    indent_level = 1;

                    INDENT; out << "bool im_temp = false; if (false) {seg_reduce: ;im_temp = true;};\n";
                    INDENT; out << "__local volatile " << writeType(sweep.type()) << " write_back[5 /* TODO: Injected g1 length*/]; // Max length is max segments pr. workgroup (not alot).\n";
                    INDENT; out << "__local volatile " << writeType(sweep.type()) << " a[DIM1];\n";
                    /* INDENT; out << "" << writeType(sweep.type()) << " *write_back = s0;\n"; */

                    INDENT; out << "size_t lid = get_local_id(0);\n";
                    /* INDENT; out << "// Rewrite thread ID's\n"; */
                    /* INDENT; out << "/1* const ulong _i2 = i1; // TODO: Verify *1/\n"; */
                    /* INDENT; out << "/1* const ulong _i1 = lid % segment_size; // TODO: Verify *1/\n"; */

                    INDENT; out << "size_t segment_size = 4; // Rounded up to power of 2\n";
                    INDENT; out << "size_t size = 4;/* TODO: Injected i2 length*/\n";
                    /* INDENT; out << "size_t segment_size; // Rounded up to power of 2\n"; */
                    /* INDENT; out << "size_t size = 4;/1* TODO: Injected i2 length*1/\n"; */
                    /* INDENT; out << "if (size < wavefront_size) {\n"; */
                    /* INDENT; out << "    segment_size = round_up_power2(size);\n"; */
                    /* INDENT; out << "}\n"; */
                    /* INDENT; out << "else{\n"; */
                    /* INDENT; out << "    segment_size = size;\n"; */
                    /* INDENT; out << "}\n"; */

                    INDENT; out << "// For each segment\n";
                    INDENT; out << "size_t sid = lid % segment_size; // Internal segment thread ID\n";
                    /* INDENT; out << "const size_t segments_per_workgroup = DIM1 / segment_size; // With 128 work-group size, this is between 4 and 64. The following loop will run between 32 to 2 times respectively.\n"; */
                    INDENT; out << "const size_t segments_per_workgroup = 32;\n";

                    INDENT; out << writeType(sweep.type()) << " acc = ";
                    jitk::sweep_identity(sweep.opcode, sweep.type()).pprint(out, true);
                    out << ";\n";

                    INDENT; out << "for (size_t segment_id = (lid/segment_size); segment_id < 5 /* TODO: Injected g1 length*/; segment_id += segments_per_workgroup){ // segment_id ~~ i2\n";
                    INDENT; out << "    // Read in data\n";
                    // Offset for each wavefront at a contigous, oversized segment. Has to work once, multiple times, and smaller than wavefront sizes.\n";
                    INDENT; out << "    const size_t increment_size = (segment_size < wavefront_size ? segment_size : wavefront_size);\n";
                    INDENT; out << "    for (int j=sid; j < size; j += increment_size) {\n";
                    /* INDENT; out << "    {\n"; */
                    INDENT; out << "        // IMPORTANT! ALL INDICES HAS TO BE REINSTANTIATED BECAUSE OF GOTO!\n";
                    INDENT; out << "        const ulong i0 = g0;\n";
                    INDENT; out << "        const ulong i1 = segment_id;\n";
                    INDENT; out << "        const ulong i2 = j; //sid\n";
                    INDENT; out << "        acc += ";
                    {
                        const bh_view &view = sweep_info.front().left_operand;
                        scope.getName(view, out);
                        if (scope.isArray(view)) {
                            write_array_subscription(scope, view, out);
                        }
                    }
                    out << ";\n";
                    INDENT; out << "    }\n";

                    INDENT; out << "    // Reduce segment\n";
                    INDENT; out << "    bool running = ((sid%2) == 0);\n";
                    INDENT; out << "    for (size_t i=1; i<=segment_size/2; i<<=1){\n";
                    INDENT; out << "        if (running){\n";
                    INDENT; out << "            running = (sid%(i<<2) == 0);\n";
                    /* INDENT; out << "            acc = OPERATOR(acc, a[lid+i]);\n"; */
                    INDENT; out << "            acc = acc + a[lid+i];\n";
                    INDENT; out << "            a[lid] = acc;\n";
                    INDENT; out << "        }\n";
                    INDENT; out << "    }\n";

                    // Writeback to result array. Saves barriers at expense of some local memory, compared calling barrier now, and fetching across wavefronts.
                    INDENT; out << "    if (sid == 0){\n";
                    /* INDENT; out << "        write_back[segment_id] = acc;\n"; */
                    /* INDENT; out << "        write_back[segment_id] = segment_id;\n"; */
                    /* INDENT; out << "        write_back[segment_id] = segment_size;\n"; */
                    INDENT; out << "    }\n";
                    INDENT; out << "}\n";
                    INDENT; out << "barrier(CLK_LOCAL_MEM_FENCE); // Synchronize before closing, so write_back access is valid;\n";
                    INDENT; out << "if (im_temp) {goto skip_block;}\n";

                    out << "// ";
                    {
                        const bh_view &view = sweep_info.front().right_operand;
                        scope.getName(view, out);
                        if (scope.isArray(view)) {
                            write_array_subscription(scope, view, out);
                        }
                    }
                    out << ";" << endl;
                    INDENT; out << "s0 = write_back[lid]; // Inject results back into outer state\n";
                    indent_level = 0;
                    INDENT; out << "}\n";
                }
                else{
                    util::spaces(out, 4 + b.rank() * 4);
                    loopHeadWriter(symbols, scope, b.getLoop(), thread_stack, out, parallelize_rank);
                    writeBlock(symbols, &scope, b.getLoop(), thread_stack, opencl, out, sweep_info, parallelize_rank);
                    util::spaces(out, 4 + b.rank() * 4);
                    out << "}\n";
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
        out << "a" << symbols.baseID(view->base);
        write_array_subscription(scope, *view, out, true);
        out << " = ";
        scope.getName(*view, out);
        out << ";\n";
    }
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
