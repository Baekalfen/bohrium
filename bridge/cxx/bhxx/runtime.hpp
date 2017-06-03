/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

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
#ifndef __BOHRIUM_BRIDGE_BHXX_RUNTIME
#define __BOHRIUM_BRIDGE_BHXX_RUNTIME
#include <iostream>
#include <sstream>

#include <bh_component.hpp>
#include <bh_instruction.hpp>

namespace bhxx {


/**
 *  Encapsulation of communication with Bohrium runtime.
 *  Implemented as a Singleton.
 *
 *  Note: Not thread-safe.
 */
class Runtime {
private:
    std::vector<bh_instruction> instr_list;      // The lazy evaluated instructions

    bohrium::ConfigParser config;                // Bohrium Configuration
    bohrium::component::ComponentFace runtime;   // The Bohrium Runtime i.e. the child of this component

    std::map<std::string, bh_opcode> extensions; // Register of extensions
    size_t extension_count;

public:
    Runtime() : config(-1), // stack level -1 is the bridge
                runtime(config.getChildLibraryPath(), 0), // and child is stack level 0
                extension_count(BH_MAX_OPCODE_ID+1) {}

    static Runtime& instance() {
        static Runtime instance;
        return instance;
    };

    // `instr_append_operand()` append an operand to the list of instruction in `instr`
    // Variadic base case: appending one array
    template <typename T>
    void instr_append_operand(bh_instruction &instr, BhArray<T> ary) {
        bh_view view;
        view.base = &ary.base->base;
        view.ndim = ary.shape.size();
        std::copy(ary.shape.begin(), ary.shape.end(), &view.shape[0]);
        std::copy(ary.stride.begin(), ary.stride.end(), &view.stride[0]);
        instr.operand.push_back(view);
    }
    // Variadic base case: appending one scalar
    template <typename T>
    void instr_append_operand(bh_instruction &instr, T scalar) {
        bh_view view;
        view.base = nullptr;
        instr.operand.push_back(view);
        bxx::assign_const_type(&instr.constant, scalar);
    }
    // Variadic case: appending one array and continue
    template <typename T, typename... Ts>
    void instr_append_operand(bh_instruction &instr, BhArray<T> ary, Ts... ops) {
        instr_append_operand(instr, ary);
        instr_append_operand(instr, ops...);
    }
    // Variadic case: appending one scalar and continue
    template <typename T, typename... Ts>
    void instr_append_operand(bh_instruction &instr, T scalar, Ts... ops) {
        instr_append_operand(instr, scalar);
        instr_append_operand(instr, ops...);
    }
    // Create and enqueue a new bh_instruction based on `opcode` and a variadic pack of BhArrays
    // and at most one scalar value
    template <typename... Ts>
    void enqueue(bh_opcode opcode, Ts... ops) {
        bh_instruction instr;
        instr.opcode = opcode;
        instr_append_operand(instr, ops...);
    }

    // Send enqueued instructions to Bohrium for execution
    void flush() {
        bh_ir bhir = bh_ir(instr_list.size(), &instr_list[0]);
        runtime.execute(&bhir);
    }
};

}
#endif
