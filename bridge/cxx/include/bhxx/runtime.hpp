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

#include "bh_instruction.hpp"
#include <bh_component.hpp>

namespace bhxx {

/**
 *  Encapsulation of communication with Bohrium runtime.
 *  Implemented as a Singleton.
 *
 *  \note  Not thread-safe.
 */
class Runtime {
  public:
    Runtime();

    // Get the singleton instance of the Runtime class
    static Runtime& instance() {
        static Runtime instance;
        return instance;
    }

    // Create and enqueue a new bh_instruction based on `opcode` and a variadic
    // pack of BhArrays and at most one scalar value
    template <typename... Ts>
    void enqueue(bh_opcode opcode, Ts&... ops) {
        BhInstruction instr(opcode);
        instr.append_operand(ops...);
        enqueue(std::move(instr));
    }

    /** Enqueue any BhInstruction object */
    void enqueue(BhInstruction instr);

    // We have to handle random specially because of the `BH_R123` scalar type
    void enqueue_random(BhArray<uint64_t>& out, uint64_t seed, uint64_t key);

    // Enqueue an extension method
    template <typename T>
    void enqueue_extmethod(const std::string& name, BhArray<T>& out, BhArray<T>& in1,
                           BhArray<T>& in2) {
        bh_opcode opcode;

        // Look for the extension opcode
        auto it = extmethods.find(name);
        if (it != extmethods.end()) {
            opcode = it->second;
        } else {
            // Add it and tell rest of Bohrium about this new extmethod
            opcode = extmethod_next_opcode_id++;
            runtime.extmethod(name.c_str(), opcode);
            extmethods.insert(std::pair<std::string, bh_opcode>(name, opcode));
        }

        // Now that we have an opcode, let's enqueue the instruction
        enqueue(opcode, out, in1, in2);
    }

    // Send enqueued instructions to Bohrium for execution
    void flush();

    ~Runtime() { flush(); }

    Runtime(Runtime&&) = default;
    Runtime& operator=(Runtime&&) = default;
    Runtime(const Runtime&)       = delete;
    Runtime& operator=(const Runtime&) = delete;

  private:
    // The lazy evaluated instructions
    std::vector<BhInstruction> instr_list;

    // Bohrium Configuration
    bohrium::ConfigParser config;

    // The Bohrium Runtime i.e. the child of this component
    bohrium::component::ComponentFace runtime;

    // Mapping an extension method name to an opcode id
    std::map<std::string, bh_opcode> extmethods;

    // The opcode id for the next new extension method
    bh_opcode extmethod_next_opcode_id;
};
}
#endif
