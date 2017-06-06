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
#pragma once
#include "bh_array.hpp"
#include <bh_instruction.hpp>
#include <type_traits>

namespace bhxx {

class BhInstruction : public bh_instruction {
  public:
    BhInstruction(bh_opcode code) : bh_instruction{} { opcode = code; }

    /** Append a single array to the list of operands */
    template <typename T>
    void append_operand(BhArray<T>& ary);

    /** Append a const array to the list of operands */
    template <typename T>
    void append_operand(const BhArray<T>& ary);

    /** Append a single scalar to the list of operands */
    template <typename T>
    void append_operand(T scalar);

    /** Append a list of operands  */
    template <typename T, typename... Ts>
    void append_operand(T& op, Ts&... ops) {
        append_operand(op);
        append_operand(ops...);
    }

    /** Append a special bh_constant */
    void append_operand(bh_constant cnt);

    /** Append a base object
     *
     * \note Only valid for BH_FREE */
    void append_operand(BhBase base);

  private:
    /** Container for a BhBase object, which is only needed for the
     *  case where this instruction represents a BH_FREE, because
     *  in this case we need to keep the BhBase alive until the
     *  queued instructions are all done, which is achieved by transferring
     *  the ownership of the BhBase to this very object.
     */
    std::unique_ptr<BhBase> base_ptr;
};
}
