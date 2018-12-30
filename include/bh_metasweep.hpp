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

#include <vector>
#include <set>

#include "bh_opcode.h"
#include <bh_view.hpp>

using namespace std;

struct bh_metasweep {
    bh_opcode opcode;
    size_t rank;
    bh_view left_operand;
    bh_view right_operand;
    int axis;

    bh_metasweep(size_t rank, const bh_opcode& opcode, bh_view& left_operand, bh_view& right_operand, int axis) : opcode(opcode), rank(rank), left_operand(left_operand), right_operand(right_operand), axis(axis) {}

    bh_metasweep(size_t rank, const bh_instruction &sweep) : rank(rank) {
        auto views = sweep.getViews();
        // TODO: Fix this abomination
        bh_view r;
        bh_view l;
        int i = 0;
        for (const bh_view &view: views) {
            if (i==0){
                r = view;
            }
            else{
                l = view;
            }
            i++;
        }

        left_operand = l;
        right_operand = r;
        opcode = sweep.opcode;
        axis = sweep.sweep_axis();
    }

/*     bh_instruction(const bh_instruction &instr) { */
/*         opcode = instr.opcode; */
/*         constant = instr.constant; */
/*         operand = instr.operand; */
/*     } */

    bool is_segment() const {
        return
            /* (right_operand.stride[right_operand.ndim-1] == 1) && // Check stride. Only one of these are necessary */
            (left_operand.stride[left_operand.ndim-1] == 1) && // Check stride. Only one of these are necessary
            (right_operand.shape[0] != 1); // Don't include vector-to-scalar reductions
    }

    bool is_scalar() const {
        return
            (right_operand.ndim == 1) &&
            (right_operand.shape[0] == 1) &&
            (left_operand.ndim == 1) &&
            (left_operand.shape[0] > 1);
    }

    int64_t input_shape() const;
    int64_t output_shape() const;

    int64_t input_ndim() const;
    int64_t output_ndim() const;

    /// Returns the axis this instruction reduces over or 'BH_MAXDIM' if 'instr' isn't a reduction
    int sweep_axis() const {
        return axis;
    }

    string pprint(bool python_notation) const {
        stringstream ss;
        if (opcode > BH_MAX_OPCODE_ID)//It is an extension method
            ss << "ExtMethod";
        else//Regular instruction
            ss << bh_opcode_text(opcode);

        for (const bh_view &v: {left_operand, right_operand}) {
            ss << " ";
            ss << v.pprint(python_notation);
        }
        return ss.str();
    }
};


/* ostream &operator<<(ostream &out, const bh_metasweep &msweep) { */
/*     out << msweep.pprint(true); */
/*     return out; */
/* } */
