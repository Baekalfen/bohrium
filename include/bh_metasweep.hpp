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

#include "bh_opcode.h"
#include <bh_view.hpp>

using namespace std;


// TODO: Could this simply be replaced by InstrB, while moving helper functions to bh_instruction?

struct bh_metasweep {
    bh_opcode opcode;
    size_t rank;
    bh_view left_operand;
    bh_view right_operand;
    int axis;
    size_t id;
    int64_t origin_id;
    int64_t base_id = -1;

    bh_metasweep(size_t rank, size_t id, const bh_instruction &sweep) : rank(rank) {
        opcode = sweep.opcode;
        assert (bh_opcode_is_sweep(opcode));
        axis = sweep.sweep_axis();
        auto views = sweep.getViews();
        left_operand = views.back(); // I know these are reversed semantically.
        right_operand = views.front();
        this->id = id;
        this->origin_id = sweep.origin_id;
    }

    bool is_segment() const {
        return false;
        return
            bh_opcode_is_reduction(opcode) &&
            (left_operand.stride[left_operand.ndim-1] == 1) && // Check stride. Segments are only for inputs with a stride of 1
            (right_operand.shape[0] != 1); // Don't include vector-to-scalar reductions
    }

    bool is_scalar() const {
        return
            bh_opcode_is_reduction(opcode) &&
            (right_operand.ndim == 1) &&
            (right_operand.shape[0] == 1) &&
            (left_operand.ndim == 1) &&
            !bh_type_is_complex(left_operand.base->type);
    }

    bh_type type() const{
        return left_operand.base->type;
    }

    int sweep_axis() const {
        return axis;
    }


    void write_op(stringstream &ss, string a, string b) const{
        const std::vector<string> ops = std::vector<string> {a,b};
        /* jitk::write_operation(bh_instruction(sweep_info.back().first, views), ops, ss, true); */
        switch (opcode) {
            case BH_BITWISE_AND_REDUCE:
                ss << ops[0] << " & " << ops[1];
                break;
            case BH_BITWISE_OR_REDUCE:
                ss << ops[0] << " | " << ops[1];
                break;
            case BH_BITWISE_XOR_REDUCE:
                ss << ops[0] << " ^ " << ops[1];
                break;
            case BH_LOGICAL_OR_REDUCE:
                ss << ops[0] << " || " << ops[1];
                break;
            case BH_LOGICAL_AND_REDUCE:
                ss << ops[0] << " && " << ops[1];
                break;
            case BH_LOGICAL_XOR_REDUCE:
                ss << ops[0] << " != !" << ops[1];
                break;
            case BH_MAXIMUM_REDUCE:
                ss <<  "max(" << ops[0] << ", " << ops[1] << ")";
                break;
            case BH_MINIMUM_REDUCE:
                ss <<  "min(" << ops[0] << ", " << ops[1] << ")";
                break;
            case BH_ADD_ACCUMULATE:
            case BH_ADD_REDUCE:
                ss << ops[0] << " + " << ops[1];
                break;
            case BH_MULTIPLY_ACCUMULATE:
            case BH_MULTIPLY_REDUCE:
                ss << ops[0] << " * " << ops[1];
                break;
            default:
                throw runtime_error("Instruction not supported.");
        }
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
        ss << " " << id;
        return ss.str();
    }

    bool operator< (bh_metasweep a) { return (origin_id>a.origin_id);}
    bool operator== (bh_metasweep a) { return (base_id==a.base_id);}
};


/* ostream &operator<<(ostream &out, const bh_metasweep &msweep) { */
/*     out << msweep.pprint(true); */
/*     return out; */
/* } */
