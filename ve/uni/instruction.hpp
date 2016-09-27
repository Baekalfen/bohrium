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

#ifndef __BH_VE_UNI_INSTRUCTION_HPP
#define __BH_VE_UNI_INSTRUCTION_HPP

#include <iostream>

#include <bh_instruction.hpp>

#include "base_db.hpp"

namespace bohrium {

// Write the source code of an instruction
void write_instr(const BaseDB &base_ids, const bh_instruction &instr, std::stringstream &out);

// Return the axis that 'instr' reduces over or 'BH_MAXDIM' if 'instr' isn't a reduction
int sweep_axis(const bh_instruction &instr);

// Write the array subscription, e.g. A[2+i0*1+i1*10], but ignore the loop-variant of 'hidden_axis' if it isn't 'BH_MAXDIM'
void write_array_subscription(const bh_view &view, std::stringstream &out, int hidden_axis=BH_MAXDIM,
                              const std::pair<int, int> axis_offset=std::make_pair(BH_MAXDIM, 0));

} // bohrium

#endif