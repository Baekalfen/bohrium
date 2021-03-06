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
#include <map>
#include <set>

#include <bh_instruction.hpp>

/* The Bohrium Internal Representation (BhIR) represents an instruction
 * batch created by the Bridge component typically. */
class BhIR
{
public:
    // The list of Bohrium instructions in topological order
    std::vector<bh_instruction> instr_list;
    // Set of sync'ed arrays
    std::set<bh_base *> _syncs;
    // Number of times to repeat this BhIR
    uint64_t _nrepeats;
    // Repeat while this base evaluate to True or is a nullptr.
    // NB: the `base->data` must point to a single element of type BH_BOOL
    bh_base *_repeat_condition;

public:
    /** The regular constructor that takes the instructions, the sync'ed arrays, and number of times to run the BhIR */
    BhIR(std::vector<bh_instruction> instr_list,
         std::set<bh_base *> syncs,
         uint64_t nrepeats = 1,
         bh_base *repeat_condition = nullptr) :
        instr_list(std::move(instr_list)),
        _syncs(std::move(syncs)),
        _nrepeats(nrepeats),
        _repeat_condition(repeat_condition) {}

    /** Constructor that takes a serialized archive. All base array pointers are updated so that they point to
     *  local base arrays in `remote2local`.
     *
     *
     * \param serialized_archive Byte vector that makes up the serialized archive. The archive should be created with
     *                           `writeSerializedArchive`, which uses `boost::archive::binary_iarchive`.
     *
     * \param remote2local Map that maps remote array bases to local bases. The map is updated to include the new
     *                     array bases encountered in this BhIR thus this map should stay allocated throughout the
     *                     whole program execution.
     *
     * \param frees On return, will contain pointers to base arrays freed in this BhIR. NB: the pointer are "remote"
     *
     * \note We use the notion of remote and local base arrays. Remote base arrays are pointers to memory on
     *       the machine that serialized `serialized_archive`. Remote base arrays cannot be de-referenced instead they
     *       act as base array IDs.
     *       Use `remote2local` to translate remote base arrays to local base arrays, which are regular base arrays
     *       that can de-referenced.
     */
    BhIR(const std::vector<char> &serialized_archive,
         std::map<const bh_base*, bh_base> &remote2local,
         std::vector<bh_base*> &data_recv,
         std::set<bh_base*> &frees);


    /** Write the BhIR into a serialized archive.
     *
     * \param known_base_arrays Set of known base arrays. The set is updated to include new base arrays in this BhIR.
     *                          The new base arrays are also serialized into the return archive.
     *
     * \param new_data On return, will contain all base arrays that points to new array data i.e. the `bh_base.data`
     *                 pointers that are unknown to the de-serializing component. The bases are order as they appear
     *                 in the BhIR, thus their data should be transferred to the de-serializing component in the order
     *                 they appear.
     */
    std::vector<char> writeSerializedArchive(std::set<bh_base*> &known_base_arrays, std::vector<bh_base*> &new_data);

    /** Returns the set of sync'ed arrays */
    const std::set<bh_base *> getSyncs() const {
        return _syncs;
    }

    /** Get number of times this BhIR should run */
    uint64_t getNRepeats() const {
        return _nrepeats;
    }

    /** Get the repeat condition array
     *  NB: can be a nullptr when the condition is always true.
     */
    bh_base * getRepeatCondition() {
        return _repeat_condition;
    }
};
