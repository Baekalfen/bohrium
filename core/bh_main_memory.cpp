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

#include <sys/mman.h>
#include <bh_main_memory.hpp>
#include <bh_malloc_cache.hpp>

using namespace std;
using namespace bohrium;

namespace {
// Allocate page-size aligned main memory.
void *main_mem_malloc(uint64_t nbytes) {
    // The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    // <http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>
    void *ret = mmap(0, nbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ret == MAP_FAILED or ret == nullptr) {
        std::stringstream ss;
        ss << "main_mem_malloc() could not allocate a data region. Returned error code: " << strerror(errno);
        throw std::runtime_error(ss.str());
    }
    return ret;
}

void main_mem_free(void *mem, uint64_t nbytes) {
    assert(mem != nullptr);
    if (munmap(mem, nbytes) != 0) {
        std::stringstream ss;
        ss << "main_mem_free() could not free a data region. " << "Returned error code: " << strerror(errno);
        throw std::runtime_error(ss.str());
    }
}

MallocCache malloc_cache(main_mem_malloc, main_mem_free);
}

void bh_data_malloc(bh_base *base) {
    if (base == nullptr) return;
    if (base->data != nullptr) return;
    base->data = malloc_cache.alloc(base->nbytes());
}

void bh_data_free(bh_base *base) {
    if (base == nullptr) return;
    if (base->data == nullptr) return;
    malloc_cache.free(base->nbytes(), base->data);
    base->data = nullptr;
}

void bh_data_malloc_stat(uint64_t &cache_lookup, uint64_t &cache_misses, uint64_t &max_memory_usage) {
    cache_lookup = malloc_cache.getTotalNumLookups();
    cache_misses = malloc_cache.getTotalNumMisses();
    max_memory_usage = malloc_cache.getMaxMemAllocated();
}