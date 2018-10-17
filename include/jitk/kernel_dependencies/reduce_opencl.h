/*
This file is part of Bohrium and copyright (c) 2018 the Bohrium
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
#ifdef cl_nv_pragma_unroll
#define NVIDIA
#define wavefront_size 32
#else
#define AMD
#define wavefront_size 64
#endif

#define OPERATOR4(a,b,c,d) (OPERATOR(a, OPERATOR(b, OPERATOR(c, d))))

inline __DATA_TYPE__ reduce_wave(__DATA_TYPE__ acc, __local volatile __DATA_TYPE__ *a, size_t limit){
    size_t lid = get_local_id(0);
    // TODO: Try iterating backwards like AMD's example
    bool running = ((lid%2) == 0);
    for (size_t i=1; i<=limit/2; i<<=1){
        if (running){
            running = (lid%(i<<2) == 0);
            acc = OPERATOR(acc, a[lid+i]);
            a[lid] = acc;
        }
    }
    return acc;
}

inline __DATA_TYPE__ reduce_workgroup_wave_elimination_quarters(__local volatile __DATA_TYPE__ *a, size_t limit){
    size_t lid = get_local_id(0);
    barrier(CLK_LOCAL_MEM_FENCE); // Other reductions don't need this, but we are reading way out of the wavefront

    // We unroll the for-loop once, to eliminate a lot of workgroups.
    size_t i;
    for (i=limit/4; i>=wavefront_size; i>>=2){
        bool running = lid < i;

        // WARN: Requires commutative property!
        if (running){
            __DATA_TYPE__ acc1 = a[lid];
            __DATA_TYPE__ acc2 = a[i+lid];
            __DATA_TYPE__ acc3 = a[i*2+lid];
            __DATA_TYPE__ acc4 = a[i*3+lid];
            // No more banking conflicts! Each lid reads its own bank every time down to wavefront size
            a[lid] = OPERATOR4(acc1, acc2, acc3, acc4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/barrier.html
        // All work-items has to execute barriers
    }

    /* size_t wavefront_size = 32; */
    size_t wavefront_number = lid / wavefront_size;
    if (wavefront_number == 0){
        // TODO: Determine at compile time.
        if (i==16){ // for-loop iterated past the wavefront limit because of uneven number of divisors for local_size (it hasn't executed. local memory is at state of 64 elements). We will run a single iteration to get to 32 elements.
            __DATA_TYPE__ acc1 = a[lid];
            __DATA_TYPE__ acc2 = a[lid+wavefront_size];
            a[lid] = OPERATOR(acc1,acc2);
        }
        return reduce_wave(a[lid], a, wavefront_size);
    }
    return -1;
}


inline void full_reduction(__DATA_TYPE__ acc, __local volatile __DATA_TYPE__ *a, __global volatile __DATA_TYPE__ *res, __global volatile unsigned int* index, __global __DATA_TYPE__* __restrict__ final_res){
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t local_size = get_local_size(0);
    size_t group_id = get_group_id(0);
    size_t group_count = get_num_groups(0);

    a[lid] = acc;
    // Barrier not needed, as data is always read from same wavefront -- not always true. Sub-function will call barrier if necessary.
    acc = reduce_workgroup_wave_elimination_quarters(a, local_size);

    if (lid == 0){
        res[group_id] = acc;
    }

    write_mem_fence(CLK_GLOBAL_MEM_FENCE);

    // Add our work-group to the finished counter, and share with remaining wavefront.
    if (lid == 0){
        uint finish_id = atomic_inc(index);
        a[0] = finish_id; // WARN: Type cast
    }

    // All wavefronts in work-group synchronizes here to determine, if they are needed for final reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    uint finish_id = a[0];

    // If we are the last work-group, fetch all other work-group's results and reduce
    if (finish_id == group_count-1){

        // Loop through all values in result array. Eventhough there might be more than work-group size
        // WARN: Requires commutative property!
        acc = NEUTRAL;
        for (size_t i=0; i < group_count; i += local_size){
            if (lid+i < group_count){
                acc = OPERATOR(acc, res[lid+i]);
            }
        }

        a[lid] = acc;
        acc = reduce_workgroup_wave_elimination_quarters(a, local_size);

        if (lid == 0){
            final_res[0] = acc;
        }
    }
}



inline void reduce_2pass_preprocess(__DATA_TYPE__ acc, __local volatile __DATA_TYPE__ *a, __global volatile __DATA_TYPE__ *res){
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t local_size = get_local_size(0);
    size_t group_id = get_group_id(0);


    a[lid] = acc;
    // Barrier not needed, as data is always read from same wavefront -- not always true. Sub-function will call barrier if necessary.
    reduce_workgroup_wave_elimination_quarters(a, local_size);

    if (lid == 0){
        res[group_id] = a[0];
    }
}


__kernel void reduce_2pass_postprocess(__local volatile __DATA_TYPE__ *a, __global __DATA_TYPE__ *__restrict__ res, __global volatile __DATA_TYPE__ *final_res, __const unsigned int group_count){
    // Loop through all values in result array. Eventhough there might be more than work-group size
    size_t lid = get_local_id(0);
    size_t local_size = get_local_size(0);

    // This final reduction is only valid as single-workgroup kernel.
    if (get_group_id(0) != 0){
        return;
    }

    __DATA_TYPE__ acc = NEUTRAL;
    for (size_t i=0; i < group_count; i += local_size){
        if (lid+i < group_count){
            acc = OPERATOR(acc, res[lid+i]);
        }
    }

    a[lid] = acc;
    reduce_workgroup_wave_elimination_quarters(a, local_size);

    if (lid == 0){
        final_res[0] = a[0];
    }
}
