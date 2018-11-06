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

#ifndef DIM2
#define DIM2 0
#endif

#ifndef DIM3
#define DIM3 0
#endif


#ifdef KERNEL_1D
#define flat_global_id get_global_id(0)
#define flat_local_id get_local_id(0)
#define flat_group_id get_group_id(0)
#define flat_local_size DIM1
#endif

#ifdef KERNEL_2D
#define flat_global_id (get_global_id(0) + get_global_size(0) * get_global_id(1))
#define flat_local_id (get_local_id(0) + DIM1 * get_local_id(1))
#define flat_group_id (get_group_id(0) + get_num_groups(0) * get_group_id(1))
#define flat_local_size (DIM1 * DIM2)
#endif

#ifdef KERNEL_3D
#define flat_global_id (get_global_id(0) + get_global_size(0) * get_global_id(1) + get_global_size(0) * get_global_size(1) * get_global_id(2))
#define flat_local_id (get_local_id(0) + DIM1 * get_local_id(1) + DIM1 * DIM2 * get_local_id(2))
#define flat_group_id (get_group_id(0) + get_num_groups(0) * get_group_id(1) + get_num_groups(0) * get_num_groups(1) * get_group_id(2))
#define flat_local_size (DIM1 * DIM2 * DIM3)
#endif

inline __DATA_TYPE__ reduce_wave(size_t lid, __DATA_TYPE__ acc, __local volatile __DATA_TYPE__ *a){
    bool running = ((lid%2) == 0);
    for (size_t i=1; i<=wavefront_size/2; i<<=1){
        if (running){
            running = (lid%(i<<2) == 0);
            acc = OPERATOR(acc, a[lid+i]);
            a[lid] = acc;
        }
    }
    return acc;
}

inline __DATA_TYPE__ reduce_workgroup_wave_elimination_quarters(size_t lid, __local volatile __DATA_TYPE__ *a, size_t limit){
    barrier(CLK_LOCAL_MEM_FENCE); // Other reductions don't need this, but we are reading way out of the wavefront

    size_t i;
    for (i=limit/4; i>=wavefront_size; i>>=2){
        bool running = lid < i;

        // WARN: Requires commutative property!
        if (running){
            __DATA_TYPE__ acc1 = a[lid];
            __DATA_TYPE__ acc2 = a[i+lid];
            __DATA_TYPE__ acc3 = a[i*2+lid];
            __DATA_TYPE__ acc4 = a[i*3+lid];
            a[lid] = OPERATOR4(acc1, acc2, acc3, acc4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/barrier.html
        // All work-items has to execute barriers
    }

    size_t wavefront_number = lid / wavefront_size;
    if (wavefront_number == 0){
        // TODO: Determine at compile time.
        if (i==16){ // for-loop iterated past the wavefront limit because of uneven number of divisors for local_size (it hasn't executed. local memory is at state of 64 elements). We will run a single iteration to get to 32 elements.
            __DATA_TYPE__ acc1 = a[lid];
            __DATA_TYPE__ acc2 = a[lid+wavefront_size];
            a[lid] = OPERATOR(acc1,acc2);
        }
        return reduce_wave(lid, a[lid], a);
    }
    return -1;
}


inline void reduce_2pass_preprocess(__DATA_TYPE__ acc, __local volatile __DATA_TYPE__ *a, __global volatile __DATA_TYPE__ *res){
    size_t lid = flat_local_id;
    size_t local_size = flat_local_size;

    a[lid] = acc;
    reduce_workgroup_wave_elimination_quarters(lid, a, local_size);

    if (lid == 0){
        size_t group_id = flat_group_id;
        res[group_id] = a[0];
    }
}


__kernel void reduce_2pass_postprocess(__global volatile __DATA_TYPE__ *final_res, __local volatile __DATA_TYPE__ *a, __global __DATA_TYPE__ *__restrict__ res, __const unsigned int group_count){
    // Loop through all values in result array. Eventhough there might be more than work-group size
    size_t lid = get_local_id(0);
    size_t local_size = get_local_size(0);

    // This final reduction is only valid as single-workgroup kernel.
    /* if (get_group_id(0) != 0){ */
    /*     return; */
    /* } */

    __DATA_TYPE__ acc = NEUTRAL;
    for (size_t i=0; i < group_count; i += local_size){
        if (lid+i < group_count){
            acc = OPERATOR(acc, res[lid+i]);
        }
    }

    a[lid] = acc;
    reduce_workgroup_wave_elimination_quarters(lid, a, local_size);

    if (lid == 0){
        final_res[0] = a[0];
    }
}



////////////////////////
// Scan operations
////////////////////////


inline void up_sweep_workgroup(__local __DATA_TYPE__* a, size_t local_size){
    size_t lid = get_local_id(0);

    for (size_t stride = 1; stride <= local_size; stride <<= 1) {
        size_t i = (lid + 1) * stride * 2 - 1;
        if (i < local_size) {
            a[i] = OPERATOR(a[i], a[i - stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void down_sweep_workgroup(__local __DATA_TYPE__* a, size_t local_size){
    size_t lid = get_local_id(0);

    for (size_t stride = local_size >> 1; stride > 0; stride >>= 1) {
        size_t i = (lid + 1) * stride * 2 - 1;
        if (i + stride < local_size) {
            a[i + stride] = OPERATOR(a[i + stride], a[i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Inclusive scan. We keep the last element: https://en.wikipedia.org/wiki/Prefix_sum
// Exclusive is not accessible through Bohrium
__kernel void scan_2pass_postprocess( __global __DATA_TYPE__ *outA, __const unsigned int length, __global __DATA_TYPE__ *inA, __local __DATA_TYPE__ *a, __global __DATA_TYPE__ *res, __const unsigned int group_count){
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t local_size = get_local_size(0);

    // TODO: This really needs optimization! Run it in another kernel, take several chunks in each workgroup, or something else
    // Find prefix for work-group
    __DATA_TYPE__ prefix = NEUTRAL;
    for (size_t i=0; i < get_group_id(0); i += local_size){
        if (lid+i < get_group_id(0)){
            prefix = OPERATOR(prefix, res[lid+i]);
        }
    }

    a[lid] = prefix;
    reduce_workgroup_wave_elimination_quarters(lid, a, local_size);
    barrier(CLK_LOCAL_MEM_FENCE);
    prefix = a[0];

    // Load in data
    __DATA_TYPE__ acc;
    if (gid > length-1) {
        acc = NEUTRAL; // Neutral element, when there is no data in global memory to read
        // NOTE: We can't just return, as this workgroup might be the last to finish, and has to finalize the scan
    }
    else{
        acc = inA[gid];
    }

    outA[gid] = acc;
    return;

    // Do scan
    a[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    up_sweep_workgroup(a, local_size);
    down_sweep_workgroup(a, local_size);

    // Add the prefix to scan result
    if (gid < length) {
        outA[gid] = OPERATOR(a[lid], prefix);
    }
}



