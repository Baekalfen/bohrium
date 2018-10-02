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

#include <vector>
#include <iostream>

#include <bh_instruction.hpp>
#include <bh_component.hpp>

#include <jitk/compiler.hpp>
#include <jitk/symbol_table.hpp>

#include "engine_opencl.hpp"

namespace fs = boost::filesystem;
using namespace std;

namespace {

#define CL_DEVICE_AUTO 1024 // More than maximum in the bitmask

map<const string, cl_device_type> device_map = {
    { "auto",        CL_DEVICE_AUTO             },
    { "gpu",         CL_DEVICE_TYPE_GPU         },
    { "accelerator", CL_DEVICE_TYPE_ACCELERATOR },
    { "default",     CL_DEVICE_TYPE_DEFAULT     },
    { "cpu",         CL_DEVICE_TYPE_CPU         }
};

// Get the OpenCL device (search order: GPU, ACCELERATOR, DEFAULT, and CPU)
cl::Device getDevice(const cl::Platform &platform, const string &default_device_type, const int &device_number) {
    vector<cl::Device> device_list;
    vector<cl::Device> valid_device_list;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &device_list);

    if(device_list.empty()){
        throw runtime_error("No OpenCL device found");
    }

    if (!util::exist(device_map, default_device_type)) {
        stringstream ss;
        ss << "'" << default_device_type << "' is not a OpenCL device type. " \
           << "Must be one of 'auto', 'gpu', 'accelerator', 'cpu', or 'default'";
        throw runtime_error(ss.str());
    } else if (device_map[default_device_type] != CL_DEVICE_AUTO) {
        for (auto &device: device_list) {
            if ((device.getInfo<CL_DEVICE_TYPE>() & device_map[default_device_type]) == device_map[default_device_type]) {
                valid_device_list.push_back(device);
            }
        }

        try {
            return valid_device_list.at(device_number);
        } catch(std::out_of_range &err) {
            stringstream ss;
            ss << "Could not find selected OpenCL device type ('" \
               << default_device_type << "') on default platform";
            throw runtime_error(ss.str());
        }
    }

    // Type was 'auto'
    for (auto &device_type: device_map) {
        for (auto &device: device_list) {
            if ((device.getInfo<CL_DEVICE_TYPE>() & device_type.second) == device_type.second) {
                valid_device_list.push_back(device);
            }
        }
    }

    try {
        return valid_device_list.at(device_number);
    } catch(std::out_of_range &err) {
        throw runtime_error("No OpenCL device of usable type found");
    }
}
}

namespace bohrium {

EngineOpenCL::EngineOpenCL(component::ComponentVE &comp, jitk::Statistics &stat) :
    EngineGPU(comp, stat),
    work_group_size_1dx(comp.config.defaultGet<cl_ulong>("work_group_size_1dx", 128)),
    work_group_size_2dx(comp.config.defaultGet<cl_ulong>("work_group_size_2dx", 32)),
    work_group_size_2dy(comp.config.defaultGet<cl_ulong>("work_group_size_2dy", 4)),
    work_group_size_3dx(comp.config.defaultGet<cl_ulong>("work_group_size_3dx", 32)),
    work_group_size_3dy(comp.config.defaultGet<cl_ulong>("work_group_size_3dy", 2)),
    work_group_size_3dz(comp.config.defaultGet<cl_ulong>("work_group_size_3dz", 2))
{
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) {
        throw runtime_error("No OpenCL platforms found");
    }

    bool found = false;
    if (platform_no == -1) {
        for (auto pform : platforms) {
            // Pick first valid platform
            try {
                // Get the device of the platform
                platform = pform;
                device = getDevice(platform, default_device_type, default_device_number);
                found = true;
            } catch(const cl::Error &err) {
                // We try next platform
            }
        }
    } else {
        if (platform_no > ((int) platforms.size()-1)) {
            std::stringstream ss;
            ss << "No such OpenCL platform. Tried to fetch #";
            ss << platform_no << " out of ";
            ss << platforms.size()-1 << "." << endl;
            throw std::runtime_error(ss.str());
        }

        platform = platforms[platform_no];
        device = getDevice(platform, default_device_type, default_device_number);
        found = true;
    }

    if (verbose) {
        cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;
    }

    if (!found) {
        throw runtime_error("Invalid OpenCL device/platform");
    }

    if(verbose) {
        cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() \
             << " ("<< device.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << ")" << endl;
    }

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);

    // Let's make sure that the directories exist
    jitk::create_directories(tmp_src_dir);

    // Write the compilation hash
    stringstream ss;
    ss << compile_flg
       << platform.getInfo<CL_PLATFORM_NAME>()
       << device.getInfo<CL_DEVICE_NAME>()
       << device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
    compilation_hash = util::hash(ss.str());

    // Initiate cache limits
    const uint64_t gpu_mem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    malloc_cache_limit_in_percent = comp.config.defaultGet<int64_t>("malloc_cache_limit", 90);
    if (malloc_cache_limit_in_percent < 0 or malloc_cache_limit_in_percent > 100) {
        throw std::runtime_error("config: `malloc_cache_limit` must be between 0 and 100");
    }
    malloc_cache_limit_in_bytes = static_cast<int64_t>(std::floor(gpu_mem * (malloc_cache_limit_in_percent/100.0)));
    malloc_cache.setLimit(static_cast<uint64_t>(malloc_cache_limit_in_bytes));
}

EngineOpenCL::~EngineOpenCL() {
    // Move JIT kernels to the cache dir
    if (not cache_bin_dir.empty()) {
        for (const auto &kernel: _programs) {
            const fs::path dst = cache_bin_dir / jitk::hash_filename(compilation_hash, kernel.first, ".clbin");
            if (not fs::exists(dst)) {
                cl_uint ndevs;
                kernel.second.getInfo(CL_PROGRAM_NUM_DEVICES, &ndevs);
                if (ndevs > 1) {
                    cout << "OpenCL warning: too many devices for caching." << endl;
                    return;
                }
                size_t bin_sizes[1];
                kernel.second.getInfo(CL_PROGRAM_BINARY_SIZES, bin_sizes);
                if (bin_sizes[0] == 0) {
                    cout << "OpenCL warning: no caching since the binary isn't available for the device." << endl;
                } else {
                    // Get the CL_PROGRAM_BINARIES and write it to a file
                    vector<unsigned char> bin(bin_sizes[0]);
                    unsigned char *bin_list[1] = {&bin[0]};
                    kernel.second.getInfo(CL_PROGRAM_BINARIES, bin_list);
                    ofstream binfile(dst.string(), ofstream::out | ofstream::binary);
                    binfile.write((const char*)&bin[0], bin.size());
                    binfile.close();
                }
            }
        }
    }

    // File clean up
    if (not verbose) {
        fs::remove_all(tmp_src_dir);
    }

    if (cache_file_max != -1 and not cache_bin_dir.empty()) {
        util::remove_old_files(cache_bin_dir, cache_file_max);
    }
}

namespace {
// Calculate the work group sizes.
// Return pair (global work size, local work size)
pair<uint32_t, uint32_t> work_ranges(uint64_t work_group_size, int64_t block_size) {
    if (numeric_limits<uint32_t>::max() <= work_group_size or
        numeric_limits<uint32_t>::max() <= block_size or
        block_size < 0) {
        stringstream ss;
        ss << "work_ranges(): sizes cannot fit in a uint32_t. work_group_size: " << work_group_size
           << ", block_size: " << block_size << ".";
        throw runtime_error(ss.str());
    }
    const auto lsize = (uint32_t) work_group_size;
    const auto rem   = (uint32_t) block_size % lsize;
    const auto gsize = (uint32_t) block_size + (rem==0?0:(lsize-rem));
    return make_pair(gsize, lsize);
}
}

pair<cl::NDRange, cl::NDRange> EngineOpenCL::NDRanges(const vector<uint64_t> &thread_stack) const {
    const auto &b = thread_stack;
    switch (b.size()) {
        case 1: {
            const auto gsize_and_lsize = work_ranges(work_group_size_1dx, b[0]);
            return make_pair(cl::NDRange(gsize_and_lsize.first), cl::NDRange(gsize_and_lsize.second));
        }
        case 2: {
            const auto gsize_and_lsize_x = work_ranges(work_group_size_2dx, b[0]);
            const auto gsize_and_lsize_y = work_ranges(work_group_size_2dy, b[1]);
            return make_pair(cl::NDRange(gsize_and_lsize_x.first, gsize_and_lsize_y.first),
                             cl::NDRange(gsize_and_lsize_x.second, gsize_and_lsize_y.second));
        }
        case 3: {
            const auto gsize_and_lsize_x = work_ranges(work_group_size_3dx, b[0]);
            const auto gsize_and_lsize_y = work_ranges(work_group_size_3dy, b[1]);
            const auto gsize_and_lsize_z = work_ranges(work_group_size_3dz, b[2]);
            return make_pair(cl::NDRange(gsize_and_lsize_x.first, gsize_and_lsize_y.first, gsize_and_lsize_z.first),
                             cl::NDRange(gsize_and_lsize_x.second, gsize_and_lsize_y.second, gsize_and_lsize_z.second));
        }
        default:
            throw runtime_error("NDRanges: maximum of three dimensions!");
    }
}

cl::Program EngineOpenCL::getFunction(const string &source) {
    uint64_t hash = util::hash(source);
    ++stat.kernel_cache_lookups;

    // Do we have the program already?
    if (_programs.find(hash) != _programs.end()) {
        return _programs.at(hash);
    }

    fs::path binfile = cache_bin_dir / jitk::hash_filename(compilation_hash, hash, ".clbin");
    cl::Program program;

    // If the binary file of the kernel doesn't exist we compile the source
    if (verbose or cache_bin_dir.empty() or not fs::exists(binfile)) {
        ++stat.kernel_cache_misses;
        std::string source_filename = jitk::hash_filename(compilation_hash, hash, ".cl");
        program = cl::Program(context, source);
        if (verbose) {
            const string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            if (not log.empty()) {
                cout << "************ Build Log ************" << endl
                     << log
                     << "^^^^^^^^^^^^^ Log END ^^^^^^^^^^^^^" << endl << endl;
            }
            fs::path srcfile = jitk::write_source2file(source, tmp_src_dir, source_filename, true);
        }
    } else { // If the binary file exist we load the binary into the program

        // First we load the binary into an vector
        vector<char> bin;
        {
            ifstream f(binfile.string(), ifstream::in | ifstream::binary);
            if (!f.is_open() or f.eof() or f.fail()) {
                throw runtime_error("Failed loading binary cache file");
            }
            f.seekg(0, std::ios_base::end);
            const std::streampos file_size = f.tellg();
            bin.resize(file_size);
            f.seekg(0, std::ios_base::beg);
            f.read(&bin[0], file_size);
        }

        // And then we load the binary into a program
        const vector<cl::Device> dev_list = {device};
        const cl::Program::Binaries bin_list = {make_pair(&bin[0], bin.size())};
        program = cl::Program(context, dev_list, bin_list);
    }

    // Finally, we build, save, and return the program
    try {
        program.build({device}, compile_flg.c_str());
    } catch (cl::Error &e) {
        cerr << "Error building: " << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        throw;
    }
    _programs[hash] = program;
    return program;
}

void EngineOpenCL::execute(const jitk::SymbolTable &symbols,
                           const std::string &source,
                           uint64_t codegen_hash,
                           const vector<uint64_t> &thread_stack,
                           const vector<const bh_instruction*> &constants) {
    // Notice, we use a "pure" hash of `source` to make sure that the `source_filename` always
    // corresponds to `source` even if `codegen_hash` is buggy.
    uint64_t hash = util::hash(source);
    std::string source_filename = jitk::hash_filename(compilation_hash, hash, ".cl");

    auto tcompile = chrono::steady_clock::now();
    string func_name; { stringstream t; t << "execute_" << codegen_hash; func_name = t.str(); }
    cl::Program program = getFunction(source);
    stat.time_compile += chrono::steady_clock::now() - tcompile;

    // Let's execute the OpenCL kernel
    cl::Kernel opencl_kernel = cl::Kernel(program, func_name.c_str());

    cl_uint i = 0;
    for (bh_base *base: symbols.getParams()) { // NB: the iteration order matters!
        opencl_kernel.setArg(i++, *getBuffer(base));
    }

    for (const bh_view *view: symbols.offsetStrideViews()) {
        uint64_t t1 = (uint64_t) view->start;
        opencl_kernel.setArg(i++, t1);
        for (int j=0; j<view->ndim; ++j) {
            uint64_t t2 = (uint64_t) view->stride[j];
            opencl_kernel.setArg(i++, t2);
        }
    }

    for (const bh_instruction *instr: constants) {
        switch (instr->constant.type) {
            case bh_type::BOOL:
                opencl_kernel.setArg(i++, instr->constant.value.bool8);
                break;
            case bh_type::INT8:
                opencl_kernel.setArg(i++, instr->constant.value.int8);
                break;
            case bh_type::INT16:
                opencl_kernel.setArg(i++, instr->constant.value.int16);
                break;
            case bh_type::INT32:
                opencl_kernel.setArg(i++, instr->constant.value.int32);
                break;
            case bh_type::INT64:
                opencl_kernel.setArg(i++, instr->constant.value.int64);
                break;
            case bh_type::UINT8:
                opencl_kernel.setArg(i++, instr->constant.value.uint8);
                break;
            case bh_type::UINT16:
                opencl_kernel.setArg(i++, instr->constant.value.uint16);
                break;
            case bh_type::UINT32:
                opencl_kernel.setArg(i++, instr->constant.value.uint32);
                break;
            case bh_type::UINT64:
                opencl_kernel.setArg(i++, instr->constant.value.uint64);
                break;
            case bh_type::FLOAT32:
                opencl_kernel.setArg(i++, instr->constant.value.float32);
                break;
            case bh_type::FLOAT64:
                opencl_kernel.setArg(i++, instr->constant.value.float64);
                break;
            case bh_type::COMPLEX64:
                opencl_kernel.setArg(i++, instr->constant.value.complex64);
                break;
            case bh_type::COMPLEX128:
                opencl_kernel.setArg(i++, instr->constant.value.complex128);
                break;
            case bh_type::R123:
                opencl_kernel.setArg(i++, instr->constant.value.r123);
                break;
            default:
                std::cerr << "Unknown OpenCL type: " << bh_type_text(instr->constant.type) << std::endl;
                throw std::runtime_error("Unknown OpenCL type");
        }
    }

    /* auto *buf = reinterpret_cast<cl::Buffer *>(malloc_cache.alloc(base->nbytes())); */
    /* bool inserted = buffers.insert(std::make_pair(base, buf)).second; */
    /* if (not inserted) { */
    /*     throw std::runtime_error("OpenCL - createBuffer(): the base already has a buffer!"); */
    /* } */
    /* return buf; */

    /* cl::Buffer *buf = createBuffer(base); */

    auto *reduction_mem = reinterpret_cast<cl::Buffer *>(malloc_cache.alloc(10*8));
    /* cout << reduction_mem << endl; */
    opencl_kernel.setArg(i++, *reduction_mem);

    auto *index_mem = reinterpret_cast<cl::Buffer *>(malloc_cache.alloc(1*4));
    opencl_kernel.setArg(i++, *index_mem);

    const auto ranges = NDRanges(thread_stack);
    auto start_exec = chrono::steady_clock::now();
    queue.enqueueNDRangeKernel(opencl_kernel, cl::NullRange, ranges.first, ranges.second);
    queue.finish();
    auto texec = chrono::steady_clock::now() - start_exec;
    stat.time_exec += texec;
    stat.time_per_kernel[source_filename].register_exec_time(texec);

    malloc_cache.free(10*8, reduction_mem);
    malloc_cache.free(1*4, index_mem);
}

// Copy 'bases' to the host (ignoring bases that isn't on the device)
void EngineOpenCL::copyToHost(const std::set<bh_base*> &bases) {
    auto tcopy = std::chrono::steady_clock::now();
    // Let's copy sync'ed arrays back to the host
    for(bh_base *base: bases) {
        if (util::exist(buffers, base)) {
            bh_data_malloc(base);
            queue.enqueueReadBuffer(*buffers.at(base), CL_FALSE, 0, (cl_ulong) base->nbytes(), base->data);
            // When syncing we assume that the host writes to the data and invalidate the device data thus
            // we have to remove its data buffer
            delBuffer(base);
        }
    }
    queue.finish();
    stat.time_copy2host += std::chrono::steady_clock::now() - tcopy;
}

// Copy 'base_list' to the device (ignoring bases that is already on the device)
void EngineOpenCL::copyToDevice(const std::set<bh_base*> &base_list) {
    // Let's update the maximum memory usage on the device
    if (prof) {
        uint64_t sum = 0;
        for (const auto &b: buffers) {
            sum += b.first->nbytes();
        }
        stat.max_memory_usage = sum > stat.max_memory_usage?sum:stat.max_memory_usage;
    }

    auto tcopy = std::chrono::steady_clock::now();
    for(bh_base *base: base_list) {
        if (not util::exist(buffers, base)) { // We shouldn't overwrite existing buffers
            cl::Buffer *buf = createBuffer(base);

            // If the host data is non-null we should copy it to the device
            if (base->data != nullptr) {
                queue.enqueueWriteBuffer(*buf, CL_FALSE, 0, (cl_ulong) base->nbytes(), base->data);
            }
        }
    }
    queue.finish();
    stat.time_copy2dev += std::chrono::steady_clock::now() - tcopy;
}

void EngineOpenCL::setConstructorFlag(std::vector<bh_instruction*> &instr_list) {
    std::set<bh_base *> constructed_arrays;
    for (auto it: buffers) {
        constructed_arrays.insert(it.first);
    }
    Engine::setConstructorFlag(instr_list, constructed_arrays);
}

// Copy all bases to the host (ignoring bases that isn't on the device)
void EngineOpenCL::copyAllBasesToHost() {
    std::set<bh_base*> bases_on_device;
    for(auto &buf_pair: buffers) {
        bases_on_device.insert(buf_pair.first);
    }
    copyToHost(bases_on_device);
}

// Delete a buffer
void EngineOpenCL::delBuffer(bh_base* base) {
    auto it = buffers.find(base);
    if (it != buffers.end()) {
        malloc_cache.free(base->nbytes(), it->second);
        buffers.erase(it);
    }
}

void EngineOpenCL::writeKernel(const jitk::LoopB &kernel,
                               const jitk::SymbolTable &symbols,
                               const vector<uint64_t> &thread_stack,
                               uint64_t codegen_hash,
                               stringstream &ss) {
    // Write the need includes
    ss << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    ss << "#include <kernel_dependencies/complex_opencl.h>\n";
    ss << "#include <kernel_dependencies/integer_operations.h>\n";
    if (symbols.useRandom()) { // Write the random function
        ss << "#include <kernel_dependencies/random123_opencl.h>\n";
    }
    ss << "\n";


    // Only handle a single sweep, which is a reduction
    auto is_reduction = false;
    auto rank0 = kernel.getLocalSubBlocks().front();
    auto sweeps = rank0->getSweeps();
    if (sweeps.size() == 1) {
        for(auto &sweep: sweeps) {
            cout << sweep << endl;
            if (bh_opcode_is_reduction(sweep->opcode)) {
                cout << "Reduction!" <<endl;
                is_reduction = true;
                /* auto s = sweeps; */
                /* s.clear(); // We'll handle it from here */
                /* kernel._sweeps = s; */

                ss << R"CONST(
#ifdef cl_nv_pragma_unroll
#define NVIDIA
#define wavefront_size 32
#else
#define AMD
#define wavefront_size 64
#endif

#define __DATA_TYPE__ long

#define LOCAL_SIZE 256

#define length 32
#define A a1

inline __DATA_TYPE__ reduce_wave(__DATA_TYPE__ acc, __local volatile __DATA_TYPE__ *a, size_t limit){
    size_t lid = get_local_id(0);
    // TODO: Try iterating backwards like AMD's example
    bool running = ((lid%2) == 0);
    for (size_t i=1; i<=limit/2; i<<=1){
        if (running){
            running = (lid%(i<<2) == 0);
            acc = acc + a[lid+i];
            a[lid] = acc;
        }
    }
    return acc;
}

inline void reduce_workgroup_wave_elimination_quarters(__local volatile __DATA_TYPE__ *a, size_t limit){
    size_t lid = get_local_id(0);
    barrier(CLK_LOCAL_MEM_FENCE); // Other reductions don't need this, but we are reading way out of the wavefront

    // We unroll the for-loop once, to eliminate a lot of workgroups.
    size_t i;
    for (i=limit/4; i>=32; i>>=2){
        __DATA_TYPE__ acc1;
        __DATA_TYPE__ acc2;
        __DATA_TYPE__ acc3;
        __DATA_TYPE__ acc4;
        bool running = lid < i;

        // WARN: Requires commutative property!
        if (running){
            acc1 = a[lid];
            acc2 = a[i+lid];
            acc3 = a[i*2+lid];
            acc4 = a[i*3+lid];
            // No more banking conflicts! Each lid reads its own bank every time down to wavefront size
            a[lid] = acc1+acc2+acc3+acc4;
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
            __DATA_TYPE__ acc1;
            __DATA_TYPE__ acc2;
            acc1 = a[lid];
            acc2 = a[lid+wavefront_size];
            a[lid] = acc1+acc2;
        }
        reduce_wave(a[lid], a, wavefront_size);
    }
}
)CONST";
            }
        }
    }




    // Write the header of the execute function
    ss << "__kernel void execute_" << codegen_hash;
    writeKernelFunctionArguments(symbols, ss, "__global", is_reduction);
    ss << " {\n";

    if (is_reduction){
        util::spaces(ss, 4);
        ss << "long reduction = 0;\n"; // TODO: Get type from kernel._block_list._sweeps and insert neutral element
    }

    // Write the IDs of the threaded blocks
    if (not thread_stack.empty()) {
        util::spaces(ss, 4);
        ss << "// The IDs of the threaded blocks: \n";
        for (unsigned int i=0; i < thread_stack.size(); ++i) {
            util::spaces(ss, 4);
            ss << "const " << writeType(bh_type::UINT32) << " g" << i << " = get_global_id(" << i << "); "
               << "if (g" << i << " >= " << thread_stack[i] << ") { return; } // Prevent overflow\n";
        }
        ss << "\n";
    }

    // Write inner blocks
    writeBlock(symbols, nullptr, kernel, thread_stack, true, ss);

    if (is_reduction){
        auto const reduction_array = "a1"; // Get this from the sweep attributes
        auto const result = "a2[0]"; // Get this from the sweep attributes

        /* util::spaces(ss, 4); */
        /* ss << "long a = " << reduction_array << "[g0] + " << reduction_array << "[g0+1];\n"; */
        util::spaces(ss, 4);
        ss << "if (g0 == 0) { \n";
        util::spaces(ss, 8);
        ss << "long a = " << reduction_array << "[g0] + " << reduction_array << "[g0+1] + a1[g0+2] + a1[g0+3];\n";
        util::spaces(ss, 8);
        ss << result <<" = a;\n";
        util::spaces(ss, 4);
        ss << "}\n";


        ss << R"KERNEL(
    __local __DATA_TYPE__ a[LOCAL_SIZE*sizeof(__DATA_TYPE__)];
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t local_size = get_local_size(0);
    size_t global_size = get_global_size(0);
    size_t group_id = get_group_id(0);
    size_t group_count = get_num_groups(0);
    size_t wavefront_id = lid % wavefront_size;
    size_t wavefront_number = lid / wavefront_size;
    size_t wavefront_count = local_size / wavefront_size;

    __DATA_TYPE__ acc;
    if (gid > length-1) {
        acc = 0; // Neutral element, when there is no data in global memory to read
        // NOTE: We can't just return, as this workgroup might be the last to finish, and has to finalize the reduction
    }
    else{
        acc = A[gid];
    }

    a[lid] = acc;
    // Barrier not needed, as data is always read from same wavefront -- not always true. Sub-function will call barrier if necessary.
    reduce_workgroup_wave_elimination_quarters(a, local_size);

    if (lid == 0){
        res[group_id] = a[0];
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
        acc = 0;
        for (size_t i=0; i < group_count; i += local_size){
            if (lid+i < group_count){
                acc = acc + res[lid+i];
            }
        }

        a[lid] = acc;
        reduce_workgroup_wave_elimination_quarters(a, local_size);

        if (lid == 0){
            res[0] = a[0];
            a2[0] = a[0];
        }
    }
}

)KERNEL";
    }
    else{
        // Close kernel
        ss << "}\n\n";
    }
}

// Writes the OpenCL specific for-loop header
void EngineOpenCL::loopHeadWriter(const jitk::SymbolTable &symbols,
                                  jitk::Scope &scope,
                                  const jitk::LoopB &block,
                                  const std::vector<uint64_t> &thread_stack,
                                  std::stringstream &out) {
    // Write the for-loop header
    std::string itername; { std::stringstream t; t << "i" << block.rank; itername = t.str(); }
    if (thread_stack.size() > static_cast<size_t >(block.rank)) {
        if (block._sweeps.size() != 0){
           for(auto &sweep: block._sweeps) {
               assert (bh_opcode_is_reduction(sweep->opcode));
           }
        }

        if (num_threads > 0 and thread_stack[block.rank] > 0) {
            if (num_threads_round_robin) {
                out << "for (" << writeType(bh_type::UINT64) << " " << itername << " = g" << block.rank << "; "
                    << itername << " < " << block.size << "; "
                    << itername << " += " << thread_stack[block.rank] << ") {";
            } else {
                const uint64_t job_size = static_cast<uint64_t>(ceil(block.size / (double)thread_stack[block.rank]));
                std::string job_start; {
                    std::stringstream t; t << "(g" << block.rank << " * " << job_size << ")"; job_start = t.str();
                }
                out << "for (" << writeType(bh_type::UINT64) << " " << itername << " = " << job_start << "; "
                    << itername << " < "  << job_start <<  " + " << job_size << " && " << itername << " < " << block.size
                    << "; ++" << itername << ") {";
            }
        } else {
            out << "{const " << writeType(bh_type::UINT64) << " " << itername << " = g" << block.rank << ";";
        }
    } else {
        out << "for (" << writeType(bh_type::UINT64) << " " << itername << " = 0; ";
        out << itername << " < " << block.size << "; ++" << itername << ") {";
    }
    out << "\n";
}

std::string EngineOpenCL::info() const {
    stringstream ss;
    ss << std::boolalpha; // Printing true/false instead of 1/0
    ss << "----"                                                                               << "\n";
    ss << "OpenCL:"                                                                            << "\n";
    ss << "  Platform no:    "; if(platform_no == -1) ss << "auto"; else ss << platform_no; ss << "\n";
    ss << "  Platform:       " << platform.getInfo<CL_PLATFORM_NAME>()                         << "\n";
    ss << "  Device type:    " << default_device_type                                          << "\n";
    ss << "  Device:         " << device.getInfo<CL_DEVICE_NAME>() << " (" \
                               << device.getInfo<CL_DEVICE_OPENCL_C_VERSION>()                 << ")\n";
    ss << "  Memory:         " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1024 / 1024    << " MB\n";
    ss << "  Malloc cache limit: " << malloc_cache_limit_in_bytes / 1024 / 1024
       << " MB (" << malloc_cache_limit_in_percent << "%)\n";
    ss << "  Compiler flags: " << compile_flg << "\n";
    ss << "  Cache dir: " << comp.config.defaultGet<string>("cache_dir", "")  << "\n";
    ss << "  Temp dir: " << jitk::get_tmp_path(comp.config)  << "\n";

    ss << "  Codegen flags:\n";
    ss << "    Index-as-var: " << comp.config.defaultGet<bool>("index_as_var", true)  << "\n";
    ss << "    Strides-as-var: " << comp.config.defaultGet<bool>("strides_as_var", true)  << "\n";
    ss << "    const-as-var: " << comp.config.defaultGet<bool>("const_as_var", true)  << "\n";
    return ss.str();
}

// Return OpenCL API types, which are used inside the JIT kernels
const std::string EngineOpenCL::writeType(bh_type dtype) {
    switch (dtype) {
        case bh_type::BOOL:       return "uchar";
        case bh_type::INT8:       return "char";
        case bh_type::INT16:      return "short";
        case bh_type::INT32:      return "int";
        case bh_type::INT64:      return "long";
        case bh_type::UINT8:      return "uchar";
        case bh_type::UINT16:     return "ushort";
        case bh_type::UINT32:     return "uint";
        case bh_type::UINT64:     return "ulong";
        case bh_type::FLOAT32:    return "float";
        case bh_type::FLOAT64:    return "double";
        case bh_type::COMPLEX64:  return "float2";
        case bh_type::COMPLEX128: return "double2";
        case bh_type::R123:       return "ulong2";
        default:
            std::cerr << "Unknown OpenCL type: " << bh_type_text(dtype) << std::endl;
            throw std::runtime_error("Unknown OpenCL type");
    }
}

} // bohrium
