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
#include "bh_type.hpp"

#include <jitk/compiler.hpp>
#include <jitk/symbol_table.hpp>
#include <jitk/instruction.hpp>

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
    work_group_size_3dz(comp.config.defaultGet<cl_ulong>("work_group_size_3dz", 2)),
    opt_access_pattern(comp.config.defaultGet<int64_t>("optimize_access_pattern", 0))
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

/*     cout << thread_stack.size() << " " << opt_access_pattern << endl; */
/*     for (int i=0; i<thread_stack.size(); i++){ */
/*         cout << thread_stack[i] <<  ", "; */
/*     } */
/*     cout << endl; */


    if (opt_access_pattern == 0){
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
    else{
        const int max_x = 1024;
        const int max_y = 1024;
        const int max_z = 64;
        const int max_size = 1024;
        switch (b.size()) {
            case 1: {
                const int x = min(max_x,(int) b[0]);
                cout << "(x)" << "(" << x << ")" << endl;
                const auto gsize_and_lsize = work_ranges(x, b[0]);
                return make_pair(cl::NDRange(gsize_and_lsize.first), cl::NDRange(gsize_and_lsize.second));
            }
            case 2: {
                const int x = min(max_x,(int) b[0]);
                const int y = min(min(max_y,(int) b[1]), max_size/x);
                cout << "(x,y)" << "(" << x << ", " << y << ")" << endl;
                const auto gsize_and_lsize_x = work_ranges(x, b[0]);
                const auto gsize_and_lsize_y = work_ranges(y, b[1]);
                return make_pair(cl::NDRange(gsize_and_lsize_x.first, gsize_and_lsize_y.first),
                                 cl::NDRange(gsize_and_lsize_x.second, gsize_and_lsize_y.second));
            }
            case 3: {
                const int x = min(max_x,(int) b[0]);
                const int y = min(min(max_y,(int) b[1]), max_size/x);
                const int z = min(min(max_z,(int) b[2]), max_size/(x*y));
                cout << "(x,y,z)" << "(" << x << ", " << y << ", " << z << ")" << endl;
                const auto gsize_and_lsize_x = work_ranges(x, b[0]);
                const auto gsize_and_lsize_y = work_ranges(y, b[1]);
                const auto gsize_and_lsize_z = work_ranges(z, b[2]);
                return make_pair(cl::NDRange(gsize_and_lsize_x.first, gsize_and_lsize_y.first, gsize_and_lsize_z.first),
                                 cl::NDRange(gsize_and_lsize_x.second, gsize_and_lsize_y.second, gsize_and_lsize_z.second));
            }
            default:
                throw runtime_error("NDRanges: maximum of three dimensions!");
        }
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
                           const vector<const bh_instruction*> &constants,
                           const std::tuple<bh_opcode, bh_view, bh_view> sweep_info) {
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



    // TODO: Test for setting!
    // Force 1D kernel on the deepest available rank
    vector<uint64_t> thread_stack2;

    if (opt_access_pattern !=0){
        size_t dims = thread_stack.size();
        size_t lowest_stride = thread_stack[dims-1];
        /* thread_stack2.push_back(lowest_stride); */

        // Fill kernel param with dims of 1 element to exploit some parallelism better
        for (int i=0; i<std::min(dims, (size_t) opt_access_pattern); i++){
            thread_stack2.push_back(thread_stack[dims-i-1]);
            /* thread_stack2.push_back(1); */
        }

        assert (thread_stack2.size() == std::min(dims, (size_t) opt_access_pattern));
    }
    else{
        thread_stack2 = thread_stack;
    }


    /* cl::Buffer *buf = createBuffer(base); */
    const auto ranges = NDRanges(thread_stack2);
    size_t local_size = ranges.second.local_size();
    /* size_t work_groups = ranges.first.dim(0) / ranges.second.dim(0); // TODO: Multi-dim kernels! */
    size_t work_groups = ranges.first.work_groups(ranges.second);
    /* cout << "local_size " << local_size << " work_groups " << work_groups << endl; */

    cl::Buffer *reduction_mem = nullptr;
    size_t dtype_size;
    // Add arguments for first reduction-pass to the end of the regular kernel
    if (std::get<0>(sweep_info) != BH_NONE) {

        dtype_size = bh_type_size(std::get<1>(sweep_info).base->type);

        if (work_groups > 1 || bh_opcode_is_accumulate(std::get<0>(sweep_info))){ // Allocate temporary memory for storing sub-results
            reduction_mem = reinterpret_cast<cl::Buffer *>(malloc_cache.alloc(work_groups*dtype_size));
            opencl_kernel.setArg(i++, *reduction_mem);
        }
        else{ // When there is only one work-group, we can calculate the reduction in one pass
            assert (bh_opcode_is_reduction(std::get<0>(sweep_info))); // not supported for scan yet
            opencl_kernel.setArg(i++, *getBuffer(std::get<2>(sweep_info).base));
        }
        opencl_kernel.setArg(i++, local_size*dtype_size, NULL); // Allocate local memory for reduction
    }

    auto start_exec = chrono::steady_clock::now();
    queue.enqueueNDRangeKernel(opencl_kernel, cl::NullRange, ranges.first, ranges.second);

    // Call a post-reduction kernel, which finalizes the reduction
    if (std::get<0>(sweep_info) != BH_NONE && (work_groups > 1 || bh_opcode_is_accumulate(std::get<0>(sweep_info)))) {
        i = 0;

        cl::Kernel post_sweep;
        size_t sweep_local_size;
        pair<cl::NDRange, cl::NDRange> post_ranges;

        if (bh_opcode_is_accumulate(std::get<0>(sweep_info))){
            sweep_local_size = local_size;
            post_ranges = NDRanges(thread_stack2);
            post_sweep = cl::Kernel(program, "scan_2pass_postprocess");

            post_sweep.setArg(i++, *getBuffer(std::get<2>(sweep_info).base)); // Output array
            bh_constant len = bh_constant((uint64_t) std::get<1>(sweep_info).shape[0], bh_type::UINT64);
            post_sweep.setArg(i++, len.value.uint32);
            post_sweep.setArg(i++, *getBuffer(std::get<1>(sweep_info).base)); // Input array
        }
        else{
            sweep_local_size = 1024;
            pair<uint32_t, uint32_t> gsize_and_lsize = work_ranges(sweep_local_size, sweep_local_size);
            post_ranges = make_pair(cl::NDRange(gsize_and_lsize.first), cl::NDRange(gsize_and_lsize.second));
            post_sweep = cl::Kernel(program, "reduce_2pass_postprocess");

            // TODO: Maybe add offsets and stride? Would this ever be required for scalar reductions?
            /* uint64_t t1 = (uint64_t) view->start; */
            /* opencl_kernel.setArg(i++, t1); */
            /* for (int j=0; j<view->ndim; ++j) { */
            /*     uint64_t t2 = (uint64_t) view->stride[j]; */
            /*     opencl_kernel.setArg(i++, t2); */
            /* } */
            post_sweep.setArg(i++, *getBuffer(std::get<2>(sweep_info).base)); // Input array
        }

        post_sweep.setArg(i++, sweep_local_size*dtype_size, NULL); // Allocate local memory for reduction
        post_sweep.setArg(i++, *reduction_mem); // Sub-results

        bh_constant wg = bh_constant((uint64_t) work_groups, bh_type::UINT64);
        post_sweep.setArg(i++, wg.value.uint32);

        queue.enqueueNDRangeKernel(post_sweep, cl::NullRange, post_ranges.first, post_ranges.second);
    }

    queue.finish();
    auto texec = chrono::steady_clock::now() - start_exec;
    stat.time_exec += texec;
    stat.time_per_kernel[source_filename].register_exec_time(texec);

    if (std::get<0>(sweep_info) != BH_NONE && work_groups > 1) {
        malloc_cache.free(work_groups*dtype_size, reduction_mem);
    }
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
                               stringstream &ss,
                               const std::tuple<bh_opcode, bh_view, bh_view> sweep_info) {

    // Not actually lowest, but the one we assume is lowest in most cases
    size_t axis_lowest_stride;
    if (opt_access_pattern == 0){
        axis_lowest_stride = -1; // Disable in writeBlock->loopHeadWriter
    }
    else{
        axis_lowest_stride = thread_stack.size()-1;
        cout << "thread_stack size: " << thread_stack.size() << " " << axis_lowest_stride << endl;
    }

    /* size_t lowest_stride = thread_stack[axis_lowest_stride]; */
    /* size_t lowest_stride = -1; // Disable */
    /* size_t axis_lowest_stride = 0; // Disable */
    /* thread_stack.clear(); */
    /* thread_stack.push_back(lowest_stride); */

    /* size_t lowest_stride = -1; */
    /* size_t axis_lowest_stride = 0; */
    /* cout << "Finding best axis" << endl; */
    /* jitk::LoopB loop = kernel._block_list[0].getLoop(); */
    /* for (jitk::Block &b: loop._block_list) { */
    /*     /1* cout << "Block:" << endl << b << endl; *1/ */

    /*     // TODO: Probably not exhaustive enough */
    /*     while (!b.isInstr()) { */
    /*         b = b.getLoop()._block_list[0]; */
    /*     } */
    /*     // TODO: If (scalar?) sweep exists, ignore */

    /*     for (const bh_view &view: b.getInstr()->getViews()) { */
    /*         for (size_t i=0; i < view.ndim; i++){ */
    /*             size_t stride = view.stride[i]; */
    /*             if (lowest_stride >= stride && axis_lowest_stride <= i){ */
    /*                 lowest_stride = stride; */
    /*                 axis_lowest_stride = i; */
    /*             } */
    /*         } */
    /*     } */
    /* } */
    /* cout << "Best parallel: " << lowest_stride << " rank: " << axis_lowest_stride << endl; */

    // Write the need includes
    ss << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    ss << "#include <kernel_dependencies/complex_opencl.h>\n";
    ss << "#include <kernel_dependencies/integer_operations.h>\n";
    if (symbols.useRandom()) { // Write the random function
        ss << "#include <kernel_dependencies/random123_opencl.h>\n";
    }

    if (std::get<0>(sweep_info) != BH_NONE) {
        ss << "#define NEUTRAL ";
        jitk::sweep_identity(std::get<0>(sweep_info), std::get<1>(sweep_info).base->type).pprint(ss, true);
        ss << "\n";

        ss << "#define OPERATOR(a,b) (";
        const std::vector<string> ops = std::vector<string> {"a", "b"};
        /* jitk::write_operation(bh_instruction(sweep_info.first, views), ops, ss, true); */
        switch (std::get<0>(sweep_info)) {
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
        ss << ")\n";
        ss << "\n";

        ss << "#define __DATA_TYPE__ " << writeType(std::get<1>(sweep_info).base->type) <<"\n";

        const auto local_range = NDRanges(thread_stack).second;
        ss << "#define KERNEL_" << local_range.dimensions() << "D\n";
        assert (local_range.dimensions() == 1); // Don't allow multi-dim before we are ready

        for (size_t i = 0; i < local_range.dimensions(); i++){
            ss << "#define DIM" << i+1 << " " << local_range.dim(i) << "\n";
        }

        ss << "#include <kernel_dependencies/reduce_opencl.h>\n";
    }
    ss << "\n";


    bool is_sweep = (std::get<0>(sweep_info) != BH_NONE);

    // Write the header of the execute function
    ss << "__kernel void execute_" << codegen_hash;
    writeKernelFunctionArguments(symbols, ss, "__global", is_sweep);
    ss << " {\n";

    if (is_sweep){
        util::spaces(ss, 4);
        ss << "__DATA_TYPE__ element;\n";
    }

    // Write the IDs and overflow guards of the threaded blocks
    if (not thread_stack.empty()) {
        util::spaces(ss, 4);
        ss << "// The IDs of the threaded blocks:\n";
        if (opt_access_pattern == 0){
            for (unsigned int i=0; i < thread_stack.size(); ++i) {
                util::spaces(ss, 4);
                // Special case for vector-to-scalar reductions and rank0 sweeps
                if (is_sweep && i == 0){
                    // NOTE: We can't just return, as this workgroup might be the last to finish, and has to finalize the reduction
                    ss << "const " << writeType(bh_type::UINT32) << " g" << i << " = get_global_id(" << i << "); "
                        << "if (g" << i << " < " << thread_stack[i] << ") { // Prevent overflow in calculations, but keep thread for reduction\n";
                }
                else{
                    ss << "const " << writeType(bh_type::UINT32) << " g" << i << " = get_global_id(" << i << "); "
                        << "if (g" << i << " >= " << thread_stack[i] << ") { return; } // Prevent overflow\n";
                }
            }
        } else {
            util::spaces(ss, 4);
            ss << "// Optimizing Access Pattern!\n";

            if (is_sweep){
                util::spaces(ss, 4);
                // NOTE: We can't just return, as this workgroup might be the last to finish, and has to finalize the reduction
                ss << "const " << writeType(bh_type::UINT32) << " g" << axis_lowest_stride << " = get_global_id(0); "
                    << "if (g" << axis_lowest_stride << " < " << thread_stack[axis_lowest_stride] << ") { // Prevent overflow in calculations, but keep thread for reduction\n";
            }
            else{
                // Injecting optimized kernel parameter to IDs
                for (int i=0; i<std::min(thread_stack.size(), (size_t) opt_access_pattern); i++){
                    util::spaces(ss, 4);
                    ss << "const " << writeType(bh_type::UINT32) << " g" << axis_lowest_stride-i << " = get_global_id(" << i << "); "
                        << "if (g" << axis_lowest_stride-i << " >= " << thread_stack[axis_lowest_stride-i] << ") { return; } // Prevent overflow\n";
                }
            }
        }
        ss << "\n";
    }

    // Write inner blocks
    writeBlock(symbols, nullptr, kernel, thread_stack, true, ss, is_sweep, axis_lowest_stride);

    if (not thread_stack.empty() && is_sweep){
        // Inject neutral element, when there is no data in global memory to read
        ss << "    }else{\n        element = NEUTRAL;\n";

        // Call special rank0 preprocessing for both reduce and scan
        ss << "    }\n\n    reduce_2pass_preprocess(element, a, res);\n";
    }
    ss << "}\n\n";
}

// Writes the OpenCL specific for-loop header
void EngineOpenCL::loopHeadWriter(const jitk::SymbolTable &symbols,
                                  jitk::Scope &scope,
                                  const jitk::LoopB &block,
                                  const std::vector<uint64_t> &thread_stack,
                                  std::stringstream &out,
                                  const size_t parallelize_rank) {

    // Write the for-loop header
    std::string itername; { std::stringstream t; t << "i" << block.rank; itername = t.str(); }

    /* cout << "thread_stack: "; */
    /* for (unsigned int i=0; i < thread_stack.size(); ++i) { */
    /*     cout << thread_stack[i] << " "; */
    /* } */
    /* cout << static_cast<size_t >(block.rank) << endl; */

    if (opt_access_pattern > 0 && parallelize_rank != -1){

        (parallelize_rank - static_cast<size_t >(block.rank)) <= opt_access_pattern;

        if (parallelize_rank >= static_cast<size_t >(block.rank) &&
            ((int64_t) static_cast<size_t >(block.rank)) > ((int64_t) parallelize_rank) - opt_access_pattern){
            out << "{const " << writeType(bh_type::UINT64) << " " << itername << " = g" << block.rank << ";";
        }
        else {
            out << "for (" << writeType(bh_type::UINT64) << " " << itername << " = 0; ";
            out << itername << " < " << block.size << "; ++" << itername << ") {";
        }
        out << "\n";
        return;
    }

    if (thread_stack.size() > static_cast<size_t >(block.rank)) {
        // If we are limiting threads. num_threads == 0 is infinite threads allowed.
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
        }
        // Classic kernel parallelism for each rank.
        else {
            out << "{const " << writeType(bh_type::UINT64) << " " << itername << " = g" << block.rank << ";";
        }
    }
    // Substitute rest of the ranks with for-loops.
    else {
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
