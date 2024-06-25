# `inlinePTX` Sample With CodePin

This sample demonstrates the usage of CodePin with inline PTX (assembly language) for SYCL code. The original CUDA* source code is migrated to SYCL for portability across GPUs from multiple vendors. 


| Area                   | Description
|:---                    |:---
| What you will learn    | How to debug and verify the SYCL equivalent of CUDA code
| Time to complete       | 15 minutes
| Category               | Concepts and Functionality

>**Note**: This sample is migrated from the NVIDIA CUDA sample. See the [inlinePTX](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/inlinePTX) sample in the NVIDIA/cuda-samples GitHub.

## Purpose

This sample demonstrates the CodePin usage experience, For a CUDA source code CodePin generates instrumented for both CUDA and migrated SYCL code. CodePin can analyze most user-defined CUDA class and generate the schema file for CUDA and SYCL classes except classes with virtual inheritance. CodePin reduces the need for extensive debugging by providing capabilities for on-the-fly functional testing during development and migration. This integration ensures that users can identify and address issues promptly, maintaining the integrity and performance of their application throughout the migration process. CodePin is distributed open-source and available at the SYCLomatic repository.

> **Note**: The sample used the open-source [SYCLomatic tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html) that assists developers in porting CUDA code to SYCL code. To finish the process, you must complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can also use the [Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html#gs.5g2aqn) available to augment Base Toolkit.

This sample contains two versions of the code and one common in the following folders:

| Folder Name          | Description
|:---                  |:---
|`dpct_output_codepin_cuda`    | Contains the migrated instrumented CUDA code from CUDA source code output from the SYCLomatic tool. 
|`dpct_output_codepin_sycl`    | Contains the migrated instrumented SYCL code from CUDA source code output from the SYCLomatic tool.
|`dpct_output`    | Contains the included files common for CUDA and SYCL.

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Ubuntu* 22.04
| Hardware (dpct_output_codepin_cuda)             |  Nvidia Tesla P100 <br> Nvidia A100 <br> Nvidia H100 
| Hardware (dpct_output_codepin_sycl)             | Intel® Gen9 <br>Intel® Gen11 <br>Intel® Xeon CPU <br>Intel® Data Center GPU Max <br> Nvidia Tesla P100 <br> Nvidia A100 <br> Nvidia H100 
| Software              | SYCLomatic (Tag - 20240618) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2024.1 <br> oneAPI for NVIDIA GPU plugin from Codeplay (to run SYCL™ applications on NVIDIA® GPUs)

For information on how to use SYCLomatic, refer to the materials at *[Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html)*.<br> How to run SYCL™ applications on NVIDIA® GPUs, refer to 
[oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin from Codeplay.


## Key Implementation Details
This sample demonstrates the migration of the following prominent CUDA features:
 - PTX Assembly

InlinePTX sample demonstrates how to implement PTX assembly (mov.u32) in SYCL kernels.


>**Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html) for general information about the migration workflow.
## CUDA source code evaluation

The inlinePTX application can embed PTX assembly (mov.u32) in a CUDA kernel.  

> **Note**: For more information on how to use the Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.vmhplg).

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.


## Migrate the Code Using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates 100% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the inlinePTX sample directory.
   ```
   cd cuda-samples/Samples/2_Concepts_and_Techniques/inlinePTX/
   ```
3. Generate a compilation database with intercept-build.
   ```
   intercept-build make
   ```
   This step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the Intel® SYCLomatic Compatibility Tool. The --in-root specifies path to the root of the source tree to be migrated. Enable CodePin with the –enable-codepin option dump a `dpct_output_codepin_cuda` (contains cuda with added instrumented code) and also dpct_output_codepin_sycl (contains migrated SYCL with added instrumented code) 

   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function --enable-codepin
   ```
## Suggested Manual Workarounds
There is one issue migrated SYCL code fails on any intel CPU as by default sub-group size is 16. But if restricted the subgroup size to 32, the issue is resolved.
The SYCL code suggestion is described below :-

 The warp size in CUDA is a fixed constant 32, but in SYCL sub-group size usually can be 16 or 32. Use intel extension [[intel::reqd_sub_group_size(32)]] to restrict the sub-group size to 32.
  ```
      dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cudaGridSize * cudaBlockSize, cudaBlockSize),
        [=](sycl::nd_item<3> item_ct1) {
            sequence_gpu(d_ptr, N, item_ct1);
        });
  
      
      Manually defined as below
   ```
      dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cudaGridSize * cudaBlockSize, cudaBlockSize),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            sequence_gpu(d_ptr, N, item_ct1);
        });
   ```

## Build and Run the `inlinePTX` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake .. or ( cmake -D INTEL_MAX_GPU=1 .. ) or ( cmake -D NVIDIA_GPU=1 .. )
   $ make
   ```
   > **Note**:
   > - By default, no flags are enabled during build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU.
   > - Enable **INTEL_MAX_GPU** flag during build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performance.
   > - Enable **NVIDIA_GPU** flag during build which supports NVIDIA GPUs.([oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin   from Codeplay is required to build for NVIDIA GPUs )

   By default, this command sequence will build the `01_dpct_output` and `02_sycl_migrated` versions of the program.
  
3. Run `01_dpct_output` on GPU.
   ```
   make run
   ```
   Run `01_dpct_output` on CPU.
   
   ```
   export ONEAPI_DEVICE_SELECTOR=opencl:cpu
   make run
   unset ONEAPI_DEVICE_SELECTOR
   ```
4. Run `02_sycl_migrated` on GPU.
   
   ```
   make run_sm
   ```
   Run `02_sycl_migrated` on CPU.
   ```
   export ONEAPI_DEVICE_SELECTOR=opencl:cpu
   make run_sm
   unset ONEAPI_DEVICE_SELECTOR
   ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
