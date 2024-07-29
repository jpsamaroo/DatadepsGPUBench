using Dagger
using DaggerGPU
platform = get(ENV, "BENCHMARK", "CUDA")
if platform == "CUDA"
    using CUDA
    if parse(Bool, get(ENV, "BENCHMARK_MULTIGPU", "false"))
        const gpu_scope = Dagger.scope(cuda_gpus=:)
    else
        const gpu_scope = Dagger.scope(cuda_gpu=1)
    end
elseif platform == "AMDGPU"
    using AMDGPU
    if parse(Bool, get(ENV, "BENCHMARK_MULTIGPU", "false"))
        const gpu_scope = Dagger.scope(rocm_gpus=:)
    else
        const gpu_scope = Dagger.scope(rocm_gpu=1)
    end
else
    error("Unknown platform $platform\nSupported platforms are: CUDA, AMDGPU")
end
#=
if !startswith(Sys.CPU_NAME, "znver")
    using MKL
end
=#
using LinearAlgebra
using BenchmarkTools
using CSV, Tables

function FMULS_GEMM(m, n, k)
    fmuls = m * n * k
    return fmuls
end
function FADDS_GEMM(m, n, k)
    fadds = m * n * k
    return fadds
end
function FLOPS_GEMM(::Type{T}, m, n, k) where T
    fmuls = FMULS_GEMM(m, n, k)
    fadds = FADDS_GEMM(m, n, k)
    if T <: Complex
        fmuls = 6*fmuls
        fadds = 2*fadds
    end
    return fmuls+fadds
end

#size, block, ib, threads, time_dagger, time_blas, flops, gflops, gflops/seonds_dagger, gflops/second s_blas, speedup BLAS/Dagger, speedup Dagger/BLAS
function gemm_bench(T)
    n_small = [1024, 2048, 4096, 8192, 10240]
    n_large = [20480, 40960, 61440, 81920, 102400]
    nb_small=[1024, 2048, 4096]
    nb_large=[4096, 5120, 6144]
    task_window = [128, 256, 512, 1024]

    p = 1 # Subdomains only in caqr
    rows = length(n_small)*length(nb_small)*length(task_window)
    rows += length(n_large)*length(nb_large)*length(task_window)
    timing = zeros(rows, 13)
    counter=1
    # mat = N
    for i in n_small
        for j in nb_small
            if j<=i
                for k in task_window
                    timing[counter, 1] = i # size
                    timing[counter, 2] = j # block
                    timing[counter, 3] = k # window size
                    timing[counter, 4] = Threads.nthreads() # threads
                    timing[counter, 5] = p # Subdomains
                    timing[counter, 6] = 0.0 # time_dagger
                    timing[counter, 7] = 0.0 # time_blas
                    timing[counter, 8] = FLOPS_GEMM(T, i, i, i)# flops
                    timing[counter, 9] = timing[counter, 8]/10^9 # gflops
                    timing[counter, 10] = 0.0 # gflops/seonds_dagger
                    timing[counter, 11] = 0.0 # gflops/seconds_blas
                    timing[counter, 12] = 0.0 # speedup
                    timing[counter, 13] = 0.0 # speeddown

                    A = rand(T, i, i);
                    C = zeros(T, i, i);

                    DA = distribute(A, Blocks(j, j)); wait.(DA.chunks);
                    DC = distribute(C, Blocks(j, j)); wait.(DC.chunks);

                    domain = 1 # only in caqr
                    BLAS.set_num_threads(1);
                    #timing[counter, 6] = @belapsed mul!($DC, $DA, $DB)
                    t = zeros(1)
                    for r in range(1, 1)
                        Dagger.with_options(;scope=gpu_scope) do
                            Base.with(Dagger.DATADEPS_REGION_SPLIT=>k,
                                  Dagger.DATADEPS_SCHEDULER=>:roundrobin) do
                                  t[r]=@belapsed mul!($DC, $DA, $DA)
                            end
                        end
                    end
                    timing[counter, 6] = minimum(t)
                    timing[counter, 10] = timing[counter, 9] / timing[counter, 6];
                    BLAS.set_num_threads(Threads.nthreads()-1);
                    #if mat != i
                    timing[counter, 7] = @belapsed mul!($C, $A, $A)
                    #    mat = i
                    #else
                    #    timing[counter, 7] = timing[counter-1, 7]
                    #end
                    timing[counter, 11] = timing[counter, 9] / timing[counter, 7];
                    timing[counter, 12] = timing[counter, 7] / timing[counter, 6]; #speedup
                    timing[counter, 13] = timing[counter, 6] / timing[counter, 7]; #speeddown
                    display(transpose(timing[counter, :]))
                    CSV.write("timing_gemm_gpu_$(T).csv",  Tables.table(transpose(timing[counter,:])), writeheader=false, append=true)
                    counter+=1
                end
            end
        end
    end

    for i in n_large
        for j in nb_large
            if j<=i
                for k in task_window
                    timing[counter, 1] = i # size
                    timing[counter, 2] = j # block
                    timing[counter, 3] = k # window size
                    timing[counter, 4] = Threads.nthreads() # threads
                    timing[counter, 5] = p # Subdomains
                    timing[counter, 6] = 0.0 # time_dagger
                    timing[counter, 7] = 0.0 # time_blas
                    timing[counter, 8] = FLOPS_GEMM(T, i, i, i)# flops
                    timing[counter, 9] = timing[counter, 8]/10^9 # gflops
                    timing[counter, 10] = 0.0 # gflops/seonds_dagger
                    timing[counter, 11] = 0.0 # gflops/seconds_blas
                    timing[counter, 12] = 0.0 # speedup
                    timing[counter, 13] = 0.0 # speeddown

                    A = rand(T, i, i);
                    C = zeros(T, i, i);

                    DA = distribute(A, Blocks(j, j)); wait.(DA.chunks);
                    DC = distribute(C, Blocks(j, j)); wait.(DC.chunks);

                    domain = 1 # only in caqr
                    BLAS.set_num_threads(1);
                    #timing[counter, 6] = @belapsed mul!($DC, $DA, $DB)
                    t = zeros(1)
                    for r in range(1, 1)
                        Dagger.with_options(;scope=gpu_scope) do
                            Base.with(Dagger.DATADEPS_REGION_SPLIT=>k,
                                  Dagger.DATADEPS_SCHEDULER=>:roundrobin) do
                                  t[r]=@belapsed mul!($DC, $DA, $DA)
                            end
                        end
                    end
                    timing[counter, 6] = minimum(t)
                    timing[counter, 10] = timing[counter, 9] / timing[counter, 6];
                    BLAS.set_num_threads(Threads.nthreads()-1);
                    #if mat != i
                    timing[counter, 7] = @belapsed mul!($C, $A, $A)
                    #    mat = i
                    #else
                    #    timing[counter, 7] = timing[counter-1, 7]
                    #end
                    timing[counter, 11] = timing[counter, 9] / timing[counter, 7];
                    timing[counter, 12] = timing[counter, 7] / timing[counter, 6]; #speedup
                    timing[counter, 13] = timing[counter, 6] / timing[counter, 7]; #speeddown
                    display(transpose(timing[counter, :]))
                    CSV.write("timing_gemm_gpu_$(T).csv",  Tables.table(transpose(timing[counter,:])), writeheader=false, append=true)
                    counter+=1
                end
            end
        end
    end
    return timing
end
T = [Float32, #=Float64=#]
#T=[ComplexF32, ComplexF64]
#T=[ComplexF64]
for t in T
    timing = gemm_bench(t)
    display(timing)
end
