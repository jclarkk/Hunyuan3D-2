#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>


__global__ void pb_solver_run_cuda_kernel(
    torch::PackedTensorAccessor32<int64_t,2,torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> X,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> Xbuf
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < A.size(0)) {
        Xbuf[index] = (X[A[index][0]] + X[A[index][1]] + X[A[index][2]] + X[A[index][3]] + B[index]) / 4.;
    }
}

void pb_solver_run_cuda(
    torch::Tensor A,
    torch::Tensor X,
    torch::Tensor B,
    torch::Tensor Xbuf,
    int num_iters
) {
    int batch_size = A.size(0);

    const int threads = 1024;
    const dim3 blocks((batch_size + threads - 1) / threads);

    auto A_ptr = A.packed_accessor32<int64_t,2,torch::RestrictPtrTraits>();
    auto X_ptr = X.packed_accessor32<float,1,torch::RestrictPtrTraits>();
    auto B_ptr = B.packed_accessor32<float,1,torch::RestrictPtrTraits>();
    auto Xbuf_ptr = Xbuf.packed_accessor32<float,1,torch::RestrictPtrTraits>();

    for (int i = 0; i < num_iters; ++i) {
        pb_solver_run_cuda_kernel<<<blocks, threads>>>(
            A_ptr,
            X_ptr,
            B_ptr,
            Xbuf_ptr
        );
        cudaDeviceSynchronize();
        std::swap(X_ptr, Xbuf_ptr);
    }
    // we may waste an iteration here, but it's fine
    return;
}