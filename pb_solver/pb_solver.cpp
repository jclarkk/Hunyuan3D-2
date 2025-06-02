#include <torch/extension.h>
#include <vector>

// Forward declaration
void pb_solver_run_cuda(torch::Tensor A, torch::Tensor X, torch::Tensor B, torch::Tensor Xbuf, int num_iters);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void pb_solver_run(torch::Tensor A, torch::Tensor X, torch::Tensor B, torch::Tensor Xbuf, int num_iters) {
    CHECK_INPUT(A);
    CHECK_INPUT(X);
    CHECK_INPUT(B);
    CHECK_INPUT(Xbuf);

    pb_solver_run_cuda(A, X, B, Xbuf, num_iters);
}

// Expose to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pb_solver_run", &pb_solver_run, "Poisson Blending Solver (CUDA)");
}