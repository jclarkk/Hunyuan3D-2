from hy3dgen.texgen.mvadapter.utils.mesh_utils.blend import PBTorchCUDAKernelBackend

# This will trigger compilation and cache the .so in TORCH_EXTENSIONS_DIR
print("Precompiling pb_solver...")
PBTorchCUDAKernelBackend()
print("Done.")
