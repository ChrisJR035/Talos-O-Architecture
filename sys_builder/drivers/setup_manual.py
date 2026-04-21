from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='talos_core',
    ext_modules=[
        CppExtension(
            name='talos_core', 
            sources=['talos_core.cpp'], 
            extra_compile_args=['-O3', '-D_GLIBCXX_USE_CXX11_ABI=1']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
