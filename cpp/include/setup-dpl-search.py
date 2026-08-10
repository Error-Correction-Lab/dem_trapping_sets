from setuptools import Extension, setup
import pybind11
import sys


if sys.platform == "win32":
    # MSVC provides std::thread through the standard runtime.
    extra_compile_args = [
        "/O2",
        "/std:c++17",
        "/EHsc",
    ]
    extra_link_args = []
else:
    # GCC/Clang require -pthread during compilation and linking.
    extra_compile_args = [
        "-O3",
        "-std=c++17",
        "-march=native",
        "-pthread",
    ]
    extra_link_args = [
        "-pthread",
    ]


ext_modules = [
    Extension(
        "dpl_search",
        ["dpl-search-binder.cpp"],
        include_dirs=[
            pybind11.get_include(),
            ".",
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]


setup(
    name="dpl_search",
    version="0.0.2",
    ext_modules=ext_modules,
    zip_safe=False,
)