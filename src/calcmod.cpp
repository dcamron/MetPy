#include <nanobind/nanobind.h>

int add(int i, int j) {
    return i + j;
}

NB_MODULE(_calc_mod, m) {
    m.doc() = "accelerator module docstring";
    m.def("add", &add, "Add two numbers");
}