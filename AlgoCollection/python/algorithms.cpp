#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_algorithms(py::module &);

namespace mcl {

PYBIND11_MODULE(algorithms, m) {
    // Optional docstring
    m.doc() = "Algorithms for missing values library";
    
    init_algorithms(m);
}
}
