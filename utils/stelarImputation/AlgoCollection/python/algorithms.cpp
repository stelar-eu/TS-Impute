// Algorithm headers
#include "../cpp/include/Algorithms/NMFMissingValueRecovery.h"
#include "../cpp/include/Algorithms/IterativeSVD.h"
#include "../cpp/include/Algorithms/SoftImpute.h"
#include "../cpp/include/Algorithms/TKCM.h"
#include "../cpp/include/Algorithms/SVT.h"
#include "../cpp/include/Algorithms/SPIRIT.h"
#include "../cpp/include/Algorithms/CDMissingValueRecovery.h"
#include "../cpp/include/Algorithms/DynaMMo.h"
#include "../cpp/include/Algorithms/GROUSE.h"
#include "../cpp/include/Algorithms/LinearImpute.h"
#include "../cpp/include/Algorithms/MeanImpute.h"
#include "../cpp/include/Algorithms/ZeroImpute.h"
#include "../cpp/include/Algorithms/ROSL.h"
#include "../cpp/include/Algorithms/PCA_MME.h"
#include "../cpp/include/Algorithms/OGDImpute.h"

// Carma
#include "carma/carma"

// Armadillo
#include <armadillo>

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Utility macro
#define DEF_ALGO(name, func, ...) \
    m.def(name, func, ##__VA_ARGS__);

// Wrappers
py::array_t<double> do_iterative_svd(const py::array_t<double>& X, uint64_t rank) {
    arma::mat X_arma = carma::arr_to_mat<double>(X);
    arma::mat result = Algorithms::IterativeSVD::recoveryIterativeSVD(X_arma, rank);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_nmf(const py::array_t<double>& input, uint64_t trunc) {
    arma::mat input_arma = carma::arr_to_mat<double>(input);
    arma::mat result = Algorithms::NMFMissingValueRecovery::doNMFRecovery(input_arma, trunc);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_soft_impute(const py::array_t<double>& X, uint64_t max_rank) {
    arma::mat X_arma = carma::arr_to_mat<double>(X);
    arma::mat result = Algorithms::SoftImpute::doSoftImpute(X_arma, max_rank);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_tkcm(const py::array_t<double>& X, uint64_t trunc) {
    arma::mat X_arma = carma::arr_to_mat<double>(X);
    arma::mat result = Algorithms::TKCM::doTKCM(X_arma, trunc);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_svt(const py::array_t<double>& X, double tau_scale) {
    arma::mat X_arma = carma::arr_to_mat<double>(X);
    arma::mat result = Algorithms::SVT::doSVT(X_arma, tau_scale);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_spirit(const py::array_t<double>& A, uint64_t k0, uint64_t w, double lambda) {
    arma::mat A_arma = carma::arr_to_mat<double>(A);
    arma::mat result = Algorithms::SPIRIT::doSpirit(A_arma, k0, w, lambda);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_mean_impute(const py::array_t<double>& input) {
    arma::mat mat = carma::arr_to_mat<double>(input);
    arma::mat result = Algorithms::MeanImpute::MeanImpute_Recovery(mat);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_zero_impute(const py::array_t<double>& input) {
    arma::mat mat = carma::arr_to_mat<double>(input);
    arma::mat result = Algorithms::ZeroImpute::ZeroImpute_Recovery(mat);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_linear_impute(const py::array_t<double>& input) {
    arma::mat mat = carma::arr_to_mat<double>(input);
    arma::mat result = Algorithms::LinearImpute::LinearImpute_Recovery(mat);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_cd_recovery(const py::array_t<double>& matrix, uint64_t truncation = 0, double eps = 1e-6) {
    arma::mat mat = carma::arr_to_mat<double>(matrix);
    arma::mat result = Algorithms::CDMissingValueRecovery::RecoverMatrix(mat, truncation, eps);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_grouse(const py::array_t<double>& input, uint64_t max_rank) {
    arma::mat mat = carma::arr_to_mat<double>(input);
    arma::mat result = Algorithms::GROUSE::doGROUSE(mat, max_rank);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_rosl(const py::array_t<double>& input, uint64_t rank, double reg) {
    arma::mat mat = carma::arr_to_mat<double>(input);
    arma::mat result = Algorithms::ROSL::ROSL_Recovery(mat, rank, reg);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_dynammo(const py::array_t<double>& X, uint64_t H = 0, uint64_t maxIter = 100, bool FAST = false) {
    arma::mat mat = carma::arr_to_mat<double>(X);
    arma::mat result = Algorithms::DynaMMo::doDynaMMo(mat, H, maxIter, FAST);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_pca_mme(const py::array_t<double>& matrix, uint64_t truncation, bool singleBlock) {
    arma::mat mat = carma::arr_to_mat<double>(matrix);
    arma::mat result = Algorithms::PCA_MME::doPCA_MME(mat, truncation, singleBlock);
    return carma::mat_to_arr(result);
}

py::array_t<double> do_ogd_impute(const py::array_t<double>& matrix, uint64_t truncation) {
    arma::mat mat = carma::arr_to_mat<double>(matrix);
    arma::mat result = Algorithms::OGDImpute::doOGDImpute(mat, truncation);
    return carma::mat_to_arr(result);
}

// PYBIND11 Module
PYBIND11_MODULE(algorithms, m) {
    m.doc() = "Algorithms for missing values library";

    DEF_ALGO("do_iterative_svd", &do_iterative_svd, py::arg("X"), py::arg("rank"));
    DEF_ALGO("do_nmf", &do_nmf, py::arg("input"), py::arg("trunc"));
    DEF_ALGO("do_soft_impute", &do_soft_impute, py::arg("X"), py::arg("max_rank"));
    DEF_ALGO("do_tkcm", &do_tkcm, py::arg("X"), py::arg("trunc"));
    DEF_ALGO("do_svt", &do_svt, py::arg("X"), py::arg("tau_scale"));
    DEF_ALGO("do_spirit", &do_spirit, py::arg("A"), py::arg("k0"), py::arg("w"), py::arg("lambda"));
    DEF_ALGO("do_mean_impute", &do_mean_impute, py::arg("input"));
    DEF_ALGO("do_zero_impute", &do_zero_impute, py::arg("input"));
    DEF_ALGO("do_linear_impute", &do_linear_impute, py::arg("input"));
    DEF_ALGO("do_cd_recovery", &do_cd_recovery, py::arg("matrix"), py::arg("truncation") = 0, py::arg("eps") = 1e-6);
    DEF_ALGO("do_grouse", &do_grouse, py::arg("input"), py::arg("max_rank"));
    DEF_ALGO("do_rosl", &do_rosl, py::arg("input"), py::arg("rank"), py::arg("reg"));
    DEF_ALGO("do_dynammo", &do_dynammo, py::arg("X"), py::arg("H") = 0, py::arg("maxIter") = 100, py::arg("FAST") = false);
    DEF_ALGO("do_pca_mme", &do_pca_mme, py::arg("matrix"), py::arg("truncation"), py::arg("singleBlock"));
    DEF_ALGO("do_ogd_impute", &do_ogd_impute, py::arg("matrix"), py::arg("truncation"));
}
