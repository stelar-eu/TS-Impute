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
#include "carma/carma.h"

#include <armadillo>

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace Algorithms {
    class IterativeSVD_Cpp : public IterativeSVD {
    public:
            using IterativeSVD::IterativeSVD;

            static py::array_t<double> doIterativeSVD(py::array_t<double> &X, uint64_t rank){
                // Convert to arma
                arma::mat X_arma = carma::arr_to_mat<double>(X);

                // Call
                arma::mat X_arma_new = IterativeSVD::recoveryIterativeSVD(X_arma, rank);

                // Convert back to numpy
                py::array_t<double> X_final = carma::mat_to_arr<double>(X_arma_new);

                return X_final;
            }
    };
	
	class NMFMissingValueRecovery_Cpp : public NMFMissingValueRecovery {
    public:
            using NMFMissingValueRecovery::NMFMissingValueRecovery;

            static py::array_t<double> doNMFRecovery(py::array_t<double> &input, uint64_t truncation){
                // Convert to arma
                arma::mat input_arma = carma::arr_to_mat<double>(input);

                // Call
                arma::mat input_arma_new = NMFMissingValueRecovery::doNMFRecovery(input_arma, truncation);

                // Convert back to numpy
                py::array_t<double> input_final = carma::mat_to_arr<double>(input_arma_new);

                return input_final;
            }
    };
	
	class SoftImpute_Cpp : public SoftImpute {
    public:
            using SoftImpute::SoftImpute;

            static py::array_t<double> doSoftImpute(py::array_t<double> &X, uint64_t max_rank){
                // Convert to arma
                arma::mat X_arma = carma::arr_to_mat<double>(X);

                // Call
                arma::mat X_arma_new = SoftImpute::doSoftImpute(X_arma, max_rank);

                // Convert back to numpy
                py::array_t<double> X_final = carma::mat_to_arr<double>(X_arma_new);

                return X_final;
            }
    };
	
	class TKCM_Cpp : public TKCM {
    public:
			using TKCM::TKCM;

            static py::array_t<double> doTKCM(py::array_t<double> &mx, uint64_t trunc){
                // Convert to arma
                arma::mat mx_arma = carma::arr_to_mat<double>(mx);
				
                // Call
                arma::mat mx_arma_new = TKCM::doTKCM(mx_arma, trunc);

                // Convert back to numpy
                py::array_t<double> mx_final = carma::mat_to_arr<double>(mx_arma_new);

                return mx_final;
            }
	};
	
	class SVT_Cpp : public SVT {
    public:
            using SVT::SVT;

            static py::array_t<double> doSVT(py::array_t<double> &X, double tauScale){
                // Convert to arma
                arma::mat X_arma = carma::arr_to_mat<double>(X);

                // Call
                arma::mat X_arma_new = SVT::doSVT(X_arma, tauScale);

                // Convert back to numpy
                py::array_t<double> X_final = carma::mat_to_arr<double>(X_arma_new);

                return X_final;
            }
    };
	
	class SPIRIT_Cpp : public SPIRIT {
    public:
            using SPIRIT::SPIRIT;

            static py::array_t<double> doSpirit(py::array_t<double> &A, uint64_t k0, uint64_t w, double lambda){
                // Convert to arma
                arma::mat A_arma = carma::arr_to_mat<double>(A);

                // Call
                arma::mat A_arma_new = SPIRIT::doSpirit(A_arma, k0, w, lambda);

                // Convert back to numpy
                py::array_t<double> A_final = carma::mat_to_arr<double>(A_arma_new);

                return A_final;
            }
    };
	
	class MeanImpute_Cpp : public MeanImpute {
    public:
            using MeanImpute::MeanImpute;

            static py::array_t<double> doMeanImpute(py::array_t<double> &input){
                // Convert to arma
                arma::mat input_arma = carma::arr_to_mat<double>(input);

                // Call
                arma::mat input_arma_new = MeanImpute::MeanImpute_Recovery(input_arma);

                // Convert back to numpy
                py::array_t<double> input_final = carma::mat_to_arr<double>(input_arma_new);

                return input_final;
            }
    };
	
	class ZeroImpute_Cpp : public ZeroImpute {
    public:
            using ZeroImpute::ZeroImpute;

            static py::array_t<double> doZeroImpute(py::array_t<double> &input){
                // Convert to arma
                arma::mat input_arma = carma::arr_to_mat<double>(input);

                // Call
                arma::mat input_arma_new = ZeroImpute::ZeroImpute_Recovery(input_arma);

                // Convert back to numpy
                py::array_t<double> input_final = carma::mat_to_arr<double>(input_arma_new);

                return input_final;
            }
    };
	
	class LinearImpute_Cpp : public LinearImpute {
    public:
            using LinearImpute::LinearImpute;

            static py::array_t<double> doLinearImpute(py::array_t<double> &input){
                // Convert to arma
                arma::mat input_arma = carma::arr_to_mat<double>(input);

                // Call
                arma::mat input_arma_new = LinearImpute::LinearImpute_Recovery(input_arma);

                // Convert back to numpy
                py::array_t<double> input_final = carma::mat_to_arr<double>(input_arma_new);

                return input_final;
            }
    };
	
	class CDMissingValueRecovery_Cpp : public CDMissingValueRecovery {
    public:
            using CDMissingValueRecovery::CDMissingValueRecovery;

            static py::array_t<double> doCDMissingValueRecovery(py::array_t<double> &matrix, uint64_t truncation = 0, double eps = 1E-6){
                // Convert to arma
                arma::mat matrix_arma = carma::arr_to_mat<double>(matrix);

                // Call
                arma::mat matrix_arma_new = CDMissingValueRecovery::RecoverMatrix(matrix_arma, truncation, eps);

                // Convert back to numpy
                py::array_t<double> matrix_final = carma::mat_to_arr<double>(matrix_arma_new);

                return matrix_final;
            }
    };
		
	class GROUSE_Cpp : public GROUSE {
    public:
            using GROUSE::GROUSE;

            static py::array_t<double> doGROUSE(py::array_t<double> &input, uint64_t maxrank){
                // Convert to arma
                arma::mat input_arma = carma::arr_to_mat<double>(input);

                // Call
                arma::mat input_arma_new = GROUSE::doGROUSE(input_arma, maxrank);

                // Convert back to numpy
                py::array_t<double> input_final = carma::mat_to_arr<double>(input_arma_new);

                return input_final;
            }
    };
			
	class ROSL_Cpp : public ROSL {
    public:
            using ROSL::ROSL;

            static py::array_t<double> doROSL(py::array_t<double> &input, uint64_t rank, double reg){
                // Convert to arma
                arma::mat input_arma = carma::arr_to_mat<double>(input);

                // Call
                arma::mat input_arma_new = ROSL::ROSL_Recovery(input_arma, rank, reg);

                // Convert back to numpy
                py::array_t<double> input_final = carma::mat_to_arr<double>(input_arma_new);

                return input_final;
            }
    };
				
	class DynaMMo_Cpp : public DynaMMo {
    public:
            using DynaMMo::DynaMMo;

            static py::array_t<double> doDynaMMo(py::array_t<double> &X, uint64_t H = 0, uint64_t maxIter = 100, bool FAST = false){
                // Convert to arma
                arma::mat X_arma = carma::arr_to_mat<double>(X);

                // Call
                arma::mat X_arma_new = DynaMMo::doDynaMMo(X_arma, H, maxIter, FAST);

                // Convert back to numpy
                py::array_t<double> X_final = carma::mat_to_arr<double>(X_arma_new);

                return X_final;
            }
    };
	
	class PCA_MME_Cpp : public PCA_MME {
    public:
            using PCA_MME::PCA_MME;

            static py::array_t<double> doPCA_MME(py::array_t<double> &matrix, uint64_t truncation, bool singleBlock){
                // Convert to arma
                arma::mat matrix_arma = carma::arr_to_mat<double>(matrix);

                // Call
                arma::mat matrix_arma_new = PCA_MME::doPCA_MME(matrix_arma, truncation, singleBlock);

                // Convert back to numpy
                py::array_t<double> matrix_final = carma::mat_to_arr<double>(matrix_arma_new);

                return matrix_final;
            }
    };
		
	class OGDImpute_Cpp : public OGDImpute {
    public:
            using OGDImpute::OGDImpute;

            static py::array_t<double> doOGDImpute(py::array_t<double> &matrix, uint64_t truncation){
                // Convert to arma
                arma::mat matrix_arma = carma::arr_to_mat<double>(matrix);

                // Call
                arma::mat matrix_arma_new = OGDImpute::doOGDImpute(matrix_arma, truncation);

                // Convert back to numpy
                py::array_t<double> matrix_final = carma::mat_to_arr<double>(matrix_arma_new);

                return matrix_final;
            }
    };
}

void init_algorithms(py::module &m) {
    py::class_<Algorithms::IterativeSVD>(m, "IterativeSVD_Parent");
    py::class_<Algorithms::IterativeSVD_Cpp, Algorithms::IterativeSVD>(m, "IterativeSVD")
    .def_static("doIterativeSVD",
        py::overload_cast<py::array_t<double>&, uint64_t>( &Algorithms::IterativeSVD_Cpp::doIterativeSVD),
        py::arg("X"), py::arg("rank"));
		
		
	py::class_<Algorithms::NMFMissingValueRecovery>(m, "NMFMissingValueRecovery_Parent");
    py::class_<Algorithms::NMFMissingValueRecovery_Cpp, Algorithms::NMFMissingValueRecovery>(m, "NMFMissingValueRecovery")
    .def_static("doNMFRecovery",
        py::overload_cast<py::array_t<double>&, uint64_t>( &Algorithms::NMFMissingValueRecovery_Cpp::doNMFRecovery),
        py::arg("input"), py::arg("truncation"));
		
		
	py::class_<Algorithms::SoftImpute>(m, "SoftImpute_Parent");
    py::class_<Algorithms::SoftImpute_Cpp, Algorithms::SoftImpute>(m, "SoftImpute")
    .def_static("doSoftImpute",
        py::overload_cast<py::array_t<double>&, uint64_t>( &Algorithms::SoftImpute_Cpp::doSoftImpute),
        py::arg("X"), py::arg("max_rank"));
		
		
	py::class_<Algorithms::TKCM>(m, "TKCM_Parent");
    py::class_<Algorithms::TKCM_Cpp, Algorithms::TKCM>(m, "TKCM")
    .def_static("doTKCM",
        py::overload_cast<py::array_t<double>&, uint64_t>( &Algorithms::TKCM_Cpp::doTKCM),
        py::arg("mx"), py::arg("trunc"));	
		
		
	py::class_<Algorithms::SVT>(m, "SVT_Parent");
    py::class_<Algorithms::SVT_Cpp, Algorithms::SVT>(m, "SVT")
    .def_static("doSVT",
        py::overload_cast<py::array_t<double>&, double>( &Algorithms::SVT_Cpp::doSVT),
        py::arg("X"), py::arg("tauScale"));		
		
		
	py::class_<Algorithms::SPIRIT>(m, "Spirit_Parent");
    py::class_<Algorithms::SPIRIT_Cpp, Algorithms::SPIRIT>(m, "Spirit")
    .def_static("doSpirit",
        py::overload_cast<py::array_t<double>&, uint64_t, uint64_t, double>( &Algorithms::SPIRIT_Cpp::doSpirit),
        py::arg("A"), py::arg("k0"), py::arg("w"), py::arg("lambda"));	
						
		
	py::class_<Algorithms::ZeroImpute>(m, "ZeroImpute_Parent");
    py::class_<Algorithms::ZeroImpute_Cpp, Algorithms::ZeroImpute>(m, "ZeroImpute")
    .def_static("doZeroImpute",
        py::overload_cast<py::array_t<double>&>( &Algorithms::ZeroImpute_Cpp::doZeroImpute),
        py::arg("input"));
						
		
	py::class_<Algorithms::MeanImpute>(m, "MeanImpute_Parent");
    py::class_<Algorithms::MeanImpute_Cpp, Algorithms::MeanImpute>(m, "MeanImpute")
    .def_static("doMeanImpute",
        py::overload_cast<py::array_t<double>&>( &Algorithms::MeanImpute_Cpp::doMeanImpute),
        py::arg("input"));
								
		
	py::class_<Algorithms::LinearImpute>(m, "LinearImpute_Parent");
    py::class_<Algorithms::LinearImpute_Cpp, Algorithms::LinearImpute>(m, "LinearImpute")
    .def_static("doLinearImpute",
        py::overload_cast<py::array_t<double>&>( &Algorithms::LinearImpute_Cpp::doLinearImpute),
        py::arg("input"));
										
		
	py::class_<Algorithms::CDMissingValueRecovery>(m, "CDMissingValueRecovery_Parent");
    py::class_<Algorithms::CDMissingValueRecovery_Cpp, Algorithms::CDMissingValueRecovery>(m, "CDMissingValueRecovery")
    .def_static("doCDMissingValueRecovery",
        py::overload_cast<py::array_t<double>&, uint64_t, double>( &Algorithms::CDMissingValueRecovery_Cpp::doCDMissingValueRecovery),
        py::arg("matrix"), py::arg("truncation"), py::arg("eps"));
		
												
	py::class_<Algorithms::GROUSE>(m, "GROUSE_Parent");
    py::class_<Algorithms::GROUSE_Cpp, Algorithms::GROUSE>(m, "GROUSE")
    .def_static("doGROUSE",
        py::overload_cast<py::array_t<double>&, uint64_t>( &Algorithms::GROUSE_Cpp::doGROUSE),
        py::arg("input"), py::arg("maxrank"));
												
		
	py::class_<Algorithms::ROSL>(m, "ROSL_Parent");
    py::class_<Algorithms::ROSL_Cpp, Algorithms::ROSL>(m, "ROSL")
    .def_static("doROSL",
        py::overload_cast<py::array_t<double>&, uint64_t, double>( &Algorithms::ROSL_Cpp::doROSL),
        py::arg("input"), py::arg("rank"), py::arg("reg"));
														
		
	py::class_<Algorithms::DynaMMo>(m, "DynaMMo_Parent");
    py::class_<Algorithms::DynaMMo_Cpp, Algorithms::DynaMMo>(m, "DynaMMo")
    .def_static("doDynaMMo",
        py::overload_cast<py::array_t<double>&, uint64_t, uint64_t, bool>( &Algorithms::DynaMMo_Cpp::doDynaMMo),
        py::arg("X"), py::arg("H"), py::arg("maxIter"), py::arg("FAST"));	
														
		
	py::class_<Algorithms::PCA_MME>(m, "PCA_MME_Parent");
    py::class_<Algorithms::PCA_MME_Cpp, Algorithms::PCA_MME>(m, "PCA_MME")
    .def_static("doPCA_MME",
        py::overload_cast<py::array_t<double>&, uint64_t, bool>( &Algorithms::PCA_MME_Cpp::doPCA_MME),
        py::arg("matrix"), py::arg("truncation"), py::arg("singleBlock"));
																
		
	py::class_<Algorithms::OGDImpute>(m, "OGDImpute_Parent");
    py::class_<Algorithms::OGDImpute_Cpp, Algorithms::OGDImpute>(m, "OGDImpute")
    .def_static("doOGDImpute",
        py::overload_cast<py::array_t<double>&, uint64_t>( &Algorithms::OGDImpute_Cpp::doOGDImpute),
        py::arg("matrix"), py::arg("truncation"));
}
