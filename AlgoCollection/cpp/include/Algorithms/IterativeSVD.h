//
// Created on 21/01/19.
//

#pragma once

#include <armadillo>

namespace Algorithms
{

class IterativeSVD
{
  public:
    static arma::mat recoveryIterativeSVD(arma::mat &X, uint64_t rank);
};

} // namespace Algorithms