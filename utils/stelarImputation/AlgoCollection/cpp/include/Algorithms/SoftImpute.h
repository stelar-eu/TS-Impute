//
// Created on 21/01/19.
//

#pragma once

#include <armadillo>

namespace Algorithms
{

class SoftImpute
{
  public:
    static arma::mat doSoftImpute(arma::mat &X, uint64_t max_rank);
};

} // namespace Algorithms
