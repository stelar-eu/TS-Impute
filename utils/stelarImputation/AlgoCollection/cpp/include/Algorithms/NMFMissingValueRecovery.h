//
// Created on 14.01.19.
//

#pragma once

#include <mlpack/core.hpp>

namespace Algorithms
{

class NMFMissingValueRecovery
{
  public:
    static arma::mat doNMFRecovery(arma::mat &matrix, uint64_t truncation);
};
} // namespace Algorithms
