#pragma once

#include <armadillo>

namespace Algorithms
{

class MeanImpute
{
  public:
    static arma::mat MeanImpute_Recovery(arma::mat &input);
    
  // other function signatyures go here
};

} // namespace Algorithms