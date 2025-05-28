#pragma once

#include <armadillo>

namespace Algorithms
{

class LinearImpute
{
  public:
    static arma::mat LinearImpute_Recovery(arma::mat &input);
    
  // other function signatyures go here
};

} // namespace Algorithms