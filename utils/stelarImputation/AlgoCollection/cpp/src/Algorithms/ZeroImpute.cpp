#include <iostream>
#include "../../include/Algorithms/ZeroImpute.h"

namespace Algorithms
{

arma::mat ZeroImpute::ZeroImpute_Recovery(arma::mat &input)
{
	// Copy state 
    arma::mat input_new = input;
	
       for (uint64_t j = 0; j < input_new.n_cols; ++j)
    {
        for (uint64_t i = 0; i < input_new.n_rows; ++i)
        {
            if (!arma::is_finite(input_new(i, j)))
            {
                input_new(i, j) = 0.0;
            }
        }
    }

	return input_new;
}    

// any other functions go here

} // namespace Algorithms
