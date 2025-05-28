#include <iostream>
#include "../../include/Algorithms/MeanImpute.h"

namespace Algorithms
{

arma::mat MeanImpute::MeanImpute_Recovery(arma::mat &input)
{
	// Copy state 
    arma::mat input_new = input;
	
    arma::vec mean = arma::zeros<arma::vec>(input_new.n_cols);
    arma::uvec values = arma::zeros<arma::uvec>(input_new.n_cols);
    for (uint64_t j = 0; j < input_new.n_cols; ++j)
    {
        for (uint64_t i = 0; i < input_new.n_rows; i++)
        {
            if (arma::is_finite(input_new(i, j)))
            {
                mean[j] += input_new(i, j);
                values[j]++;
            }
        }
        
        if (values[j] == 0) mean[j] = 0.0; // full column is missing, impute with 0
        else mean[j] /= (double)values[j];
    }
    for (uint64_t j = 0; j < input_new.n_cols; ++j)
    {
        // nothing is missing
        if (values[j] == input_new.n_rows) continue;
        for (uint64_t i = 0; i < input_new.n_rows; i++)
        {
            if (!arma::is_finite(input_new(i, j))) input_new(i, j) = mean[j];
        }
    }
	
	return input_new;
}

// any other functions go here

} // namespace Algorithms