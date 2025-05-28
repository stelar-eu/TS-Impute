//
// Created on 05.09.19.
//

#pragma once

#include <armadillo>

namespace Algorithms
{

class PCA_MME
{
  private:
    arma::mat &input;
    uint64_t m;
    uint64_t n;
    uint64_t t;
    
    double C;
    double delta;
    uint64_t k;
    uint64_t order;
    uint64_t id;
    bool done;
    
    uint64_t T;
    uint64_t B;
    std::vector<uint64_t> blocks;
    
    uint64_t nextBlock;
    arma::mat Q;
    arma::mat Qnew;
  
  public:
    explicit PCA_MME(arma::mat &_input, uint64_t _k, bool singleBlock);
  
  public:
    void runPCA_MME();
    void streamPCA_MME();
  
  private:
    void _update(const arma::vec &sample);
    void _orthonormalize();
    arma::vec _getNextSample();
	
  //
  // Static
  //
  public:
    static arma::mat doPCA_MME(arma::mat &matrix, uint64_t truncation, bool singleBlock);
};

} // namespace Algorithms


