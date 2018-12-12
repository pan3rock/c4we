#ifndef STASTISTICS_H_
#define STASTISTICS_H_

#include <armadillo>

void moment_function_4th(const arma::vec &seq,
                         const int width,
                         arma::vec &momfunc4);

void cumulants_2nd(const arma::vec &seq,
                   const int width,
                   arma::vec &cum2);

void cumulants_4th(const arma::vec &seq,
                   const int n2,
                   const int n4,
                   arma::vec &cum4);

#endif
