#include "statistics.hpp"

#include <armadillo>


void moment_function_4th(const arma::vec &seq,
                         const int width,
                         arma::vec &momfunc4)
{
  int num_seq = seq.n_rows;
  int idx = 0;
  for (int k = 0; k < width; ++k) {
    for (int j = 0; j <= k; ++j) {
      for (int i = 0; i <= j; ++i) {
        if (i == 0 && k > 0 && k == j) continue;
        double sum = 0;
        for (int n = 0; n < num_seq - k; ++n) {
          sum += seq(n) * seq(n + i) * seq(n + j) * seq(n + k);
        }
        momfunc4(idx) += sum;
        idx++;
      }
    }
  }
}

void cumulants_2nd(const arma::vec &seq,
                   const int width,
                   arma::vec &cum2)
{
  for (int i = 0; i < width; ++i) {
    double sum = 0.;
    for (int j = 0; j <= seq.n_elem - width; ++j) {
      sum += seq(j) * seq(j + i);
    }
    cum2(i) += sum;
  }
}


void cumulants_4th(const arma::vec &seq,
                   const int n2,
                   const int n4,
                   arma::vec &cum4)
{
  int num_seq = seq.n_rows;
  arma::vec cum2(n2, arma::fill::zeros);
  cumulants_2nd(seq, n2, cum2);
  double pwr = cum2(0) / num_seq;
  double scale2 = 1.0 / cum2(0);
  cum2 *= scale2;

  arma::vec mfunc4(n4, arma::fill::zeros);
  moment_function_4th(seq, n2, mfunc4);
  double scale4 = 1.0 / (num_seq * std::pow(pwr, 2));

  int idx = 0;
  for (int k = 0; k < n2; ++k) {
    for (int j = 0; j <= k; ++j) {
      for (int i = 0; i <= j; ++i) {
        if (i == 0 && k > 0 && k == j) continue;
        double m4g = cum2(i) * cum2(k - j) + cum2(j) * cum2(k - i)
          + cum2(k) * cum2(j - i);
        cum4(idx) += (scale4 * mfunc4(idx) - m4g) * (n2 - k) / n2;

        idx++;
      }
    }
  }
}
