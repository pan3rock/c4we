#include "statistics.hpp"

extern "C" {

  void calculate_c4y(const double *data_c,
                     const int nt,
                     const int nchannel,
                     const int num_c4,
                     const int nx,
                     double *c4y_c)
  {
    arma::mat data(data_c, nt, nchannel);
    arma::mat c4y(c4y_c, num_c4, nchannel, false);
    for (int i = 0; i < nchannel; ++i) {
      arma::vec trace = data.col(i);
      trace -= arma::mean(trace);
      arma::vec cum4(num_c4, arma::fill::zeros);
      cumulants_4th(trace, nx, num_c4, cum4);
      c4y.col(i) = cum4;
    }
  }


  double fitness(const double *c4y_c,
                 const int num_c4,
                 const int nchannel,
                 const double *x_c,
                 const int nx)
  {
    arma::mat c4y(c4y_c, num_c4, nchannel);
    arma::vec x(x_c, nx);

    arma::vec m4h(num_c4, arma::fill::zeros);
    moment_function_4th(x, nx, m4h);

    double misfit = 0;
    for (int i = 0; i < nchannel; ++i) {
      misfit += arma::sum(arma::pow(c4y.col(i) - m4h, 2));
    }
    return misfit;
  }

  void gradient(const double *c4y_c,
                const int num_c4,
                const int nchannel,
                const double *x_c,
                const int nx,
                double *grad_c)
  {
    arma::mat c4y(c4y_c, num_c4, nchannel);
    arma::vec x(x_c, nx);
    arma::vec grad(grad_c, nx, false);

    arma::vec m4h(num_c4, arma::fill::zeros);
    moment_function_4th(x, nx, m4h);

    arma::mat jac(nx, num_c4, arma::fill::zeros);
    int idx = 0;
    for (int k = 0; k < nx; ++k) {
      for (int j = 0; j <= k; ++j) {
        for (int i = 0; i <= j; ++i) {
          if (i == 0 && k > 0 && k == j) continue;

          arma::vec sums(nx, arma::fill::zeros);
          for (int l = 0; l < nx - k; ++l)
            sums(l) += x(l + i) * x(l + j) * x(l + k);
          for (int l = i; l < nx - k + i; ++l)
            sums(l) += x(l - i) * x(l - i + j) * x(l - i + k);
          for (int l = j; l < nx - k + j; ++l)
            sums(l) += x(l - j) * x(l - j + i) * x(l - j + k);
          for (int l = k; l < nx; ++l)
            sums(l) += x(l - k) * x(l - k + i) * x(l - k + j);

          jac.col(idx++) = sums;
        }
      }
    }

    for (int i = 0; i < nchannel; ++i) {
      arma::vec error = m4h - c4y.col(i);
      grad += 2. * jac * error;
    }
  }

}
