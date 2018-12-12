#include "problem.hpp"
#include "statistics.hpp"

#include <string>
#include <armadillo>
#include <yaml-cpp/yaml.h>


Problem::Problem(const std::string &file_config)
{
  YAML::Node config = YAML::LoadFile(file_config);
  nchannel_ = config["nchannel"].as<int>();
  nx_ = config["nx"].as<int>();
  std::string file_data = config["file_data"].as<std::string>();

  // random x0
  arma::arma_rng::set_seed_random();
  x0_ = 2. * (arma::randu<arma::vec>(nx_) - 0.5);

  // (t1, t2, t3) -> idx
  int idx = 0;
  for (int k = 0; k < nx_; ++k) {
    for (int j = 0; j <= k; ++j) {
      for (int i = 0; i <= j; ++i) {
        if (i == 0 && k > 0 && k == j) continue;
        idx++;
      }
    }
  }
  num_c4_ = idx;

  c4y_.set_size(num_c4_, nchannel_);

  // load traces
  // arma::mat data(nx_, nchannel_, arma::fill::zeros);
  arma::mat data;
  data.load(file_data);
  std::cout << data.n_rows << " " << data.n_cols << std::endl;

  // fourth-order cumulants of records
  for (int i = 0; i < nchannel_; ++i) {
    arma::vec trace = data.col(i);
    trace -= arma::mean(trace);
    arma::vec cum4(num_c4_, arma::fill::zeros);
    cumulants_4th(trace, nx_, num_c4_, cum4);
    c4y_.col(i) = cum4;
  }
}

arma::vec Problem::x0() const
{
  return x0_;
}


double Problem::fitness(const arma::vec &x)
{
  arma::vec m4h(num_c4_, arma::fill::zeros);
  moment_function_4th(x, nx_, m4h);

  double misfit = 0;
  for (int i = 0; i < nchannel_; ++i) {
    misfit += arma::sum(arma::pow(c4y_.col(i) - m4h, 2));
  }
  return misfit;
}


arma::vec Problem::gradient(const arma::vec &x)
{
  arma::vec m4h(num_c4_, arma::fill::zeros);
  moment_function_4th(x, nx_, m4h);

  arma::mat jac(nx_, num_c4_, arma::fill::zeros);
  int idx = 0;
  for (int k = 0; k < nx_; ++k) {
    for (int j = 0; j <= k; ++j) {
      for (int i = 0; i <= j; ++i) {
        if (i == 0 && k > 0 && k == j) continue;

        arma::vec sums(nx_, arma::fill::zeros);
        for (int l = 0; l < nx_ - k; ++l)
          sums(l) += x(l + i) * x(l + j) * x(l + k);
        for (int l = i; l < nx_ - k + i; ++l)
          sums(l) += x(l - i) * x(l - i + j) * x(l - i + k);
        for (int l = j; l < nx_ - k + j; ++l)
          sums(l) += x(l - j) * x(l - j + i) * x(l - j + k);
        for (int l = k; l < nx_; ++l)
          sums(l) += x(l - k) * x(l - k + i) * x(l - k + j);

        jac.col(idx++) = sums;
      }
    }
  }

  arma::vec grad(nx_, arma::fill::zeros);
  for (int i = 0; i < nchannel_; ++i) {
    arma::vec error = m4h - c4y_.col(i);
    grad += 2. * jac * error;
  }

  return grad;

}
