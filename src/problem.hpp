#ifndef PROBLEM_H_
#define PROBLEM_H_

#include <string>
#include <armadillo>

class Problem {
public:
  Problem(const std::string &file_config);
  double fitness(const arma::vec &x);
  arma::vec gradient(const arma::vec &x);
  arma::vec x0() const;
private:
  int nchannel_;
  int nx_;
  int num_c4_;
  arma::mat c4y_;
  arma::vec x0_;
};

#endif
