#include "problem.hpp"

#include "optim/optim.hpp"

#include <functional>
#include <memory>
#include <iostream>

double c4we_fn(std::shared_ptr<Problem> prob,
               const arma::vec &x,
               arma::vec *grad_out,
               void *opt_data)
{
  double obj_val = prob->fitness(x);
  if (grad_out) {
    *grad_out = prob->gradient(x);
  }
  return obj_val;
}

int main(int argc, char *argv[])
{
  using namespace std::placeholders;
  std::shared_ptr<Problem> prob = std::make_shared<Problem>("config.yml");
  arma::vec x = prob->x0();
  x.save("x0.txt", arma::raw_ascii);
  auto func_c4we = std::bind(c4we_fn, prob, _1, _2, _3);

  arma::vec grad(arma::size(x));
  func_c4we(x, &grad, nullptr);
  arma::cout << grad << arma::endl;

  bool success = optim::lbfgs(x, func_c4we, nullptr);

  if (success) {
    std::cout << "inversion is successful." << std::endl;
  } else {
    std::cout << "inversion fails." << std::endl;
  }

  std::cout << "solution: " << std::endl;
  arma::cout << x << arma::endl;

  x.save("x.txt", arma::raw_ascii);
}
