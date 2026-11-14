#ifndef CTQMC_1D_H
#define CTQMC_1D_H

#include <string>
#include <tuple>
#include <vector>
#include <armadillo>

// Physical and MC parameters (defined in ctqmc_1d.cpp)
extern double mass, hbar, k_b, lam, V0, T, beta, l;
extern int numMCsteps, numEquil, numObsInterval;
extern double Z0, NegDervZ0;

// Observable traces of the most recent CTQMC_MarkovChain() run
extern std::vector<double> last_num_trace;
extern std::vector<int> last_order_trace;

// Fix the Armadillo RNG seed used by CTQMC_MarkovChain (default: random)
void set_seed(unsigned seed);
void set_seed_random();

std::tuple<arma::vec, arma::vec> ABs(arma::vec all_ts, int order);
arma::vec GetAll_ts(arma::vec all_taus, int order);
arma::mat Make_M_matrix(arma::vec all_Bs, int order);
double DenConfigWeight(arma::vec all_As, arma::mat M, int order);
double PhiConfigWeight(arma::vec taus, int order);
double PsiConfigWeight(arma::vec all_ts, arma::vec all_As, arma::vec all_Bs, arma::mat M, int order);
std::tuple<double,double> NumConfigWeight(arma::vec taus, arma::mat M, int order, arma::vec all_ts, arma::vec all_As, arma::vec all_Bs);
arma::vec GetNewTaus(arma::vec taus, int order_old, std::string MCmove);
std::tuple<int,arma::vec,arma::vec,arma::vec,arma::vec,arma::mat,double> GetNewFeatures(arma::vec taus_old, int order_old, int order_new, std::string MCmove);
std::tuple<int,arma::vec,arma::vec,arma::vec,arma::vec,arma::mat,double> PackNewFeatures(arma::vec taus_old, int order_old, std::string MCmove);
bool MCMoveDecision(int order_old, int order_new, double configweight_old, double configweight_new, std::string MCmove);
void CTQMC_MarkovChain();
int run_simulation();

#endif
