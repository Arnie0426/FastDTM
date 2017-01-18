//
// Created by Arnie on 2016-11-15.
//

#include "dtm.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sstream>
#include <fstream>
#include <utility>
#include <functional>
#include <algorithm>
#include <cmath>
using namespace std;

VectorXf get_mvn_samples(VectorXf mean, MatrixXf cov) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigenSolver(cov);
  normal_distribution<float> dist(0, 1);
  default_random_engine gen;
  auto std_norm = [&] (float) {return dist(gen);};
  return mean + eigenSolver.eigenvectors() * eigenSolver
      .eigenvalues().cwiseSqrt().asDiagonal() *
      VectorXf::NullaryExpr(mean.size(), std_norm);
}

static float gaussianVar(float mean, float std_dev) {
  default_random_engine generator;
  normal_distribution<float> distribution(mean, std_dev);
  return distribution(generator);
}

static VectorXf softmax(VectorXf weights) {
  size_t K = weights.size();
  VectorXf soft(K);
  float MAX = weights[0];
  float norm = 0.0;
  for (size_t i = 0; i < K; i++) {
    if (weights(i) > MAX)
      MAX = weights(i);
  }

  for (size_t i = 0; i < K; i++)   {
    norm += exp(weights(i) - MAX);
  }

  for (size_t i = 0; i < K; i++) {
    soft(i) = exp(weights(i) - MAX) / (norm + 0.00001);
    if (soft(i) < 1e-30) {
      soft(i) = 0.0;
    }
  }

  return soft;
}

void DTM::build_alias_table(size_t t, size_t w) {
  AliasSamples term(phi[t].row(w));
  term_alias_samples[t][w] = term.get_samples(K);
}

DTM::DTM(const vector<vector<vector<size_t>>> &data, const vector<string> &
dictionary,
         size_t num_topics, float sgld_a, float sgld_b, float sgld_c,
         float dtm_phi_var, float dtm_eta_var, float dtm_alpha_var) : W(data),
         vocabulary(dictionary), K(num_topics), sgld_a(sgld_a), sgld_b(sgld_b),
         sgld_c(sgld_c), dtm_phi_var(dtm_phi_var), dtm_eta_var(dtm_eta_var),
         dtm_alpha_var(dtm_alpha_var) {
  V = vocabulary.size();
  T = W.size();
  D = vector<size_t>(T);
  Z = vector<vector<vector<size_t>>>(T);
  term_alias_samples =
      vector<vector<vector<size_t>>>
          (T, vector<vector<size_t>> (V, vector<size_t>(K)));
  sample_indices = vector<vector<size_t>>(T, vector<size_t>(V));


  CDK = vector<MatrixXf> (T);
  CWK = vector<MatrixXf> (T, MatrixXf::Zero(V, K));
  CK = MatrixXf::Zero(T, K);
  phi = vector<MatrixXf>(T, MatrixXf::Zero(V, K));
  eta = vector<MatrixXf>(T);
  alpha = MatrixXf::Zero(T, K);

  for (size_t t = 0; t < T; ++t) {
    D[t] = W[t].size();
    Z[t] = vector<vector<size_t>>(D[t]);
    eta[t] = MatrixXf::Zero(D[t], K);
    CDK[t] = MatrixXf::Zero(D[t], K);
    for (size_t d = 0; d < D[t]; ++d) {
      Z[t][d] = vector<size_t>(W[t][d].size());
    }
  }
}

void DTM::initialize(bool init_with_lda) {
  default_random_engine generator;
  uniform_int_distribution<size_t> uniform_topic(0, K - 1);
  u01 = uniform_real_distribution<float>(0, 1);
  float init_alpha = 50.0 / K;
  float init_beta = 0.01;
  for (size_t t = 0; t < T; ++t) {
    for (size_t d = 0; d < D[t]; ++d) {
      size_t N = W[t][d].size();
      for (size_t n = 0; n < N; ++n) {
        size_t w = W[t][d][n];
        size_t k = uniform_topic(generator);
        Z[t][d][n] = k;
        CDK[t](d, k)++;
        CWK[t](w, k)++;
        CK(t, k)++;
        eta[t](d, k) += (1 + init_alpha) / (N + K * init_alpha);
      }
    }
  }

  if (init_with_lda) {
    size_t t = 0;
    for (size_t iter = 0; iter < 50; iter++) {
      cout << "LDA Iter: " << iter << endl;
      for (size_t d = 0; d < D[t]; ++d) {
        for (size_t n = 0; n < W[t][d].size(); n++) {
          size_t k = Z[t][d][n];
          size_t w = W[t][d][n];
          CDK[t](d, k)--;
          CWK[t](w, k)--;
          CK(t, k)--;

          vector<float> prob(K);
          for (k = 0; k < K; k++) {
            prob[k] = (CDK[t](d, k) + init_alpha) * ((CWK[t](w, k) + init_beta)
                / (CK(t, k) + V * init_beta));
          }
          discrete_distribution<size_t> mult(prob.begin(), prob.end());
          k = mult(generator);
          Z[t][d][n] = k;
          CDK[t](d, k)++;
          CWK[t](w, k)++;
          CK(t, k)++;
        }
      }
    }
  }
  for (size_t t = 0; t < T; t++) {
    for (size_t w = 0; w < V; w++) {
      for (size_t k = 0; k < K; k++) {
        phi[t](w, k) = (CWK[0](w, k) + init_beta) / (CK(0, k) + V * init_beta);
      }
      build_alias_table(t, w);
    }
  }
}

void DTM::estimate(size_t num_iters) {
  default_random_engine generator;
  for (size_t iter = 0; iter < num_iters; iter++) {
    cout << "Iteration " << iter << endl;
    float eps = sgld_a * (pow(sgld_b + iter, -sgld_c));
    float xi = gaussianVar(0.0, pow(eps, 2));
    VectorXf xi_vec;
    VectorXf mean(K);
    for (size_t t = 0; t < T; t++) {
      xi_vec = VectorXf::Constant(K, xi);
      for (size_t d = 0; d < D[t]; d++) {
        size_t N = W[t][d].size();
        uniform_int_distribution<size_t> doc_dist(0, N - 1);

        // estimate eta
        VectorXf soft_eta = softmax(eta[t].row(d));
        VectorXf prior_eta = (alpha.row(t) - eta[t].row(d)) / dtm_eta_var;
        VectorXf denom_eta = N * soft_eta;
        VectorXf grad_eta = CDK[t].row(d).transpose() - denom_eta;
        eta[t].row(d) += ((eps / 2) * (grad_eta + prior_eta)) + xi_vec;


        for (size_t n = 0; n < N; n++) {
          for (size_t mh = 0; mh < 4; mh++) {
            size_t k = Z[t][d][n];
            size_t w = W[t][d][n];
            CDK[t](d, k)--;
            CWK[t](w, k)--;
            CK(t, k)--;

            size_t proposal;
            float acceptance_prob = 0.0;
            if (mh % 2 == 0) {
              // Z-proposal
              size_t index = doc_dist(generator);
              proposal = Z[t][d][index];

              acceptance_prob =
                  exp(phi[t](w, proposal)) / exp(phi[t](w, k));
            } else {
              if (sample_indices[t][w] >= K) {
                build_alias_table(t, w);
                sample_indices[t][w] = 0;
              }
              proposal = term_alias_samples[t][w][sample_indices[t][w]];
              sample_indices[t][w]++;
              acceptance_prob = exp(eta[t](d, proposal)) / exp(eta[t](d, k));
            }
            acceptance_prob = acceptance_prob > 1.0 ? 1.0 : acceptance_prob;
            if (u01(generator) >= acceptance_prob) {
              // reject proposal
              proposal = k;
            }
            Z[t][d][n] = proposal;
            CDK[t](d, proposal)++;
            CWK[t](w, proposal)++;
            CK(t, proposal)++;
          }
        }
      }

      xi_vec = VectorXf::Constant(V, xi);
      for (unsigned k = 0; k < K; ++k) {
        // sample phi
        VectorXf soft_phi = softmax(phi[t].col(k));
        VectorXf prior_phi(V);
        if (t == 0) {
          float phi_sigma = 1.0 / ((1.0 / 100) + (1 / dtm_phi_var));
          prior_phi = phi[t + 1].col(k) * (phi_sigma / dtm_phi_var);
          prior_phi = ((2 * prior_phi) - 2 * phi[t].col(k)) / dtm_phi_var;
        } else if (t == T - 1) {
          prior_phi = (phi[t - 1].col(k) - phi[t].col(k)) / dtm_phi_var;
        } else {
          prior_phi = (phi[t + 1].col(k) + phi[t - 1].col(k) - 2 * phi[t].col
              (k)) / dtm_phi_var;
        }

        VectorXf denom_phi = CK(t, k) * soft_phi;
        VectorXf grad_phi = CWK[t].col(k) - denom_phi;

        phi[t].col(k) += ((eps / 2) * (grad_phi + prior_phi)) + xi_vec;
      }

      // sample alpha
      VectorXf alpha_bar(K);
      float alpha_precision = 0.0;  // designed to be a diagonal matrix
      MatrixXf cov = MatrixXf::Identity(K, K);
      if (t == 0) {
        alpha_precision = (1.0 / 100) + (1 / dtm_alpha_var);
        float alpha_sigma = 1.0 / alpha_precision;
        alpha_bar = alpha.row(t+1) * (alpha_sigma / dtm_alpha_var);
      } else if (t == T-1) {
        alpha_bar = (alpha.row(t-1) - alpha.row(t)) / dtm_alpha_var;
        alpha_precision = 1.0 / dtm_alpha_var;
      } else {
        alpha_precision = (2 / dtm_alpha_var);
        alpha_bar = (alpha.row(t+1) - alpha.row(t-1)) / 2;
      }
      VectorXf eta_bar = eta[t].colwise().sum();
      float sigma = 1.0 / (1.0 / alpha_precision + (D[t] / dtm_eta_var));
      cov *= sigma;
      mean = (alpha_bar / alpha_precision + (eta_bar / dtm_eta_var)) * sigma;
      alpha.row(t) = get_mvn_samples(mean, cov);
      if (iter % 5 == 0) {
        diagnosis(t);
      }
    }
  }
}

void DTM::diagnosis(size_t t) {
  float perp = 0.0;
  unsigned N = 0;
  float total_log_likelihood = 0.0;
  vector<VectorXf> softmax_phi(K);
  vector<VectorXf> softmax_eta(D[t]);
  for (size_t k = 0; k < K; ++k) {
    softmax_phi[k] = softmax(phi[t].col(k));
  }
  for (size_t d = 0; d < D[t]; d++) {
    N += W[t][d].size();
    softmax_eta[d] = softmax(eta[t].row(d));
    for (size_t n = 0; n < W[t][d].size(); n++) {
      float likelihood = 0.0;
      size_t w = W[t][d][n];
      for (size_t k = 0; k < K; k++) {
        likelihood += ((softmax_eta[d](k) * (softmax_phi[k](w))));
        if (likelihood < 0)
          std::cout << "Likelihood less than 0, error" << std::endl;
      }
      total_log_likelihood += log(likelihood);
    }
  }
  cout << "Perplexity: " <<  t << "  "
       << exp(-total_log_likelihood / N) << endl;
}

void DTM::save_data(string dir) {
  for (size_t t = 0; t < T; ++t) {
    stringstream sstm;
    sstm << dir << "/time_slice_" << t << ".txt";
    string fname = sstm.str();
    ofstream myfile;

    myfile.open(fname.c_str());
    for (size_t k = 0; k < K; ++k) {
      vector<pair<float, size_t>> ranking;
      for (size_t v = 0; v < V; v++) {
        ranking.push_back(make_pair(phi[t](v, k), v));
      }
      sort(ranking.begin(), ranking.end(),
           std::greater<pair<float, size_t>>());
      myfile << "Topic " << k << "\n";
      for (size_t v = 0; v < 10; v++) {
        size_t w = ranking[v].second;
        myfile << "(" << vocabulary[w] << ", " << phi[t](w, k) << ")" << endl;
      }
      myfile << endl;
    }
    myfile.close();
  }
}