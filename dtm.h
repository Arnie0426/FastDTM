//
// Created by Arnie on 2016-11-15.
//

#ifndef DYNAMICTOPICMODELS_DTM_H
#define DYNAMICTOPICMODELS_DTM_H


#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include "alias_samples.h"

using namespace std;
using namespace Eigen;

class DTM {
 public:
  DTM(const vector<vector<vector<size_t>>> &data, const vector<string> &
  dictionary,
      size_t num_topics, float sgld_a, float sgld_b, float sgld_c,
      float dtm_phi_var, float dtm_eta_var, float dtm_alpha_var);
  void initialize(bool init_with_lda);
  void estimate(size_t num_iters);
  void save_data(string dir);
  void diagnosis(size_t t);
  void build_alias_table(size_t t, size_t w);

 private:
  vector<vector<vector<size_t>>> W;  // data
  vector<vector<vector<size_t>>> Z;  // topic indices

  vector<vector<vector<size_t>>> term_alias_samples;
  vector<vector<size_t>> sample_indices;

  vector<MatrixXf> CDK;
  vector<MatrixXf> CWK;
  MatrixXf CK;

  vector<MatrixXf> phi;
  vector<MatrixXf> eta;
  MatrixXf alpha;

  size_t V;
  size_t K;
  size_t T;
  vector<size_t> D;

  float sgld_a;
  float sgld_b;
  float sgld_c;
  float dtm_phi_var;
  float dtm_eta_var;
  float dtm_alpha_var;

  vector<string> vocabulary;

  MatrixXf denom_phi;
  MatrixXf prior_phi;
  MatrixXf denom_eta;
  MatrixXf prior_eta;

  uniform_real_distribution<float> u01;
};

#endif  //  DYNAMICTOPICMODELS_DTM_H
