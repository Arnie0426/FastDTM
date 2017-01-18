//
// Created by Arnie on 2017-01-15.
//

#include "alias_samples.h"

static const float eps = numeric_limits<float>::epsilon();

vector<size_t> AliasSamples::get_samples(size_t num_samples) const {
  vector<size_t> ret(num_samples);
  if (table.empty()) {
    cout << "Alias Tables haven't been initialized";
    return ret;
  }
  default_random_engine generator;
  size_t dim = table.size();
  uniform_real_distribution<float> u01(0, 1);
  uniform_int_distribution<> uniform_table(0, dim - 1);

  for (size_t s = 0; s < num_samples; s++) {
    // roll dice
    size_t k = uniform_table(generator);
    AliasTableCell cell = table[k];
    AliasPair ap = cell.first;

    // flip coin
    float coin = u01(generator);
    if (coin <= ap.second) {
      ret.push_back(ap.first);
    } else {
      ret.push_back(cell.second);
    }
  }
  return ret;
}

AliasSamples::AliasSamples(const VectorXf &prob) {
  vector<AliasPair> poor;
  vector<AliasPair> rich;
  size_t dim = prob.size();
  float sum = prob.sum();

  for (size_t i = 0; i < prob.size(); i++) {
    float p = prob[i];
    if (abs(p) < eps)
      continue;
    float score = p * dim / sum;
    AliasPair t(i, score);
    if (score <= 1.0) {
      poor.push_back(t);
    } else {
      rich.push_back(t);
    }
  }
  // Run Robin-hood algorithm; steal from the rich and fill poor pockets.
  while (!rich.empty() || !poor.empty()) {
    float rem = 1.0;
    AliasTableCell cell;
    AliasPair poor_pair;
    if (!poor.empty()) {
      poor_pair = poor.back();
      poor.pop_back();
      rem -= poor_pair.second;
      if (std::abs(rem) <= eps) {
        cell = make_pair(poor_pair, dim);
        table.push_back(cell);
        continue;
      }
    }
    if (!rich.empty()) {
      auto r = rich.back();
      rich.pop_back();
      size_t alias_index = r.first;
      float prob_mass = r.second;
      if (rem == 1.0) {
        cell = make_pair(make_pair(alias_index, 1.0), dim);
      } else {
        cell = make_pair(poor_pair, alias_index);
      }
      table.push_back(cell);
      prob_mass -= rem;
      if (prob_mass > 1.0) {
        rich.push_back(make_pair(alias_index, prob_mass));
      } else {
        poor.push_back(make_pair(alias_index, prob_mass));
      }
    }
  }

}