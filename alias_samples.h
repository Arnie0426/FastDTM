//
// Created by Arnie on 2017-01-15.
//
#include <queue>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

#ifndef MAIN_ALIAS_SAMPLES_H
#define MAIN_ALIAS_SAMPLES_H

typedef pair<size_t, float> AliasPair;
typedef pair<AliasPair, size_t> AliasTableCell;

class AliasSamples {
 private:
  vector<AliasTableCell> table;

 public:
  AliasSamples(const VectorXf &prob);
  vector<size_t> get_samples(size_t num_samples) const;

};


#endif //MAIN_ALIAS_SAMPLES_H
