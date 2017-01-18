#include "dtm.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;


int conf_error() {
  cout << "Couldn't open files in dtm.conf. "
      "Please check your config file." << endl;
  return -1;
}

int main(int argc, char* argv[]) {
  if (argc != 11) {
    cerr << "Usage: " << argv[0]
         << " <dtm.config path> <num_topics> <SGLD_a> <SGLD_b> <SGLD_c> "
         << "<phi_var> <eta_var> <alpha_var> <num_iter> <results_folder>"
         << endl;
    return 1;
  }
  string line;
  ifstream conf_file(argv[1]);
  vector<string> conf;

  if (!conf_file.is_open()) {
    cout << "Couldn't find dtm.conf" << endl;
    return -1;
  }
  while(getline(conf_file, line)) {
    conf.push_back(line);
  }

  vector<string> vocabulary;
  ifstream vocabulary_file(conf[0]);
  if (!vocabulary_file.is_open()) {
    return conf_error();
  }
  while(getline(vocabulary_file, line)) {
    vocabulary.push_back(line);
  }
  vector<vector<vector<size_t>>> docs(conf.size() - 1);
  for (size_t t = 0; t < docs.size(); ++t) {
    ifstream docfile(conf[t+1]);
    if (!docfile.is_open()) {
      return conf_error();
    }
    vector<vector<size_t>> docs_t;
    while(getline(docfile, line)) {
      vector<size_t> doc;
      istringstream is(line);
      size_t term;
      while (is >> term) {
        doc.push_back(term);
      }
      docs_t.push_back(doc);
    }
    cout << "Time Slice t = " << t << " Num docs: " << docs_t.size() << endl;
    docs[t] = docs_t;
  }
  size_t num_topics = atoi(argv[2]);
  float sgld_a = atof(argv[3]);
  float sgld_b = atof(argv[4]);
  float sgld_c = atof(argv[5]);
  float phi_var = atof(argv[6]);
  float eta_var = atof(argv[7]);
  float alpha_var = atof(argv[8]);
  size_t num_iters = atoi(argv[9]);

  DTM dtm(docs, vocabulary, num_topics, sgld_a, sgld_b, sgld_c, phi_var,
          eta_var, alpha_var);
  dtm.initialize(true);
  dtm.estimate(num_iters);
  dtm.save_data(argv[10]);
  return 0;
}



