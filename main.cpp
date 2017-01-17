#include "dtm.h"

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
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <dtm.config path> " << endl;
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

  DTM dtm(docs, vocabulary, 40, 0.5, 100, 0.80, 0.1, 0.1, 0.1);
  dtm.initialize(true);
  dtm.estimate(50);
  dtm.save_data("results");
  return 0;
}



