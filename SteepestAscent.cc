#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <functional>
#include <ostream>
#include <map>
#include <cassert>
#include <cmath>
#include <ctime>

/* C++11以上 */

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

using Data = std::pair<double, double>;

const int MaxIteration = 10000; // 反復回数の上限
const int ParameterSize = 2; // パラメータベクトル theta のサイズ (今回は theta = (a, b) の 2 つ)
const int DataSetNum[4] = {20, 50, 100, 1000}; // データセットの数
const double Sigma = 0.1; // 標準偏差 (これを推定するとうまくいかない)
const double Eta = 0.000001; // 学習率
const double Eps = 1e-7; // 収束判定閾値

#define SQUARE(x) (x)*(x)
#define CUBED(x) (x)*(x)*(x)

//#define DEBUG // for debug

/**************************** operator ******************************/

template <typename T, typename U>
std::ostream& operator << (std::ostream& os, std::pair<T, U>& p) {
  os << "(" << p.first << ", " << p.second << ")";
  return os;
}

template <typename T>
std::ostream& operator << (std::ostream& os, std::vector<T>& v) {
  os << "[";
  for (int i = 0; i < int(v.size()); ++i) {
    os << v[i] << (i + 1 == int(v.size()) ? "" : ", ");
  }
  os << "]";
  return os;
}

template <typename T>
std::vector<T> operator + (std::vector<T> a, std::vector<T> b) {
  assert(a.size() == b.size());

  std::vector<T> ret(int(a.size()));

  for (int i = 0; i < int(a.size()); ++i) {
    ret[i] = a[i] + b[i];
  }

  return ret;
}

template <typename T>
std::vector<double> operator * (double C, std::vector<T> a) {
  std::vector<double> ret(int(a.size()));

  for (int i = 0; i < int(a.size()); ++i) {
    ret[i] = C * a[i];
  }

  return ret;
}

/**************************** operator ******************************/

double sigmoid(const double t) {
  return 1 / (1 + exp(-t));
}

// a, b をファイルに書き込む
void write_parameter(int iteration, int data_num, std::vector<double>& theta) {
  std::ofstream ofs("parameter" + std::to_string(data_num) + "_" + std::to_string(Eta) + ".dat", std::ios::app);

  ofs << iteration << " " << theta[0] << " " << theta[1] << endl;

  ofs.close();

  return;
}

// a, b の勾配を求める関数
std::vector<double> grad_calculate(std::vector<Data>& dataset, std::vector<double>& theta) {
  std::vector<double> ret(ParameterSize, 0);

  for (auto data : dataset) {
    double e = exp(-theta[0] * data.first - theta[1]);
    ret[0] += data.first * e * (data.second * (1 + e) - 1) / (1 + e) / SQUARE(1 + e);
  }

  ret[0] /= SQUARE(Sigma);

  for (auto data : dataset) {
    double e = exp(-theta[0] * data.first - theta[1]);
    ret[1] += e * (data.second * (1 + e) - 1) / (1 + e) / SQUARE(1 + e);
  }

  ret[1] /= SQUARE(Sigma);
 
  return ret;
}

// data_num 個のデータ生成
void make_dataset(int data_num, std::vector<Data>& dataset) {
  dataset.resize(data_num);
  
  std::normal_distribution<> make_input(0.0, 1.0);
  std::normal_distribution<> make_epsilon(0.0, 0.01);
  std::random_device seed;
  std::mt19937 mt(seed());

  std::vector<double> theta = { 0.8, -0.3 }; // a, b

  double epsilon = make_epsilon(mt); // varepsilon

  for (int i = 0; i < data_num; ++i) {
    dataset[i].first = make_input(mt); // x_i
    dataset[i].second = sigmoid(theta[0] * dataset[i].first + theta[1]) + epsilon; // y_i
  }

  return;
}

int main() {

  for (int d = 0; d < 4; ++d) {
    std::vector<Data> dataset; // 入力データと教師データの組を格納する可変長配列
    std::vector<double> theta = { 0.0, 0.0 }; // theta[0] := a, theta[1] := b とする. それぞれ初期値は 0.0, 0.0

    make_dataset(DataSetNum[d], dataset); // データを生成

    clock_t start = clock();

    for (int iter = 0; iter < MaxIteration; ++iter) {
      std::vector<double> grad(ParameterSize);

      if (iter % 10 == 0) {
        write_parameter(iter, DataSetNum[d], theta);
      }

      grad = grad_calculate(dataset, theta);

      bool is_convergence = true;

#ifdef DEBUG
      if (iter % 100 == 0) {
        cerr << "grad = " << grad << endl;
        cerr << "parameter = " << theta << endl;
      }
#endif

      // 収束判定
      for (int i = 0; i < ParameterSize; ++i) {
        if (fabs(grad[i]) > Eps) {
          is_convergence = false;
          break;
        }
      }

      if (is_convergence) {
        cerr << "Convergence" << endl;
        cerr << "iter = " << iter << endl;
        break;
      }

      // パラメータの更新
      theta = theta + (Eta * grad);
    }

    clock_t end = clock();

    cout << "time = " << double(end - start) / CLOCKS_PER_SEC << endl;
    cout << "[a, b] = " << theta << endl;

#ifdef DEBUG
    for (auto data : dataset) {
      cerr << "input = " << data.first << endl;
      cerr << "label, prediction = " << data.second << ", " << sigmoid(theta[0] * data.first + theta[1]) << endl;
    }
#endif

  }
  
  return 0;
}

