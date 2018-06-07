#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <functional>
#include <ostream>
#include <map>
#include <cassert>
#include <cmath>

/* C++11以上 */

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

template <typename T>
using Matrix = std::vector<std::vector<T> >;
using Data = std::pair<double, double>;

const int MaxIteration = 10000000; // 反復回数の上限
const int ParameterSize = 2; // パラメータベクトル theta のサイズ (今回は theta = (a, b) の 2 つ)
const int DataSetNum[4] = {20, 50, 100, 1000}; // データセットの数
const double Sigma = 0.1; // 標準偏差 (これを推定するとうまくいかない)
const double Eta = 0.1; // 学習率
const double Eps = 1e-5; // 収束判定閾値

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
std::vector<T> operator + (std::vector<T>& a, std::vector<T>& b) {
  assert(a.size() == b.size());

  std::vector<T> ret(int(a.size()));

  for (int i = 0; i < int(a.size()); ++i) {
    ret[i] = a[i] + b[i];
  }

  return ret;
}

template <typename T>
std::vector<T> operator - (std::vector<T> a, std::vector<T> b) {
  assert(a.size() == b.size());

  std::vector<T> ret(int(a.size()));

  for (int i = 0; i < int(a.size()); ++i) {
    ret[i] = a[i] - b[i];
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

// l x m 行列と m x n 行列の掛け算
template <typename T>
Matrix<T> operator * (Matrix<T>& a, Matrix<T>& b) {
  assert(a[0].size() == b.size());

  Matrix<T> ret(int(a.size()), std::vector<T>(b[0].size(), T(0)));

  for (int i = 0; i < int(a.size()); ++i) {
    for (int j = 0; j < int(b[0].size()); ++j) {
      for (int k = 0; k < int(b.size()); ++k) {
        ret[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return ret;
}

// 定数と行列の積
template <typename T>
Matrix<T> operator * (double C, Matrix<T>& m) {
  Matrix<T> ret = m;
  for (int i = 0; i < int(ret.size()); ++i) {
    for (int j = 0; j < int(ret[0].size()); ++j) {
      ret[i][j] = C * m[i][j];
    }
  }

  return ret;
}

/**************************** operator ******************************/

double sigmoid(const double t) {
  return 1 / (1 + exp(-t));
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

// a, b をファイルに書き込む
void write_parameter(int iteration, int data_num, std::vector<double>& theta) {
  std::ofstream ofs("parameter" + std::to_string(data_num) + "_" + std::to_string(Eta) + ".dat", std::ios::app);

  ofs << iteration << " " << theta[0] << " " << theta[1] << endl;

  ofs.close();

  return;
}

// 行列式の値 (2x2 行列を想定)
double determinant(Matrix<double>& matrix) {
  assert(matrix.size() == matrix[0].size() and int(matrix.size()) == 2);

  double ret = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

#ifdef DEBUG
  cerr << "determinant = " << ret << endl;
#endif

  return ret;
}

// ヘッセ行列の逆行列を返す
Matrix<double> inverse_hessian(std::vector<Data>& dataset, std::vector<double>& theta) {
  Matrix<double> ret(2, std::vector<double>(2));

  // b に関して 2 階偏微分
  for (auto data : dataset) {
    double e = exp(-theta[0] * data.first - theta[1]), varep = data.second - 1 / (1 + e);

    ret[0][0] += -e * varep / SQUARE(e + 1) + 2 * SQUARE(e) * varep / CUBED(e + 1) - SQUARE(e) / SQUARE(SQUARE(e + 1));
  }
  ret[0][0] /= SQUARE(Sigma);

  // a, b で偏微分
  for (auto data : dataset) {
    double e = exp(-theta[0] * data.first - theta[1]), varep = data.second - 1 / (1 + e);

    ret[0][1] += data.first * (-e * varep / SQUARE(e + 1) + 2 * SQUARE(e) * varep / CUBED(e + 1) - SQUARE(e) / SQUARE(SQUARE(e + 1)));
  }

  ret[0][1] /= -SQUARE(Sigma);

  ret[1][0] = ret[0][1];

  // a に関して 2 階偏微分
  for (auto data : dataset) {
    double e = exp(-theta[0] * data.first - theta[1]), varep = data.second - 1 / (1 + e);

    ret[1][1] += SQUARE(data.first) * (-e * varep / SQUARE(e + 1) + 2 * SQUARE(e) * varep / CUBED(e + 1) - SQUARE(e) / SQUARE(SQUARE(e + 1)));
  }

  ret[1][1] /= SQUARE(Sigma);


  ret = (1 / determinant(ret)) * ret;

  return ret;
}

// ヘッセ行列と勾配行列の積を返す
Matrix<double> product_matrix(std::vector<Data>& dataset, std::vector<double>& theta) {
  Matrix<double> inv_hessian = inverse_hessian(dataset, theta), grad_matrix(ParameterSize, std::vector<double>(1, 0.0));

  for (auto data : dataset) {
    double e = exp(-theta[0] * data.first - theta[1]), varep = data.second - 1.0 / (1.0 + e);

    grad_matrix[0][0] += data.first * e / SQUARE(1.0 + e) * varep;
  }

  grad_matrix[0][0] /= SQUARE(Sigma);

  for (auto data : dataset) {
    double e = exp(-theta[0] * data.first - theta[1]), varep = data.second - 1.0 / (1.0 + e);

    grad_matrix[1][0] += e / SQUARE(1 + e) * varep;
  }

  grad_matrix[1][0] /= SQUARE(Sigma);


#ifdef DEBUG
  cerr << "inverse hessian" << endl;
  cerr << inv_hessian << endl;
  cerr << "grad matrix" << endl;
  cerr << grad_matrix << endl;
#endif

  return inv_hessian * grad_matrix;
}

int main() {

  for (int d = 0; d < 4; ++d) {
    std::vector<Data> dataset; // 入力データと教師データの組を格納する可変長配列
    std::vector<double> theta(ParameterSize, 0.0); // theta[0] := a, theta[1] := b とする. それぞれ初期値は 0.0 

    make_dataset(DataSetNum[d], dataset);

    clock_t start = clock();

    for (int iter = 0; iter < MaxIteration; ++iter) {
      std::vector<double> grad(ParameterSize);

      if (iter % 10 == 0) {
        write_parameter(iter, DataSetNum[d], theta);
      }

#ifdef DEBUG
      if (iter % 100 == 0) {
        cerr << "theta = " << theta << endl;
      }
#endif

      Matrix<double> prod = product_matrix(dataset, theta);

      for (int i = 0; i < ParameterSize; ++i) {
        grad[i] = prod[i][0];
      }

      bool is_convergence = true;

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
      theta = theta - (Eta * grad);
    }

    clock_t end = clock();

    cout << "time = " << double(end - start) / CLOCKS_PER_SEC << endl;
    cout << "[a, b] = " << theta << endl;

#ifdef DEBUG
    for (auto data : dataset) {
      cerr << "input = " << data.first << endl;
      cerr << "label = " << data.second << endl;
      cerr << "prediction = " << sigmoid(theta[0] * data.first + theta[1]) << endl;
    }
#endif

  }

  return 0;
}

