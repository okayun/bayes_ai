#include <iostream>
#include <algorithm>
#include <fstream>
#include <random>
#include <ostream>
#include <vector>
#include <cassert>
#include <cmath>

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

template <typename T>
constexpr T square(T x) { return x * x; };

struct Datum {
  std::vector<double> x;
  int y;

  Datum() {}
  Datum(int n) : x(n, 0), y(0) {}

  double& operator [] (const int index) {
    assert(0 <= index && index < int(x.size()));
    return x[index];
  }
};

std::ostream& operator << (std::ostream& os, Datum& d) {
  os << "x = [ ";
  for (auto xx : d.x) {
    os << xx << " ";
  }
  os << "], y = " << d.y << endl;;

  return os;
}

// const の初期化でエラー出るよ

class MetropolisHastings {
 public:
  const int N; // データセットの数
  const int ParameterSize; // a, b, c, d
  const int MaxLoop; // ループの上限
  const int BurnIn; // バーンイン
  const int Interval; // インターバル
  const double Sigma; // 標準偏差
  const std::vector<double> Prior; // ハイパーパラメータ

  std::random_device rnd; // 非決定的乱数生成器
  std::mt19937 mt; // メルセンヌ・ツイスタ
  std::vector<Datum> data; // データセット
  std::vector<double> theta; // a, b, c, d
  std::vector<std::vector<double> > parameters; // パラメータサンプル
  std::uniform_real_distribution<> dis_real; // theta の初期化に用いる

  void Init();
  double Sigmoid(int index);
  double GetLogLikelihood();
  double GetLogGaussian(int index);
  void Update(int index);
  std::vector<double> GetEap();
  void PrintEap();
  void Run();

// public:
  MetropolisHastings();
  MetropolisHastings(int loop, int burnin, int interval);
  ~MetropolisHastings();
  void Main();
  void Check();
};

void MetropolisHastings::Init() {
  // 各変数の初期化
  mt.seed(rnd());
  data.resize(N, Datum(ParameterSize - 1));
  theta.resize(ParameterSize, 0.0);
  std::uniform_real_distribution<>::param_type param(0.0, 1.0);
  dis_real.param(param);

  // データの読み込み
  std::ifstream ifs("data.csv");
  std::string s;
  std::getline(ifs, s);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < ParameterSize; ++j) {
      char delim = ',';
      if (j == ParameterSize - 1) {
        delim = '\n';
      }
      std::getline(ifs, s, delim);
      if (j == ParameterSize - 1) {
        data[i].y = std::stoi(s);
      }
      else {
        data[i][j] = std::stod(s);
      }
    }
  }
}

// index 番目のデータのシグモイド値を返す
double MetropolisHastings::Sigmoid(int index) {
  double t = 0;
  for (int i = 0; i < ParameterSize; ++i) {
    t += (i == ParameterSize - 1 ? theta[i] : data[index][i] * theta[i]);
  }
  return 1.0 / (1.0 + exp(-t));
}

// 現在の対数尤度関数の値を求める
double MetropolisHastings::GetLogLikelihood() {
  double ret = 0.0;
  for (int i = 0; i < N; ++i) {
    double p = Sigmoid(i);
    ret += log((data[i].y ? p : 1 - p));
  }
  return ret;
}

// 正規分布に基づく確率地の対数を取得
double MetropolisHastings::GetLogGaussian(int index) {
  double p = -square(theta[index] - Prior[0]) / (2 * square(Prior[1]));
  return log(exp(p) / (sqrt(2 * acos(-1)) * Prior[1]));
}

// theta[index] を更新
void MetropolisHastings::Update(int index) {
  double param = theta[index];
  double prev_p = GetLogLikelihood() + GetLogGaussian(index);
  std::normal_distribution<> generate(theta[index], Sigma); // 正規分布に従う値を生成
  theta[index] = generate(mt);
  double next_p = GetLogLikelihood() + GetLogGaussian(index);
  double alpha = exp(next_p - prev_p);
  double t = dis_real(mt);
  if (t > alpha) theta[index] = param;
}

// メトロポリスヘイスティング
void MetropolisHastings::Run() {
  for (int i = 0; i < MaxLoop; ++i) {
    // パラメータの更新
    for (int j = 0; j < ParameterSize; ++j) Update(j);

    if (i > BurnIn) {
      if (i % Interval == 0) parameters.push_back(theta);
    }
  }
}

// EAP 推定量
std::vector<double> MetropolisHastings::GetEap() {
  std::vector<double> ret(ParameterSize, 0);

  for (auto params : parameters) {
    for (int i = 0; i < ParameterSize; ++i) {
      double tmp = params[i];
      ret[i] += tmp / int(parameters.size());
    }
  }

  return ret;
}

// 表示
void MetropolisHastings::PrintEap() {
  std::string param_name = "abcd";
  std::vector<double> eap = GetEap();
  for (int i = 0; i < ParameterSize; ++i) cout << param_name[i] << " : " << eap[i] << endl;
  cout << endl;
}

// 実行
void MetropolisHastings::Main() {
  Init();
  Run();
  PrintEap();
}

// デフォルトコンストラクタ
MetropolisHastings::MetropolisHastings() {}

// 引数あり
MetropolisHastings::MetropolisHastings(int loop, int burnin, int interval) : 
  N(5000), ParameterSize(4), MaxLoop(loop), BurnIn(burnin), Interval(interval), Sigma(0.01), Prior({0.0, 1.0}) {}

MetropolisHastings::~MetropolisHastings() {}

int main() {

  int loop[2] = { 2000, 10000 }, burnin[2] = { 0, 1000 }, interval[2] = { 100, 400 };
  
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        MetropolisHastings m(loop[i], burnin[j], interval[k]);
        m.Main();
      }
    }
  }

  return 0;
}
