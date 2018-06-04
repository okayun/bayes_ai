#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <fstream>
#include <cmath>
#include <cassert>

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

const int DATA_NUM[3] = {20, 50, 100}; // 各データセットのデータ数
const int PARAMETER_NUM = 2; // パラメータの数 (今回は a, b の 2 個)

// パラメータ theta とデータ x, epsilon から y を生成
// theta のサイズは 2 と仮定
double make_y(const double x, const double epsilon, const std::vector<double> theta) {
  assert(int(theta.size()) == PARAMETER_NUM); // サイズが PARAMETER_NUM でないとき
  return (1.0 / (1.0 + exp(-theta[0] * x - theta[1])) + epsilon);
}

int main() {
  std::normal_distribution<> make_input(0.0, 1.0); // 平均 0.0, 分散 1.0 の正規分布
  std::normal_distribution<> make_epsilon(0.0, 0.01); // 平均 0.0, 分散 0.01 の正規分布
  std::random_device seed; // 非決定論的に乱数を生成
  std::mt19937 mt(seed()); // メルセンヌ・ツイスタ

  std::vector<double> theta = {0.8, -0.3}; // パラメータ a, b

  // データ数 20, 50, 100 の 3 パターンのデータセットを作成
  for (int i = 0; i < 3; ++i) {
    std::string filename = "data" + std::to_string(DATA_NUM[i]) + ".dat"; // ファイル名
    std::ofstream ofs(filename); // データセットを書き込むファイルを作成

    for (int j = 0; j < DATA_NUM[i]; ++j) {
      // データ生成
      double x = make_input(mt), epsilon = make_epsilon(mt);
      double y = make_y(x, epsilon, theta);

      ofs << x << " " << y << endl; // ファイルに書き込む
    }

    ofs.close(); // ファイルを閉じる
  }

  return 0;
}
