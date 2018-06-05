#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <ostream>
#include <cassert>

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

const int PARAMETER_NUM = 2; // パラメータの数 (今回は a, b の 2 つ)
const double EPS = 1e7; // 収束判定の閾値
const double eta = 0.01; // 学習率
const int DATA_NUM[3] = { 20, 50, 100 };

template <typename T>
std::vector<double> operator * (const double C, const std::vector<T> v) {
    std::vector<double> ret(int(v.size()));

    for (int i = 0; i < int(v.size()); ++i) {
        ret[i] = C * v[i];
    }

    return ret;
}

std::vector<double> operator + (std::vector<double> a, std::vector<double> b) {
}

// filename から input と label にデータを読み込む
void loadDataSet(std::string filename, std::vector<std::vector<double >>& input, std::vector<double>& label) {
    std::ifstream ifs(filename);

    // ファイルから読み込む
    for (int i = 0; i < int(input.size()); ++i) {
        assert(!ifs.eof());
        ifs >> input[i][0] >> label[i];
        input[i][1] = 1; // バイアス
    }

    return;
}

double sigmoid(const std::vector<double> input, const std::vector<double> theta) {
    assert(int(input.size()) == PARAMETER_NUM and int(theta.size()) == PARAMETER_NUM);

    double t = 0.0;

    for (int i = 0; i < PARAMETER_NUM; ++i) {
        t += input[i] * theta[i];
    }

    return 1.0 / (1.0 + exp(-t));
}

int main() {
    std::vector<double> theta(PARAMETER_NUM, 0.0); // パラメータ a, b
    std::vector<std::vector<double> > input(DATA_NUM[0], std::vector<double>(PARAMETER_NUM));
    std::vector<double>label(DATA_NUM[0]);
    std::string filename = "data" + std::to_string(DATA_NUM[0]) + ".dat";

    loadDataSet(filename, input, label);

    for (int iter = 1; ; iter++) {
        std::vector<double> grad(PARAMETER_NUM); // 各パラメータの勾配

        // i 番目のパラメータの勾配を求める 
        for (int i = 0; i < PARAMETER_NUM; ++i) {
            // j 番目のデータの勾配を求める
            for (int j = 0; j < DATA_NUM[0]; ++j) {
                double sig = sigmoid(input[j], theta);

                grad[i] += (label[j] - sig) * theta[i];
            }
        }

        bool is_convergence = true;

        for (int i = 0; i < PARAMETER_NUM; ++i) {
            if (abs(grad[i]) > EPS) {
                is_convergence = false;
                break;
            }
        }

        if (is_convergence) {
            break;
        }


    }

    return 0;
}
