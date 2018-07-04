import std.stdio;
import std.algorithm;
import std.array;
import std.conv;
import std.file;
import std.math;
import std.numeric;
import std.range;
import std.random;
import std.string;

immutable int N = 5000, ParameterSize = 4;
immutable double Sigma = 0.01;
immutable double[2] prior = [ 0.0, 1.0 ];
double[] theta;
double[][] data, params;
int[] y;
int[2] MaxLoop = [ 2000, 5000 ], BurnIn = [ 0, 1000 ], Interval = [ 100, 250 ];
Random gen;
auto square = (double x) => x*x;

void init() {
  theta = new double[](ParameterSize), theta[] = 0.0;
  params = new double[][](0);
  data = new double[][](N);
  y = new int[](N);
  gen = Random(unpredictableSeed);

  auto lines = readText("data.csv").splitlines;
  lines.popFront();
  foreach(i, e; lines) {
    auto line = e.split(",").array;
    data[i] ~= line[0..3].map!(to!double).array;
    y[i] = line[3].to!int;
  }
}

void mcmcRun() {
  int S = 0;
  while (8 > S) {
    init();
    run(S >> 2 & 1, S >> 1 & 1, S & 1);
    writeln("\nMaxLoop, BurnIn, Interval = ", MaxLoop[S >> 2 & 1], " ", BurnIn[S >> 1 & 1], " ", Interval[S++ & 1]);
    getEap();
  }
}

void run(int l, int m, int n) {
  for(int i = 0; i < MaxLoop[l]; ++i) {
    for (int j = 0; j < ParameterSize; ++j) update(j);
    if (i >= BurnIn[m] && i % Interval[n] == 0) params ~= theta;
  }
}

void update(int index) {
  double originalP = theta[index];
  double prevL = getLogLikelihood() + getLogGaussian(index);
  theta[index] = boxMuller(index);
  double nextL = getLogLikelihood() + getLogGaussian(index);
  double alpha = exp(nextL - prevL), threshold = uniform(0.0, 1.0, gen);
  theta[index] = (threshold > alpha ? originalP : theta[index]);
}

double boxMuller(int index) {
  return (sqrt(-2 * log(uniform(0.0, 1.0, gen))) * cos(uniform(0.0, 2 * PI, gen))) * Sigma + theta[index];
}

double sigmoid(int index) {
  return 1.0 / (1.0 + exp(-dotProduct(data[index] ~ [1.0], theta)));
}

double getLogLikelihood() {
  return reduce!("a + b")(0.0, N.iota.map!((int i) => log(y[i] == 1 ? sigmoid(i) : 1.0 - sigmoid(i))));
}

double getLogGaussian(int index) {
  return log(exp(-square(theta[index] - prior[0]) / (2 * square(prior[1]))) / (sqrt(2 * PI) * prior[1]));
}

void getEap() {
  for (int i = 0; i < ParameterSize; ++i) {
    writeln(('a'.to!int + i).to!char, " : ", reduce!("a + b")(0.0, params.map!(((double[] x) => x[i]))) / params.length);
  }
}

void main() {
  mcmcRun();
}
