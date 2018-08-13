#include <iostream>
#include <fstream>
#include <ctime>

#include <Eigen/Dense>
#include "ndarray.hpp"

using Eigen::VectorXd;
using Eigen::Map;

struct timer {
    std::clock_t started;

    timer() {
        reset();
    }

    void reset() {
        started = std::clock();
    }

    double report() {
        std::clock_t current = std::clock();
        double time_s = (current - started) / (double) CLOCKS_PER_SEC;
        started = current;
        return time_s;
    }

};

#define N_ITER 10000

int main() {


    double *buff_va = new double[10000];
    std::fill_n(buff_va, 10000, 1.0);

    double *buff_vb = new double[10000];
    std::fill_n(buff_vb, 10000, 1.0);

    double *buff_na = new double[10000];
    std::fill_n(buff_na, 10000, 1.0);

    double *buff_nb = new double[10000];
    std::fill_n(buff_nb, 10000, 1.0);


    timer t;

    VectorXd a = Map<VectorXd>(buff_va, 10000);
    VectorXd b = Map<VectorXd>(buff_vb, 10000);

    std::printf("Eigen vector mapping : %f\n", t.report());

    for (uint n = 0; n < N_ITER; n++) {
        a += b;
    }

    std::printf("Eigen addition : %f\n", t.report());

    std::ofstream file("dummy1.txt");
    for (uint i = 0; i < a.size(); i++) {
        file << a[i] << " ";
    }
    file << std::endl;

    std::printf("Saving in a file : %f\n", t.report());

    ////////////////////////////////////////////////////////////////////////////////

    ndarray<double, 2> na = ndarray<double, 2>::from_array(buff_na, 500, 20);
    ndarray<double, 2> nb = ndarray<double, 2>::from_array(buff_nb, 500, 20);

    std::printf("Ndarray vector mapping : %f\n", t.report());

    std::printf("na_size : %u\n", na.size());
    std::printf("na_id : %u\n", na.compute_id(30, 10));

    for (uint n = 0; n < N_ITER; n++) {
        na += nb;
    }

    std::printf("Ndarray addition : %f\n", t.report());



    // std::ofstream file("dummy.txt");
    // for (uint i = 0; i < na.size(); i++) {
    //     file << na[i] << " ";
    // }
    // file << std::endl;

    // std::printf("Saving in na file : %f\n", t.report());

    return 0;
}
