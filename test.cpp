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

    // ndarray_view<double, 2> na = ndarray_view<double, 2>(buff_na, 500, 20);
    ndarray_view<double, 2> na = nd::map(buff_na, 500, 20);
    ndarray_view<double, 2> nb = nd::map(buff_nb, 500, 20);
    ndarray_view<double, 6> nc = nd::map((double*)NULL, 3, 4, 5, 6, 7, 8);

    std::printf("Ndarray vector mapping : %f\n", t.report());

    std::printf("na_size : %u\n", na.size());
    std::printf("na_id : %u\n", na.compute_id(30, 10));
    std::printf("nc_id : %u\n", nc.compute_id(1, 2, 1, 2));

    for (uint n = 0; n < N_ITER; n++) {
        na += nb;
    }

    ndarray<double, 3> x(2, 3, 3);
    for (uint i = 0; i < 2*3*3; i++) {
        x.m_data[i] = i;
    }

    ndarray_view<double, 2> y = x.get_view(0);

    ndarray<double, 2> z = dot(y, y);

    print(z);
    std::cout << "\n";
    roll_axis_right(z);
    print(z);
    std::cout << "\n";

    print(x);
    std::cout << "\n";
    roll_axis_right(x);
    print(x);
    std::cout << "\n";
    roll_axis_left(x);
    print(x);
    std::cout << "\n";
    roll_axis_left(x);
    print(x);
    std::cout << "\n";

    std::printf("Ndarray addition : %f\n", t.report());

    delete[] buff_va;
    delete[] buff_vb;
    delete[] buff_na;
    delete[] buff_nb;

    // std::ofstream file("dummy.txt");
    // for (uint i = 0; i < na.size(); i++) {
    //     file << na[i] << " ";
    // }
    // file << std::endl;

    // std::printf("Saving in na file : %f\n", t.report());

    return 0;
}
