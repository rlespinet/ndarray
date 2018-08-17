#pragma once

#include <initializer_list>
#include "template_utils.hpp"

#include "simdpp/simd.h"
#include "mkl.h"

template<typename T>
struct ndarray_vtype {
    static constexpr uint npack = 1;
    typedef T type;
};

template<>
struct ndarray_vtype<float> {
    static constexpr uint npack = SIMDPP_FAST_FLOAT32_SIZE;
    typedef simdpp::float32<npack> type;
};

template<>
struct ndarray_vtype<double> {
    static constexpr uint npack = SIMDPP_FAST_FLOAT64_SIZE;
    typedef simdpp::float64<npack> type;
};


template<uint i, uint D>
struct compute_id_ {
    template<typename ...Args>
    static uint execute(const uint (&dims)[D], uint acc, uint ith_elt, Args... args) {
        return compute_id_<i+1, D>::execute(dims, acc * dims[i] + ith_elt, args...);
    }
    static uint execute(const uint (&dims)[D], uint acc) {
        return compute_id_<i+1, D>::execute(dims, acc * dims[i]);
    }
};

template<uint D>
struct compute_id_<D, D> {
    static uint execute(const uint (&dims)[D], uint acc) {
        return acc;
    }
};

template<typename T, uint D>
struct ndarray_view {

    static constexpr uint N = ndarray_vtype<T>::npack;
    typedef typename ndarray_vtype<T>::type V;

    union {
        T *m_data;
        V *v_data;
    };

    uint m_dims[D];

    ndarray_view()
        : m_data(nullptr)
        , m_dims() {
    }

    template<typename ...Args>
    ndarray_view(T* data, Args... args) {
        static_assert(are_convertible<uint, Args...>{},
                      "Dimension arguments must be of type uint");
        static_assert(sizeof...(Args) == D,
                      "Wrong number of dimensions");
        copy_arguments<0>::execute(m_dims, args...);

        m_data = data;
    }

    ndarray_view(const ndarray_view& oth) {
        assert(false);
    }

    ndarray_view(ndarray_view &&oth) {
        swap(*this, oth);
    }

    ~ndarray_view() {
    }

    ndarray_view& operator=(ndarray_view oth) {
        swap(*this, oth);
        return *this;
    }

    friend void swap(ndarray_view &m1, ndarray_view &m2) {
        using std::swap;
        swap(m1.m_dims, m2.m_dims);
        swap(m1.m_data, m2.m_data);
    }

    const T& operator[](uint id) const {
        return this->m_data[id];
    }

    T& operator[](uint id) {
        return this->m_data[id];
    }

    template<typename ...Args>
    const T& operator()(Args... args) const {
        static_assert(are_convertible<uint, Args...>{}, "Arguments must be of type uint");
        static_assert(sizeof...(args) == D, "Wrong number of dimensions");
        return this->m_data[compute_id(args...)];
    }

    template<typename ...Args>
    T& operator()(Args... args) {
        static_assert(are_convertible<uint, Args...>{}, "Arguments must be of type uint");
        static_assert(sizeof...(args) == D, "Wrong number of dimensions");
        return this->m_data[compute_id(args...)];
    }

    template<typename ...Args>
    const uint compute_id(Args... args) const {
        // return compute_id_acc(0, args...);
        return compute_id_<0, D>::execute(m_dims, 0, args...);
    }

    template<uint i>
    const uint shape() const {
        static_assert(i < D, "Wrong number of dimensions");
        return m_dims[i];
    }

    const uint size() const {
        return compute_product<0, D>::execute(m_dims);
    }

    const uint vsize() const {
        return (size() + N - 1) / N;
    }

    const T* data() const {
        return this->m_data;
    }

    T* data() {
        return this->m_data;
    }

    void fill(const T &elt) {
        for (int i = 0; i < vsize(); i++) {
            v_data[i] = simdpp::splat(elt);
        }
    }

    template<uint V, typename ...Args>
    ndarray_view<T, V> get_view(Args... args) {
        constexpr uint A = sizeof...(args);
        static_assert(V == D - A,
                      "Output of get_view has wrong dimension");

        ndarray_view<T, V> view;
        view.m_data = &m_data[compute_id(args...)];
        copy_array<A, 0, V>::execute(m_dims, view.m_dims);

        return view;
    }

    ndarray_view& operator+=(const ndarray_view& oth) {
        assert(oth.size() == size());
        for (uint i = 0; i < vsize(); i++) {
            v_data[i] = v_data[i] + oth.v_data[i];
        }
        return *this;
    }

    ndarray_view& operator+=(const T& elt) {
        for (uint i = 0; i < vsize(); i++) {
            v_data[i] = v_data[i] + elt;
        }
        return *this;
    }

    ndarray_view& operator-=(const ndarray_view& oth) {
        assert(oth.size() == size());
        for (uint i = 0; i < vsize(); i++) {
            v_data[i] = v_data[i] - oth.v_data[i];
        }
        return *this;
    }

    ndarray_view& operator-=(const T& elt) {
        for (uint i = 0; i < vsize(); i++) {
            v_data[i] = v_data[i] - elt;
        }
        return *this;
    }

};

template<typename T, uint D>
struct ndarray : public ndarray_view<T, D> {

    ndarray()
        : ndarray_view<T, D>() {
    }

    template<typename ...Args>
    ndarray(Args... args)
        : ndarray_view<T, D>(nullptr, args...) {
        uint s = this->size();
        std::cout << "new : " << s + this->N - 1 << std::endl;
        this->m_data = new T[s + this->N - 1]; // Add extra space for simd computations (N - 1)
    }

    ~ndarray() {
        if (this->m_data != nullptr) {
            std::cout << "delete : " << this->size() + this->N - 1 << std::endl;
            delete[] this->m_data;
        }
    }

};

namespace nd
{
    template<typename T, typename ...Args>
    static ndarray_view<T, sizeof...(Args)> map(T* data, Args... args) {
        return ndarray_view<T, sizeof...(Args)>(data, args...);
    }

    template<typename T, typename ...Args>
    static ndarray<T, sizeof...(Args)> create(Args... args) {
        return ndarray<T, sizeof...(Args)>(args...);
    }
}

ndarray<double, 2> dot(const ndarray_view<double, 2> &matA, const ndarray_view<double, 2> &matB) {
    ndarray<double, 2> matC(matA.m_dims[0], matB.m_dims[1]);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                matA.m_dims[0], matB.m_dims[1], matA.m_dims[1], 1.0,
                matA.m_data, matA.m_dims[1],
                matB.m_data, matB.m_dims[0],
                0.0, matC.m_data, matC.m_dims[0]);
    return matC;
}
