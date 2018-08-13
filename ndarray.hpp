#pragma once

#include <initializer_list>

#include "simdpp/simd.h"


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

template<typename T, uint D>
struct ndarray {

    static constexpr uint N = ndarray_vtype<T>::npack;
    typedef typename ndarray_vtype<T>::type V;

    union {
        T *m_data;
        V *v_data;
    };

    uint m_dims[D];

    ndarray()
        : m_data(nullptr)
        , m_dims() {
    }

    template<typename ...Args>
    ndarray(Args... args) {
        static_assert(sizeof...(args) == D, "Wrong number of dimensions");
        init_dim(args...);
        uint m_size = size();
        m_data = new T[m_size + N - 1]; // Add extra space for simd computations (N - 1)
    }

    template<uint i>
    void init_dim() {}

    template<uint i = 0, typename ...Args>
    void init_dim(uint first, Args... args) {
        m_dims[i] = first;
        init_dim<i+1>(args...);
    }

    ndarray(const ndarray& oth) {
        // std::cout << "sq_ndarray_copy" << std::endl;
        assert(false);
    }

    ndarray(ndarray &&oth) {
        // std::cout << "ndarray_move" << std::endl;
        swap(*this, oth);
    }

    template<typename ...Args>
    static ndarray from_array(T *data, Args... args) {
        ndarray<T, sizeof...(args)> array;
        array.m_data = data;
        array.init_dim(args...);
        return array;
    }

    ~ndarray() {
        // std::cout << "ndarray_delete" << std::endl;
        if (m_data != nullptr) {
            delete[] m_data;
        }
    }

    ndarray& operator=(ndarray oth) {
        // std::cout << "sq_ndarray_eq" << std::endl;
        swap(*this, oth);
        return *this;
    }

    friend void swap(ndarray &m1, ndarray &m2) {
        using std::swap;
        // std::cout << "ndarray_swap" << std::endl;
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
        static_assert(sizeof...(args) == D, "Wrong number of dimensions");
        uint id = compute_id(args...);
        return this->m_data[id];
    }

    template<typename ...Args>
    T& operator()(Args... args) {
        static_assert(sizeof...(args) == D, "Wrong number of dimensions");
        uint id = compute_id(args...);
        return this->m_data[id];
    }


    template<uint i = 0, typename ...Args>
    const uint compute_id_acc(uint id, uint first, Args... args) const {
        assert(first < m_dims[i]);
        return compute_id_acc<i+1>(id * m_dims[i] + first, args...);
    }

    template<uint i = 0>
    const uint compute_id_acc(uint id) const {
        if constexpr (i < D) {
            return compute_id_acc<i+1>(id * m_dims[i]);
        } else {
            return id;
        }
    }

    template<typename ...Args>
    const uint compute_id(Args... args) const {
        return compute_id_acc(0, args...);
    }

    // template<uint i = 0>
    // const uint compute_id(uint first) const {return first;}

    // template<uint i = D - 1, typename ...Args>
    // const uint compute_id(uint first, Args... args) const {
    //     return last + m_dims[i] * compute_id<i-1>(args...);
    // }

    template<uint i>
    const uint shape() const {
        static_assert(i < D, "Wrong number of dimensions");
        return m_dims[i];
    }

    template<uint i = 0>
    constexpr uint compute_size() const {
        if constexpr (i >= D) {
            return 1;
        } else {
            return m_dims[i] * compute_size<i+1>();
        }
    }

    const uint size() const {
        return compute_size();
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

    ndarray& operator+=(const ndarray& oth) {
        assert(oth.size() == size());
        for (uint i = 0; i < vsize(); i++) {
            v_data[i] = v_data[i] + oth.v_data[i];
        }
        return *this;
    }

    ndarray& operator-=(const ndarray& oth) {
        assert(oth.size() == size());
        for (uint i = 0; i < vsize(); i++) {
            v_data[i] = v_data[i] - oth.v_data[i];
        }
        return *this;
    }


};
