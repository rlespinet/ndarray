#pragma once

template<class T, class...>
struct are_convertible : std::true_type
{};

template<class T, class U, class... V>
struct are_convertible<T, U, V...>
    : std::integral_constant<bool, std::is_convertible<U,T>{} && are_convertible<T, V...>{}>
{};


template<uint start, uint len>
struct compute_product {
    template<uint N>
    static uint execute(const uint (&array)[N]) {
        return array[start] * compute_product<start + 1, len - 1>::execute(array);
    }
};

template<uint start>
struct compute_product<start, 0> {
    template<uint N>
    static uint execute(const uint (&array)[N]) {
        return 1;
    }
};

template<uint i>
struct copy_arguments {
    template<uint N, typename ...Args>
    static void execute(uint (&array)[N], uint ith_elt, Args... args) {
        array[i] = ith_elt;
        copy_arguments<i+1>::execute(array, args...);
    }

    template<uint N>
    static void execute(uint (&array)[N]) {
    }
};

template<uint id_src, uint id_dest, uint len>
struct copy_array {
    template<uint N_src, uint N_dest>
    static void execute(uint (&src)[N_src], uint (&dest)[N_dest]) {
        dest[id_dest] = src[id_src];
        copy_array<id_src+1, id_dest+1, len-1>::execute(src, dest);
    }
};

template<uint id_src, uint id_dest>
struct copy_array<id_src, id_dest, 0> {
    template<uint N_src, uint N_dest>
    static void execute(uint (&src)[N_src], uint (&dest)[N_dest]) {
    }
};
