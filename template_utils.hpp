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
    static void forward(const uint (&src)[N_src], uint (&dest)[N_dest]) {
        dest[id_dest] = src[id_src];
        copy_array<id_src+1, id_dest+1, len-1>::forward(src, dest);
    }
    template<uint N_src, uint N_dest>
    static void backward(const uint (&src)[N_src], uint (&dest)[N_dest]) {
        dest[id_dest + len - 1] = src[id_src + len - 1];
        copy_array<id_src, id_dest, len-1>::backward(src, dest);
    }
};

template<uint id_src, uint id_dest>
struct copy_array<id_src, id_dest, 0> {
    template<uint N_src, uint N_dest>
    static void forward(const uint (&src)[N_src], uint (&dest)[N_dest]) {
    }
    template<uint N_src, uint N_dest>
    static void backward(const uint (&src)[N_src], uint (&dest)[N_dest]) {
    }
};

template<uint start, uint len>
struct roll_array_right {
    template<uint N>
    static void execute(uint (&array)[N]) {
        uint last_elt = array[start + len - 1];
        copy_array<start, start+1, len-1>::backward(array, array);
        array[start] = last_elt;
    }
};

template<uint start, uint len>
struct roll_array_left {
    template<uint N>
    static void execute(uint (&array)[N]) {
        uint first_elt = array[start];
        copy_array<start+1, start, len-1>::forward(array, array);
        array[start + len - 1] = first_elt;
    }
};
