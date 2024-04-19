# cython: language_level=3
from libc.string cimport strstr


cpdef list calculate_frequencies(str main_string, list substrings, int len_seq):
    cdef list counts = []
    cdef bytes main_str_bytes = main_string.encode('utf-8')
    cdef char* main_str = main_str_bytes
    cdef int count
    cdef bytes sub_str_bytes
    cdef char* sub_str
    cdef double val

    if len_seq <= 0:
        raise ValueError("len_seq must be greater than 0")

    for substring in substrings:
        sub_str_bytes = substring.encode('utf-8')
        sub_str = sub_str_bytes
        count = 0

        if strstr(main_str, sub_str) != NULL:
            count = main_string.count(substring)
            val = <double>count / len_seq
        else:
            val = 0.0
        counts.append(val)

    return counts
