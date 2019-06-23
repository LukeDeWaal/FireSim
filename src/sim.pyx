from __future__ import print_function
import cython as cyt
from cython.view cimport array as cvarray



cdef class Simulation(object):

    cdef public:
        int time

        float retardant_efficiency
        float k
        float gust_factor
        float elevation_shift_intensity
        float maxfuel

        tuple forest_size





