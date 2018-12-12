import numpy as np
from ctypes import c_char_p, c_double, c_int, POINTER
from numpy.ctypeslib import load_library, ndpointer
import os

def get_root_dir():
    path = os.path.dirname(os.path.realpath(__file__)) + "/../"
    return path


def calculate_c4y():
    libdir = get_root_dir() + "lib"
    libpython = load_library("libpython", libdir)
    type_array2d = ndpointer(dtype=np.float64, ndim=2, flags="C")
    functor = libpython.calculate_c4y
    functor.argtypes = ([type_array2d,
                         c_int,
                         c_int,
                         c_int,
                         c_int,
                         type_array2d])
    functor.restype = None

    return functor


def fitness():
    libdir = get_root_dir() + "lib"
    libpython = load_library("libpython", libdir)
    type_array1d = ndpointer(dtype=np.float64, ndim=1, flags="C")
    type_array2d = ndpointer(dtype=np.float64, ndim=2, flags="C")
    functor = libpython.fitness
    functor.argtypes = ([type_array2d,
                         c_int,
                         c_int,
                         type_array1d,
                         c_int])
    functor.restype = c_double

    return functor


def gradient():
    libdir = get_root_dir() + "lib"
    libpython = load_library("libpython", libdir)
    type_array1d = ndpointer(dtype=np.float64, ndim=1, flags="C")
    type_array2d = ndpointer(dtype=np.float64, ndim=2, flags="C")
    functor = libpython.gradient
    functor.argtypes = ([type_array2d,
                         c_int,
                         c_int,
                         type_array1d,
                         c_int,
                         type_array1d])
    functor.restype = None

    return functor
