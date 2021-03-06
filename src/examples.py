#!/usr/bin/env python3

import time
import arrayfire as af
import numpy as np

def time_fn(fn, *args, **kwargs):
    start_time = time.time()
    result = fn(*args, **kwargs)
    end_time = time.time()

    return (end_time - start_time, result)









# Test de produit scalaire
n = 1000000
print('')
print('==== Dot product of {}-vectors ===='.format(n))

af.set_backend('cpu')
v1 = af.random.randu(n)
v2 = af.random.randu(n)
time_dot_product_host = af.timer.timeit(af.blas.dot, v1, v2)
print('Host time: {}'.format(time_dot_product_host))

af.set_backend('cuda')
v1 = af.random.randu(n)
v2 = af.random.randu(n)
time_dot_product_device = af.timer.timeit(af.blas.dot, v1, v2)
print('Device time: {}'.format(time_dot_product_device))
print('Speedup: {0:.2f}x'.format(time_dot_product_host / time_dot_product_device))















# Matrice-Vecteur
n = 10000

print('')
print('==== Multiply a matrix of {}x{} with a {}-vector ===='.format(n, n, n))

af.set_backend('cpu')
m = af.random.randu(n, n)
a = af.random.randu(n, 1)
time_matmul_host = af.timer.timeit(af.blas.matmul, m, a)
print('Host time: {}'.format(time_matmul_host))

af.set_backend('cuda')
m = af.random.randu(n, n)
a = af.random.randu(n, 1)
time_matmul_device = af.timer.timeit(af.blas.matmul, m, a)
print('Device time: {}'.format(time_matmul_device))
print('Speedup: {0:.2f}x'.format(time_matmul_host / time_matmul_device))













# Matrice-Matrice
n = 1000

print('')
print('==== Multiply matrices of {}x{} ===='.format(n, n))

af.set_backend('cpu')
m = af.random.randu(n, n)
a = af.random.randu(n, n)
time_matmul2_host = af.timer.timeit(af.blas.matmul, m, a)
print('Host time: {}'.format(time_matmul2_host))

af.set_backend('cuda')
m = af.random.randu(n, n)
a = af.random.randu(n, 1)
time_matmul2_device = af.timer.timeit(af.blas.matmul, m, a)
print('Device time: {}'.format(time_matmul_device))
print('Speedup: {0:.2f}x'.format(time_matmul2_host / time_matmul2_device))











# Solve linear equation
n = 500
print('')
print('==== Solve a {}x{} linear equation ===='.format(n, n, n))

af.set_backend('cpu')
m = af.random.randu(n, n)
a = af.random.randu(n, 1)
b = af.blas.matmul(m, a)

time_les_host = af.timer.timeit(af.solve, m, b)
print('Host time: {}'.format(time_les_host))


af.set_backend('cuda')
m = af.random.randu(n, n)
a = af.random.randu(n, 1)
b = af.blas.matmul(m, a)

time_les_device = af.timer.timeit(af.solve, m, b)
print('Device time: {}'.format(time_les_device))
print('Speedup: {0:.2f}x'.format(time_les_host / time_les_device))










# Test de sort
size_of_vector = 1000000
print('')
print('==== Sort an array of {} floats ===='.format(size_of_vector))

af.set_backend('cpu')
v = af.random.randu(size_of_vector)
time_sort_host = af.timer.timeit(af.algorithm.sort, v)
print('Host time: {}'.format(time_sort_host))

af.set_backend('cuda')
v = af.random.randu(size_of_vector)
time_sort_device = af.timer.timeit(af.algorithm.sort, v)
print('Device time: {}'.format(time_sort_device))
print('Speedup: {0:.2f}x'.format(time_sort_host / time_sort_device))








# Test de convolution
n = 1000
print('')
print('=== 2d convolution on a {}x{} matrix ==='.format(n, n))

af.set_backend('cpu')
m = af.random.randu(n, n)

numpy_kernel = np.array([[0.1, 0.2, 0.1],
                         [0.2, 0.3, 0.2],
                         [0.1, 0.2, 0.1]])
af_kernel = af.Array(numpy_kernel.ctypes.data, numpy_kernel.shape, numpy_kernel.dtype.char)


time_convolve_host = af.timer.timeit(af.convolve, m, af_kernel)
print('Host time: {}'.format(time_convolve_host))


af.set_backend('cuda')
af_kernel = af.Array(numpy_kernel.ctypes.data, numpy_kernel.shape, numpy_kernel.dtype.char)
m = af.random.randu(n, n)

time_convolve_device = af.timer.timeit(af.convolve, m, af_kernel)
print('Device time: {}'.format(time_convolve_device))
print('Speedup: {0:.2f}x'.format(time_convolve_host / time_convolve_device))









# Interop numpy
print('')
print('Numpy interop...')
af.set_backend('cuda')
m = np.random.rand(1000, 1000)

arrayfire_m = af.np_to_af_array(m)
(maxs, idxs) = af.algorithm.imax(arrayfire_m, 1)
print('Done')
