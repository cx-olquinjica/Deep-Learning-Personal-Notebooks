#! /Users/admin/miniconda3/envs/d2l/bin/python

import numpy as np


a = np.arange(15).reshape(3, 5)
print(a)

# in numpy dimensions are called axes
# for example the 3D space  [1, 2, 1], has one one axis. That axis has 3 elements in it
# so it has a length of 3
print("array dimention:", a.ndim)

""" ndarray.shape
Returns the dimmeion of the array. This is a tuple of integers indication the size 
of the array in each dimension of axes"""

print("the shape of the array", a.shape)

"""ndarray.size 
returns the total number of element of the arrray. This is equal to the product of the 
elements in shape."""

print("the size of the array", a.size)

""" ndarray.dtype
an object describing the type of the elements in the array."""

print("the data type of the array is", a.dtype)

print(type(a))


#############################################################################################

# Array Creation
""" One can create an array from a regular python list or tuple using the array function.
the type of the resulting array is deduced from the type of elements in the sequence."""
b = np.array([1, 2, 3, 4, 5])
print("array b, dtype, b.size, b.shpe, b.ndim",b, b.dtype, b.size, b.shape, b.ndim)


# array transforms sequences of sequences into two-dimensional arrays
# sequences of sequences of sequences into three-dimensional array, and so on
# Example below: 

c = np.array([(1.5, 2, 3), (4, 5, 6)])
print()
print(c.ndim)


"""The function zeros creates an arra full of zeros, the function ones
creates an array full of ones, and the empty creates an array full whose initial 
content is random and depends on the state of the memory. """
print()
d = np.zeros(shape=(3, 4))
print(d)
print()
e = np.ones(shape=(3, 5))
print(e)

""" To create sequences of numbers, NumPy provides the arange function which analogous to the
Python built-in range, but returns an array."""
print()
f = np.arange(10).reshape(2, 5)
print(f)
print()

g = np.arange(24).reshape(2, 3, 4) # 3d array
print("3d array:", g)
print()

# basic operations

a = np.array([20, 30, 40, 50])
b = np.arange(4)
c = a - b
print("substraction:", c)
print()
print("power:", b**2)
print()
print(" testing if @ and dot produce the same result:")
print("@: ", a @ b)
print()
print("dot:", a.dot(b))
print()
print(a @ b == a.dot(b))
print()

# multidimenstion array can have one index per axis

# these indices are given in a tuple separated by commas

i = np.arange(12).reshape(4, 3)
print(i)


