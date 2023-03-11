#! /Users/admin/miniconda3/envs/d2l/bin/python 

"""... everything became much clearer when I started writing code."""

# Chapter I: Real-valued circuits


# strategy #1: Random Local Search

from random import uniform 
from decimal import Decimal

## circuit with single gate for now

def forwardMultiplyGate(x, y): return x * y


x, y = -2, 3 # some input values

tweak_amount = 0.01

best_out = Decimal('-Infinity')
best_x, best_y = x, y

# try changing x, y randomly small amounts and keep track of waht works best

for k in range(100):
    x_try = x + tweak_amount * (uniform(0,1) * 2 -1) # tweak x a bit
    y_try = y - tweak_amount * (uniform(0,1) * 2 - 1) # tweak y a bit


    out = forwardMultiplyGate(x_try, y_try)

    if out > best_out:
        # best imporvement yet! keep track of that x and y
        best_out =  out
        best_x, best_y = x_try, y_try

print(f"Best value of x is {x_try}")
print()
print(f"Best value of y is {y_try}")
print()
print("best value for the functions",forwardMultiplyGate(best_x, best_y))

# Strategy #2: Numerical Gradient

""" The derivative can be thought of as a force on each input as we pull the output to become higher.
A nice intuition about numerical gradient descent is to think of:

    Imagine taking the output value that comes out from the circuit and tugging on it in the positive 
    direction. This positive tension will in turn translate through the gate and induce forces on the inputs
    which in our case, when talking about nn it will refer to the parameters. So this tension will tell us 
    how we should change the parameters to increase the output value."""


