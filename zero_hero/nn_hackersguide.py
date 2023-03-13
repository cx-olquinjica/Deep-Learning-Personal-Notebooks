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

print("Numerical Gradient Part:")
print()
x, y = -2, 3 # some input values
out = forwardMultiplyGate(x, y)
print(out)

# now based on the formula for derivatives introducing h

h = 0.0001;

# compute derivative with respect to x

xph = x + h;
out2 = forwardMultiplyGate(xph, y)
x_derivative = (out2 - out) / h
print("the derivative of x: ", x_derivative)

# compute derivative with respect to y
print()
yph = y + h

out3 = forwardMultiplyGate(x, yph)
y_derivative = (out3 - out)/h
print("the derivative of y: ", y_derivative)
print()

""" From this calculations we see that the derivative wrt to x is +3. I'm making the positive sign explicit
because it indicates THE CIRCUIT IS TUGGING ON X TO BECOME HIGHER. The actual vaue, 3 can be interpreted as 
the force of that tug. 

By the way, we usually talk about the derivative with respect to a single input, or about a gradient with 
respect to all the inputs. The gradient is just made up of the derivatives of all the inputs concatenated 
in a vector(i.e a list).


Crucially, notice that if we let the inputs respond to the tug by following the gradient a tiny amount
(i.e we just add the derivative on top of every input), we can see that the value increases, as expected: """

# gradient a tiny amount
step_size = 0.01 # learning rate
out = forwardMultiplyGate(x, y) # before: -6
x = x + step_size * x_derivative # x becomes -1.97
y = y + step_size * y_derivative # y becomes 2.98
out_new = forwardMultiplyGate(x, y) # -5.87!! Exiciting
print(out_new)


## Strategy #3: Analytic Gradient

x, y = -2,  3
out = forwardMultiplyGate(x, y) # before -6
x_gradient = y
y_gradient = x
step_size = 0.01
x = x + step_size * x_gradient # -1.97
y = y + step_size * y_gradient # 2.98
out_new = forwardMultiplyGate(x, y)
print()
print("analytical solution:", out_new)
print()

""" In practice by the way, all nn libraries always compute the analytic gradient, but the correctness of
the implementation is verified by comparing it to the numerical gradient. That is because the numerical 
gradient is very easy to evaluate (but can be a bit expensive to compute), while the analyic gradient can contain bugs
at times, but it is extremely efficient to compute. As we will see, evaluating the gradient (i.e while doing backprop, 
or backwards pass) will turn out to cost about as much as evaluating the forward pass."""

## Recursive Case: Circuits with Multiple Gates

""" The expression we are computing now is: f(x, y, z) = (x + y) * z."""

def forwardAddGate(a, b): return a + b

def forwardCircuit(x, y, z): 
    q = forwardAddGate(x, y)
    f = forwardMultiplyGate(q, z)
    return f

x, y, z = -2, 5, -4
f = forwardCircuit(x, y, z) # -12 
print(f)
print()

## Backpropagation

# gradient of the MULTIPLY gate with respect to its inputs
q = forwardAddGate(x, y)
f = forwardMultiplyGate(q, z)
derivative_f_wrt_z = q  # 3
derivative_f_wrt_q = z # -4
# derivative of the ADD gate wrt to its inputs

derivative_q_wrt_x = 1.0
derivative_q_wrt_y = 1.0

# chain rule

derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q # -4
derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q # -4

# final gradient, from above [-4, -4, 3]

gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

# let the inputs respond to the force/tug

x = x + step_size * derivative_f_wrt_x # -2.04
y = y + step_size * derivative_f_wrt_y # 4.96
z = z + step_size * derivative_f_wrt_y # -3.97

# our circuit now better give higher output

q = forwardAddGate(x, y) # q becomes 2.92
f = forwardMultiplyGate(q, z) # output is -11.59, up from -12! !Nice


