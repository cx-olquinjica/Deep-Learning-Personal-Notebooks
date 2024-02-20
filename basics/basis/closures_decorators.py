#! /Users/admin/miniconda3/envs/d2l/bin/python

""" Important Concepts: 
    First Class Functions: allows us to treat functions as any other object in python. 
    That means we can pass functions as arguments to other functions, return functions
    and we can assign functions to variables. 

    Closures: allow us to take advantage of first class functions, and return an 
    inner functions that REMEMBERS and has access to variables local to the scope
    in which they were created. 

    Decorators: a functions that takes another function as an argument, adds some
    kind of functionality and then returns another function. It does all of this 
    without altering the source code of the original function that was passed in.
    """

def outer_function(msg): 
    """ A function to higlight the inner working behind closure."""
    message = msg                                      # Free variable
    
    def inner_function(): 
        print(message)
    return inner_function                             # Returning a funtion without executing it lacks ()


def decorator_function(original_function): 
    """ A simple decorator function."""
    def wrapper_function(): 
        print('wrapper executed this before {}'.format(original_function.__name__))
        return original_function()
    return wrapper_function

def decorator_function_with_arguments(original_function): 
    """highlights how to handle original functions with arguments."""
    def wrapper_function(*args, **kwargs):              # a must if the function will receive arguments
        print('wrapper executed this before {}'.format(original_function.__name__))
        return original_function(*args, **kwargs)
    return wrapper_function

class decorator_class(object): 
    """Creating a decorator using a class."""
    def __init__(self, original_function): 
        self.original_function = original_function

    def __call__(self, *args, **kwargs): 
        print('call method executed this before {}'.format(self.original_function.__name__))
        return self.original_function(*args, **kwargs)

# Decorator with argurments 

def prefix_decorator(prefix): 
    """ highlights decorators that take arguments @decorator(argument)"""
    def decorator_function(original_function): 
        def wrapper_function(*args, **kwargs): 
            print(prefix, 'Executed Before', original_function.__name__)
            result = original_function(*args, **kwargs)
            print(prefix, 'Executed after', original_function.__name__, '\n')
            return result 
        return wrapper_function
    return decorator_function


# Practical Example: 

def my_logger(original_func): 
    import logging 
    logging.basicConfig(filename='{}.log'.format(original_func.__name__), level=logging.INFO)
    
    def wrapper(*args, **kwargs): 
        logging.info(
                'Ran with args: {}, and kwargs: {}'.format(args,kwargs))
        return original_func(*args, **kwargs)
    return wrapper 

def my_timer(original_func): 
    import time 
    def wrapper(*args, **kwargs): 
        start = time.time()
        result = original_func(*args, **kwargs)
        end = time.time() - start 
        print('{} ran in: {} sec'.format(original_func.__name__, end)) 
        return result 
    return wrapper 

# Understanding the @property decorator

class Circle: 
    def __init__(self, radius): 
        self._radius = radius
    
    @property 
    def radius(self): 
        return self._radius 
    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value


def display(): 
    print("display function ran!")

@decorator_class
def display_basic(): 
    print("display_basic function ran!")

#@decorator_function_with_arguments
#@decorator_class
#@my_logger
@prefix_decorator('LOG:')
def display_info(name, age): 
    print(f'display info ran with arguments {name} and {age}')


@my_timer
def display_time(): 
    from datetime import datetime 
    time = datetime.now()
    print(time)


# Underrstaing @d2l_add_to_class decorator in d2l 

def add_to_class(Class):                 # When add_to_class is called with a class as an argument, it returns the wrapper function.
    def wrapper(obj):                    # Then, when the wrapper function is called with an object (typically a method) as its argument, it dynamically adds that object as an attribute to the class Class. 
        setattr(Class, obj.__name__, obj) # The attribute's name is determined by obj.__name__, which is the name of the method being added.
    return wrapper

if __name__ == '__main__': 
    my_func = outer_function("Hello World!")          # This won't execute the inner function will only save it in the my_func
    my_func()                                         # Now executing it 
    # decorator example 
    decorator_display = decorator_function(display)  
    decorator_display()
    """ having 
    @decorator_function
    def display(): 
        print("display function!") is equal to display = decorator_function(display)"""
    display_basic()
    display_info('angle', '56')
    display()
    display_time()
    C = Circle(5) 
    print(C.radius)


