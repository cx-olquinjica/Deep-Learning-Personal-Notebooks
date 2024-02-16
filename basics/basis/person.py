#! /Users/admin/miniconda3/envs/d2l/bin/python




class Person: 
    def __init__(self, name, job=None, pay=0):
        self.name = name 
        self.job = job 
        self.pay = pay 
    
    def lastName(self): 
        return self.name.split()[-1]

    def giveRaise(self, percent): 
        self.pay = int(self.pay) * (1 + percent)

    def __str__(self): 
        #return '[Person: %s, %s]' % (self.name, self.pay)
        return f'[Person: {self.name}, {self.pay}]'


class Manager(Person): 
    def __init__(self, name, pay): 
        super().__init__( name, 'mgr', pay) # customizing the constructor/ best method
        #Person.__init__(self, name, 'mgr', pay) another way to call/customize the init method

    def giveRaise(self, percent, bonus=.10): 
        Person.giveRaise(self, percent + bonus)


# Operator Overloading
class Number: 
    def __init__(self, start): 
        self.data = start
    def __sub__(self, other): 
        return Number(self.data - other) 

# User-defined iterables 

class Squares: 
    def __init__(self, start, stop): 
        self.value = start - 1 
        self.stop = stop 
    def __iter__(self):                    # get iterable object on iter 
        return self 
    def __next__(self): 
        if self.value == self.stop: 
            raise StopIteration 
        self.value += 1 
        return self.value ** 2 

if __name__ == '__main__': 
    bob = Person('Bob Smith') 
    sue = Person('Sue Jones', job='dev', pay=1000000)
    print(bob ) 
    print(sue)
    print(bob.lastName(), sue.lastName())
    sue.giveRaise(.10)
    print(sue)
    tom = Manager('Tom Jones', 50000) #Make a manager: __ini__
    tom.giveRaise(.10)                      # Runs custom version 
    print(tom.lastName())                   # Runs inherited method 
    print(tom)                              # Runs inherited __repr__
    X = Number(5)
    Y = X - 2                               # Here Y is new Number instance
    print(Y.data)


    for i in Squares(1, 5):                # for calls iter which calls __iter__ 
        print(i, end=' ')                  # Each iteration calls __next__

