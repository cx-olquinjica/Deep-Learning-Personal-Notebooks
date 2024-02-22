#! /Users/admin/miniconda3/envs/d2l/bin/python

"""Homegrown timing tools for function calls. 
Does total-time, best-time, and best-of-totals time. 
"""

import time, sys


timer = time.clock if sys.platform[:3] == 'win'  else time.time

def total(reps,func, *pargs, **kargs): 
    """Total time to run func() reps time. 
    Returns (Total time, last result).
    """

    repslist = list(range(reps))                                  
    start = timer()
    for i in repslist: 
        ret = func(*pargs, **kargs)
    elapsed = timer() - start 
    return (elapsed, ret) 

def bestof(reps, func, *pargs, **kargs): 
    """
    Quickest func() among reps runs. 
    Returns (best time, last result)
    """

    best = 2 ** 32
    for i in range(reps): 
        start = timer()
        ret = func(*pargs, **kargs)
        elapsed = timer() - start 
        if elapsed < best: best = elapsed 
    return (best, ret)

def bestoftotal(reps1, reps2, func, *pargs, **kargs): 
    """
    Best of totals: 
    (best of reps11 of (total of reps2 runs of func))
    """
    return bestof(reps1, total, reps2, func, *pargs, **kargs)


# time relative speeds of the list construction techniques

#reps = 10000
#repslist = list(range(reps))

def forLoop(): 
    res = []
    for x in repslist: 
        res.append(abs(x))
    return res

def listComp(): 
    return [abs(x) for x in repslist]

def mapCall(): 
    return list(map(abs, repslist))

def genExpr(): 
    return list(abs(x) for x in repslist)

def genFunc(): 
    def gen(): 
        for x in repslist: 
            yield abs(x)
    return list(gen())



if __name__ == '__main__': 
    #print(total(1000, pow, 2, 100)[0])
    reps = 10000
    repslist = list(range(reps))
    print(sys.version)
    for test in (forLoop, listComp, mapCall, genExpr, genFunc): 
        (bestof, (total, result)) = bestoftotal(5, 1000, test)
        print ('%-9s: %.5f => [%s...%s]' % (test.__name__, bestof, result[0], result[-1]))

