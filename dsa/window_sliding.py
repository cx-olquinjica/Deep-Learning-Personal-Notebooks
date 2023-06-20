#! /Users/admin/miniconda3/envs/d2l/bin/python


def getMaxSum(arr, k):
    maxSum = 0
    windowSum = 0
    start = 0

    for i in range(len(arr)): 
        windowSum += arr[i]

        if ((i - start + 1) == k):
            maxSum = max(maxSum, windowSum)
            windowSum -= arr[start]
            start += 1
    return maxSum

