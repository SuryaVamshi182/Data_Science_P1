# Given two arrays, write a python function to return the intersection of the two X=[1,5,9,0] Y=[3,0,2,9] => returns [9,0]
# Convert the arrays into sets   => Sets automatically remove duplicates    => Find the intersection of the sets    => Convert back to arrays
X=[1,5,9,0]
Y=[3,0,2,9]
def intersection(X,Y):
    return list(set(X)&set(Y))
print(intersection(X,Y))

# Given an array find all the duplicates in the array For example: input: [1,2,3,1,3,6,5] output: [1,3]
arr = [1,2,3,1,3,6,5]

set1 = set()        # Keeps track of elements already seen
res = set()         # Stores elements that are duplicates

for i in arr:
    if i in set1:
        res.add(i)
    else:
        set1.add(i)
        
print(list(res))

# Given an integer array, return the maximum product of any three numbers in the array
import heapq

arr = [1, 10, -5, 1, -100]

def max_three(arr):
    a = heapq.nlargest(3, arr)      # 3 largest numbers
    b = heapq.nsmallest(2, arr)     # 2 smallest numbers (possibly negative)
    
    return max(
        a[0] * a[1] * a[2],         # 3 largest
        b[0] * b[1] * a[0]          # two smallest + largest
    )

print(max_three(arr))

# Given an integer array, find the sum of the largest contiguous subarray within the array A = [0,-1,-5,-2,3,14] it should return 17 because of [3,14]
# Kadane's Algorithm
# At every index we ask is it better to extend the array or start fresh?
arr = [0, -1, -5, -2, 3, 14]

def max_subarray(arr):
    max_sum = 0
    curr_sum = 0
    
    for num in arr:
        curr_sum += num
        max_sum = max(max_sum, curr_sum)
        
        if curr_sum<0:
            curr_sum = 0
            
    return max_sum

print(max_subarray(arr))

# Given and integer n and an integer k, output a list of all the combination of k numbers choosen from 1 to n. For example if 
# n=3 and k=2 [1,2],[1,3],[2,3]
from itertools import combinations

n=3
k=2

def find_combination(k,n):
    list_num = []
    comb = combinations(range(1, n+1), k)
    
    for i in comb:
        list_num.append(list(i))
        
    print("(k:{}, n:{}):".format(k,n))
    print(list_num, "\n")
print(find_combination(k,n))
