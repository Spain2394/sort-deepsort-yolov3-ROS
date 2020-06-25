#!/usr/bin/env python

a_buff_limit = 4

a = [1,2,3,4,4,4,5,6]
b = [1,2,3,4,5,6,7,8]

a_size = len(a)
b_size = len(b)

i = 0 
j = 0 

buff_thresh = 4

# x ^3 complexity 
while a_size > buff_thresh: 
    i = 0
    j = 0
    while i < a_size:
        j = 0
        while j < b_size: 
            if a[i] == b[i]:
                a.pop(i)
                a_size = a_size - 1
                b.pop(j)
                b_size = b_size - 1
                break
            j += 1
        i += 1
            
print(a)
print(b)
print("still %s items in a, and %s in b" % (str(len(a)), str(len(b))))            
            
        
