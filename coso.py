import numpy as np
a = int(input('\ta > '))
b = int(input('\tb > '))
k = 0
c = 0
e = a - 1
f = b - 1
m = np.zeros((a, b))
while(k != (a * b)):
    if k < a * b:
        for i in range(f, c - 1, -1):
            k += 1
            m[c][i] = k
    
    if k < a * b:
        for i in range(c + 1, e):
            k = k + 1
            m[i, c] = k
    if k < a * b:
        if c == f:
            k = k + 1
            m[e, f] = k
        for i in range(c, f):
            k = k + 1
            m[e, i] = k
    if k < a * b:
        for i in range(e, c, -1):
            k = k + 1
            m[i, f] = k
        e -= 1
        f -= 1
    c += 1

print(m)