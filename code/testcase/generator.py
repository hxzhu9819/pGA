import math
import random 
import sys

n = int(sys.argv[2])
m = 100
def dist(a,b):
    return math.sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 )

points = [ [random.randint(0,m), random.randint(0,m) ] for i in range(n) ]
dist_mat = [ [ int(dist(points[i], points[j]))for i in range(n) ] for j in range(n) ]
# print(dist_mat)
with open(str(sys.argv[1]), "w") as f:
    f.write(str(n))
    f.write('\n')
    for line in dist_mat:
        line = [str(x) for x in line]
        f.write(" ".join(line))
        f.write('\n')

