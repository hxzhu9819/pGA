# python tsplib_loader.py <inputfile.tsp> <outputfilename> <dimension>
import math
import random 
import sys

n = int(sys.argv[3])
m = 100
def dist(a,b):
    return math.sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 )

points = []
ignore = True
with open(str(sys.argv[1]), "r") as f:
    for line in f:
        if line.startswith("NODE_COORD_SECTION"):
            ignore = False
        elif ignore:
            continue
        else:
            point = list(map(float, line.split()))
            points.append(point[1:])

print(points)


dist_mat = [ [ int(dist(points[i], points[j]))for i in range(n) ] for j in range(n) ]
# print(dist_mat)
with open(str(sys.argv[2]), "w") as f:
    f.write(str(n))
    f.write('\n')
    for line in dist_mat:
        line = [str(x) for x in line]
        f.write(" ".join(line))
        f.write('\n')

