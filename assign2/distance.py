

def distance(target, source, insertcost, deletecost, replacecost):
    n = len(target) + 1
    m = len(source) + 1

    # set up dist and init values
    dist = [ [0 for j in range(m)] for i in range(n)]
    for i in range(1,n):
        dist[i][0] = dist[i-1][0] + insertcost
    for j in range(1,m):
        dist[0][j] = dist[0][j-1] + deletecost

    # align source and target strings
    for j in range(1,m):
        #check(dist)
        for i in range(1,n):
            inscost = insertcost + dist[i][j-1]
            delcost = deletecost + dist[i-1][j]
            add = 0 if source[j-1] == target[i-1] else replacecost
            substcost = add + dist[i-1][j-1]
            dist[i][j] = min(inscost, delcost, substcost)

    # return min edit distance
    return dist

def min_op(dist, i, j):
    coordinates = []
    inscost = dist[i][j-1]
    delcost = dist[i-1][j]
    substcost = dist[i-1][j-1]

    minimum = min(inscost, delcost, substcost)

    if inscost == minimum:
        coordinates.append((i,j-1,"I"))
    if delcost == minimum:
        coordinates.append((i-1,j, "D"))
    if substcost == minimum:
        coordinates.append((i-1,j-1, "S" if dist[i][j] != substcost else "A"))

    return coordinates

def backtrace(dist, i, j, ops, paths):

    if i == 0 or j == 0:
        paths.append(list(reversed(ops)))
        return
    
    coordinates = min_op(dist, i, j)
    for coord in coordinates:
        next_i, next_j, op = coord
        backtrace(dist, next_i, next_j, ops + [op], paths)
        
def visualize(source, target, ops):
    t = 0
    s = 0
    tar = ""
    sou = ""
    mid = ""

    for op in ops:
        if op == "I":
            sou += "_ "
            tar += target[t] + " "
            t += 1
            mid += "  "
        elif op == "D":
            sou += source[s] + " "
            s += 1
            tar += "_ "
            mid += "  "
        else:
            sou += source[s] + " "
            s += 1
            tar += target[t] + " "
            t += 1
            mid += "| " if op == "A" else "  "

    while t < len(target):
        tar += target[t] + " "
        t += 1

    while s < len(source):
        sou += source[s] + " "
        s += 1

    print "%s\n%s\n%s" % (sou, mid, tar)

def check(dist):
    print
    for row in reversed(dist):
        print " ".join(["%2s(%s)" % el for el in row])

if __name__ == '__main__':
    from sys import argv
    if len(argv) > 2:
        n = len(argv[1]) + 1
        m = len(argv[2]) + 1
        dist = distance(argv[1], argv[2], 1, 1, 2)
        print "levenshtein distance =", dist[n-1][m-1]
        paths = []
        backtrace(dist, n-1, m-1, [], paths)
        # check(dist)
        print "paths", paths
        #visualize(argv[1], argv[2], ops)
       

