

def distance(target, source, insertcost, deletecost, replacecost):
    n = len(target) + 1
    m = len(source) + 1

    # set up dist and init values
    dist = [ [(0,"") for j in range(m)] for i in range(n)]
    for i in range(1,n):
        dist[i][0] = dist[i-1][0][0] + insertcost, ""
    for j in range(1,m):
        dist[0][j] = dist[0][j-1][0] + deletecost, ""

    # align source and target strings
    for j in range(1,m):
        #check(dist)
        for i in range(1,n):
            inscost = insertcost + dist[i][j-1][0]
            delcost = deletecost + dist[i-1][j][0]
            add = 0 if source[j-1] == target[i-1] else replacecost
            substcost = add + dist[i-1][j-1][0]
            dist[i][j] = min_op(inscost, delcost, substcost, add)

    # return min edit distance
    return dist

def min_op(inscost, delcost, substcost, add):
    minimum = min(inscost, delcost, substcost)
    op = None
    if minimum == substcost:
        op = "S" if add > 0 else "A" 
    elif minimum == delcost:
        op = "D"
    elif minimum == inscost:
        op = "I"
   
    return minimum, op

def backtrace(dist, m, n):
    ops = []
    j = m - 1
    i = n - 1

    while i > 0 and j > 0:
        op = dist[i][j][1]
        ops.append(op)
        if op == "I":
            j = j - 1
        elif op == "D":
            i = i - 1
        else:
            i = i -1
            j = j -1
    
    inscost = dist[i][j-1][0]
    delcost = dist[i-1][j][0]
    substcost = dist[i-1][j-1][0]

    minimum = min(inscost, delcost, substcost)

    if minimum == substcost:
        ops.append("S")
    elif minimum == delcost:
        ops.append("D")
    else:
        ops.append("I")

    return list(reversed(ops))

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
        print "levenshtein distance =", dist[n-1][m-1][0]
        ops = backtrace(dist, m, n)
        visualize(argv[1], argv[2], ops)
       

