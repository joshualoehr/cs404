################################################################################
# CSCI 404 - Assignment 2 - Josh Loehr & Robin Cosbey - distance.py            # 
#                                                                              # 
# Computes minimum number of edits (insertions, deletions, substitutions) that #
# can convert an input source string to an input target string. Outputs the    #
# minimum edit Levenshtein distance, and N visualizations of equivalent        #
# alignments.                                                                  #
#                                                                              # 
# usage: distance.py [-h] [-n N] target source                                 #
#                                                                              # 
# positional arguments:                                                        #
#       target      The input target string.                                   #
#       source      The input source string.                                   #
#                                                                              # 
# optional arguments:                                                          #
#       -h, --help  show this help message and exit                            #
#       -n N        The maximum number of alignments to print.                 #
#                                                                              #
################################################################################

import argparse

def argumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help="The input target string.")
    parser.add_argument('source', type=str, help="The input source string.")
    parser.add_argument('-n', type=int, default=50, help="The maximum number of alignments to print.")
    return parser.parse_args()

def distance(insertcost, deletecost, replacecost):
    """Populates the dist matrix with the results of the minimum edit distance algorithm
    Args:
    insertcost -- cost of insertion operation
    deletecost -- cost of deletion operation
    replacecost -- cost of substitution operation
    """
    tlen = len(target) + 1
    slen = len(source) + 1

    dist = [ [0 for t in range(tlen)] for s in range(slen)]
    for t in range(1,tlen):
        dist[0][t] = dist[0][t-1] + insertcost
    for s in range(1,slen):
        dist[s][0] = dist[s-1][0] + deletecost

    for s in range(1,slen):
        for t in range(1,tlen):
            inscost = insertcost + dist[s-1][t]
            delcost = deletecost + dist[s][t-1]
            add = 0 if source[s-1] == target[t-1] else replacecost
            substcost = add + dist[s-1][t-1]
            dist[s][t] = min(inscost, delcost, substcost)

    return dist

def min_op(dist, s, t, insertcost, deletecost, replacecost):
    """Returns a list of triples containing the coordinates and operation symbol
        of the next step in the backtrace of the dist matrix
    Args:
    dist -- the minimum edit distance matrix
    s -- current dist matrix index corresponding to the source string
    t -- current dist matrix index corresponding to the target string
    insertcost -- cost of insertion operation
    deletecost -- cost of deletion operation
    replacecost -- cost of substitution operation
    """
    coordinates = []

    inscost = dist[s][t-1] + insertcost
    delcost = dist[s-1][t] + deletecost
    substcost = dist[s-1][t-1] + replacecost

    if inscost == dist[s][t]:
        coordinates.append((s,t-1,"I"))
    if delcost == dist[s][t]:
        coordinates.append((s-1,t, "D"))
    if substcost == dist[s][t]:
        coordinates.append((s-1,t-1, "S" if replacecost > 0 else "A"))

    return coordinates

def backtrace(dist, s, t, ops, paths):
    """Recursively traverses the dist matrix collecting minimum edit alignments
    Args:
    dist -- the minimum edit distance matrix
    s -- current dist matrix index corresponding to the source string
    t -- current dist matrix index corresponding to the target string
    ops -- accumulated list of operations for a single path
    paths -- list of all accumulated ops lists
    """

    if s == 0 or t == 0:
        if s > 0:
             ops.append("D")
        elif t > 0:
             ops.append("I")
        cost = sum([1 for op in ops if op == 'I' or op == 'D'])
        cost += sum([2 for op in ops if op == 'S'])
        if cost == dist[-1][-1]:
            paths.append(list(reversed(ops)))
        return
    
    coordinates = min_op(dist, s, t, 1, 1, 0 if source[s-1] == target[t-1] else 2)
    for coord in coordinates:
        next_s, next_t, op = coord
        backtrace(dist, next_s, next_t, ops + [op], paths)
        
def visualize(path):
    """Prints the alignment of the target and source strings based on
        a given path
    Args:
    path -- a single accumulated list of operations through the dist matrix
    """
    t = 0
    s = 0
    tar = ""
    sou = ""
    mid = ""

    for op in path:
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

    print "%s\n%s\n%s" % (tar, mid, sou)

if __name__ == '__main__':
    args = argumentParser()
    source = args.source
    target = args.target

    slen = len(source) + 1
    tlen = len(target) + 1
    dist = distance(1, 1, 2)
    print "levenshtein distance =", dist[slen-1][tlen-1]

    paths = []
    backtrace(dist, slen-1, tlen-1, [], paths)

    for path in paths[:args.n]:
       visualize(path)
       print
       

