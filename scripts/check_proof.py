#!/usr/bin/env python3

import sys
import getopt
import re
from sage.all import *


# Usage: sage ./check_proof.py -i path/to/input -b path/to/basis -p path/to/proof -e (0 if non-expanded proof, >0 else) 

def read_input(argv):
 
    if len(argv) != 9:
        raise ValueError("Required Inputs: -i path to input file, -b path to output basis, -p path to output proof, -e whether the proof is expanded or not")

    opts, args = getopt.getopt(argv[1:], "i:b:p:e:")
    
    input = None
    basis = None
    proof = None
    expanded = None
    
    for opt, arg in opts:
        if opt == "-i":
            input = open(arg, 'r')
        elif opt == "-b":
            basis = open(arg, 'r')
        elif opt == "-p":
            proof = open(arg, 'r')
        elif opt == "-e":
            expanded = int(arg) > 0
            
    return input, basis, proof, expanded

def replace_indices(s, F, G):
    # Replace [ik] with F[k]
    def replace_ik(match):
        index = int(match.group(1))
        return "(" + str(F[index]) + ")"
    
    # Replace [k] with G[k]
    def replace_k(match):
        index = int(match.group(1))
        return "(" + str(G[index]) + ")"

    # First replace [ik]
    s = re.sub(r'\[i(\d+)\]', replace_ik, s)
    
    # Then replace [k]
    s = re.sub(r'\[(\d+)\]', replace_k, s)
    
    return s
    
if __name__ == '__main__':  
    
    input, basis, proof, expanded = read_input(sys.argv)    
    
    vars = []
    while True:
        line = input.readline().rstrip("\n")
        vars += [v.strip() for v in line.split(",") if v.strip()]
        if not line.rstrip().endswith(","):
            break

    char = int(input.readline())
        
    K = QQ
    if char != 0:
        K = GF(char)
    R = FreeAlgebra(K,vars)
    
    F = [R(l.rstrip().strip(",")) for l in input]
    G = []

    lines_proof = proof.readlines()
    lines_basis = basis.readlines()

    if len(lines_proof) != len(lines_basis):
        print("Different number of lines: %d vs %d" % (len(lines_proof), len(lines_basis)))
        print("TEST FAILED")
        sys.exit(1)
    
    for i in range(len(lines_proof)):
        line_proof = lines_proof[i].rstrip().strip(",").split("+")
        line_basis = lines_basis[i].rstrip().strip(",")
        
        print("Line", i, ": ", end="")
        
        f_basis = R(line_basis)
                
        if expanded:
            f_proof = sum(R(replace_indices(l, F, F)) for l in line_proof)
        else:
            f_proof = sum(R(replace_indices(l, F, G)) for l in line_proof)
        
        if(f_basis != f_proof):
            print("FAILED! ", f_basis, " != ", f_proof)
            print(f_basis - f_proof)
            sys.exit(1)
        print("CORRECT")
        G.append(f_basis)    

                    