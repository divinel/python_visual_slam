#!/usr/bin/env python3
import numpy as np

def parse_P3x4(line):
    data = line.split(' ')
    assert len(data) == 13
    p_mat = np.array([float(val) for val in data[1:]])
    p_mat.resize(3,4)
    return p_mat

def get_R_t_from_P(P, invK):
    KR = P[:, 0:3]
    Kt = P[:, 3:]
    return (invK.dot(KR), invK.dot(Kt))

def load_calib_params(file):
    with open(file) as f:
        lines = f.readlines()
        Pmats = [parse_P3x4(line) for line in lines]
        K = Pmats[0][:,0:3]
        invK = np.linalg.inv(K)
        R_ts = [get_R_t_from_P(P, invK) for P in Pmats]
        return K, R_ts