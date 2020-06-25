import numpy as np

def orthonomalize_rot_mat(R):
    '''
        orthonomalize rotation matrix R to avoid numerical side effect
        U, D, Vt = svd(R)
        R = U * Vt
    '''
    u, d, vt = np.linalg.svd(R)
    ortho_R = u.dot(vt)
    return ortho_R

def get_inverse_pose(pose):
    '''
        pose is (R, T) where R is 3x3, T is 3x1
        return (R^T, -R^T * T)
    '''
    Rt = pose[0].T
    inv_t = -Rt.dot(pose[1])
    return (Rt, inv_t)

def get_3x4_pose_mat(pose):
    '''
        pose is (R, T) where R is 3x3, T is 3x1
        return [R, T]
    '''
    res = np.zeros((3,4))
    res[:, :3] = pose[0]
    res[:, 3:] = pose[1]
    return res