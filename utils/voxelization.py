import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def passthrough(velo, geom, center):
        q = (center[0] + geom['W1'] < velo[:, 0]) * (velo[:, 0] < center[0] + geom['W2']) * \
            (center[1] + geom['L1'] < velo[:, 1]) * (velo[:, 1] < center[1] + geom['L2']) * \
            (center[2] + geom['H1'] < velo[:, 2]) * (velo[:, 2] < center[2] + geom['H2'])
        indices = np.where(q)[0]
        if indices.shape[0] == velo.shape[0]:
            return velo[indices, :]
        else:
            indices_copy = np.tile(indices[500], (velo.shape[0]-indices.shape[0]))
            indices = np.concatenate([indices, indices_copy], axis=0)
            return velo[indices, :]

def voxelization(velo, geom, center, training=False):
    velo_processed = np.zeros(geom['input_shape'], dtype=np.float32)
    velo = passthrough(velo, geom, center)
    velo_processed = np.zeros(geom['input_shape'], dtype=np.float32)
    # intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
    x_indices = ((velo[:, 1] - (center[1] + geom['L1'])) / geom['grid_size']).astype(np.int32)
    y_indices = ((velo[:, 0] - (center[0] + geom['W1'])) / geom['grid_size']).astype(np.int32)
    z_indices = ((velo[:, 2] - (center[2] + geom['H1'])) / geom['grid_size']).astype(np.int32)
    velo_processed[x_indices, y_indices, z_indices] = 1
    
    # velo_processed = velo_processed.transpose(2, 0, 1)
    if training:
        indices_maps = np.stack([x_indices, y_indices, z_indices], axis=1)
        return velo_processed, indices_maps
    return velo_processed

def devoxelization(voxel, geom, center):
    indices = np.argwhere(voxel==1)
    velo = np.zeros((indices.shape[0], 3))
    velo[:, 0] = indices[:, 1] * geom['grid_size'] + center[0] + geom['W1']
    velo[:, 1] = indices[:, 0] * geom['grid_size'] + center[1] + geom['L1'] 
    velo[:, 2] = indices[:, 2] * geom['grid_size'] + center[2] + geom['H1']
    return velo

def devoxelization_torch(indices, center):
    velo = torch.zeros((indices.shape[0], 3), dtype=torch.float32, device=indices.device)

    velo[:, 0] = indices[:, 1] * 0.04 + center[0] - 0.32
    velo[:, 1] = indices[:, 0] * 0.04 + center[1] - 0.32
    velo[:, 2] = indices[:, 2] * 0.04 + center[2] - 0.32
    
    return velo

def devoxelization_train(indices_maps, indices, cloud):
    raw_indices = []
    count = 0
    non_res = torch.tensor([0]).to('cuda:0')
    for i in range(indices.shape[0]):
        res = torch.argwhere((indices_maps == indices[i]).all(dim=1))
        if res.shape[0] != 0:
            res = res[0]
            count += 1
        else :
            res = non_res
        raw_indices.append(res)
        # TODO: Need to be better: Loss of compare the occupancy and the pred_voxel
    print('pred_voxel meet occupancy count: ', count)
    raw_indices = torch.stack(raw_indices).squeeze()
    seed_xyz = cloud[raw_indices]

    return seed_xyz, raw_indices