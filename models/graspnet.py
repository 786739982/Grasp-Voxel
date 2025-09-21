""" GraspNet baseline model definition.
    Author: chenxi-wang
"""

"""
Points to be optimization:
    1. xyz or voxel_pc predicted in the VoxelHead module can be prior to the EulerHead module.
    2. EulerHead module should be able to predict the grasp angle, width, and tolerance.
    3. EulerHead module should be predicting the Pitch/Yaw/Roll angles.
    4. VoxelHead module should predict Voxel with tow scores of objectness.
    5. The method that 'topk_flat_indices' can be optimization
    6. " views_cls, angles_cls = self.eulerhead(fused_feature) " maybe can be used where output is derictly views and angles with input is 1024 voxels
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from backbone import Pointnet2Backbone
from modules import ApproachNet, CloudCrop, OperationNet, ToleranceNet, VoxelHead, EulerHead, PoseNet
from loss import get_loss
from loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from voxel import BackBone, Bottleneck
from voxelization import voxelization, devoxelization_train

class GraspNetStage1(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, geom=None):
        super().__init__()
        # self.backbone = Pointnet2Backbone(input_feature_dim)
        self.yaw_backbone = BackBone(Bottleneck, [3, 6, 6, 3], geom, use_bn=True)
        self.pitch_backbone = BackBone(Bottleneck, [3, 6, 6, 3], geom, use_bn=True)
        self.roll_backbone = BackBone(Bottleneck, [3, 6, 6, 3], geom, use_bn=True)
        # self.vpmodule = ApproachNet(num_view, 160)
        self.voxelhead = VoxelHead(480, 160)
        self.posenet = PoseNet(480, 300, 12)
        # self.eulerhead = EulerHead(480, 300, 12)

    def forward(self, end_points):
        voxel_yaw = end_points['voxel_pc']
        voxel_pitch = voxel_yaw.clone()
        voxel_pitch = voxel_pitch.transpose(2, 3).contiguous()
        voxel_roll = voxel_yaw.clone()
        voxel_roll = voxel_roll.transpose(1, 3).contiguous()
        
        # seed_features, seed_xyz, end_points = self.backbone(pointcloud, end_points)
        yaw_feature = self.yaw_backbone(voxel_yaw)
        pitch_features = self.pitch_backbone(voxel_pitch)
        roll_features = self.roll_backbone(voxel_roll)
        # end_points = self.vpmodule(yaw_feature, pitch_features, roll_features, end_points)
        
        ## Assign the predicted euler angles to the 3D Voxel
        # shape = yaw_feature.shape
        # euler_tensor = torch.zeros((shape[0], shape[1], shape[2], shape[3], 3), device=yaw_feature.device, dtype=torch.int64)
        # euler_tensor[:, :, :, :, 0] = end_points['yaw_inds'].unsqueeze(-1)
        # euler_tensor[:, :, :, :, 1] = end_points['pitch_inds'].unsqueeze(-2)
        # euler_tensor[:, :, :, :, 2] = end_points['roll_inds'].unsqueeze(-3)
        # euler_score = torch.zeros((shape[0], shape[1], shape[2], shape[3], 3), device=yaw_feature.device, dtype=torch.float32)
        # euler_score[:, :, :, :, 0] = end_points['yaw_score'].unsqueeze(-1)
        # euler_score[:, :, :, :, 1] = end_points['pitch_score'].unsqueeze(-2)
        # euler_score[:, :, :, :, 2] = end_points['roll_score'].unsqueeze(-3)
        
        ## Pred Voxel
        fused_feature = torch.cat((yaw_feature, pitch_features, roll_features), dim=-1)
        # positive_voxel, negative_voxel = self.voxelhead(fused_feature)
        positive_voxel = self.voxelhead(fused_feature)
        # score_voxel = torch.stack([negative_voxel, positive_voxel], dim=4)
        # views_cls, angles_cls = self.eulerhead(fused_feature)
        fp2_xyz = []
        fp2_inds = []
        voxel_score = []
        # euler_inds = []
        # euler_inds_score = []
        indices = []
        # devoxelization
        for i in range(positive_voxel.shape[0]):
            topk_values, topk_flat_indices = torch.topk(positive_voxel[i].view(-1), 1024)
            topk_indices = torch.unravel_index(topk_flat_indices, positive_voxel[i].shape)
            indices_ = torch.stack(topk_indices, dim=1)
            seed_xyz, fp2_ind = devoxelization_train(end_points['indices_maps'][i], indices_, end_points['point_clouds'][i])
            x, y, z = indices_[:, 0], indices_[:, 1], indices_[:, 2]
            score = positive_voxel[i][x, y, z]
            # euler_ = euler_tensor[i][x, y, z]
            # euler_inds_score_ = euler_score[i][x, y, z]
            indices.append(indices_)
            # euler_inds_score.append(euler_inds_score_)
            fp2_xyz.append(seed_xyz)
            fp2_inds.append(fp2_ind)
            voxel_score.append(score)
            # euler_inds.append(euler_)

        view_cls, angle_cls, end_points = self.posenet(fused_feature, indices, end_points)

        end_points['fp2_xyz'] = torch.stack(fp2_xyz)
        end_points['fp2_inds'] = torch.stack(fp2_inds)
        end_points['indices'] = torch.stack(indices)
        end_points['voxel_score'] = torch.stack(voxel_score)
        # objectness_score = torch.stack(objectness_score)
        # end_points['objectness_score'] = objectness_score.permute(0, 2, 1)
        # end_points['euler_inds'] = torch.stack(euler_inds)
        # end_points['euler_inds_score'] = torch.stack(euler_inds_score)

        return end_points


class GraspNetStage2(nn.Module):
    def __init__(self, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=True):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training
        self.crop = CloudCrop(64, 3, cylinder_radius, hmin, hmax_list)
        self.operation = OperationNet(num_angle, num_depth)
        self.tolerance = ToleranceNet(num_angle, num_depth)
    
    def forward(self, end_points):
        pointcloud = end_points['point_clouds']
        if self.is_training:
            grasp_top_views_rot, _, _, _, end_points = match_grasp_view_and_label(end_points)
            seed_xyz = end_points['batch_grasp_point']
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']
            seed_xyz = end_points['fp2_xyz']

        vp_features = self.crop(seed_xyz, pointcloud, grasp_top_views_rot)
        end_points = self.operation(vp_features, end_points)
        end_points = self.tolerance(vp_features, end_points)

        return end_points

class GraspNet(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], geom=None, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.stage1 = GraspNetStage1(input_feature_dim, num_view, geom)
        # self.grasp_generator = GraspNetStage2(num_angle, num_depth, cylinder_radius, hmin, hmax_list, is_training)

    def forward(self, end_points):
        end_points = self.stage1(end_points)
        if self.is_training:
            end_points = process_grasp_labels(end_points)
            end_points = match_grasp_view_and_label(end_points)
        # end_points = self.grasp_generator(end_points)
        return end_points

def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        ## load predictions
        objectness_score = end_points['objectness_score'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_center = end_points['fp2_xyz'][i].float()
        approaching = end_points['grasp_top_view_xyz'][i].float()
        print('approaching:', approaching, approaching.shape)
        grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i]
        grasp_width = 1.2 * end_points['grasp_width_pred'][i]
        grasp_width = torch.clamp(grasp_width, min=0, max=GRASP_MAX_WIDTH)
        grasp_tolerance = end_points['grasp_tolerance_pred'][i]

        ## slice preds by angle
        # grasp angle
        grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)
        grasp_angle = grasp_angle_class.float() / 12 * np.pi
        # grasp score & width & tolerance
        grasp_angle_class_ = grasp_angle_class.unsqueeze(0)
        grasp_score = torch.gather(grasp_score, 0, grasp_angle_class_).squeeze(0)
        grasp_width = torch.gather(grasp_width, 0, grasp_angle_class_).squeeze(0)
        grasp_tolerance = torch.gather(grasp_tolerance, 0, grasp_angle_class_).squeeze(0)

        ## slice preds by score/depth
        # grasp depth
        grasp_depth_class = torch.argmax(grasp_score, 1, keepdims=True)
        grasp_depth = (grasp_depth_class.float()+1) * 0.01
        # grasp score & angle & width & tolerance
        grasp_score = torch.gather(grasp_score, 1, grasp_depth_class)
        grasp_angle = torch.gather(grasp_angle, 1, grasp_depth_class)
        grasp_width = torch.gather(grasp_width, 1, grasp_depth_class)
        grasp_tolerance = torch.gather(grasp_tolerance, 1, grasp_depth_class)

        ## slice preds by objectness
        objectness_pred = torch.argmax(objectness_score, 0)
        objectness_mask = (objectness_pred==1)
        grasp_score = grasp_score[objectness_mask]
        grasp_width = grasp_width[objectness_mask]
        grasp_depth = grasp_depth[objectness_mask]
        approaching = approaching[objectness_mask]
        grasp_angle = grasp_angle[objectness_mask]
        grasp_center = grasp_center[objectness_mask]
        grasp_tolerance = grasp_tolerance[objectness_mask]
        grasp_score = grasp_score * grasp_tolerance / GRASP_MAX_TOLERANCE

        ## convert to rotation matrix
        Ns = grasp_angle.size(0)
        approaching_ = approaching.view(Ns, 3)
        grasp_angle_ = grasp_angle.view(Ns)
        rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
        rotation_matrix = rotation_matrix.view(Ns, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids], axis=-1))
    return grasp_preds