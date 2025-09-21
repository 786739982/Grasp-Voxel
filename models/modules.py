""" Modules for GraspNet baseline model.
    Author: chenxi-wang
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import pytorch_utils as pt_utils
from pointnet2_utils import CylinderQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim):
        """ Approach vector estimation from seed point features.

            Input:
                num_view: [int]
                    number of views generated from each each seed point
                seed_feature_dim: [int]
                    number of channels of seed point features
        """
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        # self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        # self.conv2 = nn.Conv1d(self.in_dim, 2+self.num_view, 1)
        # self.conv3 = nn.Conv1d(2+self.num_view, 2+self.num_view, 1)
        self.yaw_conv1 = nn.Conv2d(self.in_dim, self.in_dim, 1)
        self.yaw_conv2 = nn.Conv2d(self.in_dim, self.num_view, 1)
        self.yaw_conv3 = nn.Conv2d(self.num_view, self.num_view, 1)
        self.yaw_bn1 = nn.BatchNorm2d(self.in_dim)
        self.yaw_bn2 = nn.BatchNorm2d(self.num_view)

        self.pitch_conv1 = nn.Conv2d(self.in_dim, self.in_dim, 1)
        self.pitch_conv2 = nn.Conv2d(self.in_dim, self.num_view, 1)
        self.pitch_conv3 = nn.Conv2d(self.num_view, self.num_view, 1)
        self.pitch_bn1 = nn.BatchNorm2d(self.in_dim)
        self.pitch_bn2 = nn.BatchNorm2d(self.num_view)

        self.roll_conv1 = nn.Conv2d(self.in_dim, self.in_dim, 1)
        self.roll_conv2 = nn.Conv2d(self.in_dim, self.num_view, 1)
        self.roll_conv3 = nn.Conv2d(self.num_view, self.num_view, 1)
        self.roll_bn1 = nn.BatchNorm2d(self.in_dim)
        self.roll_bn2 = nn.BatchNorm2d(self.num_view)

    def forward(self, yaw_features, pitch_features, roll_features, end_points):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                yaw_features: [torch.FloatTensor, (batch_size,feature_dim,num_seed)
                    features of seed points
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        
        yaw_features = yaw_features.permute(0, 3, 1, 2)
        pitch_features = pitch_features.permute(0, 3, 1, 2)
        roll_features = roll_features.permute(0, 3, 1, 2)

        yaw_features = F.relu(self.yaw_bn1(self.yaw_conv1(yaw_features)), inplace=True)
        yaw_features = F.relu(self.yaw_bn2(self.yaw_conv2(yaw_features)), inplace=True)
        yaw_features = self.yaw_conv3(yaw_features)
        yaw_features = yaw_features.permute(0, 2, 3, 1)  # (B, H, W, yaw_angles)
        top_yaw_scores, top_yaw_inds = torch.max(yaw_features, dim=3) 
        end_points['yaw_score'] = top_yaw_scores
        end_points['yaw_inds'] = top_yaw_inds

        pitch_features = F.relu(self.pitch_bn1(self.pitch_conv1(pitch_features)), inplace=True)
        pitch_features = F.relu(self.pitch_bn2(self.pitch_conv2(pitch_features)), inplace=True)
        pitch_features = self.pitch_conv3(pitch_features)
        pitch_features = pitch_features.permute(0, 2, 3, 1)  # (B, H, W, pitch_angles)
        top_pitch_scores, top_pitch_inds = torch.max(pitch_features, dim=3)
        end_points['pitch_score'] = top_pitch_scores
        end_points['pitch_inds'] = top_pitch_inds

        roll_features = F.relu(self.roll_bn1(self.roll_conv1(roll_features)), inplace=True)
        roll_features = F.relu(self.roll_bn2(self.roll_conv2(roll_features)), inplace=True)
        roll_features = self.roll_conv3(roll_features)
        roll_features = roll_features.permute(0, 2, 3, 1)  # (B, H, W, roll_angles)
        top_roll_scores, top_roll_inds = torch.max(roll_features, dim=3)
        end_points['roll_score'] = top_roll_scores
        end_points['roll_inds'] = top_roll_inds

        # print(view_score.min(), view_score.max(), view_score.mean())
        # top_view_scores, top_view_inds = torch.max(view_score, dim=2) # (B, num_seed)
        # top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
        # template_views = generate_grasp_views(self.num_view).to(features.device) # (num_view, 3)
        # template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous() #(B, num_seed, num_view, 3)
        # vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2) #(B, num_seed, 3)
        # vp_xyz_ = vp_xyz.view(-1, 3)
        # batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        # vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
        # end_points['grasp_top_view_inds'] = top_view_inds
        # end_points['grasp_top_view_score'] = top_view_scores
        # end_points['grasp_top_view_xyz'] = vp_xyz
        # end_points['grasp_top_view_rot'] = vp_rot

        return end_points

class PoseNet(nn.Module):
    def __init__(self, input_dim, num_view, num_angle):
        """ 6D Pose estimation from seed voxel features.

            Input:
                num_view: [int]
                    number of views generated from each each seed point
                seed_feature_dim: [int]
                    number of channels of seed point features
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 512)
        self.fc3 = nn.Linear(512, 512)  #TODO: Attention mechanism Maybe
        self.conv1 = nn.Conv1d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(256, 512, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.objectness = nn.Linear(512, 2)
        self.head_views = nn.Linear(512, num_view)
        self.head_angles = nn.Linear(512, num_angle)
        self.head_offset = nn.Linear(512, 144)
        self.head_tolerance = nn.Linear(512, 48)
        self.num_view = num_view
        self.num_angle = num_angle

    def forward(self, fused_feature, indices, end_points):
        """ Forward pass.

            Input:
                fused_feature: [torch.FloatTensor, (batch_size,H,W,input_dim)]
                    features of grouped points in different depths
                indices: [torch.LongTensor, (batch_size,num_seed,3)]
                    indices of seed points in the voxel space
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        fused_feature = F.relu(self.fc1(fused_feature))
        fused_feature = F.relu(self.fc2(fused_feature))
        fused_feature = F.relu(self.fc3(fused_feature))
        seeds_features = []
        for i in range(fused_feature.size(0)):
            seed_features = fused_feature[i][indices[i][:, 0], indices[i][:, 1], indices[i][:, 2]]
            seeds_features.append(seed_features)
        seeds_features = torch.stack(seeds_features, dim=0)
        B, num_seed = seeds_features.size()
        seeds_features = seeds_features.unsqueeze(1)
        seeds_features = F.relu(self.bn1(self.conv1(seeds_features)))
        seeds_features = F.relu(self.bn2(self.conv2(seeds_features)))
        seeds_features = F.relu(self.bn3(self.conv3(seeds_features)))
        seeds_features = seeds_features.squeeze(1)
        seeds_features = seeds_features.permute(0, 2, 1)
        objectness_score = self.objectness(seeds_features)
        view_score = self.head_views(seeds_features)
        angle_score = self.head_angles(seeds_features)
        offset_score = self.head_offset(seeds_features)
        offset_score = offset_score.view(-1, seeds_features.size(1), 36, 4).permute(0, 2, 1, 3)
        tolerance_score = self.head_tolerance(seeds_features)
        tolerance_score = tolerance_score.view(-1, seeds_features.size(1), 12, 4).permute(0, 2, 1, 3)

        end_points['objectness_score'] = objectness_score.permute(0, 2, 1)
        end_points['view_score'] = view_score
        end_points['angle_score'] = angle_score
        print('view_score', view_score, view_score.size())
        top_view_scores, top_view_inds = torch.max(view_score, dim=2) # (B, num_seed)
        top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
        template_views = generate_grasp_views(self.num_view).to(view_score.device) # (num_view, 3)
        template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous() #(B, num_seed, num_view, 3)
        vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2) #(B, num_seed, 3)
        end_points['grasp_top_view_inds'] = top_view_inds
        end_points['grasp_score_pred'] = offset_score[:, 0:self.num_angle]
        end_points['grasp_angle_cls_pred'] = offset_score[:, self.num_angle:2*self.num_angle]
        end_points['grasp_width_pred'] = offset_score[:, 2*self.num_angle:3*self.num_angle]
        end_points['grasp_tolerance_pred'] = tolerance_score
        end_points['grasp_top_view_xyz'] = vp_xyz
        print('top_view_inds', top_view_inds, top_view_inds.size())
        print('top_view_scores', top_view_scores, top_view_scores.size())
        print('vp_xyz', vp_xyz, vp_xyz.size())

        return view_score, angle_score, end_points

class CloudCrop(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.

        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04]):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [self.in_dim, 64, 128, 256]
        
        self.groupers = []
        for hmax in hmax_list:
            self.groupers.append(CylinderQueryAndGroup(
                cylinder_radius, hmin, hmax, nsample, use_xyz=True
            ))
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz, pointcloud, vp_rot):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                pointcloud: [torch.FloatTensor, (batch_size,num_seed,3)]
                    the points to be cropped
                vp_rot: [torch.FloatTensor, (batch_size,num_seed,3,3)]
                    rotation matrices generated from approach vectors

            Output:
                vp_features: [torch.FloatTensor, (batch_size,num_features,num_seed,num_depth)]
                    features of grouped points in different depths
        """
        B, num_seed, _, _ = vp_rot.size()
        num_depth = len(self.groupers)
        grouped_features = []
        for grouper in self.groupers:
            grouped_features.append(grouper(
                pointcloud, seed_xyz, vp_rot
            )) # (batch_size, feature_dim, num_seed, nsample)
        grouped_features = torch.stack(grouped_features, dim=3) # (batch_size, feature_dim, num_seed, num_depth, nsample)
        grouped_features = grouped_features.view(B, -1, num_seed*num_depth, self.nsample) # (batch_size, feature_dim, num_seed*num_depth, nsample)

        vp_features = self.mlps(
            grouped_features
        ) # (batch_size, mlps[-1], num_seed*num_depth, nsample)
        vp_features = F.max_pool2d(
            vp_features, kernel_size=[1, vp_features.size(3)]
        ) # (batch_size, mlps[-1], num_seed*num_depth, 1)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        return vp_features

        
class OperationNet(nn.Module):
    """ Grasp configure estimation.

        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """
    def __init__(self, num_angle, num_depth):
        # Output:
        # scores(num_angle)
        # angle class (num_angle)
        # width (num_angle)
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 3*num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed*num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0:self.num_angle]
        end_points['grasp_angle_cls_pred'] = vp_features[:, self.num_angle:2*self.num_angle]
        end_points['grasp_width_pred'] = vp_features[:, 2*self.num_angle:3*self.num_angle]
        return end_points

    
class ToleranceNet(nn.Module):
    """ Grasp tolerance prediction.
    
        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """
    def __init__(self, num_angle, num_depth):
        # Output:
        # tolerance (num_angle)
        super().__init__()
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed*num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        end_points['grasp_tolerance_pred'] = vp_features
        return end_points
    

class VoxelHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VoxelHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 512)
        self.positive = nn.Linear(512, output_dim)
        # self.negative = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, fused_feature):
        fused_feature = F.relu(self.fc1(fused_feature))
        fused_feature = F.relu(self.fc2(fused_feature))
        positive = torch.sigmoid(self.positive(fused_feature))
        # negative = self.negative(fused_feature)
        
        return positive


class EulerHead(nn.Module):
    def __init__(self, input_dim, num_view, num_angle):
        super(EulerHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 512)
        self.approach = nn.Sequential(
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, num_view)
                        )
        self.angle = nn.Sequential(
                            nn.Linear(512, 512),
                            nn.ReLU(),  
                            nn.Linear(512, num_angle)
                        )
        self.relu = nn.ReLU()
        

    def forward(self, fused_feature):
        fused_feature = F.relu(self.fc1(fused_feature))
        fused_feature = F.relu(self.fc2(fused_feature))
        approach_cls = self.approach(fused_feature)
        angle_cls = self.angle(fused_feature)

        return approach_cls, angle_cls

