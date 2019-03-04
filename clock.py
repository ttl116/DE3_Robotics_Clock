#!/usr/bin/env python

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Inverse Kinematics Pick and Place Demo
"""
import argparse
import struct
import sys
import copy

import rospy
import rospkg
import numpy as np

from tf.transformations import *

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface

import threading

import time

exitFlag = 0

LEFT = True
RIGHT = False
PICKUP = True
PUTDOWN = False

#------------------------------------------------------------------
#Initialising Pick and Place class


class PickAndPlace(object):
    def __init__(self, limb, hover_distance = 0.2, verbose=True):
        self._limb_name = limb # string
        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_open()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            print('ERROR')
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            print('ik_request, final layer')
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        elif self._limb_name == 'left':
            rospy.logerr("No Joint Angles provided for move_to_joint_positions for left arm. Moving back to starting position.")
            self._limb.move_to_joint_positions({'left_w0': 0.50139,
                                                 'left_w1': 1.40508,
                                                 'left_w2': -0.30773584,
                                                 'left_e0': -1.69264941,
                                                 'left_e1': 1.8151054,
                                                 'left_s0': 0.75142667,
                                                 'left_s1': -1.08935144})
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions for right arm. Moving back to starting position.")
            self._limb.move_to_joint_positions({'right_w0': 0.0164238,
                                                  'right_w1': 1.17581551,
                                                  'right_w2': -0.37732007,
                                                  'right_e0':  0.022188204,
                                                  'right_e1': 2.03023082,
                                                  'right_s0': 0.4355969,
                                                  'right_s1': -1.6393228})

    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(1.0)

    def _approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self.ik_request(approach)
        self._guarded_move_to_joint_position(joint_angles)

    def _retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        self._guarded_move_to_joint_position(joint_angles)

    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)

    def pick(self, pose):
        # open the gripper
        self.gripper_open()
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # close gripper
        self.gripper_close()
        # retract to clear object
        self._retract()

    def place(self, pose):
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # open the gripper
        self.gripper_open()
        # retract to clear object
        self._retract()

#---------------------------------------------------------------------------------
#Loading Models

def load_gazebo_models(table_pose=Pose(position=Point(x=1.35, y=-0.09, z=-0.1)),
                       table_reference_frame="world",
                       # Hour Hand
                       brick1_pose=Pose(position=Point(x=0.4125, y=-0.590, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 1 HA0
                       brick1_reference_frame="world",
                       brick2_pose=Pose(position=Point(x=0.510, y=-0.4475, z=0.7)), # brick 2 HA1
                       brick2_reference_frame="world",
                       brick3_pose=Pose(position=Point(x=0.720, y=-0.4475, z=0.7)), # brick 3 HA2
                       brick3_reference_frame="world",
                       brick4_pose=Pose(position=Point(x=0.8175, y=-0.590, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 4 HA3
                       brick4_reference_frame="world",
                       brick5_pose=Pose(position=Point(x=0.720, y=-0.7325, z=0.7)), # brick 5 HA4
                       brick5_reference_frame="world",
                       brick6_pose=Pose(position=Point(x=0.510, y=-0.7325, z=0.7)), # brick 6 HA5
                       brick6_reference_frame="world",
                       # brick7_pose=Pose(position=Point(x=0.6050, y=-0.59, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 7 HA6
                       # brick7_reference_frame="world",
                       brick8_pose=Pose(position=Point(x=0.5125, y=-0.200, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 8 HB0
                       brick8_reference_frame="world",
                       brick9_pose=Pose(position=Point(x=0.6100, y=-0.0575, z=0.7)), # brick 9 HB1
                       brick9_reference_frame="world",
                       brick10_pose=Pose(position=Point(x=0.820, y=-0.0575, z=0.7),), # brick 10 HB2
                       brick10_reference_frame="world",
                       brick11_pose=Pose(position=Point(x=0.9175, y=-0.200, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 11 HB3
                       brick11_reference_frame="world",
                       brick12_pose=Pose(position=Point(x=0.8200, y=-0.3425, z=0.7)), # brick 12 HB4
                       brick12_reference_frame="world",
                       brick13_pose=Pose(position=Point(x=0.6100, y=-0.3425, z=0.7)), # brick 13 HB5
                       brick13_reference_frame="world",
                       # brick14_pose=Pose(position=Point(x=0.7050, y=-0.2000, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 14 HB6
                       # brick14_reference_frame="world",
                       # Minute Hand
                       brick15_pose=Pose(position=Point(x=0.5125, y=0.200, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 15 MA0
                       brick15_reference_frame="world",
                       brick16_pose=Pose(position=Point(x=0.6100, y=0.3425, z=0.7)), # brick 16 MA1
                       brick16_reference_frame="world",
                       brick17_pose=Pose(position=Point(x=0.820, y=0.3425, z=0.7)), # brick 17 MA2
                       brick17_reference_frame="world",
                       brick18_pose=Pose(position=Point(x=0.9175, y=0.200, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 18 MA3
                       brick18_reference_frame="world",
                       brick19_pose=Pose(position=Point(x=0.8200, y=0.05750, z=0.7)), # brick 19 MA4
                       brick19_reference_frame="world",
                       brick20_pose=Pose(position=Point(x=0.6100, y=0.0575, z=0.7)), # brick 20 MA5
                       brick20_reference_frame="world",
                       # brick21_pose=Pose(position=Point(x=0.7050, y=0.2000, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 21 MA6
                       # brick21_reference_frame="world",
                       brick22_pose=Pose(position=Point(x=0.4125, y=0.590, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 22 MB0
                       brick22_reference_frame="world",
                       brick23_pose=Pose(position=Point(x=0.5100, y=0.7325, z=0.7)), # brick 23 MB1
                       brick23_reference_frame="world",
                       brick24_pose=Pose(position=Point(x=0.7200, y=0.7325, z=0.7)), # brick 24 MB2
                       brick24_reference_frame="world",
                       brick25_pose=Pose(position=Point(x=0.8175, y=0.5900, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 25 MB3
                       brick25_reference_frame="world",
                       brick26_pose=Pose(position=Point(x=0.7200, y=0.4475, z=0.7)), # brick 26 MB4
                       brick26_reference_frame="world",
                       brick27_pose=Pose(position=Point(x=0.5100, y=0.4475, z=0.7)), # brick 27 MB5
                       brick27_reference_frame="world",
                       # brick28_pose=Pose(position=Point(x=0.6050, y=0.5900, z=0.7),orientation=Quaternion(x=0, y=0, z=0.70738827, w=0.70738827)), # brick 28 MB6
                       # brick28_reference_frame="world",
                       brick29_pose=Pose(position=Point(x=0.220, y=0.5975, z=0.7)), # brick 29 MS0
                       brick29_reference_frame="world",
                       brick30_pose=Pose(position=Point(x=0.220, y=0.7025, z=0.7)), # brick 30 MS1
                       brick30_reference_frame="world",):

    # Get Models' Path
    model_path = rospkg.RosPack().get_path('baxter_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "tables_newnew/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    # Load Brick1 SDF
    brick1_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick1_file:
        brick1_xml=brick1_file.read().replace('\n', '')
    # Load Brick2 SDF
    brick2_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick2_file:
        brick2_xml=brick2_file.read().replace('\n', '')
    # Load Brick3 SDF
    brick3_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick3_file:
        brick3_xml=brick3_file.read().replace('\n', '')
    # Load Brick4 SDF
    brick4_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick4_file:
        brick4_xml=brick4_file.read().replace('\n', '')
    # Load Brick5 SDF
    brick5_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick5_file:
        brick5_xml=brick5_file.read().replace('\n', '')
    # Load Brick6 SDF
    brick6_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick6_file:
        brick6_xml=brick6_file.read().replace('\n', '')
    # Load Brick7 SDF
    # brick7_xml = ''
    # with open (model_path + "new_brick/model.sdf", "r") as brick7_file:
    #     brick7_xml=brick7_file.read().replace('\n', '')
    # Load Brick8 SDF
    brick8_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick8_file:
        brick8_xml=brick8_file.read().replace('\n', '')
    # Load Brick9 SDF
    brick9_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick9_file:
        brick9_xml=brick9_file.read().replace('\n', '')
    # Load Brick10 SDF
    brick10_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick10_file:
        brick10_xml=brick10_file.read().replace('\n', '')
    # Load Brick11 SDF
    brick11_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick11_file:
        brick11_xml=brick11_file.read().replace('\n', '')
    # Load Brick12 SDF
    brick12_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick12_file:
        brick12_xml=brick12_file.read().replace('\n', '')
    # Load Brick13 SDF
    brick13_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick13_file:
        brick13_xml=brick13_file.read().replace('\n', '')
    # Load Brick14 SDF
    # brick14_xml = ''
    # with open (model_path + "new_brick/model.sdf", "r") as brick14_file:
    #     brick14_xml=brick14_file.read().replace('\n', '')
    # Load Brick15 SDF
    brick15_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick15_file:
        brick15_xml=brick15_file.read().replace('\n', '')
    # Load Brick16 SDF
    brick16_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick16_file:
        brick16_xml=brick16_file.read().replace('\n', '')
    # Load Brick17 SDF
    brick17_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick17_file:
        brick17_xml=brick17_file.read().replace('\n', '')
    # Load Brick18 SDF
    brick18_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick18_file:
        brick18_xml=brick18_file.read().replace('\n', '')
    # Load Brick19 SDF
    brick19_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick19_file:
        brick19_xml=brick19_file.read().replace('\n', '')
    # Load Brick20 SDF
    brick20_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick20_file:
        brick20_xml=brick20_file.read().replace('\n', '')
    # Load Brick21 SDF
    # brick21_xml = ''
    # with open (model_path + "new_brick/model.sdf", "r") as brick21_file:
    #     brick21_xml=brick21_file.read().replace('\n', '')
    # Load Brick22 SDF
    brick22_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick22_file:
        brick22_xml=brick22_file.read().replace('\n', '')
    # Load Brick23 SDF
    brick23_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick23_file:
        brick23_xml=brick23_file.read().replace('\n', '')
    # Load Brick24 SDF
    brick24_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick24_file:
        brick24_xml=brick24_file.read().replace('\n', '')
    # Load Brick25 SDF
    brick25_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick25_file:
        brick25_xml=brick25_file.read().replace('\n', '')
    # Load Brick26 SDF
    brick26_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick26_file:
        brick26_xml=brick26_file.read().replace('\n', '')
    # Load Brick27 SDF
    brick27_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick27_file:
        brick27_xml=brick27_file.read().replace('\n', '')
    # Load Brick28 SDF
    # brick28_xml = ''
    # with open (model_path + "new_brick/model.sdf", "r") as brick28_file:
    #     brick28_xml=brick28_file.read().replace('\n', '')
    # Load Brick29 SDF
    brick29_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick29_file:
        brick29_xml=brick29_file.read().replace('\n', '')
    # Load Brick28 SDF
    brick30_xml = ''
    with open (model_path + "new_brick/model.sdf", "r") as brick30_file:
        brick30_xml=brick30_file.read().replace('\n', '')


    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("tables", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick1 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick1 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick1 = spawn_sdf("brick1", brick1_xml, "/",
                               brick1_pose, brick1_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick2 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick2 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick2 = spawn_sdf("brick2", brick2_xml, "/",
                               brick2_pose, brick2_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick3 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick3 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick3 = spawn_sdf("brick3", brick3_xml, "/",
                               brick3_pose, brick3_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick4 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick4 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick4 = spawn_sdf("brick4", brick4_xml, "/",
                               brick4_pose, brick4_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick5 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick5 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick5 = spawn_sdf("brick5", brick5_xml, "/",
                               brick5_pose, brick5_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick6 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick6 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick6 = spawn_sdf("brick6", brick6_xml, "/",
                               brick6_pose, brick6_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # # Spawn Brick7 SDF
    # rospy.wait_for_service('/gazebo/spawn_sdf_model')
    # try:
    #     spawn_brick7 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    #     resp_brick7 = spawn_sdf("brick7", brick7_xml, "/",
    #                            brick7_pose, brick7_reference_frame)
    # except rospy.ServiceException, e:
    #     rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick8 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick8 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick8 = spawn_sdf("brick8", brick8_xml, "/",
                               brick8_pose, brick8_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick9 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick9 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick9 = spawn_sdf("brick9", brick9_xml, "/",
                               brick9_pose, brick9_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick10 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick10 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick10 = spawn_sdf("brick10", brick10_xml, "/",
                               brick10_pose, brick10_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick11 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick11 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick11 = spawn_sdf("brick11", brick11_xml, "/",
                               brick11_pose, brick11_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick12 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick12 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick12 = spawn_sdf("brick12", brick12_xml, "/",
                               brick12_pose, brick12_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick13 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick13 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick13 = spawn_sdf("brick13", brick13_xml, "/",
                               brick13_pose, brick13_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick14 SDF
    # rospy.wait_for_service('/gazebo/spawn_sdf_model')
    # try:
    #     spawn_brick14 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    #     resp_brick14 = spawn_sdf("brick14", brick14_xml, "/",
    #                            brick14_pose, brick14_reference_frame)
    # except rospy.ServiceException, e:
    #     rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick15 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick15 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick15 = spawn_sdf("brick15", brick15_xml, "/",
                               brick15_pose, brick15_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick16 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick16 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick16 = spawn_sdf("brick16", brick16_xml, "/",
                               brick16_pose, brick16_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick17 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick17 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick17 = spawn_sdf("brick17", brick17_xml, "/",
                               brick17_pose, brick17_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick18 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick18 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick18 = spawn_sdf("brick18", brick18_xml, "/",
                               brick18_pose, brick18_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick19 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick19 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick19 = spawn_sdf("brick19", brick19_xml, "/",
                               brick19_pose, brick19_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick20 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick20 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick20 = spawn_sdf("brick20", brick20_xml, "/",
                               brick20_pose, brick20_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick21 SDF
    # rospy.wait_for_service('/gazebo/spawn_sdf_model')
    # try:
    #     spawn_brick21 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    #     resp_brick21 = spawn_sdf("brick21", brick21_xml, "/",
    #                            brick21_pose, brick21_reference_frame)
    # except rospy.ServiceException, e:
    #     rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick22 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick22 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick22 = spawn_sdf("brick22", brick22_xml, "/",
                               brick22_pose, brick22_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick23 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick23 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick23 = spawn_sdf("brick23", brick23_xml, "/",
                               brick23_pose, brick23_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick24 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick24 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick24 = spawn_sdf("brick24", brick24_xml, "/",
                               brick24_pose, brick24_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick25 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick25 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick25 = spawn_sdf("brick25", brick25_xml, "/",
                               brick25_pose, brick25_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick26 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick26 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick26 = spawn_sdf("brick26", brick26_xml, "/",
                               brick26_pose, brick26_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick27 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick27 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick27 = spawn_sdf("brick27", brick27_xml, "/",
                               brick27_pose, brick27_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick28 SDF
    # rospy.wait_for_service('/gazebo/spawn_sdf_model')
    # try:
    #     spawn_brick28 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    #     resp_brick28 = spawn_sdf("brick28", brick28_xml, "/",
    #                            brick28_pose, brick28_reference_frame)
    # except rospy.ServiceException, e:
    #     rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Brick29 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick29 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick29 = spawn_sdf("brick29", brick29_xml, "/",
                               brick29_pose, brick29_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # # Spawn Brick30 SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_brick30 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_brick30 = spawn_sdf("brick30", brick30_xml, "/",
                               brick30_pose, brick30_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))


def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("tables")
        resp_delete = delete_model("brick1")
        resp_delete = delete_model("brick2")
        resp_delete = delete_model("brick3")
        resp_delete = delete_model("brick4")
        resp_delete = delete_model("brick5")
        resp_delete = delete_model("brick6")
        # resp_delete = delete_model("brick7")
        resp_delete = delete_model("brick8")
        resp_delete = delete_model("brick9")
        resp_delete = delete_model("brick10")
        resp_delete = delete_model("brick11")
        resp_delete = delete_model("brick12")
        resp_delete = delete_model("brick13")
        # resp_delete = delete_model("brick14")
        resp_delete = delete_model("brick15")
        resp_delete = delete_model("brick16")
        resp_delete = delete_model("brick17")
        resp_delete = delete_model("brick18")
        resp_delete = delete_model("brick19")
        resp_delete = delete_model("brick20")
        # resp_delete = delete_model("brick21")
        resp_delete = delete_model("brick22")
        resp_delete = delete_model("brick23")
        resp_delete = delete_model("brick24")
        resp_delete = delete_model("brick25")
        resp_delete = delete_model("brick26")
        resp_delete = delete_model("brick27")
        # resp_delete = delete_model("brick28")
        resp_delete = delete_model("brick29")
        resp_delete = delete_model("brick30")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))


#------------------------------------------------------------------------------------------
## Multithreading class

class myThread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      if self.name == "Thread-1":
         print ("Starting " + self.name)
         pick_and_place_left_threaded ()
         print ("Exiting " + self.name)
      else:
         print ("Starting " + self.name)
         pick_and_place_right_threaded ()
         print ("Exiting " + self.name) 

# Pick and place functions to go into multithreading 

def pick_and_place_left_threaded (): # Minute hand
    global idx
    global pnp
    global block_poses
    Display = "00"
    while not rospy.is_shutdown():
      # Time = time.ctime()[14:16]
      Time = "57" # to test
      if Time != Display: 
        Display = Time_change(LEFT, Display,Time) 
    return 0

def pick_and_place_right_threaded (): # Hour hand
    global idxR
    global pnpR
    global block_posesR
    Display = "00"
    while not rospy.is_shutdown():
      # Time = time.ctime()[11:13]
      Time = "12" # to test
      if Time != Display: 
        Display = Time_change(RIGHT, Display,Time)
    return 0

#----------------------------------------------------------------------------

def move_brick (side, brick_position, pickUp):
    global pnp
    global pnpR
    global block_poses
    global block_posesR
    if side == LEFT:
        print ("add brick LEFT, brick_position: " + str(brick_position) + " " + ("picking" if pickUp else "placing"))
        if BrickPlaces[brick_position][2] == 0:
            block_poses.append(Pose(
                    position=Point(x=(BrickPlaces[brick_position][0]*0.001), y=(BrickPlaces[brick_position][1]*0.001), z=0.05),
                    orientation=overhead_orientation))
        else:
            block_poses.append(Pose(
                position=Point(x=(BrickPlaces[brick_position][0]*0.001), y=(BrickPlaces[brick_position][1]*0.001), z=0.05),        
                orientation=Quaternion(x=overhead_orientation2[0],y=overhead_orientation2[1],z=overhead_orientation2[2],w=overhead_orientation2[3])))               
        if pickUp:
          pnp.pick(block_poses[-1])
        else:
          pnp.place(block_poses[-1])
    else:
        print ("add brick RIGHT, brick_position: " + str(brick_position) + " " + ("picking" if pickUp else "placing"))
        if BrickPlaces[brick_position][2] == 0:
            block_posesR.append(Pose(
                    position=Point(x=(BrickPlaces[brick_position][0]*0.001), y=(BrickPlaces[brick_position][1]*0.001), z=0.05),
                    orientation=overhead_orientation))
        else:
            block_posesR.append(Pose(
                position=Point(x=(BrickPlaces[brick_position][0]*0.001), y=(BrickPlaces[brick_position][1]*0.001), z=0.05),        
                orientation=Quaternion(x=overhead_orientation2[0],y=overhead_orientation2[1],z=overhead_orientation2[2],w=overhead_orientation2[3])))   
        if pickUp:
          pnpR.pick(block_posesR[-1])
        else:
          pnpR.place(block_posesR[-1])


def change_bricks (side, BricksIncoming,BricksOutgoing,Hstore,Mstore,Digit):
    global PrefixList


    #While we still want bricks

    Store = Mstore
    StorePrefix = 'MS'
    if side == RIGHT:
        Store = Hstore
        StorePrefix = 'HS'
        Digit += 2
    print ("change_bricks BricksIncoming: " + str(BricksIncoming))
    print ("change_bricks BricksOutgoing: " + str(BricksOutgoing))
    while BricksIncoming != []:
        #Where do we get them from?
        if BricksOutgoing == []:  #If we don't want to get rid of any get them from store
            #get more from the store
            brick_position = StorePrefix+str(Store)
            Store -= 1
            move_brick(side, brick_position, PICKUP)
        else:
            brick_position = PrefixList[Digit]+str(BricksOutgoing.pop())  #If we do, get it from the number
            move_brick(side, brick_position, PICKUP)
        brick_position = PrefixList[Digit]+str(BricksIncoming.pop()) 
        move_brick(side, brick_position, PUTDOWN)    # Either way, place it where it needs to be
        
    #While want to get rid of bricks
    while BricksOutgoing != []: 
        brick_position = PrefixList[Digit]+str(BricksOutgoing.pop())  #Pick Brick to go
        move_brick(side, brick_position, PICKUP)
        Store += 1
        brick_position = StorePrefix+str(Store) #Put it in storage
        move_brick(side, brick_position, PUTDOWN)
        #Put it in the store
    if side == RIGHT:
        return (Store, Mstore)
    else:
        return (Hstore, Store)



#------------------------------------------------------------------------
def initial_bricks():
    global BrickPlaces
    global SwapDictionary
    global Hstore
    global Mstore
    global PrefixList
    global StorePrefix
    global block_poses
    global block_posesR
    global BricksIncoming
    global BricksOutgoing

    print ("Initialised BrickPlaces")
    BrickPlaces = {#Hour Digit 1
                'HA0':(412.5,-590,1),
                'HA1':(510.0,-447.5,0),
                'HA2':(720.0,-447.5,0),
                'HA3':(817.5,-590,1),
                'HA4':(720,-732.5,0),
                'HA5':(510,-732.5,0),
                'HA6':(605,-590,1),
                #Hour Digit 2
                'HB0':(512.5,-200,1),
                'HB1':(610,-57.5,0),
                'HB2':(820,-57.5,0),
                'HB3':(917.5,-200,1),
                'HB4':(820,-342.5,0),
                'HB5':(610,-342.5,0),
                'HB6':(705,-200,1),
                #Hour Digit Store
                'HS0':(220,-497.5,0),
                'HS1':(220,-602.5,0),
                'HS2':(220,-707.5,0),
                'HS3':(220,-812.5,0),
                'HS4':(220,-917.5,0),
                'HS5':(10,-497.5,0),
                'HS6':(10,-602.5,0),
                'HS7':(10,-707.5,0),
                'HS8':(10,-812.5,0),
                #Minute Digit 1
                'MA0':(512.5,200,1),
                'MA1':(610,342.5,0),
                'MA2':(820,342.5,0),
                'MA3':(917.5,200,1),
                'MA4':(820,57.5,0),
                'MA5':(610,57.5,0),
                'MA6':(705,200,1),
                #Minute Digit 2
                'MB0':(412.5,590,1),
                'MB1':(510,732.5,0),
                'MB2':(720,732.5,0),
                'MB3':(817.5,590,1),
                'MB4':(720,447.5,0),
                'MB5':(510,447.5,0),
                'MB6':(605,590,1),
                #Minute Digit Store
                'MS0':(220,597.5,0),
                'MS1':(220,702.5,0),
                'MS2':(220,807.5,0),
                'MS3':(220,912.5,0),
                'MS4':(220,1017.5,0),
                'MS5':(10,597.5,0),
                'MS6':(10,702.5,0),
                'MS7':(10,807.5,0),
                'MS8':(10,912.5,0)}

#'FromTo':((To Go),(To Come))
    SwapDictionary = {'00':([],[]),     #<------This has been added since last test. Please make sure it is in the code.
                  '01':([0,3,4,5],[]),
                  '02':([2,5],[6]),
                  '03':([4,5],[6]),
                  '04':([0,3,5],[6]),
                  '05':([1,4],[6]),
                  '06':([1],[6]),
                  '07':([3,4,5],[]),
                  '08':([],[6]),
                  '09':([4],[6]),
                  '12':([2],[0,3,4,6]),
                  '23':([4],[2]),
                  '34':([0,3],[5]),
                  '45':([1],[0,3]),
                  '56':([],[4]),
                  '67':([3,4,5,6],[1]),
                  '78':([],[3,4,5,6]),
                  '89':([4],[]),
                  '90':([6],[4]),
                  '20':([6],[2,5]),
                  '30':([6],[4,5]),
                  '50':([6],[1,4])}

    print ("Initialised PrefixList")

    PrefixList = ['MB','MA','HB','HA']
    StorePrefix = ['MS','MS','HS','HS']

    block_poses = list()
    block_posesR = list()

    BricksIncoming = []

    BricksOutgoing = []

    Hstore = 1 #How many bricks do we have in the store (Hour)

    Mstore = 1 #How many bricks do we have in the store (Minute)

def string_change (side, SwapString):
  print ("string_change SwapString: " + str(SwapString))
  global SwapDictionary
  global Hstore
  global Mstore
  Digit = 0
  while SwapString != []:
      swap = SwapString.pop()
      print ("string_change swap: " + str(swap))
      come = list(SwapDictionary[swap][1])
      go = list(SwapDictionary[swap][0])
      Hstore, Mstore = change_bricks (side, come,go,Hstore,Mstore,Digit)
      Digit += 1
      print("Hstore, Mstore: " + str(Hstore) + ", " + str(Mstore))

def Time_change (side, display, time):
  print ("Time_change display: " + str(display) + " Time: " + str(time))
  String = []
  Dis = list(display)
  Tim = list(time)
  while Dis != []:
      D = Dis.pop(0)
      T = Tim.pop(0)
      if D != T:
          String.append(D+T)
          while Dis != []:
              D = Dis.pop(0)
              T = Tim.pop(0)
              String.append(D+T)
  string_change (side, String)
  display = time
  return display


def main():
    global pnp
    global pnpR
    global idx
    global idxR
    global overhead_orientation
    global overhead_orientation2
    global overhead_orientation3
    """RSDK Inverse Kinematics Pick and Place Example
    A Pick and Place example using the Rethink Inverse Kinematics
    Service which returns the joint angles a requested Cartesian Pose.
    This ROS Service client is used to request both pick and place
    poses in the /base frame of the robot.
    Note: This is a highly scripted and tuned demo. The object location
    is "known" and movement is done completely open loop. It is expected
    behavior that Baxter will eventually mis-pick or drop the block. You
    can improve on this demo by adding perception and feedback to close
    the loop.
    """
    rospy.init_node("ik_pick_and_place_demo")

    # Remove models from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_models)

    # Wait for the All Clear from emulator startup
    rospy.wait_for_message("/robot/sim/started", Empty)

    limb = 'left'
    hover_distance = 0.2 # meters
    # Starting Joint angles for left arm
    starting_joint_angles = {'left_w0': 0.50139,
                             'left_w1': 1.40508,
                             'left_w2': -0.30773584,
                             'left_e0': -1.69264941,
                             'left_e1': 1.8151054,
                             'left_s0': 0.75142667,
                             'left_s1': -1.08935144}
    pnp = PickAndPlace(limb, hover_distance)
    print ("Left limb picking up")
    # An orientation for gripper fingers to be overhead and parallel to the obj
    overhead_orientation = Quaternion(
                             x=-0.0249590815779,
                             y=0.999649402929,
                             z=0.00737916180073,
                             w=0.00486450832011)
    #print('overhead')
    #print(overhead_orientation)
    #print(type(overhead_orientation))

    orig = np.array([-0.0249590815779,0.999649402929,0.00737916180073,0.00486450832011]) #quaternion_from_euler(0,1,0)
    #print(type(orig))
    overhead_orientation2 = quaternion_multiply(quaternion_from_euler(0,0,1.57),orig)
    #print(type(overhead_orientation2))
    overhead2 = Quaternion(x=overhead_orientation2[0], y=overhead_orientation2[1], z=overhead_orientation2[2], w=overhead_orientation2[3])
    #print('overhead2: ')
    #print(overhead2)
    #print(type(overhead2))
    #overhead_orientation2.normalize()

    orig = np.array([-0.0249590815779,0.999649402929,0.00737916180073,0.00486450832011])
    overhead_orientation3 = quaternion_multiply(quaternion_from_euler(0,0,-1.57),orig)
    #print('Orient 3: ')
    #print(overhead_orientation3)
    #print(type(overhead_orientation3))
    overhead3 = Quaternion(x=overhead_orientation3[0],y=overhead_orientation3[1], z=overhead_orientation3[2], w=overhead_orientation3[3])
    #print('overhead3: ')
    #print(overhead3)
    #print(type(overhead3))

    limbR = 'right'
    hover_distance = 0.15 # meters
    # Starting Joint angles for right arm
    starting_joint_anglesR = {'right_w0': 0.0164238,
                             'right_w1': 1.17581551,
                             'right_w2': -0.37732007,
                             'right_e0':  0.022188204,
                             'right_e1': 2.03023082,
                             'right_s0': 0.4355969,
                             'right_s1': -1.6393228}

    pnpR = PickAndPlace(limbR, hover_distance)
    print ("Right limb picking up")
    # An orientation for gripper fingers to be overhead and parallel to the obj

    #####################
    #block_poses = list() # -  defined in initalise_bricks
    # The Pose of the block in its initial location.
    # You may wish to replace these poses with estimates
    # from a perception node.

    #M-- pick/place 0 uses overhead_orientation
    #block_poses.append(Pose(
     #   position=Point(x=0.7, y=0.30, z=0.05),
      #  orientation=Quaternion(x=overhead_orientation2[0],y=overhead_orientation2[1],z=overhead_orientation2[2],w=overhead_orientation2[3])))
    #block_poses.append(Pose(
     #   position=Point(x=0.7, y=0.30, z=0.05),
      #  orientation=overhead_orientation))
    # block_posesR = list()  # -  defined in initalise_bricks
    # The Pose of the block in its initial location

    # Move to the desired starting angles
    pnp.move_to_start(starting_joint_angles)
    pnpR.move_to_start(starting_joint_anglesR)

    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    print("Initialising Bricks")
    initial_bricks ()
    print("Loading Gazebo Models")
    load_gazebo_models()

    #Create 2 seperate indexes
    idx = 0
    idxR = 0

    #Threading
    threadLock = threading.Lock()
    threads = []
    thread1 = myThread(1, "Thread-1", 1)
    thread2 = myThread(2, "Thread-2", 2)

    thread1.start()
    thread2.start()

    threads.append(thread1)
    threads.append(thread2)

    for t in threads:
        t.join()

    print ("Exiting Main Thread")

if __name__ == '__main__':
	sys.exit(main())
