# the key idea is that we utilize the observation from mcc-ho.
# which is a two stage foundation model that extract the hand pose from human
# and the relative ankle pose between human hand and object 6dof pose
import os
import sys





# thus we also need to policy to deploy this two state based part

# for the first policy, the observation is the object's 6dof pose and bounding box points
#  the output is the hand end effector pose


# the second policy, the observation is the geometry of object (mainly sparse sampled point clouds with means-std)
# the output is the robotic hand grasping pose



