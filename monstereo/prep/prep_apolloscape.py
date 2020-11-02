import os
import sys
import time
import math
import copy
import json
import logging
from collections import defaultdict
import datetime
import numpy as np
import torch
from ..utils import correct_angle, normalize_hwl, pixel_to_camera, to_spherical #,append_cluster

from ..network.process import preprocess_monoloco

from ..utils import K, KPS_MAPPING , APOLLO_CLUSTERS ,car_id2name, intrinsic_vec_to_mat, car_projection, pifpaf_info_extractor, keypoint_expander, keypoints_to_cad_model





def extract_box_average(boxes_3d):
    boxes_np = np.array(boxes_3d)
    means = np.mean(boxes_np[:, 3:], axis=0)
    stds = np.std(boxes_np[:, 3:], axis=0)
    print(means)
    print(stds)




def bbox_gt_extract(bbox_3d, kk):
    zc = np.mean(bbox_3d[:,2])
    
    #take the top right corner and the bottom left corner of the bounding box in the 3D spac
    corners_3d = np.array([[np.min(bbox_3d[:,0]), np.min(bbox_3d[:,1]), zc], [np.max(bbox_3d[:,0]), np.max(bbox_3d[:,1]), zc] ])
    
    box_2d = []
    
    for xyz in corners_3d:
        xx, yy, zz = np.dot(kk, xyz)
        uu = xx / zz
        vv = yy / zz
        box_2d.append(uu)
        box_2d.append(vv)

    return box_2d
    
def append_cluster(dic_jo, phase, xx, ys, kps):
    """Append the annotation based on its distance"""

    for clst in APOLLO_CLUSTERS:
        try:
            if ys[3] <= int(clst):
                dic_jo[phase]['clst'][clst]['kps'].append(kps)
                dic_jo[phase]['clst'][clst]['X'].append(xx)
                dic_jo[phase]['clst'][clst]['Y'].append(ys)
                break
        except ValueError:
            dic_jo[phase]['clst'][clst]['kps'].append(kps)
            dic_jo[phase]['clst'][clst]['X'].append(xx)
            dic_jo[phase]['clst'][clst]['Y'].append(ys)

    #print(dic_jo[phase]['clst'].keys())
    """if ys[3] <= 10:
        dic_jo[phase]['clst']['10']['kps'].append(kps)
        dic_jo[phase]['clst']['10']['X'].append(xx)
        dic_jo[phase]['clst']['10']['Y'].append(ys)
    elif ys[3] <= 20:
        dic_jo[phase]['clst']['20']['kps'].append(kps)
        dic_jo[phase]['clst']['20']['X'].append(xx)
        dic_jo[phase]['clst']['20']['Y'].append(ys)
    elif ys[3] <= 30:
        dic_jo[phase]['clst']['30']['kps'].append(kps)
        dic_jo[phase]['clst']['30']['X'].append(xx)
        dic_jo[phase]['clst']['30']['Y'].append(ys)
    elif ys[3] < 50:
        dic_jo[phase]['clst']['50']['kps'].append(kps)
        dic_jo[phase]['clst']['50']['X'].append(xx)
        dic_jo[phase]['clst']['50']['Y'].append(ys)
    else:
        dic_jo[phase]['clst']['>50']['kps'].append(kps)
        dic_jo[phase]['clst']['>50']['X'].append(xx)
        dic_jo[phase]['clst']['>50']['Y'].append(ys)"""




class PreprocessApolloscape:


    """Preprocess apolloscape dataset"""


    dic_jo = {'train': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                            clst=defaultdict(lambda: defaultdict(list))),
              'val': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                          clst=defaultdict(lambda: defaultdict(list))),
                          }
    dic_names = defaultdict(lambda: defaultdict(list))

    def __init__(self, dir_ann, dataset, kps_3d = False, buffer=20, radius=200):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.buffer = buffer
        self.radius = radius
        
        self.kps_3d = kps_3d
        
        self.dir_ann = dir_ann

        dir_apollo = os.path.join('data', 'apolloscape')
        dir_out = os.path.join('data', 'arrays')

        assert os.path.exists(dir_apollo), "apollo directory does not exists"
        assert os.path.exists(self.dir_ann), "The annotations directory does not exists"
        assert os.path.exists(dir_out), "Joints directory does not exists"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_joints = os.path.join(dir_out, 'joints-apolloscape-' + dataset + '-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-apolloscape-' + dataset + '-' + now_time + '.json')

        
        self.path  = os.path.join(dir_apollo, dataset)
        self.scenes, self.train_scenes, self.validation_scenes = factory(dataset, dir_apollo)

    def run(self):
        """
        Prepare arrays for training
        """
        cnt_scenes  = cnt_ann = 0
        start = time.time()
        
        for ii, scene in enumerate(self.scenes):
            
            if ii ==-100:
                print("val_scenes",self.validation_scenes)

            if ii==(-100):
                print("BREAK")
                break

            cnt_scenes +=1 
            
            scene_id = scene.split("/")[-1].split(".")[0]
            camera_id = scene_id.split("_")[-1]
            car_poses = os.path.join(self.path, "car_poses", scene_id + ".json")
            
            #print(scene_id, self.train_scenes[:5])
            if scene_id+".jpg" in self.train_scenes:
                phase = 'train'
            elif scene_id+".jpg" in self.validation_scenes:
                phase ='val'
            else:    
                print("phase name not in training or validation split")
                continue
                
            kk = K["Camera_"+camera_id]#intrinsic_vec_to_mat( K["Camera_"+camera_id])
            
            path_im = scene
            
            # Run IoU with pifpaf detections and save
            path_pif = os.path.join(self.dir_ann, scene_id+".jpg" + '.predictions.json')
                        
            if os.path.isfile(path_pif) and True:
                boxes_gt_list, boxes_3d_list, kps_list, ys_list, car_model_list  = self.extract_ground_truth_pifpaf(car_poses,camera_id, scene_id, path_pif)

   
            self.dic_names[scene_id+".jpg"]['boxes'] = copy.deepcopy(list(boxes_gt_list))
            self.dic_names[scene_id+".jpg"]['ys'] = copy.deepcopy(ys_list)
            self.dic_names[scene_id+".jpg"]['car_model'] = copy.deepcopy(car_model_list)
            self.dic_names[scene_id+".jpg"]['K'] = copy.deepcopy(intrinsic_vec_to_mat(kk).tolist())

            
            for kps, ys, boxes_gt, boxes_3d in zip(kps_list, ys_list, boxes_gt_list, boxes_3d_list):
                
                kps = [kps.transpose().tolist()]
                #kk = list(intrinsic_vec_to_mat(kk).transpose())
                
                self.dic_jo[phase]['kps'].append(kps)
                inp = preprocess_monoloco(kps,  intrinsic_vec_to_mat(kk).tolist(), kps_3d = self.kps_3d).view(-1).tolist()
                
                #print("inp 2.0 \n", inp)
                self.dic_jo[phase]['X'].append(inp)
                self.dic_jo[phase]['Y'].append(list(ys))
                self.dic_jo[phase]['names'].append(scene_id+".jpg")  # One image name for each annotation
                self.dic_jo[phase]['boxes_3d'].append(list(boxes_3d))
                self.dic_jo[phase]['K'].append(intrinsic_vec_to_mat(kk).tolist())
                
                append_cluster(self.dic_jo, phase, list(inp), list(ys), kps)
                cnt_ann += 1
                sys.stdout.write('\r' + 'Saved annotations {}'.format(cnt_ann) + '\t')

        with open(os.path.join(self.path_joints), 'w') as f:
            json.dump(self.dic_jo, f)
        with open(os.path.join(self.path_names), 'w') as f:
            json.dump(self.dic_names, f)
        end = time.time()

        extract_box_average(self.dic_jo['train']['boxes_3d'])
        print("\nSaved {} annotations for {} scenes. Total time: {:.1f} minutes".format(cnt_ann, cnt_scenes, (end-start)/60))
        print("\nOutput files:\n{}\n{}\n".format(self.path_names, self.path_joints))    
          
                 
    def extract_ground_truth_pifpaf(self, car_poses,camera_id, scene_id, path_pif):
        with open(car_poses) as json_file:
            data = json.load(json_file) #open the pose of the cars

        dic_vertices = {}
        dic_boxes = {}
        dic_poses = {}
        vertices_to_keypoints = {}
        dic_keypoints = {}
        dic_car_model = {}

        #Extract the boxes, vertices and the poses of each cars                   
        for car_index, car in enumerate(data):
            name_car = car_id2name[car['car_id']].name 
            car_model = os.path.join(self.path, "car_models_json",name_car+".json")

            intrinsic_matrix = intrinsic_vec_to_mat(K["Camera_"+camera_id])

            vertices_r, triangles, bbox_3d , w, l, h = car_projection(car_model, np.array([1,1,1]), T = np.array(car['pose']),  turn_over = True, bbox = True
                                                           )
            vertices_2d = np.matmul(vertices_r,intrinsic_matrix.transpose()) # Projected vertices on the 2D plane
            
            box_gt = bbox_gt_extract(bbox_3d, intrinsic_matrix)  # Take the overal bounding box in the 2D space
            
            dic_vertices[car_index] = vertices_2d
            dic_boxes[car_index] = [box_gt, w, l, h]
            dic_poses[car_index] = np.array(car['pose'])
            dic_car_model[car_index] = car_model

        new_keypoints = None
        
        #  print("DIC_BOXES", dic_boxes)
                           
        if "sample" not in self.path:
                           
            car_keypoints = os.path.join(self.path, "keypoints", scene_id)
            
            keypoints_list = []
            boxes_gt_list = []  # Countain the the 2D bounding box of the vehicles
            boxes_3d_list = []
            ys_list = []     
            car_model_list = []          
            
            keypoints_pifpaf = pifpaf_info_extractor(path_pif)
            
            #Compute the similarity between each set of car models in the 3D space and the set of keypoints from in the 2D space
            for index_keypoints, keypoints in enumerate(keypoints_pifpaf):

                dic_keypoints[index_keypoints] = keypoints

                k_t_c, index_cad, count = keypoints_to_cad_model(keypoints, dic_vertices, radius = self.radius)

                if index_cad not in vertices_to_keypoints.keys():
                    vertices_to_keypoints[index_cad] = [index_keypoints, count, k_t_c]
                elif vertices_to_keypoints[index_cad][1] < count:
                    vertices_to_keypoints[index_cad] = [index_keypoints, count, k_t_c]
            
            for index_cad, (index_keypoints, count, k_t_c) in vertices_to_keypoints.items()   :

                if (index_cad != -1 and count >=1):

                    keypoints = dic_keypoints[index_keypoints]
                    vertices_2d = dic_vertices[index_cad]
                    
                    new_keypoints = keypoint_expander(vertices_2d, keypoints, self.buffer ,self.kps_3d)

                    if self.kps_3d :
                        keypoints_list.append( new_keypoints[:,[1,2,3,0]])
                    else:
                        keypoints_list.append( new_keypoints[:,[1,2,0]])
                                        
                    boxes_gt_list.append(dic_boxes[index_cad][0]) #2D corners of the bounding box
                    
                    
                    w, l, h = dic_boxes[index_cad][1:]
                    pitch, yaw, roll, xc, yc, zc = dic_poses[index_cad] # Center position of the car and its orientation
                    
                    boxes_3d_list.append([xc, yc, zc, w, l, h])
                    
                    # CHECK IF IT IS PI or 2PI
                    yaw = yaw%np.pi
                    
                    sin, cos, _ = correct_angle(yaw, [xc, yc, zc])
                    
                    if True :
                        rtp = to_spherical([xc, yc, zc])
                        r, theta, psi = rtp # With r =d = np.linalg.norm([xc,yc,zc]) -> conversion to spherical coordinates 
                        #print("THETA, PSI, R", theta, psi, r)
                        ys_list.append([theta, psi, zc, r, h, w, l, sin, cos, yaw])
                    else:
                        ys_list.append([xc, yc, zc, np.linalg.norm([xc, yc, zc]), h, w, l, sin, cos, yaw])
                    
                    car_model_list.append(dic_car_model[index_cad])
                    
            return boxes_gt_list, boxes_3d_list, keypoints_list, ys_list, car_model_list
        
        
        else:
            for index_cad, _ in dic_vertices.items() :
                
                boxes_gt_list.append(dic_boxes[index_cad][0]) #2D corners of the bounding box
                    
                w, l, h = dic_boxes[index_cad][1:]
                pitch, yaw, roll, xc, yc, zc = dic_poses[index_cad] # Center position of the car and its orientation

                boxes_3d_list.append([xc, yc, zc, w, l, h])
                yaw = yaw%np.pi

                ys_list.append([xc, yc, zc, np.linalg.norm([xc, yc, zc]), h, w, l, np.sin(yaw), np.cos(yaw), yaw])
            
            return boxes_gt_list, boxes_3d_list, None, ys_lists
            
    
            
def factory(dataset, dir_apollo):
    """Define dataset type and split training and validation"""

    assert dataset in ['train', '3d_car_instance_sample']

    path = os.path.join(dir_apollo, dataset)
    
    if dataset == 'train':
        
        with open(os.path.join(path, "split", "train-list.txt"), "r") as file:
            train_scenes = file.read().splitlines()
        #scenes = [scene for scene in scenes if scene['token'] in train_scenes]
        with open(os.path.join(path, "split", "validation-list.txt"), "r") as file:
            validation_scenes = file.read().splitlines()
            
    elif dataset == '3d_car_instance_sample':
        with open(os.path.join(path, "split", "train-list.txt"), "r") as file:
            train_scenes = file.read().splitlines()
        #scenes = [scene for scene in scenes if scene['token'] in train_scenes]
        with open(os.path.join(path, "split", "validation-list.txt"), "r") as file:
            validation_scenes = file.read().splitlines()
    
    path_img = os.path.join(path, "images")
    scenes = [os.path.join(path_img, file) for file in os.listdir(path_img) if file.endswith(".jpg")]
    
    return scenes, train_scenes, validation_scenes


"""
def extract_ground_truth(self, car_poses,camera_id, scene_id):
        
        with open(car_poses) as json_file:
            data = json.load(json_file) #open the pose of the cars

        dic_vertices = {}
        dic_boxes = {}
        dic_poses = {}
        vertices_to_keypoints = {}
        dic_keypoints = {}

        #Extract the boxes, vertices and the poses of each cars                   
        for car_index, car in enumerate(data):
            name_car = car_id2name[car['car_id']].name 
            car_model = os.path.join(self.path, "car_models_json",name_car+".json")

            intrinsic_matrix = intrinsic_vec_to_mat(K["Camera_"+camera_id])

            vertices_r, triangles, bbox_3d , w, l, h = car_projection(car_model, np.array([1,1,1]), T = np.array(car['pose']),  turn_over = True, bbox = True
                                                           )
            vertices_2d = np.matmul(vertices_r,intrinsic_matrix.transpose()) # Projected vertices on the 2D plane
            
            box_gt = bbox_gt_extract(bbox_3d, intrinsic_matrix)  # Take the overal bounding box in the 2D space
            
            dic_vertices[car_index] = vertices_2d
            dic_boxes[car_index] = [box_gt, w, l, h]
            dic_poses[car_index] = np.array(car['pose'])

        new_keypoints = None
        
        #  print("DIC_BOXES", dic_boxes)
                           
        if "sample" not in self.path:
                           
            car_keypoints = os.path.join(self.path, "keypoints", scene_id)
            
            keypoints_list = []
            boxes_gt_list = []  # Countain the the 2D bounding box of the vehicles
            boxes_3d_list = []
            ys_list = []               
            
            #Compute the similarity between each set of car models in the 3D space and the set of keypoints in the 2D space
            for index_keypoints, file in enumerate(os.listdir(car_keypoints)):
                if file.endswith(".txt"):
                    keypoints = np.loadtxt(os.path.join(car_keypoints, file))

                    dic_keypoints[index_keypoints] = keypoints

                    k_t_c, index_cad, count = keypoints_to_cad_model(keypoints, dic_vertices, radius = self.radius)

                    if index_cad not in vertices_to_keypoints.keys():
                        vertices_to_keypoints[index_cad] = [index_keypoints, count, k_t_c]
                    elif vertices_to_keypoints[index_cad][1] < count:
                        vertices_to_keypoints[index_cad] = [index_keypoints, count, k_t_c]
            
            for index_cad, (index_keypoints, count, k_t_c) in vertices_to_keypoints.items()   :

                if (index_cad != -1 and count >=1):

                    keypoints = dic_keypoints[index_keypoints]
                    vertices_2d = dic_vertices[index_cad]
                    new_keypoints = keypoint_expander(vertices_2d, keypoints, self.buffer)
                    #Pixel keypoints with a Z component
                    
                    #Return an ordered list of the keypoints in the 3D space, their Bounding box in the 3D space and their pose in the 3D space
                    keypoints_list.append( new_keypoints)
                    
                    
                    boxes_gt_list.append(dic_boxes[index_cad][0]) #2D corners of the bounding box
                    
                    
                    w, l, h = dic_boxes[index_cad][1:]
                    pitch, yaw, roll, xc, yc, zc = dic_poses[index_cad] # Center position of the car and its orientation
                    
                    boxes_3d_list.append([xc, yc, zc, w, l, h])
                    
                    # CHECK IF IT IS PI or 2PI
                    yaw = yaw%np.pi
                    
                    sin, cos, _ = correct_angle(yaw, [xc, yc, zc])
                    
                    if True :
                        rtp = to_spherical([xc, yc, zc])
                        r, theta, phi = rtp # With r =d = np.linalg.norm([xc,yc,zc]) -> conversion to spherical coordinates 
                        ys_list.append([theta, phi, zc, r, h, w, l, sin, cos, yaw])
                    else:
                        ys_list.append([xc, yc, zc, np.linalg.norm([xc, yc, zc]), h, w, l, sin, cos, yaw])
                    
                    
            
            return boxes_gt_list, boxes_3d_list, keypoints_list, ys_list
        
        else:
            
            for index_cad, _ in dic_vertices.items() :
                
                boxes_gt_list.append(dic_boxes[index_cad][0]) #2D corners of the bounding box
                    
                w, l, h = dic_boxes[index_cad][1:]
                pitch, yaw, roll, xc, yc, zc = dic_poses[index_cad] # Center position of the car and its orientation

                boxes_3d_list.append([xc, yc, zc, w, l, h])
                yaw = yaw%np.pi

                ys_list.append([xc, yc, zc, np.linalg.norm([xc, yc, zc]), h, w, l, np.sin(yaw), np.cos(yaw), yaw])
            
            
            return boxes_gt_list, boxes_3d_list, None, ys_lists



def kps_apolloscape_to_monoloco(kps, kps_3d = False):
    new_kps = []
    for i in range(67):
        if i not in kps[:,0]:
            new_kps.append([i, -100, -100, -100])
        else:
            new_kps.append(kps[np.where(kps[:,0] == i)[0], :][0])
    new_kps = np.array(new_kps)
    
    new_kps = new_kps[KPS_MAPPING] if kps_3d else new_kps[KPS_MAPPING][:,:3] #Remove or let the third dimension
    
    for i in range(len(new_kps)):
        if new_kps[i,1] < 0:
            new_kps[i, 0] = 0.0 # If the keypoint is not visible in the dataset, the confidence is of 0
        else:
            new_kps[i, 0] = 1.0 # If the keypoint is visible in the dataset, its confidence is of 1.0
        
    if kps_3d:
        return new_kps[:,[1,2,3,0]]
    else:
        return new_kps[:,[1,2,0]]


def bbox_3d_extract(vertices):
    
    min_x = np.min(vertices[:,0])
    max_x= np.max(vertices[:,0])
    min_y = np.min(vertices[:,1])
    max_y= np.max(vertices[:,1])
    min_z = np.min(vertices[:,2])
    max_z= np.max(vertices[:,2])
    
    h = max_y - min_y
    w = max_x - min_x
    l = max_z- min_z
    
    bbox_3d=[]
    for x in [min_x, max_x]:
        for y in [min_y, max_y]:
            for z in [min_z, max_z]:
                bbox_3d.append([x,y,z])
                
    
    return(np.array(bbox_3d), w, l, h)

def bbox_3d_extract(vertices):
    min_x = np.min(vertices[:,0])
    max_x= np.max(vertices[:,0])
    min_y = np.min(vertices[:,1])
    max_y= np.max(vertices[:,1])
    min_z = np.min(vertices[:,2])
    max_z= np.max(vertices[:,2])
    
    bbox_3d=[]
    for x in [min_x, max_x]:
        for y in [min_y, max_y]:
            for z in [min_z, max_z]:
                bbox_3d.append([x,y,z])
    return(np.array(bbox_3d))

"""