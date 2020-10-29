
#pylint: disable=too-many-branches

"""
Run MonoLoco/MonStereo and converts annotations into KITTI format
"""

import os
import math
from collections import defaultdict

import torch
from PIL import Image


from ..network import Loco
from ..network.process import preprocess_pifpaf
from ..network.geom_baseline import geometric_coordinates
from ..utils import get_keypoints, pixel_to_camera, factory_file, factory_basename, make_new_directory, get_category, \
    xyz_from_distance, read_and_rewrite
from .stereo_baselines import baselines_association
from .reid_baseline import get_reid_features, ReID

NUM_ANNOTATIONS = 7481

class GenerateKitti:

    METHODS = ['monstereo', 'monoloco_pp', 'monoloco', 'geometric']
    METHODS = ['monoloco_pp']


    def __init__(self, model, dir_ann, p_dropout=0.2, n_dropout=0, hidden_size=1024, vehicles = False, model_mono = None):

        # Load monoloco
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if 'monstereo' in self.METHODS:
            self.monstereo = Loco(model=model, net='monstereo', device=device, n_dropout=n_dropout, p_dropout=p_dropout,
                                  linear_size=hidden_size, vehicles=vehicles)
        

        if 'monoloco_pp' in self.METHODS:
            
            if len(self.METHODS)<=1:
                model_mono_pp = model


            #model_mono_pp = 'data/models/ms-201021-1825.pkl' #KITTI human
            if model_mono is not None:
                model_mono_pp = model_mono
            else:
                model_mono_pp = None#'data/models/ms-201022-1548-vehicles.pkl' # KITTI vehicle
            # model_mono_pp = 'data/models/stereoloco-200608-1550.pkl'  # nuScenes_pp
            self.monoloco_pp = Loco(model=model_mono_pp, net='monoloco_pp', device=device, n_dropout=n_dropout,
                                    p_dropout=p_dropout, vehicles = vehicles, linear_size=hidden_size)

        if 'monoloco' in self.METHODS:
            model_mono = 'data/models/monoloco-190717-0952.pkl'  # KITTI
            # model_mono = 'data/models/monoloco-190719-0923.pkl'  # NuScenes
            self.monoloco = Loco(model=model_mono, net='monoloco', device=device, n_dropout=n_dropout,
                                 p_dropout=p_dropout, linear_size=256)
        self.dir_ann = dir_ann

        self.vehicles = vehicles
        # Extract list of pifpaf files in validation images
        self.dir_gt = os.path.join('data', 'kitti', 'training','label_2')
        self.dir_gt_new = os.path.join('data', 'kitti', 'gt_new')
        self.set_basename = factory_basename(dir_ann, self.dir_gt)
        self.dir_kk = os.path.join('data', 'kitti', 'training', 'calib')
        self.dir_byc = '/data/lorenzo-data/kitti/object_detection/left'

        # For quick testing
        # ------------------------------------------------------------------------------------------------------------
        # self.set_basename = ('001782',)
        # self.set_basename = ('002282',)
        # ------------------------------------------------------------------------------------------------------------

        # Calculate stereo baselines
        # self.baselines = ['pose', 'reid']
        self.baselines = []
        self.cnt_disparity = defaultdict(int)
        self.cnt_no_stereo = 0
        self.dir_images = os.path.join('data', 'kitti', 'training','image_2')
        self.dir_images_r = os.path.join('data', 'kitti', 'training','images_3')
        # ReID Baseline
        if 'reid' in self.baselines:
            weights_path = 'data/models/reid_model_market.pkl'
            self.reid_net = ReID(weights_path=weights_path, device=device, num_classes=751, height=256, width=128)

    def run(self):
        """Run Monoloco and save txt files for KITTI evaluation"""

        cnt_ann = cnt_file = cnt_no_file = 0
        dir_out = {key: os.path.join('data', 'kitti', key) for key in self.METHODS}
        print("\n")
        for key in self.METHODS:
            make_new_directory(dir_out[key])

        for key in self.baselines:
            dir_out[key] = os.path.join('data', 'kitti', key)
            make_new_directory(dir_out[key])
            print("Created empty output directory for {}".format(key))

        
        create_empty_files(dir_out, self.METHODS)

        # Run monoloco over the list of images
        
        for i, basename in enumerate(self.set_basename):

            
            path_calib = os.path.join(self.dir_kk, basename + '.txt')
            annotations, kk, tt = factory_file(path_calib, self.dir_ann, basename)

            path_im = os.path.join(self.dir_images, basename + '.png')

            if i == 0:
                with Image.open(path_im) as im:
                    width, height = im.size

            #!bookmark
            min_conf = 0.1
            min_conf = 0.35*min_conf
            boxes, keypoints = preprocess_pifpaf(annotations, im_size=(width, height), min_conf=min_conf)
            cat = get_category(keypoints, os.path.join(self.dir_byc, basename + '.json'))
            
            if keypoints:
                annotations_r, _, _ = factory_file(path_calib, self.dir_ann, basename, mode='right')
                _, keypoints_r = preprocess_pifpaf(annotations_r, im_size=(width, height), min_conf=min_conf)

                cnt_ann += len(boxes)
                cnt_file += 1
                all_inputs, all_outputs = {}, {}

                # STEREOLOCO
                if 'monstereo' in self.METHODS:
                    dic_out = self.monstereo.forward(keypoints, kk, keypoints_r=keypoints_r)
                    all_outputs['monstereo'] = [dic_out['xyzd'], dic_out['bi'], dic_out['epi'],
                                                dic_out['yaw'], dic_out['h'], dic_out['w'], dic_out['l']]

                # MONOLOCO++
                if 'monoloco_pp' in self.METHODS:
                    dic_out = self.monoloco_pp.forward(keypoints, kk)
                    all_outputs['monoloco_pp'] = [dic_out['xyzd'], dic_out['bi'], dic_out['epi'],
                                                  dic_out['yaw'], dic_out['h'], dic_out['w'], dic_out['l']]
                    zzs = [float(el[2]) for el in dic_out['xyzd']]

                # MONOLOCO
                if 'monoloco' in self.METHODS:
                    dic_out = self.monoloco.forward(keypoints, kk)
                    zzs_geom, xy_centers = geometric_coordinates(keypoints, kk, average_y=0.48)
                    all_outputs['monoloco'] = [dic_out['d'], dic_out['bi'], dic_out['epi']] + [zzs_geom, xy_centers]
                    all_outputs['geometric'] = all_outputs['monoloco']

                params = [kk, tt]

                for key in self.METHODS:
                    path_txt = {key: os.path.join(dir_out[key], basename + '.txt')}
                    save_txts(path_txt[key], boxes, all_outputs[key], params, mode=key, cat=cat, vehicles = self.vehicles)

                # STEREO BASELINES
                if self.baselines:
                    dic_xyz = self._run_stereo_baselines(basename, boxes, keypoints, zzs, path_calib)

                    for key in dic_xyz:
                        all_outputs[key] = all_outputs['monoloco'].copy()
                        all_outputs[key][0] = dic_xyz[key]
                        all_inputs[key] = boxes

                        path_txt[key] = os.path.join(dir_out[key], basename + '.txt')
                        save_txts(path_txt[key], all_inputs[key], all_outputs[key], params, mode='baseline', cat=cat, vehicles = self.vehicles)

        print("\nSaved in {} txt {} annotations. Not found {} images".format(cnt_file, cnt_ann, cnt_no_file))

        if 'monstereo' in self.METHODS:
            print("STEREO:")
            for key in self.baselines:
                print("Annotations corrected using {} baseline: {:.1f}%".format(
                    key, self.cnt_disparity[key] / cnt_ann * 100))
            print("Maximum possible stereo associations: {:.1f}%".format(self.cnt_disparity['max'] / cnt_ann * 100))
            print("Not found {}/{} stereo files".format(self.cnt_no_stereo, cnt_file))

          # Create empty files for official evaluation

    def _run_stereo_baselines(self, basename, boxes, keypoints, zzs, path_calib):

        annotations_r, _, _ = factory_file(path_calib, self.dir_ann, basename, mode='right')
        boxes_r, keypoints_r = preprocess_pifpaf(annotations_r, im_size=(1242, 374))
        _, kk, tt = factory_file(path_calib, self.dir_ann, basename)

        uv_centers = get_keypoints(keypoints, mode='bottom')  # Kitti uses the bottom center to calculate depth
        xy_centers = pixel_to_camera(uv_centers, kk, 1)

        # Stereo baselines
        if keypoints_r:
            path_image = os.path.join(self.dir_images, basename + '.png')
            path_image_r = os.path.join(self.dir_images_r, basename + '.png')
            reid_features = get_reid_features(self.reid_net, boxes, boxes_r, path_image, path_image_r)
            dic_zzs, cnt = baselines_association(self.baselines, zzs, keypoints, keypoints_r, reid_features)

            for key in cnt:
                self.cnt_disparity[key] += cnt[key]

        else:
            self.cnt_no_stereo += 1
            dic_zzs = {key: zzs for key in self.baselines}

        # Combine the stereo zz with x, y from 2D detection (no MonoLoco involved)
        dic_xyz = defaultdict(list)
        for key in dic_zzs:
            for idx, zz_base in enumerate(dic_zzs[key]):
                xx = float(xy_centers[idx][0]) * zz_base
                yy = float(xy_centers[idx][1]) * zz_base
                dic_xyz[key].append([xx, yy, zz_base])

        return dic_xyz


def save_txts(path_txt, all_inputs, all_outputs, all_params, mode='monoloco', cat=None, vehicles = False):

    assert mode in ('monoloco', 'monstereo', 'geometric', 'baseline', 'monoloco_pp')

    if mode in ('monstereo', 'monoloco_pp'):
        xyzd, bis, epis, yaws, hs, ws, ls = all_outputs[:]
        xyz = xyzd[:, 0:3]
        tt = [0, 0, 0]
    elif mode in ('monoloco', 'geometric'):
        tt = [0, 0, 0]
        dds, bis, epis, zzs_geom, xy_centers = all_outputs[:]
        xyz = xyz_from_distance(dds, xy_centers)
    else:
        _, tt = all_params[:]
        xyz, bis, epis, zzs_geom, xy_centers = all_outputs[:]
    uv_boxes = all_inputs[:]
    assert len(uv_boxes) == len(list(xyz)), "Number of inputs different from number of outputs"

    with open(path_txt, "w+") as ff:
        for idx, uv_box in enumerate(uv_boxes):

            xx = float(xyz[idx][0]) - tt[0]
            yy = float(xyz[idx][1]) - tt[1]
            zz = float(xyz[idx][2]) - tt[2]

            if mode == 'geometric':
                zz = zzs_geom[idx]

            cam_0 = [xx, yy, zz]
            bi = float(bis[idx])
            epi = float(epis[idx])
            if mode in ('monstereo', 'monoloco_pp'):
                alpha, ry = float(yaws[0][idx]), float(yaws[1][idx])
                hwl = [float(hs[idx]), float(ws[idx]), float(ls[idx])]
            else:
                alpha, ry, hwl = -10., -10., [0, 0, 0]

            # Set the scale to obtain (approximately) same recall at evaluation¨
            #!bookmark
            n = 2.5
            if mode == 'monstereo':
                conf_scale = 0.03
                conf_scale = n*conf_scale
            elif mode == 'monoloco_pp':
                conf_scale = 0.033
                conf_scale = n*conf_scale
            else:
                conf_scale = 0.055

            conf = conf_scale * (uv_box[-1]) / (bi / math.sqrt(xx ** 2 + yy ** 2 + zz ** 2))

            output_list = [alpha] + uv_box[:-1] + hwl + cam_0 + [ry, conf, bi, epi]
            category = cat[idx]

            if vehicles:
                if category < 0.1:
                    ff.write("%s " % 'Car')
                else:
                    ff.write("%s " % 'Van')
            else:
                if category < 0.1:
                    ff.write("%s " % 'Pedestrian')
                else:
                    ff.write("%s " % 'Cyclist')

            ff.write("%i %i " % (-1, -1))
            for el in output_list:
                ff.write("%f " % el)
            ff.write("\n")


def create_empty_files(dir_out, methods_2 = ['monoloco_pp', 'monstereo']):
    """Create empty txt files to run official kitti metrics on MonStereo and all other methods"""

    methods = ['pseudo-lidar', 'monopsr', '3dop', 'm3d', 'oc-stereo', 'e2e']
    methods = []
    dirs = [os.path.join('data', 'kitti', method) for method in methods]
    dirs_orig = [os.path.join('data', 'kitti', method + '-orig') for method in methods]

    for di, di_orig in zip(dirs, dirs_orig):
        make_new_directory(di)

        for i in range(NUM_ANNOTATIONS):
            name = "0" * (6 - len(str(i))) + str(i) + '.txt'
            path_orig = os.path.join(di_orig, name)
            path = os.path.join(di, name)

            # If the file exits, rewrite in new folder, otherwise create empty file
            read_and_rewrite(path_orig, path)

    
    for method in methods_2:
        for i in range(NUM_ANNOTATIONS):
            name = "0" * (6 - len(str(i))) + str(i) + '.txt'
            ff = open(os.path.join(dir_out[method], name), "a+")
            ff.close()
