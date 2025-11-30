import json
import math
import os
import os.path as osp
import random

import numpy as np
class infostruct(object):
    pass


class Tmars(object):

    def __init__(self, root='/data/datasets/', min_seq_len=0,seq_len=8,mode='sjj_dense'):
        root = '/media/jqzhu/e/jjsu/datasets'
        self.root = root
        self.train_info = osp.join(self.root, 'train_info')
        self.gallery_info = osp.join(self.root, 'test_gallery_info')
        self.query_info = osp.join(self.root, 'test_query_info')
        self.seq_len = seq_len
        self.mode = mode

        self._check_before_run()

        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_cams, num_train_vids,_,_,caption_len_train = \
            self._process_data(self.train_info, relabel=True, min_seq_len=min_seq_len,sampler = 'rss_train')

        query, num_query_tracklets, num_query_pids, num_query_imgs, num_query_cams, num_query_vids, query_pid,query_camid,caption_len_query = \
            self._process_data(self.query_info, relabel=False, min_seq_len=min_seq_len,sampler = self.mode)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_vids,gallery_pid,gallery_camid,caption_len_gallery = \
            self._process_data(self.gallery_info, relabel=False, min_seq_len=min_seq_len,sampler = self.mode)

        self.num_class = num_train_pids

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets
        num_total_captions = caption_len_gallery + caption_len_query + caption_len_train

        print("=> TMARS loaded")
        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset   | # ids | # tracklets | # captions")
        print("  -------------------------------------------")
        print("  train    | {:5d} | {:8d} | {:8d}".format(num_train_pids, num_train_tracklets,caption_len_train))
        print("  test     | {:5d} | {:8d} | {:8d}".format(634, num_query_tracklets+num_gallery_tracklets,caption_len_query+caption_len_gallery))
        print("  -------------------------------------------")
        print("  total    | {:5d} | {:8d} | {:8d}".format(num_total_pids, num_total_tracklets,num_total_captions))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  -------------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

        self.queryinfo = infostruct()
        self.queryinfo.pid = query_pid
        self.queryinfo.camid = query_camid
        self.queryinfo.tranum = num_query_imgs

        self.galleryinfo = infostruct()
        self.galleryinfo.pid = gallery_pid
        self.galleryinfo.camid = gallery_camid
        self.galleryinfo.tranum = num_gallery_imgs

        self.num_train_cams = num_train_cams
        self.num_camera = 6


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_info):
            raise RuntimeError("'{}' is not available".format(self.train_info))
        if not osp.exists(self.query_info):
            raise RuntimeError("'{}' is not available".format(self.query_info))
        if not osp.exists(self.gallery_info):
            raise RuntimeError("'{}' is not available".format(self.gallery_info))



    def _process_data(self, fpath, relabel=False, min_seq_len=0,sampler = None):
        pid_list = []
        data_list = []
        cam_list = []
        num_imgs_per_tracklet = []
        caption_list = []
        for file in os.listdir(fpath):
            file_path = os.path.join(fpath, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 处理每个JSON对象
                    for item in data:
                        img_paths = item.get('img_path', [])
                        length = len(img_paths)
                        frame_indices = list(range(length))
                        person_id = item.get('person_id', '')
                        captions_detail = item.get('captions', [])
                        caption_list.append(captions_detail)
                        cam_id = int(img_paths[0].split('/')[-1].split('C')[1][0])
                        person_id = int(person_id)
                        pid_list.append(person_id)

                        assert 1 <= cam_id <= 6
                        cam_id -= 1
                        cam_list.append(cam_id)
                        pnames = [img_name[:4] for img_name in img_paths]

                        # make sure image names correspond to the same person
                        assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"
                        # make sure all images are captured under the same camera
                        camnames = [img_name[5] for img_name in img_paths]
                        assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"
                        num_imgs_per_tracklet.append(len(img_paths))
                        selected_index = []
                        if len(img_paths) >= min_seq_len:
                            if length < self.seq_len:
                                inter_val = 1 #number of image per seq
                                strip = frame_indices + [frame_indices[-1]] * (self.seq_len - length)
                            else:
                                inter_val = math.ceil(length / self.seq_len)
                                strip = frame_indices + [frame_indices[-1]] * (inter_val * self.seq_len - length)
                            num_strip = list(range(len(strip)))

                            pools = []
                            if sampler == 'rss_train':
                                for i in range(inter_val):
                                    for s in range(self.seq_len):
                                        pool = num_strip[inter_val * s:inter_val * (s + 1)]
                                        available_indices = [idx for idx in pool if idx not in selected_index]
                                        selected_idx = random.choice(available_indices)
                                        pools.append(strip[selected_idx])
                                        selected_index.append(selected_idx)
                                    img_tuple = tuple([os.path.join(self.root, img_paths[p].split('datasets/')[1]) for p in pools])
                                    data_list.append((img_tuple,person_id,cam_id,captions_detail))
                                    pools = []

                            elif sampler == 'rss':
                                pools = []
                                for s in range(self.seq_len):
                                    pool = num_strip[inter_val * s:inter_val * (s + 1)]
                                    available_indices = [idx for idx in pool if idx not in selected_index]
                                    selected_idx = available_indices[0]
                                    pools.append(strip[selected_idx])
                                img_tuple = tuple([os.path.join(self.root, img_paths[p].split('datasets/')[1]) for p in pools])
                                # print('img_tuple',img_tuple)
                                data_list.append((img_tuple,person_id,cam_id,captions_detail))

                            elif sampler == 'dense':
                                dense_sequences = []

                                if length <= 80:
                                    for start in range(0, length, self.seq_len):
                                        new_paths = []
                                        end = min(start + self.seq_len, length)
                                        segment = img_paths[start:end]

                                        for path in segment:
                                            path_parts = path.split('datasets')
                                            new_path = os.path.join(self.root, path_parts[1].lstrip('/'))
                                            new_paths.append(new_path)
                                        # 如果最后一段不足self.seq_len长度，用最后一帧填充
                                        if len(new_paths) < self.seq_len:
                                            new_paths += [img_paths[-1]] * (self.seq_len - len(segment))
                                        dense_sequences.append(new_paths)
                                    data_list.append((dense_sequences, person_id, cam_id, captions_detail))
                                else:
                                    dense_sequences = []
                                    scale = math.ceil(length / 80)
                                    total_segments = length // self.seq_len

                                    # 进行二次抽样
                                    for i in range(0, total_segments, scale):
                                        new_paths = []
                                        start = i * self.seq_len
                                        end = min(start + self.seq_len, length)
                                        segment = img_paths[start:end]
                                        for path in segment:
                                            path_parts = path.split('datasets')
                                            new_path = os.path.join(self.root, path_parts[1].lstrip('/'))
                                            new_paths.append(new_path)
                                        # 如果最后一段不足self.seq_len长度，用最后一帧填充
                                        if len(new_paths) < self.seq_len:
                                            new_paths += [img_paths[-1]] * (self.seq_len - len(segment))
                                        dense_sequences.append(new_paths)
                                    data_list.append((dense_sequences, person_id, cam_id, captions_detail))

            except json.JSONDecodeError as e:
                print(f"Error reading {file_path}: {e}")

        pid_set = set(pid_list)
        cam_set = set(cam_list)
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_set)}
            # print(pid2label)
            for i in range(len(data_list)):
                # 将元组转换为列表
                temp_list = list(data_list[i])
                temp_list[1] = pid2label[temp_list[1]]
                # 将列表转换回元组
                data_list[i] = tuple(temp_list)

        num_pids = len(pid_set)
        num_camids = len(cam_set)
        num_tracklets = len(data_list)
        caption_len = len(caption_list)
        return data_list, num_tracklets, num_pids, num_imgs_per_tracklet, num_camids, 1,pid_list,cam_list, caption_len

    def get_imagedata_info(self, data):
        pids, cams, tracks= [], [], []
        num_imgs = 0

        for img_paths,pid,camid,caption in data:
            num_imgs += len(img_paths)
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)

        return num_pids, num_imgs, num_cams, 0