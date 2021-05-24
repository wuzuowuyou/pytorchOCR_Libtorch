# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import sys
import pathlib

# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))

import numpy as np

sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torchocr.networks import build_model
from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.utils import CTCLabelConverter

suffix_list = [".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"]

def get_file_path_list(path):
    cnt_ = 0
    imagePathList = []
    for root, dir, files in os.walk(path):
        for file in files:
            try:
                full_path = os.path.join(root, file)
                suffix_img = full_path.rsplit(".",1)
                if(len(suffix_img)<2):
                    continue
                suffix_img = "." + suffix_img[-1]
                if suffix_img in suffix_list:
                    cnt_ += 1
                    print (cnt_, "  :: ", full_path)
                    imagePathList.append(full_path)

            except IOError:
                continue

    print("=====end get_file_path_list============================\n")
    return imagePathList

def get_label(img_path):
    pos_2 = img_path.rfind(".")
    pos_1 = img_path.rfind("_")
    label = img_path[pos_1+1:pos_2]
    label = label.replace("@", "/")
    return label


class RecInfer:
    def __init__(self, model_path, batch_size=16):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        self.model = build_model(cfg['model'])
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.process = RecDataProcess(cfg['dataset']['train']['dataset'])
        self.converter = CTCLabelConverter(cfg['dataset']['alphabet'])
        self.batch_size = batch_size

    def predict(self, imgs):
        # 预处理根据训练来
        if not isinstance(imgs,list):
            imgs = [imgs]
        imgs = [self.process.normalize_img(self.process.resize_with_specific_height(img)) for img in imgs]
        widths = np.array([img.shape[1] for img in imgs])
        idxs = np.argsort(widths)
        txts = []
        for idx in range(0, len(imgs), self.batch_size):
            batch_idxs = idxs[idx:min(len(imgs), idx+self.batch_size)]
            batch_imgs = [self.process.width_pad_img(imgs[idx], imgs[batch_idxs[-1]].shape[1]) for idx in batch_idxs]
            batch_imgs = np.stack(batch_imgs)
            tensor = torch.from_numpy(batch_imgs.transpose([0,3, 1, 2])).float()
            tensor = tensor.to(self.device)
            with torch.no_grad():
                out = self.model(tensor)

                traced_script_module = torch.jit.trace(self.model, tensor)
                traced_script_module.save("./model.pt")

                out = out.softmax(dim=2)
            out = out.cpu().numpy()
            txts.extend([self.converter.decode(np.expand_dims(txt, 0)) for txt in out])
        #按输入图像的顺序排序
        idxs = np.argsort(idxs)
        out_txts = [txts[idx] for idx in idxs]
        return out_txts


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='PytorchOCR infer')
    parser.add_argument('--model_path', required=False, type=str, help='rec model path')
    parser.add_argument('--img_path', required=False, type=str, help='img path for predict')
    args = parser.parse_args()

    args.model_path = "./ch_rec_server_crnn_res34.pth"
    args.img_path = "./1.jpg"
    return args


if __name__ == '__main__':
    import cv2
    import random

    args = init_args()
    model = RecInfer(args.model_path)
    path_img_dir = "./img_dir/"

    list_path_img = get_file_path_list(path_img_dir)
    random.shuffle(list_path_img)

    cnt_all = 0
    cnt_right = 0
    for cnt, path_img in enumerate(list_path_img):
        print(cnt, path_img)

        # if -1 != path_img.find("々"):
        #     continue

        img_src = cv2.imread(path_img)
        if img_src is None: 
            continue

        cnt_all += 1

        ans = get_label(path_img)
        out = model.predict(img_src)
        print(out)

        rec = out[0][0][0]

        if ans == rec:
            cnt_right += 1

        print("ans=", ans)
        print("rec=", rec, "\n")

        print("cnt_right=", cnt_right)
        print("cnt_all=", cnt_all)

        print("ratio=", cnt_right * 1.0 / cnt_all)

        #
        cv2.imshow("src", img_src)
        cv2.waitKey(0)

    print("============end===========================================")
    print("end:: cnt_right=", cnt_right)
    print("end:: cnt_all=", cnt_all)
    print("end:: ratio=", cnt_right * 1.0 / cnt_all)