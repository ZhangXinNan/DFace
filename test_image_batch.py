
import os
import argparse
import csv
import numpy as np
import cv2
import torch
from dface.core.detect_zx import create_mtcnn_net, MtcnnDetector
# import dface.core.vision as vision


def get_images(in_dir):
    img_path_list = []
    for d in os.listdir(in_dir):
        sub_dir = os.path.join(in_dir, d)
        if not os.path.isdir(sub_dir):
            continue
        for filename in os.listdir(sub_dir):
            name, suffix = os.path.splitext(filename)
            if suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            img_path = os.path.join(sub_dir, filename)
            img_path_list.append((d, filename, img_path))

        # break
    return img_path_list


def draw_img(img, boxes, landmarks):
    img_draw = img.copy()
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i, :4].astype(np.int)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 255))

    for i in range(landmarks.shape[0]):
        p = landmarks[i].reshape((5, 2)).astype(np.int)
        for j in range(5):
            cv2.circle(img_draw, (p[j, 0], p[j, 1]), radius=2, color=(0, 0, 255), thickness=2)
    return img_draw


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='/media/zhangxin/DATA/data_public/face/widerface/WIDER_val/images')
    parser.add_argument('--out_dir', default='/home/zhangxin/data_public/widerface/val_results_cpu')
    parser.add_argument('--result', default='/home/zhangxin/data_public/widerface/result_cpu.csv')
    parser.add_argument('--result_faces', default='/home/zhangxin/data_public/widerface/result_cpu_faces.csv')
    parser.add_argument('--gpu_id', default=-1, type=int)
    return parser.parse_args()


def main(args):

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./model_store/pnet_epoch.pt",
                                        r_model_path="./model_store/rnet_epoch.pt",
                                        o_model_path="./model_store/onet_epoch.pt",
                                        use_cuda=args.gpu_id == 0)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img_path_list = get_images(args.in_dir)
    print(len(img_path_list))
    fi = open(args.result, 'w')
    fi2 = open(args.result_faces, 'w')
    writer = csv.writer(fi)
    writer2 = csv.writer(fi2)
    writer.writerow(['子目录名', '图片名', 'width', 'height', 'bboxes', 'landmarks', 'pnet', 'rnet', 'onet', 'sum'])
    writer2.writerow(['face num', 'img num', 'pnet', 'rnet', 'onet', 'sum'])
    time_pnet_map = [[] for _ in range(100)]
    time_rnet_map = [[] for _ in range(100)]
    time_onet_map = [[] for _ in range(100)]
    time_sum_map = [[] for _ in range(100)]
    for i, (d, filename, img_path) in enumerate(img_path_list):
        print(i, d, filename, img_path)
        img = cv2.imread(img_path, 1)
        bboxs, landmarks, time_list = mtcnn_detector.detect_face2(img)
        h, w = img.shape[:2]
        t1, t2, t3 = time_list
        writer.writerow([d, filename, w, h, bboxs.shape[0], landmarks.shape[0], t1, t2, t3, t1+t2+t3])

        fn = bboxs.shape[0] if bboxs.shape[0] < 99 else 99
        time_pnet_map[fn].append(t1)
        time_rnet_map[fn].append(t2)
        time_onet_map[fn].append(t3)
        time_sum_map[fn].append(t1 + t2 + t3)
        print(img.shape, bboxs.shape, landmarks.shape)
        # vision.vis_face(img_bg, bboxs, landmarks)
        '''
        img_draw = draw_img(img, bboxs, landmarks)

        sub_dir = os.path.join(args.out_dir, d)
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)
        cv2.imwrite(os.path.join(args.out_dir, d, filename), img_draw)
        '''
        # break

    for i in range(100):
        if len(time_sum_map[i]) < 1:
            writer2.writerow([i, 0, 0, 0, 0, 0])
            continue
        writer2.writerow([i, len(time_sum_map[i]),
                          np.average(time_pnet_map[i]),
                          np.average(time_rnet_map[i]),
                          np.average(time_onet_map[i]),
                          np.average(time_sum_map[i])])
    fi.close()
    fi2.close()


if __name__ == '__main__':
    main(get_args())
