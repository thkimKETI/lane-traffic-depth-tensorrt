import cv2
import torch
import random
import time
import numpy as np
import tensorrt as trt
import sys
import os
import pycuda.driver as cuda
import pycuda.autoinit
import argparse

from weights.depth.transform import load_image
from collections import OrderedDict,namedtuple

class TrafficInfraDetector:
    def __init__(self, weight_file, show_result):
        
        self.names = ['TS_ELS', 'AR_ST8', 'AR_RGT', 'RM_CHR', 'AR_ELS', 'AR_UTR', 'TS_SPD', 'TL_GRN', 'RM_CRW', 'RM_STL', 'RM_NUM', 'TL_RED', 'TL_YEL', 'TL_GRA', 'AR_LFT', 'TL_REA', 'AR_STR8', 'AR_STLF', 'TL_AR', 'TL_YAR']
        self.colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(self.names)}
        self.device = torch.device('cuda:0')
        
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        
        with open(weight_file, 'rb') as f, trt.Runtime(logger) as runtime:
            self.TL_det_model = runtime.deserialize_cuda_engine(f.read())
  
        self.bindings = OrderedDict()
        
        for index in range(self.TL_det_model.num_bindings):
            name = self.TL_det_model.get_binding_name(index)
            dtype = trt.nptype(self.TL_det_model.get_binding_dtype(index))
            shape = tuple(self.TL_det_model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.TL_det_model.create_execution_context()
        self.show_result = show_result
        
        
    def calculate_area(self, box):
        x1, y1, x2, y2 = box
        return (x2-x1)*(y2-y1) 
    
    def postprocess(self, boxes, r, dwdh):
        dwdh = torch.tensor(dwdh*2).to(boxes.device)
        boxes -= dwdh
        boxes /= r
        return boxes
        
    def letterbox(self, im, new_shape=(1280, 1280), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        if auto: 
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    
        dw /= 2 
        dh /= 2
    
        if shape[::-1] != new_unpad:  
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
        return im, r, (dw, dh)
        
        
    def refine_boxes(self, boxes, ratio, dwdh):
        for idx, box in enumerate(boxes):
            box = self.postprocess(box, ratio, dwdh).round().int()
        return boxes
        
    def inference(self, img_ori, img_disp):
        image = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im = torch.from_numpy(im).to(self.device)
        im /= 255
        
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data
        boxes = self.bindings['det_boxes'].data
        scores = self.bindings['det_scores'].data
        classes = self.bindings['det_classes'].data
        
        boxes = boxes[0,:nums[0][0]]
        scores = scores[0,:nums[0][0]]
        classes = classes[0,:nums[0][0]]

        #for demo log output
        demo = [] 

        for box,score,cl in zip(boxes,scores,classes):
            box = self.postprocess(box,ratio,dwdh).round().int()
            name = self.names[cl]

            demo.append(name)
            
            color = self.colors[name]
            name += ' ' + str(round(float(score),3))
            cv2.rectangle(img_disp,box[:2].tolist(),box[2:].tolist(),color,2)
            cv2.putText(img_disp,name,(int(box[0]), int(box[1]) - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=2)

        if len(demo) > 0:
            now = time.localtime()
            print(time.strftime('%c', now)) 
            print("Detected Traffic Infra. Objects: "+ str(demo))

        return img_disp

        #cv2.imshow('RESULT', img_disp)
        #cv2.waitKey(1)


class LaneDetector: 
    def __init__(self, weight_file):
        self.device = torch.device('cuda:0')
        self.logger = trt.Logger(trt.Logger.WARNING) 

        self.n_strips = 72 - 1 
        self.n_offsets = 72 
        self.cut_height = 270 
        self.img_w = 800
        self.img_h = 320
        self.conf_thresh = 0.4
        self.nms_thresh = 50
        self.nms_topk = 4 
        self.anchor_ys = [ 1 - i / self.n_strips for i in range(self.n_offsets)] 
        self.ori_w = 1920
        self.ori_h = 1080

        with open(weight_file, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context() 

        self.bindings = {} 
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index) 
            dtype = trt.nptype(self.engine.get_binding_dtype(index)) 
            shape = tuple(self.engine.get_binding_shape(index)) 
            data = cuda.pagelocked_empty(trt.volume(shape), dtype=dtype) 
            ptr = cuda.mem_alloc(data.nbytes) 
            self.bindings[name] = {'data': data, 'ptr': ptr, 'shape': shape} 

        self.input_ptr = next(self.bindings[name]['ptr'] for name in self.bindings if self.engine.binding_is_input(name)) 
        self.output_ptr = next(self.bindings[name]['ptr'] for name in self.bindings if not self.engine.binding_is_input(name)) 
        self.stream = cuda.Stream()

    def preprocess(self, img):
        img = img[self.cut_height:, :, :]
        img = cv2.resize(img, (self.img_w, self.img_h)) 
        img = (img / 255.0).astype(np.float32) 
        img = img.transpose(2,0,1)[None] 
        return img 

    def postprocess(self, pred):
        lanes = [] 
        for img_id, lane_id in zip(*np.where(pred[..., 1]>self.conf_thresh)):
            lane = pred[img_id, lane_id]
            lanes.append(lane.tolist())
        lanes = sorted(lanes, key=lambda x:x[1], reverse=True)
        lanes = self._nms(lanes) 
        lanes_points = self._decode(lanes)
        return lanes_points[:self.nms_topk] 

    def _decode(self, lanes):
        lanes_points = [] 
        for lane in lanes:
            start = int((1-lane[2])*self.n_strips+0.5) 
            end = start + int(lane[5]+0.5)-1
            end = min(end, self.n_strips) 
            points = [] 
            for i in range(start, end+1):
                y = self.anchor_ys[i]
                factor = self.cut_height/self.ori_h
                ys = (1-factor) * y + factor
                points.append([lane[i+6], ys])
            points = torch.from_numpy(np.array(points))
            lanes_points.append(points)
        return lanes_points

    def _nms(self, lanes):
        remove_flags = [False] * len(lanes) 
        keep_lanes = [] 

        for i, ilane in enumerate(lanes):
            if remove_flags[i]:
                continue

            keep_lanes.append(ilane) 
            for j in range(i+1, len(lanes)):
                if remove_flags[j]:
                    continue 

                jlane = lanes[j]
                if self._lane_iou(ilane, jlane) < self.nms_thresh:
                    remove_flags[j] = True
        return keep_lanes

    def _lane_iou(self, lane_a, lane_b):
        start_a = int((1-lane_a[2])*self.n_strips+0.5)
        start_b = int((1-lane_b[2])*self.n_strips+0.5)
        start = max(start_a, start_b) 

        end_a = start_a + int(lane_a[5]+0.5) -1 
        end_b = start_b + int(lane_b[5]+0.5) -1 
        end = min(min(end_a, end_b), self.n_strips) 
        dist = 0 
        for i in range(start, end+1):
            dist += abs((lane_a[i+6] - lane_b[i+6]) * (self.img_w-1)) 
        dist = dist / float(end-start+1) 
        return dist 


    def inference(self, input_image, img_disp):
        img_pre = self.preprocess(input_image)
        np.copyto(self.bindings['images']['data'], img_pre.ravel())

        # Perform inference
        cuda.memcpy_htod_async(self.input_ptr, self.bindings['images']['data'], self.stream)
        self.context.execute_async_v2(bindings=list(b['ptr'] for b in self.bindings.values()), stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.bindings['output']['data'], self.output_ptr, self.stream)
        self.stream.synchronize()

        # Postprocess the output
        output = self.bindings['output']['data']
        pred = output.reshape((1, 192, 78))
        lanes_points = self.postprocess(pred) 

        for points in lanes_points:
            points[:,0] *= img_disp.shape[1]
            points[:,1] *= img_disp.shape[0] 
            points = points.numpy().round().astype(int) 

            for point in points:
                cv2.circle(img_disp, point, 3, color=(0,255,0), thickness=3) 


        
class DepthEstimator:
    def __init__(self, weight_file):
        self.device = torch.device('cuda:0')  # Initialize CUDA device
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load TensorRT engine
        with open(weight_file, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers for bindings
        self.bindings = {}
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            shape = tuple(self.engine.get_binding_shape(index))
            data = cuda.pagelocked_empty(trt.volume(shape), dtype=dtype)
            ptr = cuda.mem_alloc(data.nbytes)
            self.bindings[name] = {'data': data, 'ptr': ptr, 'shape': shape}
        
        # Automatically identify input and output bindings
        self.input_ptr = next(self.bindings[name]['ptr'] for name in self.bindings if self.engine.binding_is_input(name))
        self.output_ptr = next(self.bindings[name]['ptr'] for name in self.bindings if not self.engine.binding_is_input(name))
        self.stream = cuda.Stream()

    def inference(self, input_image):
        """
        Runs inference on the provided input image.

        Args:
            input_image (np.ndarray): Preprocessed input image.

        Returns:
            np.ndarray: Post-processed depth map as an 8-bit grayscale image.
        """
        # Copy input image to the pagelocked memory
        input_image = load_image(input_image)
        #print(input_image)
        np.copyto(self.bindings['input']['data'], input_image.ravel())
        
        # Perform inference
        cuda.memcpy_htod_async(self.input_ptr, self.bindings['input']['data'], self.stream)
        self.context.execute_async_v2(bindings=list(b['ptr'] for b in self.bindings.values()), stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.bindings['output']['data'], self.output_ptr, self.stream)
        self.stream.synchronize()
        
        # Postprocess the output
        output = self.bindings['output']['data']
        depth = np.reshape(output, self.bindings['output']['shape'][2:])
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0  # Normalize to [0, 255]
        depth = depth.astype(np.uint8)
        return depth
        
            
def add_logo(frame, logo, x=2, y=2):
    """ 프레임에 로고를 추가하는 함수 """
    logo_height, logo_width = logo.shape[:2]
    overlay = np.copy(frame)
    overlay[y:y+logo_height, x:x+logo_width] = logo
    return cv2.addWeighted(overlay, 1, frame, 0, 0)

def resize_with_aspect_ratio(image, target_height, inter=cv2.INTER_AREA):
    """ 높이를 기준으로 이미지의 비율을 유지하면서 크기 조정 """
    (h, w) = image.shape[:2]
    ratio = target_height / float(h)
    dim = (int(w * ratio), target_height)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default="./videos/SIHEUNG.mp4")
    args = parser.parse_args()


    obj_det = TrafficInfraDetector('./weights/object/object.engine', True)
    lane_det = LaneDetector('./weights/lane/lane.engine')
    depth_estimator = DepthEstimator('./weights/depth/depth.engine') 
    cap = cv2.VideoCapture(args.video) 
    proc_idx = 0 

    logo = cv2.imread('./assets/keti_logo.png')
    logo = resize_with_aspect_ratio(logo, 60)
    
    while True:
        success, img = cap.read()
        proc_idx += 1 
        
        if proc_idx % 1 == 0: 
            img_disp = img.copy()
            #static obj. det.
            lane_det.inference(img, img_disp)
            img_disp = obj_det.inference(img, img_disp)      
            depth = depth_estimator.inference(img)
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
       
            # 디스플레이용 리사이즈
            target_height = 480
            img_disp = resize_with_aspect_ratio(img_disp, target_height)

            # depth_disp를 작은 사이즈로 만들고 위치 조정
            overlay_scale = 0.33  # 예: 전체의 1/3 크기
            depth_disp = cv2.resize(depth, (int(img_disp.shape[1] * overlay_scale), int(img_disp.shape[0] * overlay_scale)))

            # canvas 설정 (img_disp만 사용)
            logo_height, logo_width  = logo.shape[:2]
            canvas_height = target_height + logo_height
            canvas_width = img_disp.shape[1]
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            canvas = add_logo(canvas, logo)
            text_x = logo_width + 10
            text_y = 30
            line_1 = "Traffic Infrastructure Perception Module" 
            line_2 = "Taehyeon Kim, Senior Researcher (taehyeon.kim@keti.re.kr)" 
            cv2.putText(canvas, line_1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(canvas, line_2, (text_x, text_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            # 본 영상 (좌측 하단)
            canvas[logo_height:logo_height+target_height, :img_disp.shape[1]] = img_disp

            # depth overlay (우측 상단)
            depth_h, depth_w = depth_disp.shape[:2]
            overlay_y = logo_height
            overlay_x = img_disp.shape[1] - depth_w - 10  # 우측 여백
            canvas[overlay_y:overlay_y+depth_h, overlay_x:overlay_x+depth_w] = depth_disp

            cv2.imshow('Traffic Infrastructure Detection and Depth Estimation', canvas)

            #cv2.imwrite(f"NEW_{proc_idx}.jpg", canvas)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    

    







