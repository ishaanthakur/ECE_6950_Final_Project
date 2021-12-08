#!/usr/bin/python
import sys, getopt
import asyncore
import traceback
from RobotMoveCommands import parse_command, receiveResponse, sendObject
# from final_soc_distance import Social_Distance_Detection
import numpy as np
import pickle
import socket
import struct
import cv2
import time
import imutils
from scipy.spatial import distance as dist
from Detection_dir.detection import detect_people
from Detection_dir import social_distancing_config as config
import os
import json


mc_ip_address = '224.0.0.1'
#mc_ip_address = '192.168.0.190'
#mode = "tof"
mode = "rs2"
#mode = "tof"
ports = {"rs" : 1024,"tof" : 1025, "ext": 1028, "int": 1029, "rs2": 1024}
port = ports[mode]
chunk_size = 4096



#######
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = list(net.getLayerNames())

ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
# vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None



#######


def main(argv):
    multi_cast_message(mc_ip_address, port, 'boop')
        
#UDP client for each camera server 
class ImageClient(asyncore.dispatcher):
    def __init__(self, server, source):   
        asyncore.dispatcher.__init__(self, server)
        self.start_time = time.time()
        self.address = server.getsockname()[0]
        self.port = source[1]
        self.buffer = bytearray()
        self.windowName = self.port
        self.RunYoLo = True
        self.wait_stabilise = 7
        self.frames_for_positive = 0
        self.frames_for_negative = 0
        self.YoloMode = "Rotate"
        self.frame_count_to_change_mode = 10
        self.dist_to_violator = 100
        self.TimeToReachDest = 0
        
        # open cv window which is unique to the port 
        if mode == "rs2":
            cv2.namedWindow("RSCam"+str(self.windowName))
        self.remainingBytes = 0
        self.frame_id = 0
       
    def handle_read(self):
        if self.remainingBytes == 0:
            # get the expected frame size
            recieved = self.recv(4)
            self.frame_length = struct.unpack('<I', recieved)[0]
            self.remainingBytes = self.frame_length
        
        # request the frame data until the frame is completely in buffer
        data = self.recv(self.remainingBytes)
        self.buffer += data
        self.remainingBytes -= len(data)
        # once the frame is fully recived, process/display it
        if len(self.buffer) == self.frame_length:
            self.handle_frame()

    def handle_frame(self):
        global LABELS,net,ln, writer
        # convert the frame from string to numerical data
        fps = 1/(time.time()-self.start_time)
        
        print(fps)
        self.start_time = time.time()
        try:
            imdata = pickle.loads(self.buffer)
            print('MODE: ', self.YoloMode)
            if mode == "rs2":
                if self.YoloMode == "Rotate":
                    self.command = "i display green.png"
                    sendObject(parse_command(self.command))
                    receiveResponse(self.command, 28200)
                    if self.wait_stabilise <= 0:
                        self.command = "left 600"
                        self.wait_stabilise = 7
                    else:
                        self.command = "stop"
                        self.wait_stabilise -= 1
                    sendObject(parse_command(self.command))
                    receiveResponse(self.command, 28200)
                    image_frm_cam = cv2.resize(imdata, (640,480))
                    frame = image_frm_cam
                    frame = imutils.resize(frame, width=700)
                    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
                    FOCAL_LENGTH = 621
                    violate = set()
                    detect_in_other_coor = False
                    if len(results) >= 2:
                        dist_camera = []
                        for item, res_items in enumerate(results):
                            height_bnd_box = res_items[1][3] - res_items[1][1]
                            dist_frm_camera = (6 * FOCAL_LENGTH) / (height_bnd_box)
                            dist_camera.append((item,dist_frm_camera))

                        dist_camera.sort(key = lambda x: x[1])
                        in_between_dist = [(dist_camera[i+1][1] - dist_camera[i][1], dist_camera[i][1], dist_camera[i][0]) for i in range(len(dist_camera[1])-1)]
                        final_res = []
                        for dist_bet in in_between_dist:
                            if dist_bet[0] < 3:
                                
                                print('FIRST VIOLATION', dist_bet[1])
                                # print('results', results, dist_bet)
                                final_res.append(results[dist_bet[2]])
                                if dist_bet[2] + 1 >= len(results):
                                    final_res.append(results[dist_bet[2] - 1])
                                else:
                                    final_res.append(results[dist_bet[2] + 1])
                                detect_in_other_coor = True
                                results = final_res
                                break
                        centroids = np.array([r[2] for r in results])
                        D = dist.cdist(centroids, centroids, metric="euclidean")

                        for i in range(0, D.shape[0]):
                            for j in range(i + 1, D.shape[1]):
                                if D[i, j] < config.MIN_DISTANCE and detect_in_other_coor:
                                    violate.add(i)
                                    violate.add(j)
                                    self.YoloMode = "ViolationConfirmation"
                                    self.wait_stabilise = 7

                    for (i, (prob, bbox, centroid)) in enumerate(results):
                        (startX, startY, endX, endY) = bbox
                        (cX, cY) = centroid
                        color = (0, 255, 0)

                        if i in violate:
                            color = (0, 0, 255)

                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        cv2.circle(frame, (cX, cY), 5, color, 1)

                    text = "Social Distancing Violations: {}".format(len(violate))
                    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

                    cv2.imshow("Frame", frame)

                elif self.YoloMode == "ViolationConfirmation":
                    self.command = "i display orange.png"
                    sendObject(parse_command(self.command))
                    receiveResponse(self.command, 28200)
                    self.command = "stop"
                    sendObject(parse_command(self.command))
                    receiveResponse(self.command, 28200)
                    image_frm_cam = cv2.resize(imdata, (640,480))
                    frame = image_frm_cam
                    frame = imutils.resize(frame, width=700)
                    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
                    FOCAL_LENGTH = 621
                    violate = set()
                    detect_in_other_coor = False
                    if len(results) >= 2:
                        dist_camera = []
                        for item, res_items in enumerate(results):
                            height_bnd_box = res_items[1][3] - res_items[1][1]
                            dist_frm_camera = (6 * FOCAL_LENGTH) / (height_bnd_box)
                            dist_camera.append((item,dist_frm_camera))

                        dist_camera.sort(key = lambda x: x[1])
                        in_between_dist = [(dist_camera[i+1][1] - dist_camera[i][1], dist_camera[i][1], dist_camera[i][0]) for i in range(len(dist_camera[1])-1)]
                        final_res = []
                        for dist_bet in in_between_dist:
                            if dist_bet[0] < 3:
                                
                                print('FIRST VIOLATION', dist_bet[1])
                                # print('results', results, dist_bet)
                                final_res.append(results[dist_bet[2]])
                                if dist_bet[2] + 1 >= len(results):
                                    final_res.append(results[dist_bet[2] - 1])
                                else:
                                    final_res.append(results[dist_bet[2] + 1])
                                detect_in_other_coor = True
                                results = final_res
                                break
                        centroids = np.array([r[2] for r in results])
                        D = dist.cdist(centroids, centroids, metric="euclidean")

                        for i in range(0, D.shape[0]):
                            for j in range(i + 1, D.shape[1]):
                                if D[i, j] < config.MIN_DISTANCE and detect_in_other_coor:
                                    self.frames_for_positive += 1
                                    self.frames_for_negative = 0
                                    violate.add(i)
                                    violate.add(j)
                                else:
                                    self.frames_for_negative += 1
                                    self.frames_for_positive = 0
                    else:
                        self.frames_for_negative += 1
                        self.frames_for_positive = 0

                    for (i, (prob, bbox, centroid)) in enumerate(results):
                        (startX, startY, endX, endY) = bbox
                        (cX, cY) = centroid
                        color = (0, 255, 0)

                        if i in violate:
                            color = (0, 0, 255)

                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        cv2.circle(frame, (cX, cY), 5, color, 1)

                    text = "Social Distancing Violations: {}".format(len(violate))
                    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

                    cv2.imshow("Frame", frame)

                    if self.frames_for_positive >= self.frame_count_to_change_mode:
                        self.frames_for_positive = 0
                        self.YoloMode = "Centralize"
                    elif self.frames_for_negative >= self.frame_count_to_change_mode:
                        self.frames_for_negative = 0
                        self.YoloMode = "Rotate"

                elif self.YoloMode == "Centralize":
                    self.command = "i display red.jpg"
                    sendObject(parse_command(self.command))
                    receiveResponse(self.command, 28200)
                    image_frm_cam = cv2.resize(imdata, (640,480))
                    frame = image_frm_cam
                    frame = imutils.resize(frame, width=700)
                    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
                    FOCAL_LENGTH = 621
                    violate = set()
                    detect_in_other_coor = False
                    if len(results) >= 2:
                        dist_camera = []
                        for item, res_items in enumerate(results):
                            height_bnd_box = res_items[1][3] - res_items[1][1]
                            dist_frm_camera = (6 * FOCAL_LENGTH) / (height_bnd_box)
                            dist_camera.append((item,dist_frm_camera))

                        dist_camera.sort(key = lambda x: x[1])
                        in_between_dist = [(dist_camera[i+1][1] - dist_camera[i][1], dist_camera[i][1], dist_camera[i][0]) for i in range(len(dist_camera[1])-1)]
                        final_res = []
                        for dist_bet in in_between_dist:
                            if dist_bet[0] < 3:
                                print('FIRST VIOLATION', dist_bet[1])
                                final_res.append(results[dist_bet[2]])
                                if dist_bet[2] + 1 >= len(results):
                                    final_res.append(results[dist_bet[2] - 1])
                                else:
                                    final_res.append(results[dist_bet[2] + 1])
                                detect_in_other_coor = True
                                results = final_res
                                break
                        centroids = np.array([r[2] for r in results])
                        D = dist.cdist(centroids, centroids, metric="euclidean")

                                                
                        for i in range(0, D.shape[0]):
                            for j in range(i + 1, D.shape[1]):
                                if D[i, j] < config.MIN_DISTANCE and detect_in_other_coor:
                                    violate.add(i)
                                    violate.add(j)
                                    

                    x_coordinates_centroid = []
                    for (i, (prob, bbox, centroid)) in enumerate(results):
                        (startX, startY, endX, endY) = bbox
                        (cX, cY) = centroid
                        color = (0, 255, 0)

                        if i in violate:
                            color = (0, 0, 255)
                            x_coordinates_centroid.append(cX)
                            

                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        cv2.circle(frame, (cX, cY), 5, color, 1)

                    if len(x_coordinates_centroid) > 0:
                        x_centroid_avg = np.sum(x_coordinates_centroid)/ (len(x_coordinates_centroid))
                    else:
                        x_centroid_avg = 350
                        self.YoloMode = "ViolationConfirmation"
                    diff_in_x = 350 - x_centroid_avg
                    if self.wait_stabilise > 0:
                        self.wait_stabilise -= 1
                        self.command = "stop"
                        sendObject(parse_command(self.command))
                        receiveResponse(self.command, 28200)
                    elif diff_in_x >= 50 and self.wait_stabilise <= 0:
                        self.command = "left 550"
                        sendObject(parse_command(self.command))
                        receiveResponse(self.command, 28200)
                        self.wait_stabilise = 7
                        time.sleep(0.2)
                        self.command = "stop"
                        sendObject(parse_command(self.command))
                        receiveResponse(self.command, 28200)
                    elif diff_in_x <= -50:
                        self.command = "right 550"
                        sendObject(parse_command(self.command))
                        receiveResponse(self.command, 28200)
                        self.wait_stabilise = 7
                        time.sleep(0.2)
                        self.command = "stop"
                        sendObject(parse_command(self.command))
                        receiveResponse(self.command, 28200)
                    elif not self.YoloMode == "ViolationConfirmation":
                        self.YoloMode = "CalculateDistance"

                    text = "Social Distancing Violations: {}".format(len(violate))
                    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

                    cv2.imshow("Frame", frame)
                elif self.YoloMode == "CalculateDistance":
                    self.command = "i display red.jpg"
                    sendObject(parse_command(self.command))
                    receiveResponse(self.command, 28200)
                    image_frm_cam = cv2.resize(imdata, (640,480))
                    frame = image_frm_cam
                    frame = imutils.resize(frame, width=700)
                    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
                    FOCAL_LENGTH = 621
                    violate = set()
                    detect_in_other_coor = False
                    if len(results) >= 2:
                        dist_camera = []
                        for item, res_items in enumerate(results):
                            height_bnd_box = res_items[1][3] - res_items[1][1]
                            dist_frm_camera = (6 * FOCAL_LENGTH) / (height_bnd_box)
                            dist_camera.append((item,dist_frm_camera))

                        dist_camera.sort(key = lambda x: x[1])
                        in_between_dist = [(dist_camera[i+1][1] - dist_camera[i][1], dist_camera[i][1], dist_camera[i][0]) for i in range(len(dist_camera[1])-1)]
                        final_res = []
                        for dist_bet in in_between_dist:
                            if dist_bet[0] < 3:
                                self.dist_to_violator = dist_bet[1]
                                print('FIRST VIOLATION', dist_bet[1])
                                final_res.append(results[dist_bet[2]])
                                if dist_bet[2] + 1 >= len(results):
                                    final_res.append(results[dist_bet[2] - 1])
                                else:
                                    final_res.append(results[dist_bet[2] + 1])
                                detect_in_other_coor = True
                                results = final_res
                                break
                        centroids = np.array([r[2] for r in results])
                        D = dist.cdist(centroids, centroids, metric="euclidean")

                                                
                        for i in range(0, D.shape[0]):
                            for j in range(i + 1, D.shape[1]):
                                if D[i, j] < config.MIN_DISTANCE and detect_in_other_coor:
                                    violate.add(i)
                                    violate.add(j)
                                    

                    x_coordinates_centroid = []
                    for (i, (prob, bbox, centroid)) in enumerate(results):
                        (startX, startY, endX, endY) = bbox
                        (cX, cY) = centroid
                        color = (0, 255, 0)
                        if i in violate:
                            color = (0, 0, 255)
                            x_coordinates_centroid.append(cX)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        cv2.circle(frame, (cX, cY), 5, color, 1)

                    text = "Social Distancing Violations: {}".format(len(violate))
                    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

                    cv2.imshow("Frame", frame)
                    self.YoloMode = "MoveTowardsTheEnemy"
                elif self.YoloMode == "MoveTowardsTheEnemy":
                    self.command = "i display red.jpg"
                    sendObject(parse_command(self.command))
                    receiveResponse(self.command, 28200)
                    some_speed = 1
                    self.TimeToReachDest = (self.dist_to_violator - 7) / some_speed
                    self.command = "front 500"
                    sendObject(parse_command(self.command))
                    receiveResponse(self.command, 28200)
                    if(self.TimeToReachDest >= 4):
                        time.sleep(int(self.TimeToReachDest / 2))
                    else:
                        time.sleep(self.TimeToReachDest)
                    self.command = "stop"
                    sendObject(parse_command(self.command))
                    receiveResponse(self.command, 28200)
                    time.sleep(1)
                    if(self.TimeToReachDest < 4):
                        self.command = "tts social_distancing_violated_calling_police"
                        sendObject(parse_command(self.command))
                        receiveResponse(self.command, 28200)
                        time.sleep(5)
                        self.YoloMode = "Rotate"
                    else:
                        self.YoloMode = "Centralize"
                else:
                    pass
                cv2.waitKey(1)
        except Exception as e:
            print("EXCEPTION", e)
            traceback.print_exc()
            
        self.buffer = bytearray()
        self.frame_id += 1
    def readable(self):
        return True

    
class ExtStreamingClient(asyncore.dispatcher):
    def __init__(self):
        asyncore.dispatcher.__init__(self)
        self.server_address = ('', port)
        # create a socket for TCP connection between the client and server
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5)
        
        self.bind(self.server_address) 	
        self.listen(10)

    def writable(self):
        return False # don't want write notifies

    def readable(self):
        return True
        
    def handle_connect(self):
        print("connection recvied")

    def handle_accept(self):
        pair = self.accept()
        #print(self.recv(10))
        if pair is not None:
            sock, addr = pair
            print ('Incoming connection from %s' % repr(addr))
            # when a connection is attempted, delegate image receival to the ImageClient 
            handler = ImageClient(sock, addr)

class IntStreamingClient(asyncore.dispatcher):
    def __init__(self):
        asyncore.dispatcher.__init__(self)
        self.server_address = ('', port)
        # create a socket for TCP connection between the client and server
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5)

        self.bind(self.server_address)
        self.listen(10)

    def writable(self):
        return False # don't want write notifies

    def readable(self):
        return True

    def handle_connect(self):
        print("connection recvied")

    def handle_accept(self):
        pair = self.accept()
        #print(self.recv(10))
        if pair is not None:
            sock, addr = pair
            print ('Incoming connection from %s' % repr(addr))
            # when a connection is attempted, delegate image receival to the ImageClient
            handler = ImageClient(sock, addr)

class RSStreamingClient(asyncore.dispatcher):
    def __init__(self):
        asyncore.dispatcher.__init__(self)
        self.server_address = ('', port)
        # create a socket for TCP connection between the client and server
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5)

        self.bind(self.server_address)
        self.listen(10)

    def writable(self):
        return False # don't want write notifies

    def readable(self):
        return True

    def handle_connect(self):
        print("connection recvied")

    def handle_accept(self):
        pair = self.accept()
        #print(self.recv(10))
        if pair is not None:
            sock, addr = pair
            print ('Incoming connection from %s' % repr(addr))
            # when a connection is attempted, delegate image receival to the ImageClient
            handler = ImageClient(sock, addr)

class TOFStreamingClient(asyncore.dispatcher):
    def __init__(self):
        asyncore.dispatcher.__init__(self)
        self.server_address = ('', port)
        # create a socket for TCP connection between the client and server
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5)

        self.bind(self.server_address)
        self.listen(10)

    def writable(self):
        return False # don't want write notifies

    def readable(self):
        return True

    def handle_connect(self):
        print("connection recvied")

    def handle_accept(self):
        pair = self.accept()
        #print(self.recv(10))
        if pair is not None:
            sock, addr = pair
            print ('Incoming connection from %s' % repr(addr))
            # when a connection is attempted, delegate image receival to the ImageClient
            handler = ImageClient(sock, addr)

def multi_cast_message(ip_address, port, message):
    # send the multicast message
    multicast_group = (ip_address, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    connections = {}
    try:
        # Send data to the multicast group
        print('sending "%s"' % message + str(multicast_group))
        sent = sock.sendto(message.encode(), multicast_group)
        print('message sent')
        # defer waiting for a response using Asyncore
        if mode == "rs" or "rs2":
            client = RSStreamingClient()
        elif mode == "ext":
            client = ExtStreamingClient()
        elif mode == "tof":
            client = TOFStreamingClient()
        elif mode == "int":
            client = IntStreamingClient()
        asyncore.loop()

        # Look for responses from all recipients
        
    except socket.timeout:
        print('timed out, no more responses')
    finally:
        print(sys.stderr, 'closing socket')
        sock.close()

if __name__ == '__main__':
    main(sys.argv[1:])
