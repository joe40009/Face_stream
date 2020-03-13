import socket
import threading

import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
import time
import sys
import numpy
import Recommand_auto

import ipywidgets as widgets
from matplotlib import pyplot as plt
from PIL import Image

import tensorflow as tf
import facenet
import os
import pymysql
import datetime
import argparse
import yaml

from agmodel import select_model, get_checkpoint

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/default.config')
args = parser.parse_args()

print('-----------------------------------------------------------------')
print("reading config...")
print('-----------------------------------------------------------------')
with open(args.config, 'r') as cf:
    config = yaml.safe_load(cf)
    
print('-----------------------------------------------------------------')
print("check dirs...")
print('-----------------------------------------------------------------')
if not os.path.isdir(config["unknow_face_save_dir"]["unknow_face"]):
    os.makedirs(config["unknow_face_save_dir"]["unknow_face"])
if not os.path.isdir(config["unknow_face_save_dir"]["unknow_face_raw"]):
    os.mskedirs(config["unknow_face_save_dir"]["unknow_face_raw"])
if not os.path.isdir(config["unknow_face_save_dir"]["unknow_face_env"]):
    os.makedirs(config["unknow_face_save_dir"]["unknow_face_env"])
if not os.path.isdir(config["unknow_face_log_dir"]):
    os.makedirs(config["unknow_face_log_dir"])
    
print('-----------------------------------------------------------------')
print("loading face model...")
print('-----------------------------------------------------------------')
sess = tf.Session()
with sess.as_default():
    facenet.load_model(config["face_model_dir"])
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

print('-----------------------------------------------------------------')
print("loading age model...")
print('-----------------------------------------------------------------')
tf.reset_default_graph()
sess_ag = tf.Session()
with sess_ag.as_default():
    model_fn = select_model(config['age_gender_model']['model_select'])    
    images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
    if config['age_gender_model']['mode'] == 'age':
        label_list = AGE_LIST
    if config['age_gender_model']['mode'] == 'gender':
        label_list = GENDER_LIST
    nlabels = len(label_list)
    logits = model_fn(nlabels, images, 1, False)
    init = tf.global_variables_initializer()
    checkpoint_path = config['age_gender_model']['model_dir']
    model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, None, 'checkpoint')
    saver = tf.train.Saver()
    print(model_checkpoint_path, global_step)
    saver.restore(sess_ag, model_checkpoint_path)
    softmax_output = tf.nn.softmax(logits)

print('-----------------------------------------------------------------')
print("loading recommand model...")
print('-----------------------------------------------------------------')
normalize_data, model = Recommand_auto.recommand_init()
all_probability = Recommand_auto.recommand_all(model, normalize_data)

normalize_new_data, new_model = Recommand_auto.recommand_init( 
    normalize_data_path_name=config["unknow_recommend"]["normalize_data_path_name"],
    model_data_path_name=config["unknow_recommend"]["model_data_path_name"] )
new_all_probability = Recommand_auto.recommand_all(new_model, normalize_new_data)
print('-----------------------------------------------------------------')
print("connect facedb...")
print('-----------------------------------------------------------------')
db = pymysql.connect(host=config["facedb"]["ipaddr"], port=int(config["facedb"]["port"]), user=config["facedb"]["user"], passwd=config["facedb"]["passwd"], db=config["facedb"]["db"] )

cursor = db.cursor()
facecolstr = ''
facecol_keyl = ["face_id", "face_nm", "face_encode", "cust_id"]
for facecol_key in facecol_keyl:
    facecolstr = facecolstr + config['facedb']["column"][facecol_key] + ','
cursor.execute('select ' + facecolstr[:-1] + ' from ' + config['facedb']['table'])

face_results = cursor.fetchall()
embl = []
fnm = []
cust_idl = []
for face_list in face_results:
    emb = []
    idxf = str(face_list[2], encoding='utf-8').split(',')[:-1]
    fnm.append(face_list[1])
    cust_idl.append(face_list[3])
    for flist in idxf:
        emb.append(float(flist))
    embl.append(emb)
print('connect facedb success')

print('-----------------------------------------------------------------')
print("start bind...")
print('-----------------------------------------------------------------')
bind_ip = config["bind"]["ip"]# receive from all
bind_port = int(config["bind"]["port"])
bufsize=4096
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
server.bind((bind_ip, bind_port))
server.listen(5)  # max backlog of connections

print ('Listening on {}:{}'.format(bind_ip, bind_port))

def facedetectROI_img(rawframe):
    face_cascade = cv2.CascadeClassifier(config["face_ROI_xml_path"])    
    facesl = []
    bb = []
    drawframw = rawframe.copy()
    gray = cv2.cvtColor(drawframw, cv2.COLOR_BGR2GRAY)
    facess = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(facess) > 0:
        for (x,y,w,h) in facess:
            facesl.append(drawframw[y:y+h, x:x+w])
            bb.append([x,y,w,h])            
    else:
        facesl = []
        bb = []    
    return facesl, bb , drawframw


def handle_client_connection2(client_socket):
    payload_size = struct.calcsize(">L")
    data = b""
    rerunCounter=0
    ifReRun =False
    counter=0
    
    
    while True:
        try:
            while len(data) < payload_size:
                print("Recv: {}".format(len(data)))
                data += client_socket.recv(4096)
                if (len(data) == 0):
                    rerunCounter+=1

                if(rerunCounter==10):
                    ifReRun =True
                    break

            if(ifReRun == True):
                print("Socket Stopped")
                break

            print("Done Recv: {}".format(len(data)))
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            print("msg_size: {}".format(msg_size))
            while len(data) < msg_size:
                data += client_socket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            print("data_collected")
            print('-----------------------------------------------------------------')
###################################################################################################################          
            framel, bb, drawframe = facedetectROI_img(frame)
            tStart = time.time()
            if len(framel) > 0:
                msgl = ''
                num = 0
                for frame in framel:
                    try:
                        frame_size = int(config["frame_model_size"])
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_160 = cv2.resize(frame_rgb, (frame_size, frame_size), interpolation=cv2.INTER_CUBIC)

                        prewhiten_face = facenet.prewhiten(frame_160)
                        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
                        face_512 = sess.run(embeddings, feed_dict=feed_dict)[0]
                        ddl = []
                        for valf in embl:
                            dd = np.sqrt(np.sum(np.square(valf - face_512)))
                            ddl.append(dd)

                        fn = fnm[ddl.index(min(ddl))]                        
                        cust_id = int(cust_idl[ddl.index(min(ddl))])
                        
                        img_ag = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                        image_batch = img_ag
                        image_batch = cv2.resize(image_batch, (227, 227), interpolation=cv2.INTER_CUBIC)
                        batch_results = sess_ag.run(softmax_output, feed_dict={images:[image_batch]})
                        best = np.argmax(batch_results)
                        output = batch_results[0]
                        best_choice = (label_list[best], output[best])
                        
                        thres = float(config["threshold"])                        
                        if min(ddl) < thres:
                            print( 'dd: ' + str(min(ddl)))
                            _string = Recommand_auto.final_output(all_probability[cust_id])
                            msg = 'id: ' + fn + ':' + str(round(min(ddl), 4)) + '@' + str(best_choice) + ' | ' + _string + '_' + str(bb[num][0]) + ',' + str(bb[num][1]) + ',' + str(bb[num][2]) + ',' + str(bb[num][3])
                        else:
                            new_string = Recommand_auto.final_output(new_all_probability[cust_id])                            
                            msg = 'id: ' + 'unknow' + ':' + str(round(min(ddl), 4)) + '@' + str(best_choice) + ' | ' + new_string + '_' + str(bb[num][0]) + ',' + str(bb[num][1]) + ',' + str(bb[num][2]) + ',' + str(bb[num][3])
                            drawframe2 = drawframe.copy()
                            drawframe2 = cv2.rectangle(drawframe2,(bb[num][0],bb[num][1]),(bb[num][0]+bb[num][2],bb[num][1]+bb[num][3]),(255,0,0),2)
                            drawframe2 = cv2.putText(drawframe2, 'id: ' + str(fn), (bb[num][0], bb[num][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            ukn = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d:%H:%M:%S.%f')
                            uknl = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
                            unknow_face_save_dir0 = config["unknow_face_save_dir"]["unknow_face"]
                            unknow_face_save_dir1 = config["unknow_face_save_dir"]["unknow_face_env"]
                            unknow_face_save_dir2 = config["unknow_face_save_dir"]["unknow_face_raw"]
                            
                            cv2.imwrite(unknow_face_save_dir0 + ukn + '_' + fn + '_' + str(min(ddl)) + '.jpg', cv2.cvtColor(frame_160, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(unknow_face_save_dir1 + ukn + '_' + fn + '_' + str(min(ddl)) + '_env.jpg', drawframe2)
                            cv2.imwrite(unknow_face_save_dir2 + ukn + '_' + fn + '_' + str(min(ddl)) + '_raw.jpg', drawframe)
                            if config["select_save"] in ["0", "2"]:
                                with open(config["unknow_face_log_dir"] + uknl + '_unknow_face_log.txt', 'a', encoding='utf-8') as uft:
                                    uft.writelines(ukn + ',' + fn + ',' + 'NA' + ',' 
                                              + unknow_face_save_dir2 + ukn + '_' + fn + '_' + str(min(ddl)) + '_raw.jpg' + ','
                                              + unknow_face_save_dir1 + ukn + '_' + fn + '_' + str(min(ddl)) + '_env.jpg' + ','
                                              + unknow_face_save_dir0 + ukn + '_' + fn + '_' + str(min(ddl)) + '.jpg' + ','
                                              + str(bb[num][0]) + '_' + str(bb[num][1]) + '_' + str(bb[num][2]) + '_' + str(bb[num][3]) + '_' + ','
                                              + str(thres) + ','
                                              + str(min(ddl)) + '\n')
                                    
                            if config["select_save"] in ["1", "2"]:
                                unfacecolstr = ''
                                unfacecol_keyl = ["logtime", "mod_no", "ck_no", "face_raw", "face_env", "face", "bb", "thres", "eud"]
                                for unfacecol_key in unfacecol_keyl:
                                    unfacecolstr = unfacecolstr + config['unknow_face_savedb']["column"][unfacecol_key] + ',' 
                                
                                cursor.execute("INSERT INTO " + config["unknow_face_savedb"]["table"] + "(" + unfacecolstr[:-1] + ") VALUES('"
                                               + str(datetime.datetime.now()) + "','" + fn + "','" + 'NA' + "','"
                                               + unknow_face_save_dir2 + ukn + '_' + fn + '_' + str(min(ddl)) + '_raw.jpg' + "','"
                                               + unknow_face_save_dir1 + ukn + '_' + fn + '_' + str(min(ddl)) + '_env.jpg' + "','"
                                               + unknow_face_save_dir0 + ukn + '_' + fn + '_' + str(min(ddl)) + '.jpg' + "','"
                                               + str(bb[num][0]) + '_' + str(bb[num][1]) + '_' + str(bb[num][2]) + '_' + str(bb[num][3]) + '_' + "','"
                                               + str(thres) + "','"
                                               + str(min(ddl)) + "'" + ")")
                            db.commit() 
                        print( 'face_msg: ' + msg)
                    except Exception as e:
                        print(e)
                        msg = 'error'

                    msgl = msgl + msg + '_'
                    print('-----------------------------------------------------------------')
                    num += 1
            else:
                msgl = 'None'
            tEnd = time.time()
            print("spend time: %f sec" %(tEnd - tStart))
            print( 'send msg: ' + msgl)
##############################################################################################################
            client_socket.send(msgl.encode('utf-8') )
            counter+=1
        except:
            print("Exception")
            continue
        finally:
            print("Finally")
            print('****************************NEXT*********************************')

    client_socket.close()
    # cv2.destroyAllWindows()
print('-----------------------------------------------------------------')
print("start server to listening...")
print('-----------------------------------------------------------------')
while True:
    print('Listening on {}:{}'.format(bind_ip, bind_port))
    client_sock, address = server.accept()
    print ('Accepted connection from {}:{}'.format(address[0], address[1]))
    client_handler = threading.Thread(
        target=handle_client_connection2,
        args=(client_sock,)  # without comma you'd get a... TypeError: handle_client_connection() argument after * must be a sequence, not _socketobject
    )
    client_handler.start()
    
    
