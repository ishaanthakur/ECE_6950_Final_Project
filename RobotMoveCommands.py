
import numpy as np
import cv2
import time
import time, socket, json, pickle, ComparableDict

def sendObject(message):
    # local host IP '127.0.0.1'
    #host = '10.49.33.92'
    host = '192.168.0.173'
    # Define the port on which you want to connect
    port = 65432
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # connect to server on local computer
    s.connect((host,port))
    s.send(pickle.dumps(message))
    # s.send(json.dumps(message).encode())
    s.close()


def receiveResponse(command, port):
    command = command.split()
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("",port))
    s.listen(5)
    connection, address = s.accept()
    data = []
    while True:
        packet = connection.recv(4096)
        if not packet:
            break
        data.append(packet)
    s.close()
    data_arr = pickle.loads(b"".join(data))
    #print(data_arr)
    if command[0] == "image" and command[1] == "get" or command[0] == "int" and command[1] == "single" or command[0] == "ext" and command[1] == "single" or command[0] == "rs" and command[1] == "single":
        try:
            if (command[0]=="int" or command[0]=="ext"):
                cv2.imshow("image",data_arr["data"][:, :, ::-1])
            if (command[0]=="image" or command[0]=="rs"):
                cv2.imshow("image",data_arr["data"]/np.amax(data_arr["data"]))
            cv2.waitKey(1)
        except:
            pass
    #    elif command[0] == "ext" and command[1] == "streaming":
    #        sendAgainAndAgain()
    elif(command[0] == "front1") or (command[0] == "back1"):
        time.sleep(1)
        sendObject(parse_command("stop"))
        receiveResponse("stop", 28200)
    elif(command[0] == "left1") or (command[0] == "right1"):
        time.sleep(1)
        sendObject(parse_command("stop"))
        receiveResponse("stop", 28200)
    # else:
    #     try:
    #         # print(data_arr["data"])
    #     except:
    #         pass
    # image = cv2.imread('137.png')

def parse_command(command):
	command = command.split()
	if command[0] == "motor" or command[0] == 'm':
		return {
		"id" : int(time.time()),
		"cmd" : "SetSpeedCommand", 
		"priority" : 1,#, Priority, type int
		"leftSpeed" : -1*int(command[1]), ## Left speed value, type int
		"rightSpeed" : int(command[2]), ## Right speed value, type int
		"receivingPort" : 28200
		}
	elif command[0] == "hello" or command[0] == 'h':
		return{
		"id" : int(time.time()),
		"cmd" : "PrintHelloCommand", #Command name
		"type" : 'hello',
		"height": 480,
		"width": 640,
		"count_period" : 10,
		"colorize": 1,
		"priority" : 1,#, Priority, type int
		"receivingPort" : 28200
		}
	elif command[0] == "sleep" or command[0] == 's':
		return{
		"id" : int(time.time()),
		"cmd" : "SleepTwentySecsCommand", #Command name
		"type" : 'hello',
		"height": 480,
		"width": 640,
		"count_period" : 10,
		"colorize": 1,
		"priority" : 1,#, Priority, type int
		"receivingPort" : 28200
		}
	elif command[0] == "front" or command[0] == 'f':
		return {
		"id" : int(time.time()),
		"cmd" : "SetSpeedCommand", 
		"priority" : 1,#, Priority, type int
		"leftSpeed" : -1*int(command[1]), ## Left speed value, type int
		"rightSpeed" : int(command[1]), ## Right speed value, type int
		"receivingPort" : 28200
		}
	elif command[0] == "back":
		return {
		"id" : int(time.time()),
		"cmd" : "SetSpeedCommand", 
		"priority" : 1,#, Priority, type int
		"leftSpeed" : int(command[1]), ## Left speed value, type int
		"rightSpeed" : -1*int(command[1]), ## Right speed value, type int
		"receivingPort" : 28200
		}
	elif command[0] == "left":
		return {
		"id" : int(time.time()),
		"cmd" : "SetSpeedCommand", 
		"priority" : 1,#, Priority, type int
		"leftSpeed" : int(command[1]), ## Left speed value, type int
		"rightSpeed" : int(command[1]), ## Right speed value, type int
		"receivingPort" : 28200
		}
	elif command[0] == "right":
		return {
		"id" : int(time.time()),
		"cmd" : "SetSpeedCommand", 
		"priority" : 1,#, Priority, type int
		"leftSpeed" : -1*int(command[1]), ## Left speed value, type int
		"rightSpeed" : -1*int(command[1]), ## Right speed value, type int
		"receivingPort" : 28200
		}
	elif command[0] == "stop" or command[0] == "s":
		return {
		"id" : int(time.time()),
		"cmd" : "SetSpeedCommand", 
		"priority" : 1,#, Priority, type int
		"leftSpeed" : 0, ## Left speed value, type int
		"rightSpeed" : 0, ## Right speed value, type int
		"receivingPort" : 28200
		}
	elif command[0] == "tts":
		return {
		"id" : int(time.time()),
		"cmd" : "TextToSpeechCommand", 
		"text" : " ".join(command[1:]),
		"priority" : 1,#, Priority, type int
		}
	elif command[0] == "image" or command[0] == 'i':
		if len(command) == 3:
			name = command[2]
		else:
			name = ""
		return {
		"id" : int(time.time()),
		"cmd" : "ImageCommand", #Command name
		"type" : command[1],#list, get, upload, display
		"name" : name,
		"priority" : 1,#, Priority, type int
		"receivingPort" : 28200
		}
	elif command[0] == "int":
		return {
		"id" : int(time.time()),
		"cmd" : "InternalCameraCommand", #Command name
		"type" : command[1],#single, continuous-start, continuous-stop
		"fps": 15,
		"priority" : 1,#, Priority, type int
		"receivingPort" : 28200,
		"height": 640,
		"width": 480
		}
	elif command[0] == "ext":
		return {
		"id" : int(time.time()),
		"cmd" : "ExternalCameraCommand", #Command name
		"type" : command[1],#single, continuous-start, continuous-stop
		"fps": 10,
		"priority" : 1,#, Priority, type int
		"receivingPort" : 28200,
		"height": 64,
		"width": 48
		}
	elif command[0] == "rs":
		return {
		"id" : int(time.time()),
		"cmd" : "RealSenseCommand", #Command name
		"type" : command[1],#single, continuous-start, continuous-stop
		"priority" : 1,#, Priority, type int
		"receivingPort" : 28200
		}
	elif command[0] == "tof":
		return {
		"id" : int(time.time()),
		"cmd" : "TimeofFlightCommand", #Command name
		"type" : command[1],#single, stream
		"priority" : 1,#, Priority, type int
		"receivingPort" : 28200
		}
	elif command[0] == "bc":
		return {
			"id" : int(time.time()),
			"cmd" : "BladderCommand", #Command name
			"action" : command[1],#single, stream
			"priority" : 1,#, Priority, type int
			"receivingPort" : 28200
		}
	elif command[0] == "front1":
		return {
			"id" : int(time.time()),
			"cmd" : "SetSpeedCommand",
			"priority" : 1,#, Priority, type int
			"leftSpeed" : -1*int(command[1]), ## Left speed value, type int
			"rightSpeed" : int(command[1]), ## Right speed value, type int
			"receivingPort" : 28200
		}
	elif command[0] == "back1":
		return {
			"id" : int(time.time()),
			"cmd" : "SetSpeedCommand",
			"priority" : 1,#, Priority, type int
			"leftSpeed" : int(command[1]), ## Left speed value, type int
			"rightSpeed" : -1*int(command[1]), ## Right speed value, type int
			"receivingPort" : 28200
		}
	elif command[0] == "left1":
		return {
			"id" : int(time.time()),
			"cmd" : "SetSpeedCommand",
			"priority" : 1,#, Priority, type int
			"leftSpeed" : -1*int(command[1]), ## Left speed value, type int
			"rightSpeed" : -1*int(command[1]), ## Right speed value, type int
			"receivingPort" : 28200
		}
	elif command[0] == "right1":
		return {
			"id" : int(time.time()),
			"cmd" : "SetSpeedCommand",
			"priority" : 1,#, Priority, type int
			"leftSpeed" : int(command[1]), ## Left speed value, type int
			"rightSpeed" : int(command[1]), ## Right speed value, type int
			"receivingPort" : 28200
		}
