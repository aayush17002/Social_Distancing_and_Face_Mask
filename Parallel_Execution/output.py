import socket

def connections():
    PORT1 = 8400
    PORT2 = 8450
    #Bind the server
	server_conn=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	server_conn.bind((socket.gethostname(),PORT2))
	server_conn.listen(5)
    #Bind the server
	input_conn=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	input_conn.bind((socket.gethostname(),PORT1))
	input_conn.listen(5)
	print('Socket now listening')
    return server_conn, input_conn


server_conn, input_conn = connections()

conn_server,addr=server_conn.accept()
print('Connected by', addr)

conn_input,addr=server_conn.accept()
print('Connected by', addr)

while True:
	result_server = ""
	while True:
		msg = conn_server.recv(1024)
		if len(msg) > 0:
			result_server += msg.decode("utf-8")
			break
	print("server: ",result_server)
	result_input = ""
	while True:
		msg = conn_input.recv(1024)
		if len(msg) > 0:
			result_input += msg.decode("utf-8")
			break
	print("input: ",result_input)
