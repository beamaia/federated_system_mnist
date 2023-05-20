import grpc
import client_pb2 
import client_pb2_grpc

import server_pb2
import server_pb2_grpc

from concurrent import futures

import time 

class Client(client_pb2_grpc.apiServicer):
    # def load_data(): pass
    # def load_data(): pass
    def __init__(self):
        self.uuid = "2"
        self.ipv4 = "localhost"
        self.port = 5051

        self.channel = grpc.insecure_channel('localhost:8080')
        self.stub = server_pb2_grpc.apiStub(self.channel)

    def train_model(self, request, context):
        print("Training model.")
        time.sleep(5)
        return client_pb2.void()
    
    def connect_to_aggregator(self):
        grpc_server = self.server()
        self.connect()
        print("Client connected.")
        grpc_server.wait_for_termination()
        print('a')

    def server(self):
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        client_pb2_grpc.add_apiServicer_to_server(Client(), grpc_server)
        grpc_server.add_insecure_port(f'{self.ipv4}:{self.port}')
        grpc_server.start()
        return grpc_server

    def connect(self):
        res = self.stub.add_trainer(server_pb2.trainer_request(uuid=self.uuid, ipv4=self.ipv4, port=self.port))
        print(res)
        print("Connected to aggregator.")
        return res

    

if __name__ == '__main__':
    client = Client()
    print("Client started.")
    client.connect_to_aggregator()

