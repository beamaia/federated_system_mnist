import grpc
import client_pb2 
import client_pb2_grpc

import server_pb2
import server_pb2_grpc

from concurrent import futures

    
class Client(client_pb2_grpc.apiServicer):
    # def load_data(): pass
    # def load_data(): pass
    def __init__(self):
        self.uuid = "2"
        self.ipv4 = "locahost"
        self.port = 5050

        self.channel = grpc.insecure_channel('localhost:8080')
        self.stub = server_pb2_grpc.apiStub(self.channel)

    def train_model(self, request):
        print("Training model.")
        return self.stub.train_model(request)
    
    def connect_to_aggregator(self):
        self.connect()
        print("Client connected.")
        self.server()
        print('a')

    def server(self):
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        client_pb2_grpc.add_apiServicer_to_server(Client(), grpc_server)
        grpc_server.add_insecure_port('[::]:5050')
        grpc_server.start()
        grpc_server.wait_for_termination()

    def connect(self):
        res = self.stub.add_trainer(server_pb2.trainer_request(uuid=self.uuid, ipv4=self.ipv4, port=self.port))
        return res

    

if __name__ == '__main__':
    client = Client()
    print("Client started.")
    client.connect_to_aggregator()

