import grpc
import server_pb2 
import server_pb2_grpc

import client_pb2 
import client_pb2_grpc

from concurrent import futures


def server():
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    server_pb2_grpc.add_apiServicer_to_server(TrainerAgregator(), grpc_server)
    grpc_server.add_insecure_port('[::]:8080')
    grpc_server.start()
    grpc_server.wait_for_termination()

'''
opensTrainin
add_trainers
load_data
receive_responses
calculate_average

repeat?
'''

class TrainerAgregator(server_pb2_grpc.apiServicer):     
    trainers = []
    
    def connect_to_trainer(self):
        print("Connecting to trainer.")
        if len(self.trainers) == 0:
            print("No trainers connected.")
            return False
        else:
            print("Trainer connected.")
            info = self.trainers[0]
            channel = grpc.insecure_channel(info["ipv4"] + ":" + str(info["port"]))
            stub = client_pb2_grpc.apiStub(channel)

            return stub.train_model(client_pb2.void())

    def add_trainer(self, request, context):
        self.trainers.append({
            "uuid": request.uuid,
            "port": request.port,
            "ipv4": request.ipv4
        })
        # server_pb2_grpc.success(result=True)

        print("Trainer added.")
        print(self.trainers)

        self.connect_to_trainer()
        return server_pb2.void()

    # def load_data(): pass
    # def load_data(): pass


if __name__ == '__main__':
    print("Starting server. Listening on port 8080.")
    server()


   