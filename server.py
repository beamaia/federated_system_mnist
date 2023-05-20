import grpc
import trainer_grpc_pb2
from concurrent import futures
import trainer_agregator_grpc_pb2_grpc

def server():
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    trainer_agregator_grpc_pb2_grpc.TrainerAgregator(MineServicer(), grpc_server)
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

class TrainerAgregator(trainer_agregator_grpc_pb2_grpc.apiServer):     
    trainers = []

    def add_trainer(self, request, context):
        self.trainers.append({
            "uuid": request.uuid,
            "port": request.port,
            "ipv4": request.ipv4
        })
        trainer_grpc_pb2.success(result=True)

    def load_data(): pass
    def load_data(): pass


if __name__ == '__main__':
    print("Starting server. Listening on port 8080.")
    server()


   