import grpc
import server_pb2 
import server_pb2_grpc

import client_pb2 
import client_pb2_grpc

from concurrent import futures

from random import sample, randint

def server():
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    server_pb2_grpc.add_apiServicer_to_server(TrainerAgregator(), grpc_server)
    grpc_server.add_insecure_port('localhost:8080')
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
import threading

global MIN_CLIENTS_PER_ROUND
global MIN_CLIENTS_TOTAL
global MAX_ROUNDS

class SemaphoredList:
    list = []

    

class TrainerAgregator(server_pb2_grpc.apiServicer):     
    trainers = []
    TRAINING_MODE = False
    current_round = 0

    def __init__(self):
        self.current_round = 1

        threading.Thread(target=self.train).start()



    def train(self):
        while True:
            if self.TRAINING_MODE:
                print("Training model.")
    # def check_init():
    #     while True:
    #         sem.acquire()
    #         if len(trainers) > MIN_CLIENTS_PER_ROUND :
                
        
    def connect_to_trainer(self):
        print("Connecting to trainer.")
        if len(self.trainers) == 0:
            print("No trainers connected.")
            return False
        else:
            print("Trainer connected.")
            try:
                info = self.trainers[0]
                channel = grpc.insecure_channel(info["ipv4"] + ":" + str(info["port"]))
                stub = client_pb2_grpc.apiStub(channel)

                return stub.train_model(client_pb2.void())
            except Exception as e:
                print(e)
                return False
                
    def init_round(self):
        selected_trainers = sample(self.trainers, randint(MIN_CLIENTS_PER_ROUND, len(self.trainers)))
        
        # add pra func de threads
        for trainer in selected_trainers:
            channel = grpc.insecure_channel(trainer["ipv4"] + ":" + str(trainer["port"]))
            stub = client_pb2_grpc.apiStub(channel)
            stub.train_model(client_pb2.void())

        pass

    def add_trainer(self, request, context):
        self.trainers.append({
            "uuid": request.uuid,
            "port": request.port,
            "ipv4": request.ipv4
        })
        

        print("Trainer added.")
        print(self.trainers)

        if len(self.trainers) == 2: self.TRAINING_MODE = True

        # se chegou na quantidade necessaria para treino, inicia
        # e bloquea o recebimento de novos treinadores
        self.connect_to_trainer()
        print("Finished training")
        
        return server_pb2.success(result=True)

    # def load_data(): pass
    # def load_data(): pass


if __name__ == '__main__':
    MIN_CLIENTS_PER_ROUND = 3
    MIN_CLIENTS_TOTAL = 5
    MAX_ROUNDS = 10
    
    print("Starting server. Listening on port 8080.")
    server()


   