# GRPC imports
import grpc
import server_pb2 
import server_pb2_grpc
import client_pb2 
import client_pb2_grpc
from concurrent import futures

import threading

# Other imports
from random import sample, randint
import time
import argparse
import numpy as np

# Custom imports
from custom_thread import CustomThread
from model import define_model, ModelBase64Encoder, ModelBase64Decoder


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
    training_mode = False
    current_round = 0

    def __init__(self, min_clients_per_round=2, max_clients_total=5, max_rounds=10, \
                 accuracy_threshold = 0.9, input_shape=(28, 28, 1), num_classes=10):
        
        # Configuring federated learning parameters
        self.min_clients_per_round = min_clients_per_round
        self.max_clients_total = max_clients_total
        self.max_rounds = max_rounds
        self.accuracy_threshold = accuracy_threshold

        # Starting base model with no pretrained weights
        self.model = define_model(input_shape, num_classes)

        # Starting train function that verifies if training mode is on
        threading.Thread(target=self.train).start()

    def federeated_train(self, results):
        models = [ModelBase64Decoder(result.weights) for result in results]

        sample_sizes = [result.sample_amount for result in results]
        weights_list = [[weight.numpy() for weight in model.weights] for model in models]

        new_weights = []
        for layers in zip(*weights_list):
            new_layer = np.average(layers, axis=0, weights=sample_sizes)
            new_weights.append(new_layer)
            
        # new_weights = [np.average(layer, axis=0, weights=sample_sizes) for layer in zip(*weights_list)]
        # new_weights = [[np.average(layer, axis=0, weights=sample_size) for layer in weights] for weights, sample_size in zip(weights_list, sample_sizes)]
        # new_weights = np.average(weights_list, axis=1, weights=sample_sizes)
        return new_weights
    
    def train(self):
        while True:
            if self.training_mode:
                for round in range(self.max_rounds):
                    print("Starting round {}.".format(round))
                    
                    self.current_round = round
                    results = self.init_round()
                    
                    # Calculate fed avg and update weights
                    avg_weights = self.federeated_train(results)
                    self.model.set_weights(avg_weights)

                    accuracies = self.test_round()
                    print("Accuracies: {}".format(accuracies))
                    accuracy_mean = np.mean(accuracies)

                    if accuracy_mean >= self.accuracy_threshold:
                        break
                    # send model to all clients
                    # receive metrics from clients
                    # compares model metrics with threshold
                    # if metrics are good enough, stop training
                    
                self.training_mode = False
            time.sleep(5)

    def test_round(self):
      
        threads = []
        for trainer in self.trainers:
            print(trainer)
            threads.append(CustomThread(target=self.test_models, args=(trainer)))

        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()   

        return [t.value.accuracy for t in threads]
    
    def test_models(self, ipv4, port, uuid):
        channel = grpc.insecure_channel(ipv4 + ":" + str(port))
        stub = client_pb2_grpc.apiStub(channel)
        model_weights = ModelBase64Encoder(self.model)
        return stub.test_model(client_pb2.models_weights(weights=model_weights, number_of_trainers=len(self.trainers)))
        

    def train_models(self, ipv4, port, uuid, number_of_trainers):
        channel = grpc.insecure_channel(ipv4 + ":" + str(port))
        stub = client_pb2_grpc.apiStub(channel)
        model_weights = ModelBase64Encoder(self.model)
        return stub.train_model(client_pb2.models_weights_input(weights=model_weights, round_number=self.current_round, number_of_trainers=number_of_trainers))
        

    def init_round(self):
        print("Initializing round.")
        number_of_trainers =  randint(self.min_clients_per_round, len(self.trainers))
        selected_trainers = sample(self.trainers, number_of_trainers)

        threads = []
        for trainer in selected_trainers:
            aux = trainer.copy()
            aux["number_of_trainers"] = number_of_trainers
            threads.append(CustomThread(target=self.train_models, args=(aux)))

        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()   

        return [t.value for t in threads]


    def add_trainer(self, request, context):        
        self.trainers.append({
            "uuid": request.uuid,
            "port": request.port,
            "ipv4": request.ipv4
        })
        
        print("Trainer added.")
        print(self.trainers[-1])

        if len(self.trainers) == self.max_clients_total: self.training_mode = True
        
        return server_pb2.success(result=1)


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Start the server.')
    parser.add_argument('--min_clients_per_round', type=int, default=2, help='Minimum number of clients per round.')
    parser.add_argument('--max_clients_total', type=int, default=3, help='Maximum number of clients per round.')
    parser.add_argument('--max_rounds', type=int, default=10, help='Maximum number of rounds.')
    parser.add_argument('--accuracy_threshold', type=float, default=0.9, help='Minimum accuracy threshold.')
    args = parser.parse_args()
    return args

def server(args):
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    server_pb2_grpc.add_apiServicer_to_server(
                                TrainerAgregator(min_clients_per_round=args.min_clients_per_round,
                                                 max_clients_total=args.max_clients_total,
                                                 max_rounds=args.max_rounds,
                                                 accuracy_threshold=args.accuracy_threshold), grpc_server)
    grpc_server.add_insecure_port('localhost:8080')
    grpc_server.start()
    grpc_server.wait_for_termination()


if __name__ == '__main__':
    args = parse_args()
    print("Starting server. Listening on port 8080.")
    server(args)


   