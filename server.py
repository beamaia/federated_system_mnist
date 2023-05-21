# GRPC imports
import grpc
import server_pb2 
import server_pb2_grpc
import client_pb2 
import client_pb2_grpc
from concurrent import futures

import threading

from tensorflow.keras.models import clone_model
# Other imports
from random import sample, randint
import argparse
import numpy as np
import time
import datetime

# Custom imports
from custom_thread import CustomThread
from model import define_model, ModelBase64Encoder, ModelBase64Decoder

import uuid


class TrainerAgregator(server_pb2_grpc.apiServicer):     
    trainers = []
    training_mode = False
    current_round = 0

    def __init__(self, min_clients_per_round=2, max_clients_total=5, max_rounds=10, \
                 accuracy_threshold = 0.90, timeout = 300, save_model = False, \
                 input_shape=(28, 28, 1), num_classes=10):
        
        # Configuring federated learning parameters
        self.min_clients_per_round = min_clients_per_round
        self.max_clients_total = max_clients_total
        self.max_rounds = max_rounds
        self.accuracy_threshold = accuracy_threshold

        # Starting base model with no pretrained weights
        self.model = define_model(input_shape, num_classes)
        
        self.timeout = timeout
        self.save_model = save_model

        # Starting train function that verifies if training mode is on
        threading.Thread(target=self.__train).start()

    #####################################################################
    def add_trainer(self, request, context):
        self.trainers.append({
            "uuid": request.uuid,
            "port": request.port,
            "ipv4": request.ipv4
        })
        
        print(f"[{datetime.datetime.now()}] Trainer added with information:")
        print(f"[{datetime.datetime.now()}] UUID: {request.uuid}")
        print(f"[{datetime.datetime.now()}] Port: {request.port}")
        print(f"[{datetime.datetime.now()}] IPv4: {request.ipv4}")
        
        if len(self.trainers) == self.max_clients_total: self.training_mode = True
        
        return server_pb2.success(result=1)
    
    def train_models(self, ipv4, port, uuid, number_of_trainers):
        channel = grpc.insecure_channel(ipv4 + ":" + str(port))
        stub = client_pb2_grpc.apiStub(channel)
        model_weights = ModelBase64Encoder(self.model.get_weights())
        return stub.train_model(client_pb2.models_weights_input(weights=model_weights, round_number=self.current_round, number_of_trainers=number_of_trainers, training_session=self.uuid))
    
    def test_models(self, ipv4, port, uuid, model_weights):
        channel = grpc.insecure_channel(ipv4 + ":" + str(port))
        stub = client_pb2_grpc.apiStub(channel)
        return stub.test_model(client_pb2.models_weights_input(weights=model_weights, round_number=self.current_round, number_of_trainers=len(self.trainers), training_session=self.uuid))    
    
    def finish_training_round(self, ipv4, port, uuid):
        print(f"[{datetime.datetime.now()}] Finishing training for trainer {uuid}.")
        channel = grpc.insecure_channel(ipv4+ ":" + str(port))
        stub = client_pb2_grpc.apiStub(channel)
        return stub.finish_training(client_pb2.finish_message(end=1))
    
    #####################################################################

    def __restart_config(self):
        if not self.save_model:
            self.model = define_model((28,28,1), 10)   
        self.current_round = 0 

    
    def __train(self):
        while True:
            if self.training_mode:
                self.uuid = str(uuid.uuid4())
                print(f"[{datetime.datetime.now()}] Generating uuid for training session.")
                print(f"[{datetime.datetime.now()}] UUID: {self.uuid}")

                for round in range(self.max_rounds):
                    print(f"[{datetime.datetime.now()}]********************************************************")
                    print(f"[{datetime.datetime.now()}] Starting round {round+1}/{self.max_rounds}.")
                    
                    self.current_round = round
                    results = self.__init_round()

                    avg_weights = self.__federeated_train(results)
                    self.model.set_weights(avg_weights)

                    accuracies = self.__test_round()
                    accuracy_mean = np.mean(accuracies)
                    print(f"[{datetime.datetime.now()}] Mean accuracy: {accuracy_mean * 100: 0.2f}.")

                    if accuracy_mean >= self.accuracy_threshold:
                        print(f"[{datetime.datetime.now()}] Accuracy threshold reached. Stopping rounds.")
                        break
                
                print(f"[{datetime.datetime.now()}] Training finished. Next round in {self.timeout} seconds.")
                self.training_mode = False    
                
                self.__finish_training()
                self.__restart_config() 

                time.sleep(self.timeout)
                
    def __init_round(self):
        number_of_trainers =  randint(self.min_clients_per_round, len(self.trainers))
        selected_trainers = sample(self.trainers, number_of_trainers)

        print(f"[{datetime.datetime.now()}] Starting training for {number_of_trainers} trainers.")

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

    def __federeated_train(self, results):
        print(f"[{datetime.datetime.now()}] Calculating federated average.")

        weights_list = [ModelBase64Decoder(result.weights) for result in results]
        sample_sizes = [result.sample_amount for result in results]

        new_weights = []
        for layers in zip(*weights_list):
            aggreagation = []
            for layer, sample_size in zip(layers, sample_sizes):
                if isinstance(aggreagation, list) and not len(aggreagation):
                    aggreagation = layer * sample_size
                else:
                    aggreagation += layer * sample_size
            new_layer = aggreagation / sum(sample_sizes)
            new_weights.append(new_layer)
            
        return new_weights
    
    def __test_round(self):
        threads = []
        model_weights = ModelBase64Encoder(self.model.get_weights())

        for trainer in self.trainers:
            aux = trainer.copy()
            aux["model_weights"] = model_weights
            threads.append(CustomThread(target=self.test_models, args=(aux)))

        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()   

        return [t.value.accuracy for t in threads]
    

    def __finish_training(self):
        threads = []
        while len(self.trainers):
            trainer = self.trainers.pop()
            threads.append(CustomThread(target=self.finish_training_round, args=(trainer)))

        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()


#########################################################################################


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Start the server.')
    parser.add_argument('--min_clients_per_round', type=int, default=5, help='Minimum number of clients per round.')
    parser.add_argument('--max_clients_total', type=int, default=10, help='Maximum number of clients per round.')
    parser.add_argument('--max_rounds', type=int, default=10, help='Maximum number of rounds.')
    parser.add_argument('--accuracy_threshold', type=float, default=0.99, help='Minimum accuracy threshold.')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds for the server between training sessions.')
    parser.add_argument('--save_model', action='store_true', default=False, help='Save the model after training. This means that the server will use this model for next training sessions.')
    args = parser.parse_args()
    return args

def server(args):
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    server_pb2_grpc.add_apiServicer_to_server(
                                TrainerAgregator(min_clients_per_round=args.min_clients_per_round,
                                                 max_clients_total=args.max_clients_total,
                                                 max_rounds=args.max_rounds,
                                                 accuracy_threshold=args.accuracy_threshold,
                                                 save_model=args.save_model,
                                                 timeout=args.timeout), grpc_server)
    
    grpc_server.add_insecure_port('localhost:8080')
    grpc_server.start()
    grpc_server.wait_for_termination()


if __name__ == '__main__':
    args = parse_args()
    print(f"[{datetime.datetime.now()}] Starting server. Listening on port 8080.")
    server(args)


   