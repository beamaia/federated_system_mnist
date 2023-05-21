# GRPC imports
import grpc
import client_pb2 
import client_pb2_grpc
import server_pb2
import server_pb2_grpc
from concurrent import futures

import threading

# ML imports
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import numpy as np

# Other imports
from random import randint
import time 
import datetime
import uuid

import argparse
import csv

import os
# Custom imports
from model import ModelBase64Encoder, ModelBase64Decoder, define_model


class Client(client_pb2_grpc.apiServicer):
    def __init__(self, ipv4 = "localhost", store_training_data = False, store_test_data = False):
        self.uuid = str(uuid.uuid4())
        self.ipv4 = ipv4
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()

        self.model = define_model((28, 28, 1), 10)
        self.opt = SGD(learning_rate=0.01, momentum=0.9)
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])

        self.listen = True

        while True:
            self.port = randint(1000, 9999)
            if self.port != 8080:
                break

        self.channel = grpc.insecure_channel('localhost:8080')
        self.stub = server_pb2_grpc.apiStub(self.channel)

        self.stop_event = threading.Event()
        self.grpc_server = self.server(self.stop_event)

        self.store_training_data = store_training_data
        self.store_test_data = store_test_data

        threading.Thread(target=self.listen_to_server).start()
    
    ##############################################################################################################
    def train_model(self, request, context):
        print(f"[{datetime.datetime.now()}]********************************************************")
        print(f"[{datetime.datetime.now()}] Training model. Round number: " + str(request.round_number))
 
        percentage = int(1 / request.number_of_trainers * 100)
        min_lim = min(5, percentage)
        random_number = randint(min_lim, percentage) / 100
        
        sample_size_train = int(random_number * len(self.x_train))
        print(f"[{datetime.datetime.now()}] Sample size: {sample_size_train}")

        idx_train = np.random.choice(np.arange(len(self.x_train)), sample_size_train, replace=False)
        x_train = self.x_train[idx_train]
        y_train = self.y_train.numpy()[idx_train]

        model_weights = ModelBase64Decoder(request.weights)
        self.model.set_weights(model_weights)

        history = self.model.fit(x_train, y_train, batch_size=32 ,epochs=1, verbose=False)
        model_weights = ModelBase64Encoder(self.model.get_weights())

        print(f"[{datetime.datetime.now()}] Training finished. Results:")
        print(f"[{datetime.datetime.now()}] Accuracy: {history.history['accuracy'][0]}")
        print(f"[{datetime.datetime.now()}] Loss: {history.history['loss'][0]}")

        if self.store_training_data:
            self.store_information(history.history['loss'][0], history.history['accuracy'][0], request.round_number, request.training_session)
        
        return client_pb2.models_weights_output(weights=model_weights, sample_amount=sample_size_train)

    def test_model(self, request, context):
        print(f"[{datetime.datetime.now()}]********************************************************")
        print(f"[{datetime.datetime.now()}] Testing model.")

        sample_size_test = int((1/request.number_of_trainers)*len(self.x_test))
        idx_test = np.random.choice(np.arange(len(self.x_test)), sample_size_test, replace=False)
        x_test = self.x_test[idx_test]
        y_test = self.y_test.numpy()[idx_test]
        
        model_weights = ModelBase64Decoder(request.weights)
        self.model.set_weights(model_weights)
        results = self.model.evaluate(x_test, y_test, batch_size=32, verbose=False)

        print(f"[{datetime.datetime.now()}] Testing finished. Results:")
        print(f"[{datetime.datetime.now()}] Accuracy: {results[1]}")
        print(f"[{datetime.datetime.now()}] Loss: {results[0]}")

        if self.store_test_data:
            self.store_information(results[0], results[1], request.round_number, request.training_session, train=False)

        return client_pb2.metrics_results(accuracy=results[1])
    

    def finish_training(self, request, context):
        if request.end == 1:
            print(f"[{datetime.datetime.now()}] Training finished. Shutting down server, this could take a while...")
            self.listen = False
            return client_pb2.finish_message(end=1)
        return client_pb2.finish_message(end=0)
    
    ##############################################################################################################
    def connect(self):
        return self.stub.add_trainer(server_pb2.trainer_request(uuid=self.uuid, ipv4=self.ipv4, port=self.port))
    
    ##############################################################################################################
    def listen_to_server(self):
        while True:
            if not self.listen:
                time.sleep(5)
                self.stop_event.set()
                break
            time.sleep(60)

    def connect_to_aggregator(self):
        res = self.connect()
        if res.result == 1:
            print(f"[{datetime.datetime.now()}] Connected to aggregator.")
            self.stop_event.wait()
            self.grpc_server.stop(10)
        else:
            print(f"[{datetime.datetime.now()}] Failed to connect to aggregator.")


    def server(self, events=None):
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        client_pb2_grpc.add_apiServicer_to_server(self, grpc_server)
        grpc_server.add_insecure_port(f'{self.ipv4}:{self.port}')
        grpc_server.start()

        print(f"[{datetime.datetime.now()}] Server started.")
        print(f"[{datetime.datetime.now()}] Client listening on {self.ipv4}:{self.port}.")
        return grpc_server

    # processamento de dados
    def load_data(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_train = x_train / 255.0
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        x_test = x_test / 255.0

        y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
        y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
    
        return (x_train, y_train), (x_test, y_test)

    def store_information(self, loss, accuracy, round, session_uuid, train=True):
        now = datetime.datetime.now()
        
        file_name = "train" if train else "test"
        file_name += "_results.csv"

        new_file = os.path.isfile(file_name)

        headers = ['uuid', 'ipv4','port', 'accuracy', 'loss', 'round', 'timestamp', 'session_uuid']

        with open (file_name,'a') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
            if not new_file:
                writer.writeheader()
            writer.writerow({'uuid': self.uuid, 
                             'ipv4': self.ipv4, 
                             'port': self.port, 
                             'accuracy':accuracy, 
                             'loss': loss, 
                             'round': round, 
                             'timestamp': now, 
                             'session_uuid': session_uuid})

def parse_args():
    parser = argparse.ArgumentParser(description='Client')
    parser.add_argument('--ipv4', type=str, default='localhost', help='IPv4 address of the client')
    parser.add_argument('--save_train', action='store_true', default=False, help='Plot the training results')
    parser.add_argument('--save_test', action='store_true', default=False, help='Plot the testing results')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    client = Client(store_training_data=args.save_train, store_test_data=args.save_test, ipv4=args.ipv4)
    print(f"[{datetime.datetime.now()}] Client started.")
    client.connect_to_aggregator()

