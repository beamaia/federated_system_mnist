# GRPC imports
import grpc
import client_pb2 
import client_pb2_grpc
import server_pb2
import server_pb2_grpc
from concurrent import futures

# ML imports
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import numpy as np

# Other imports
from random import randint
import time 

# Custom imports
from model import ModelBase64Encoder, ModelBase64Decoder


class Client(client_pb2_grpc.apiServicer):
    def __init__(self):
        self.uuid = "2"
        self.ipv4 = "localhost"
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()
        while True:
            self.port = randint(1000, 9999)
            if self.port != 8080:
                break

        self.channel = grpc.insecure_channel('localhost:8080')
        self.stub = server_pb2_grpc.apiStub(self.channel)

    def train_model(self, request, context):
        print("Training model. Round number: " + str(request.round_number))

        # 
        percentage = int(1 / request.number_of_trainers * 100)
        min_lim = min(5, percentage)
        random_number = randint(min_lim, percentage) / 100
        
        sample_size_train = int(random_number * len(self.x_train))


        idx_train = np.random.choice(np.arange(len(self.x_train)), sample_size_train, replace=False)
        x_train = self.x_train[idx_train]
        y_train = self.y_train.numpy()[idx_train]

        model = ModelBase64Decoder(request.weights)
        opt = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(x_train, y_train, batch_size=4 ,epochs=1, verbose=2)
        model_weights = ModelBase64Encoder(model)

        return client_pb2.models_weights_output(weights=model_weights, sample_amount=sample_size_train)

    def connect_to_aggregator(self):
        grpc_server = self.server()
        print("Server started.")
        print(f"Client listening on {self.ipv4}:{self.port}.")
        res = self.connect()
        if res.result == 1:
            print("Connected to aggregator.")
            grpc_server.wait_for_termination()
        else:
            print("Failed to connect to aggregator.")


    def server(self):
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        client_pb2_grpc.add_apiServicer_to_server(Client(), grpc_server)
        grpc_server.add_insecure_port(f'{self.ipv4}:{self.port}')
        grpc_server.start()
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

    def connect(self):
        res = self.stub.add_trainer(server_pb2.trainer_request(uuid=self.uuid, ipv4=self.ipv4, port=self.port))
        return res

    def test_model(self, request, context):
        print("Testing model.")

        sample_size_test = int((1/request.number_of_trainers)*len(self.x_test))
        idx_test = np.random.choice(np.arange(len(self.x_test)), sample_size_test, replace=False)
        x_test = self.x_test[idx_test]
        y_test = self.y_test.numpy()[idx_test]
        
        model = ModelBase64Decoder(request.weights)
        opt = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        results = model.evaluate(x_test, y_test, batch_size=4, verbose=False)
        print(results)

        return client_pb2.metrics_results(accuracy=results[1])

if __name__ == '__main__':
    client = Client()
    print("Client started.")
    client.connect_to_aggregator()

