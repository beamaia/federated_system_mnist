# Federated Learning
This is a Federated Learning implementation using the algorithm Federated Average.

Implemeted by: Beatriz Maia, Iago Cerqueira & Sophie Dilhon

## Running the application
### Environment Config
The application was made using Python 3.10 and a few libraries that you may need to install.
It is recommended to use a virtual environment, for that you may run the following commands:

```sh
python -m venv {environment}
```

where {environment} can be any name of your choice. After creating it, it has to be activated. On Linux and Mac, use the following command:

```sh
source /{environment}/bin/activate
```

and on Windows:

```sh
.\{environment}\Scripts\activate
```

Finally, install the dependencies with

```sh
pip install -r requirements.txt
```

### Execution
To execute the system, first run one of the following scripts. They are responsible to create the files used by the clients and server communication via grpc.

```sh
# Mac or Linux
./config.sh

# Windows
.\config.bat
```

To run the server and clients, execute the following commands in different terminals. 
```sh
python server.py --min_clients_per_round {n} --max_clients_total {m} --max_rounds {r} --accuracy_threshold {a}
python client.py
```

Several clients can be created. The flags are not obligatory, the server will use default values if no argument is passed.

## Implementation

### Communication
To stablish the communication between server and clients and vice versa, the grpc lib was used, and two proto files were created. The first, [server.proto](proto/server.proto), is responsible for implementing the server's avaiable methods, and the other one, [client.proto](proto/client.proto), is responsible for implementing the clients's avaiable methods.

#### Methods

The server only has one method, which is responsible for receiving and saving the client's data, and checking if the training can start.
```sh
add_trainer(trainer_request) returns (success);
```

As for the client, three methods were created. 
- The first is responsible for training the model with the local data, and it returns to the server the model's weights. 
    ```sh
    train_model(models_weights_input) returns (models_weights_output);
    ```
- The second one, receives the aggregated weights and tests the model with the local data.
    ```sh
    test_model(models_weights) returns (metrics_results);
    ```
- The last one ends the client's server
    ```sh
    finish_training(finish_message) returns (finish_message);
    ```

### Server

### Client

## Analysis