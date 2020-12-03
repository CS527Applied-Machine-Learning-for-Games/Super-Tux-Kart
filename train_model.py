import numpy as np
from network import network
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 15
MODEL_NAME = 'SuperTuxKart_Model.model'

model = network(WIDTH, HEIGHT, LR)

for e in range(EPOCHS):

    train_data = np.load('training_data_balance_data.npy',allow_pickle=True)

    train = train_data[:-1000]
    test = train_data[-1000:]

    X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=400, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)