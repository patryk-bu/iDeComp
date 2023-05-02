# importing necessary libraries
import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, accuracy_score
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator

from data_loader_ordered import read_data

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.applications import Xception, EfficientNetV2S, InceptionV3, VGG16
from keras.applications.inception_v3 import preprocess_input

import warnings
warnings.filterwarnings('ignore')

# setting TensorFlow running parameters
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.run_functions_eagerly(True)


# function returning image passed as parameter randomly augmented
def random_aug(img):
    # random rotation
    img = np.rot90(img, np.random.choice([0, 1, 2, 3]))
    # random flipping
    if np.random.choice([0, 1]):
        img = np.flipud(img)
    if np.random.choice([0, 1]):
        img = np.fliplr(img)
    # returns augmented image
    return img


# function for getting base pre-trained CNN
def get_base(model_func=None, base_trainable=True, freeze_before=None):
    base_model = model_func(
        weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # if base_trainable is True set the base model to trainable
    if base_trainable:
        base_model.trainable = True
    if freeze_before:
        trainable = False
        for layer in base_model.layers:
            if layer.name.startswith(freeze_before):
                trainable = True
            if not trainable:
                layer.trainable = False
    else:
        base_model.trainable = False

    return base_model


# function for creating new model based on pre-trained CNN with modified classification layer
def create_model(base, num_classes, dropout=0, n_hidden=1024,
                 activation='relu', kernel_reg='l2'):
    # create Sequential model
    model = Sequential()
    # add base pre-trained CNN
    model.add(base)
    # add GlobalAveragePooling2D layer
    model.add(GlobalAveragePooling2D())
    # add fully connected layer with "n_hidden" filters
    model.add(Dense(n_hidden, activation='relu', kernel_regularizer='l2'))
    # add dropout layer if dropout parameter > 0
    if dropout:
        model.add(Dropout(dropout))
    # add classification layer
    model.add(Dense(num_classes, activation='softmax'))
    return model


# function returning classification report for DeTraC model
def get_detrac_clf_report(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    for n in range(0, 16, 2):
        y_true = np.where(y_true == n + 1, n, y_true)
        y_pred = np.where(y_pred == n + 1, n, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return report


def get_decay_fn(decay_factor=0.9, decay_rate=10):
    def step_decay(epoch, lr):
        if epoch % decay_rate == 0 and epoch != 0:
            return lr * decay_factor
        return lr

    return step_decay


# function calculating DeTraC accuracy
def detrac_accuracy(y_true_T, y_pred_T):
    y_true = y_true_T.numpy()
    y_pred = y_pred_T.numpy()
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    for n in range(0, 16, 2):
        y_true = np.where(y_true == n + 1, n, y_true)
        y_pred = np.where(y_pred == n + 1, n, y_pred)
    return accuracy_score(y_true, y_pred)


# function returning pre-processed set
def get_and_preprocess_set(X, y, n_classes):
    X = X
    y = y
    # calling preprocess_input function
    X = preprocess_input(X)
    # labels are changed to categorical values
    y = np_utils.to_categorical(y, n_classes)
    # X and y returned
    return X, y


# run experiment function
def run_experiment(optim_fn,
                   lr_schedule, base_params, dense_params, lr_params, augmented, data_dir,
                   epochs=None,
                   batch_size=None):
    # sets DATA_DIR_DECOMP variable to directory storing decomposed data split into training,
    # validation and testing subsets
    DATA_DIR_DECOMP = data_dir + '/decomposed_split/'
    # loading testing set
    X_test_d, y_test_d, num_classes_d_test = read_data(DATA_DIR_DECOMP + 'test', skip_classes=[], img_size=IMG_SIZE)
    X_test_d, y_test_d = get_and_preprocess_set(X_test_d, y_test_d, num_classes_d_test)
    # loading validation set
    X_val_d, y_val_d, num_classes_d_val = read_data(DATA_DIR_DECOMP + 'val', skip_classes=[], img_size=IMG_SIZE)
    X_val_d, y_val_d = get_and_preprocess_set(X_val_d, y_val_d, num_classes_d_val)
    # if augmented parameter is True augmented data is loaded from "DATA_DIR_DECOMP/train_aug
    if augmented:
        print('----------------TRAIN AUGMENTED----------------')
        X_train_d, y_train_d, num_classes_d_train = read_data(DATA_DIR_DECOMP + 'train_aug', skip_classes=[],
                                                              img_size=IMG_SIZE)
    # if augmented parameter is False non-augmented data is loaded from "DATA_DIR_DECOMP/train
    else:
        print('----------------TRAIN NON-AUGMENTED----------------')
        X_train_d, y_train_d, num_classes_d_train = read_data(DATA_DIR_DECOMP + 'train', skip_classes=[],
                                                              img_size=IMG_SIZE)
    X_train_d, y_train_d = get_and_preprocess_set(X_train_d, y_train_d, num_classes_d_train)

    # num_classes_d is set as number of decomposed classes
    num_classes_d = num_classes_d_train

    # train_datagen is instantiated as ImageDataGenerator with random_aug function declared as preprocessing_function
    # will apply random augmentation to training data
    train_datagen = ImageDataGenerator(
        preprocessing_function=random_aug)
    # base model is instantiated
    base_model = get_base(**base_params)
    # creating DeTraC model
    model_detrac = create_model(base_model, num_classes_d, **dense_params)
    # optimizer is set
    optimizer = optim_fn(**lr_params)
    # metrics to return whilst training are declared
    metrics = ['accuracy', detrac_accuracy]
    # model is declared with categorical_crossentropy loss function, optimizer and metrics are set
    model_detrac.compile(loss='categorical_crossentropy',
                         optimizer=optimizer, metrics=metrics,
                         run_eagerly=True)
    # model training with training and validation set
    # training history set to history_d
    history_d = model_detrac.fit(
        train_datagen.flow((X_train_d, y_train_d), batch_size=batch_size),
        validation_data=(X_val_d, y_val_d),
        steps_per_epoch=(len(X_train_d) // batch_size),
        epochs=epochs, callbacks=[lr_schedule], verbose=1)
    # predicitons for testing set are made
    y_pred = model_detrac.predict(X_test_d)
    # classification report for predictions on testing set are returned
    clf_report = get_detrac_clf_report(y_test_d, y_pred)
    # changing print statement based on whether DeTraC or Conditional DeTraC was used
    if augmented:
        print('Conditional DetraC Test Report:')
    else:
        print('DetraC Test Report:')
    print(clf_report)

    del model_detrac
    gc.collect()
    # returns history_d
    return history_d

"""### Main"""

# declaring directory storing dataset
DATA_DIR = 'data/ZHANG_PNEU/'
# setting image size to be used for experiment
IMG_SIZE = 150

# setting parameters for base pre-trained CNN
base_params = {
    'model_func': Xception,
    'base_trainable': True,
    'freeze_before': "block14"
}

# setting parameters for modified final layer of pre-trained CNN
dense_params = {
    'n_hidden': 1024,
    'dropout': 0,
    'activation': 'relu',
    'kernel_reg': 'l1'
}

# setting learning rate and momentum hyper-parameters
lr_params = {
    'learning_rate': 1e-2,
    'momentum': 0.9
}

# setting additional hyper-parameters such as number of epochs and batch size
other_params = {
    'epochs': 30,
    'batch_size': 32
}

# setting decay parameters
decay_params = {
    'decay_factor': 0.9,
    'decay_rate': 10
}

# further setting of hyper-parameters
OPTIM_FN = SGD
step_decay = get_decay_fn(**decay_params)
LR_SCHEDULE = LearningRateScheduler(step_decay)

# runs experiment with non-augmented data
history_d = run_experiment(
    OPTIM_FN, LR_SCHEDULE, base_params, dense_params, lr_params, False, DATA_DIR
    , **other_params)

# plot results of experiment
history_dict = history_d.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, loss_values, label='Training loss')
plt.plot(epochs, val_loss_values, label='Validation loss')
plt.title('DeTraC Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, label='Training acc')
plt.plot(epochs, val_acc_values, label='Validation acc')
plt.title('DeTraC Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

history_d = run_experiment(
    OPTIM_FN, LR_SCHEDULE, base_params, dense_params, lr_params, True, DATA_DIR
    , **other_params)

# plot results of augmented data
history_dict = history_d.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, loss_values, label='Training loss')
plt.plot(epochs, val_loss_values, label='Validation loss')
plt.title('iDeComp Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, label='Training acc')
plt.plot(epochs, val_acc_values, label='Validation acc')
plt.title('iDeComp Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
