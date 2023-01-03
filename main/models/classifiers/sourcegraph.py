import tensorflow as tf
from keras import Model
import logging

import psutil
import os

from keras import backend as k
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Input, Embedding, Concatenate, Dropout, TimeDistributed, Dense

from models.layers.attention import AttentionLayer

from keras.regularizers import l2

from util import ensure_path, tf_gpu_housekeeping

class MemoryUsageCallback(Callback):
  '''Monitor memory usage on epoch begin and end.'''

  def on_epoch_begin(self,epoch,logs=None):
    print('**Epoch {}**'.format(epoch))
    print('Memory usage on epoch begin: {}'.format(psutil.Process(os.getpid()).memory_info().rss))

  def on_epoch_end(self,epoch,logs=None):
    print('Memory usage on epoch end:   {}'.format(psutil.Process(os.getpid()).memory_info().rss))

class SourceGraph(Model):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.code_rep = ['ast', 'cfg', 'cdg', 'ddg']
        self.properties = ['source','path', 'value']

        #self.att_scores = []
        self.inputs = []

    def _setup_input_layer(self, params, MAX_CONTEXT):
        source_input = Input(shape = (MAX_CONTEXT,), dtype = tf.float64)
        path_input = Input(shape = (MAX_CONTEXT,), dtype = tf.float64)
        value_input = Input(shape = (MAX_CONTEXT,), dtype = tf.float64)
        
        self.inputs += [source_input, path_input, value_input]

        EMBEDDING_TOKEN_DIM = params["embedding"]["EMBEDDING_TOKEN_DIM"]
        EMBEDDING_PATH_DIM = params["embedding"]["EMBEDDING_PATH_DIM"]

        Embedding_source = Embedding(MAX_CONTEXT, EMBEDDING_TOKEN_DIM, mask_zero= True)
        Embedding_path = Embedding(MAX_CONTEXT, EMBEDDING_PATH_DIM, mask_zero=True)
        Embedding_value = Embedding(MAX_CONTEXT, EMBEDDING_TOKEN_DIM, mask_zero= True)
        
        '''attention_mask = Concatenate()([Embedding_source.compute_mask(source_input), 
                                    Embedding_path.compute_mask(path_input), 
                                    Embedding_value.compute_mask(value_input)])'''

        context_layer = Concatenate()([Embedding_source(source_input), Embedding_path(path_input), Embedding_value(value_input)])
        context_layer = Dropout(params["dropout"])(context_layer)

        context_after_dense = TimeDistributed(Dense(2 * EMBEDDING_TOKEN_DIM + EMBEDDING_TOKEN_DIM, use_bias= False, activation= "tanh"))(context_layer)

        code_vector, att_score = AttentionLayer()([context_after_dense])

        #self.att_scores.append(att_score)

        return code_vector
    def _setup_classifier_layer(self, layer, params):
        params = params["final_dense"]

        layer = Dense(units = params["units"],
                        activation= params["activation"],
                        use_bias = params["use_bias"],
                        bias_regularizer= l2(params["bias_regularizer"]),
                        kernel_regularizer= l2(params["kernel_regularizer"]))(layer)
        
        return layer
    def build_model(self, model_name = None, verbose = False, extract_representation = True):
        if not model_name:
            self.model_name = self.args["models"]
        else:
            self.model_name = model_name

        if verbose:
            logging.info("Building model {}".format(self.model_name))
        
        params = self.args["models"][self.model_name]
        MAX_CONTEXT = self.args["split"]["MAX_CONTEXT"]

        code_vectors = []
        #Input Layer
        for rep in self.code_rep:
            layer = self._setup_input_layer(params, MAX_CONTEXT[rep])
            code_vectors.append(layer)

        classifier = Concatenate()(code_vectors)
        
        outputs = self._setup_classifier_layer(classifier, params)
        
        #prev_cls = classifier

        #outputs = [classifier] + self.att_scores
        #loss    = [params["loss"]] + [None] * len(self.att_scores)

        #if extract_representation:
        #    outputs.append(prev_cls)
        #    loss.append(None)

        model = Model(name=self.model_name, inputs=self.inputs, outputs=outputs)
        
        model.compile(loss=params["loss"],
                      optimizer=params["optimizer"],
                      metrics=params["metrics"])

        self.model = model
        if verbose:
            logging.info(model.summary())

        return model
    
    def fit(self, train_set, test_set):
        
        X_train, Y_train = train_set
        
        X_test, Y_test = test_set

        params = self.args["train"]

        checkpoint_dir = self.args["weights"] + "{}_weights.h5".format(self.model_name)
        
        if (params["callback"]):
            callback = ModelCheckpoint(filepath = checkpoint_dir, 
                                                        save_best_only = True,
                                                        mode = "min", 
                                                        save_freq= "epoch",
                                                        verbose = 1)

        inputs = []
        valids = []
        for rep in self.code_rep:
            inputs+=[X_train[rep + '_source'], X_train[rep + '_path'], X_train[rep + '_value']]
            valids+=[X_test[rep + '_source'], X_test[rep + '_path'], X_test[rep + '_value']]
        self.model.fit(inputs, 
                    Y_train, 
                    validation_data=(valids, Y_test),
                    epochs= params["epoch"],
                    batch_size = params["batch_size"],
                    callbacks = [callback, MemoryUsageCallback()])
            
    def load_weights(self):
        if not self.model:
            raise Exception("Model not initialized!")
        
        path = self.args["weights"] + "{}_weights.h5".format(self.model_name)
        self.model.load_weights(path)

    def predict(self, dataset):
        X , Y  = dataset
        inputs = []

        for rep in self.code_rep:
            inputs+=[X[rep + '_source'], X[rep + '_path'], X[rep + '_value']]

        return self.model.predict(inputs)
    
    def predict_problem():

        return 0
        
    def serialize(self, path):
        ensure_path(path)
        self.model.save_models(path)

    def deserialize(self, path):
        self.model.load_weights(path)
    
    def __name__(self):
        return self.model_name
    
