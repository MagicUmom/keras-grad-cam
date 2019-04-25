from keras.models import Model
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.models import load_model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os
import warnings
from tqdm import tqdm

class gradCam():
    """
        # Arguments:

            model_path: str.
                the model path for keras model_load
            image_path: str or list.
                ** remember use absolute path **
                if you have only 1 image for test, you can use absolute path in string type.
                if you have multiple path for test, you can put them into list.
            focus_layer: str.
                set the focus_layer in model's layer which is you want to see.
            output_dir: str.
                ** relative path**
                save the output image to this output_dir.
            resize: tuple
                resize to (xxx,yyy)

        # Examples:
            input_model = "resnet50_.h5"
            image_path = ["~/images/00004000_000.png",
                         "~/images/00003000_000.png",
                         "~/images/00002000_000.png"]
            focus_layer = "activation_49"
            output_dir = "result"
            resize = (512,512)

            gradcam = gradCam(input_model, image_path, focus_layer, output_dir, resize )
    """


    def __init__(self, input_model, image_path, focus_layer, output_dir = "result_dir", resize = (256,256), preprocess_method = 1):
        self.input_model = input_model

        if isinstance(image_path, list):
            self.image_list = image_path
        elif isinstance(image_path, str):
            if not os.path.isfile(image_path):
                raise ValueError(image_path + " image_path is not found !")
                self.image_list = list()
                self.image_list.append(image_path)
        else:
            raise ValueError("Cannot parse " + image_path)

        self.focus_layer = focus_layer

        if not os.path.isdir(os.path.join(os.getcwd(),output_dir)):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                # if e.errno != errno.EEXIST:
                raise
        self.output_dir = os.path.join(os.getcwd(),output_dir)

        self.resize = resize

        ## TODO :
        self.preprocess_method = preprocess_method



    def preprocess_input(self, x):
        """
            TODO: when self.preprocess_method = 0
                use imagenet preprocessing method
                return x : -1~1
        """
        # normalize to 0-1
        return x/255.0

    def target_category_loss(self, x, category_index, nb_classes):
        return tf.multiply(x, K.one_hot([category_index], nb_classes))

    def target_category_loss_output_shape(self, input_shape):
        return input_shape

    def normalize(self, x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    def load_image(self, path):
        img = image.load_img(path, target_size=self.resize)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        return x

    def register_gradient(self):
        if "GuidedBackProp" not in ops._gradient_registry._registry:
            @ops.RegisterGradient("GuidedBackProp")
            def _GuidedBackProp(op, grad):
                dtype = op.inputs[0].dtype
                return grad * tf.cast(grad > 0., dtype) * \
                    tf.cast(op.inputs[0] > 0., dtype)

    def compile_saliency_function(self, model, activation_layer='block5_conv3'):
        input_img = model.input
        layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
        layer_output = layer_dict[activation_layer].output
        max_output = K.max(layer_output, axis=3)
        saliency = K.gradients(K.sum(max_output), input_img)[0]
        return K.function([input_img, K.learning_phase()], [saliency])

    def modify_backprop(self, model, name):
        g = tf.get_default_graph()
        with g.gradient_override_map({'Relu': name}):

            # get layers that have an activation
            layer_dict = [layer for layer in model.layers[1:]
            if hasattr(layer, 'activation')]

            # replace relu activation
            for layer in layer_dict:
                if layer.activation == keras.activations.relu:
                    layer.activation = tf.nn.relu

            # re-instanciate a new model
            new_model = VGG16(weights='imagenet')
        return new_model

    def deprocess_image(self, x):
        '''
        Same normalization as in:
        https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
        '''
        if np.ndim(x) > 3:
            x = np.squeeze(x)
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_dim_ordering() == 'th':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def _compute_gradients(self, tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

    def output_heatmap_img(self, cam, image_path):
        save_img_name = image_path.split('/')[-1].split('.')[0]
        cv2.imwrite( os.path.join(self.output_dir, save_img_name + "grad_cam.jpg"), cam)

    def caculate_cam_heatmap(self, output, grads_val, image):
        weights = np.mean(grads_val, axis = (0, 1))
        cam = np.ones(output.shape[0 : 2], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam = cv2.resize(cam, self.resize)
        cam = np.maximum(cam, 0)
        heatmap = cam / np.max(cam)

        #Return to BGR [0..255] from the preprocessed image
        image = image[0, :]
        image -= np.min(image)
        image = np.minimum(image, 255)
        image = image*255

        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        cam = np.float32(cam) + np.float32(image)
        cam = 255 * cam / np.max(cam)
        return (np.uint8(cam), heatmap)

    def grad_cam(self):
        """
        input_model

        """
        layer_name = self.focus_layer

        for image_path in tqdm(self.image_list):
            if not os.path.isfile(image_path):
                print(image_path + " image_path is not found !")
                warnings.warn(image_path + " image_path is not found !")
                continue

            image = self.load_image(image_path)

            predictions = self.input_model.predict(image)
            predicted_class = np.argmax(predictions)
            category_index = predicted_class
            nb_classes = len(predictions)
            assert nb_classes > 0 , "len(predictions) must be greater than 0"

            target_layer = lambda x: self.target_category_loss(x, category_index, nb_classes)
            x = Lambda(target_layer, output_shape = self.target_category_loss_output_shape)(self.input_model.output)
            model = Model(inputs=self.input_model.input, outputs=x)
            #     model.summary()
            loss = K.sum(model.output)
            try :
                conv_output =  [l for l in model.layers if l.name == layer_name][0].output
            except Exception as err:
                print(err)
                raise ValueError(layer_name + " is not found")

            grads = self.normalize(self._compute_gradients(loss, [conv_output])[0])
            gradient_function = K.function([model.input], [conv_output, grads])

            output, grads_val = gradient_function([image])
            output, grads_val = output[0, :], grads_val[0, :, :, :]

            cam, heatmap = self.caculate_cam_heatmap(output, grads_val, image)
            self.output_heatmap_img(cam, image_path)
