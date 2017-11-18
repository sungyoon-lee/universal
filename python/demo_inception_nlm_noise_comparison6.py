import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
import matplotlib.pyplot as plt
import sys, getopt
import zipfile
from timeit import time
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle
from skimage import img_as_float


# Only used in the case no pb file is ready
if sys.version_info[0] >= 3: # python >=3
    from urllib.request import urlretrieve
else: # python <=2
    from urllib import urlretrieve


from universal_pert import universal_perturbation
device = '/gpu:0' # choose gpu
num_classes = 10

def jacobian(y_flat, x, inds):
    n = num_classes # Not really necessary, just a quick fix.
    loop_vars = [
         tf.constant(0, tf.int32),
         tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j,_: j < n,
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))), # \nabla_x f_j ?
        loop_vars)
    return jacobian.stack()

if __name__ == '__main__':

    # Parse arguments (input?)
    argv = sys.argv[1:]

    # Default values
    path_train_imagenet = '/datasets2/ILSVRC2012/train'
    path_test_image = 'data/test_im6.jpg'
    
    try:
        opts, args = getopt.getopt(argv,"i:t:",["test_image=","training_path="])
    except getopt.GetoptError:
        print ('python ' + sys.argv[0] + ' -i <test image> -t <imagenet training path>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-t':
            path_train_imagenet = arg
        if opt == '-i':
            path_test_image = arg

    with tf.device(device):
        persisted_sess = tf.Session()
        inception_model_path = os.path.join('data', 'tensorflow_inception_graph.pb') # concatenation

        if os.path.isfile(inception_model_path) == 0: # no file
            print('Downloading Inception model...')
            # Retrieving from url to the given path
            urlretrieve ("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip", os.path.join('data', 'inception5h.zip'))
            # Unzipping the file
            zip_ref = zipfile.ZipFile(os.path.join('data', 'inception5h.zip'), 'r')
            zip_ref.extract('tensorflow_inception_graph.pb', 'data')
            zip_ref.close()

        model = os.path.join(inception_model_path)

        # Load the Inception model
        with gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        persisted_sess.graph.get_operations()

        persisted_input = persisted_sess.graph.get_tensor_by_name("input:0")
        persisted_output = persisted_sess.graph.get_tensor_by_name("softmax2_pre_activation:0")

        print('>> Computing feedforward function...')
        def f(image_inp): return persisted_sess.run(persisted_output, feed_dict={persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))})

        file_perturbation = os.path.join('data', 'universal.npy')

        if os.path.isfile(file_perturbation) == 0:

            # TODO: Optimize this construction part!
            print('>> Compiling the gradient tensorflow functions. This might take some time...')
            y_flat = tf.reshape(persisted_output, (-1,))
            inds = tf.placeholder(tf.int32, shape=(num_classes,))
            dydx = jacobian(y_flat,persisted_input,inds)

            print('>> Computing gradient function...')
            def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)

            # Load/Create data
            datafile = os.path.join('data', 'imagenet_data.npy')
            if os.path.isfile(datafile) == 0:
                print('>> Creating pre-processed imagenet data...')
                X = create_imagenet_npy(path_train_imagenet) ## ERROR

                print('>> Saving the pre-processed imagenet data')
                if not os.path.exists('data'):
                    os.makedirs('data')

                # Save the pre-processed images
                # Caution: This can take a lot of space. Comment this part to discard saving.
                np.save(os.path.join('data', 'imagenet_data.npy'), X)

            else:
                print('>> Pre-processed imagenet data detected')
                X = np.load(datafile)

            # Running universal perturbation
            v = universal_perturbation(X, f, grad_fs, delta=0.2, num_classes=num_classes)

            # Saving the universal perturbation
            np.save(os.path.join(file_perturbation), v)

        else:
            print('>> Found a pre-computed universal perturbation! Retrieving it from ", file_perturbation')
            v = np.load(file_perturbation)

        print('>> Testing the universal perturbation on an image')

        # Test the perturbation on the image
        labels = open(os.path.join('data', 'labels.txt'), 'r').read().split('\n')

        image_original = preprocess_image_batch([path_test_image], img_size=(256, 256), crop_size=(224, 224), color_mode="rgb")
        label_original = np.argmax(f(image_original), axis=1).flatten()
        str_label_original = labels[np.int(label_original)-1].split(',')[0]
        
        #print(image_original.shape)
        #print(np.max(undo_image_avg(image_original).astype(dtype='uint8')))
        #print(np.min(undo_image_avg(image_original).astype(dtype='uint8')))

        image_nlm2 = denoise_nl_means(undo_image_avg(image_original[0,:,:,:])/255,7,11,0.02, multichannel=True)
        image_nlm2 = image_nlm2*255
        label_nlm2 = np.argmax(f(image_nlm2), axis=1).flatten()
        str_label_nlm2 = labels[np.int(label_nlm2)-1].split(',')[0]

        image_nlm = denoise_nl_means(undo_image_avg(image_original[0,:,:,:])/255,7,11,0.05, multichannel=True)
        image_nlm = image_nlm*255
        label_nlm = np.argmax(f(image_nlm), axis=1).flatten()
        str_label_nlm = labels[np.int(label_nlm)-1].split(',')[0]

        image_nlm10 = denoise_nl_means(undo_image_avg(image_original[0,:,:,:])/255,7,11,0.1, multichannel=True)
        image_nlm10 = image_nlm10*255
        label_nlm10 = np.argmax(f(image_nlm10), axis=1).flatten()
        str_label_nlm10 = labels[np.int(label_nlm10)-1].split(',')[0]
        # Clip the perturbation to make sure images fit in uint8
        clipped_v = np.clip(undo_image_avg(image_original[0,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(image_original[0,:,:,:]), 0, 255)

        image_perturbed = image_original + clipped_v[None, :, :, :]
        label_perturbed = np.argmax(f(image_perturbed), axis=1).flatten()
        str_label_perturbed = labels[np.int(label_perturbed)-1].split(',')[0]

        image_denoised2 = denoise_nl_means(undo_image_avg(image_perturbed[0,:,:,:])/255,7,11,0.01, multichannel=True)
        image_denoised2 = image_denoised2*255
        label_denoised2 = np.argmax(f(image_denoised2), axis=1).flatten()
        str_label_denoised2 = labels[np.int(label_denoised2)-1].split(',')[0]
        
        image_denoised = denoise_nl_means(undo_image_avg(image_perturbed[0,:,:,:])/255,7,11,0.03, multichannel=True)
        image_denoised = image_denoised*255
        label_denoised = np.argmax(f(image_denoised), axis=1).flatten()
        str_label_denoised = labels[np.int(label_denoised)-1].split(',')[0]

        image_denoised10 = denoise_nl_means(undo_image_avg(image_perturbed[0,:,:,:])/255,7,11,0.04, multichannel=True)
        image_denoised10 = image_denoised10*255
        label_denoised10 = np.argmax(f(image_denoised10), axis=1).flatten()
        str_label_denoised10 = labels[np.int(label_denoised)-1].split(',')[0]
        # Show original, perturbed image, denoised image and noise
        plt.figure()
        plt.subplot(2, 4, 1)
        plt.imshow(undo_image_avg(image_original[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_original)
        
        plt.subplot(2, 4, 2)
        plt.imshow(image_nlm2[:, :, :].astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_nlm2)

        plt.subplot(2, 4, 3)
        plt.imshow(image_nlm[:, :, :].astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_nlm)

        plt.subplot(2, 4, 4)
        plt.imshow(image_nlm10[:, :, :].astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_nlm10)

        plt.subplot(2, 4, 5)
        plt.imshow(undo_image_avg(image_perturbed[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_perturbed)

        plt.subplot(2, 4, 6)
        plt.imshow(image_denoised2[:, :, :].astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_denoised2)
        
        plt.subplot(2, 4, 7)
        plt.imshow(image_denoised[:, :, :].astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_denoised)
        
        plt.subplot(2, 4, 8)
        plt.imshow(image_denoised10[:, :, :].astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_denoised10)
        
        plt.show()
