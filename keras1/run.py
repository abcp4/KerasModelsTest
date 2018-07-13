import numpy as np
import argparse
import traceback
import os

np.random.seed(1337)  # for reproducibility

import util
import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Path to data dir')
    parser.add_argument('--model', type=str, required=True, help='Base model architecture', choices=[
        config.MODEL_RESNET50,
        config.MODEL_RESNET152,
        config.MODEL_INCEPTION_V3,
        config.MODEL_VGG16,
        config.MODEL_CNN,
        config.MODEL_VGGFACE
        ])
    parser.add_argument('--nb_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--freeze_layers_number', type=int, help='will freeze the first N layers and unfreeze the rest')
    return parser.parse_args()


def init():
    util.lock()
    util.set_img_format()
    util.override_keras_directory_iterator_next()
    util.set_classes_from_train_dir()
    util.set_samples_info()

    if util.get_keras_backend_name() != 'theano':
        util.tf_allow_growth()

    if not os.path.exists(config.trained_dir):
        os.mkdir(config.trained_dir)


def run():
    model = util.get_model_class_instance(
        class_weight=util.get_class_weight(config.train_dir),
        nb_epoch=args.nb_epoch,
        batch_size=args.batch_size,
        freeze_layers_number=args.freeze_layers_number)

    #config.train_dir = 'data/sorted/test/'
    #model.load()
    #model.test()
    model.train()
    #model.extract()
    model.evaluate()

    print('Training is finished!')
    #model.save('cnn_model.h5')


if __name__ == '__main__':
    try:
        args = parse_args()
        if args.data_dir:
            config.data_dir = args.data_dir
            config.set_paths()
        if args.model:
            config.model = args.model

        init()
        run()
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        util.unlock()
