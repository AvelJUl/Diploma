import os
import random
import datetime
import re
import math
import logging

import multiprocessing

import numpy as np
import tensorflow as tf

import keras
import keras.backend as k_backend
import keras.layers as k_layers
import keras.engine as k_engine
import keras.models as k_models

import mrccn.utils as utils

from collections import OrderedDict
from distutils.version import LooseVersion

# Требуется TensorFlow 1.3+ и Keras 2.0.8+.
assert LooseVersion(tf.__version__) >= LooseVersion('1.3')
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


################################################################################
#  Утилиты
################################################################################

def log(text, array=None):
    """
    Логирование текстового сообщения. И, опционально, если предоставляется
    массив Numpy, то логируется его размерность, минимальное и максимальное
    значения.
    :param text: текстовое сообщение.
    :param array: массив Numpy
    :return:
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


class BatchNormalization(k_layers.BatchNormalization):
    """
    Пакетная нормализация -- метод, который позволяет повысить
    производительность и стабилизировать работу искусственных нейронных сетей.
    Суть данного метода заключается в том, что некоторым слоям нейронной сети
    на вход подаются данные, предварительно обработанные и имеющие нулевое
    математическое ожидание и единичную дисперсию.

    Пакетная нормализация оказывает негативное влияние на обучение, если пакеты
    маленькие, поэтому этот слой часто замораживается (через настройку в классе
    Config) и функционирует как линейный слой.
    """

    def call(self, inputs, training=None):
        """
        :param inputs: входной тензор (любого ранга).
        :param training: None -- обучение.
                         False -- заморозка слоя. слой нормализует свои входные
                         данные, используя среднее значение и дисперсию своей
                         скользящей статистики, полученной во время обучения.
                         True -- слой нормализует свои входные данные,
                         используя среднее значение и дисперсию текущей партии
                         входных данных.
        :return:
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """
    Вычисляет ширину и высоту каждой ступени магистральной сети.
    :return: [N, (height, width)]. Где N количество ступеней
    """
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
          int(math.ceil(image_shape[1] / stride))]
         for stride in config.BACKBONE_STRIDES])


################################################################################
#  Resnet граф
################################################################################


def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    """
    Блок, который не имеет сверточного слоя при anchor
    :param input_tensor: входной тензор
    :param kernel_size: по умолчанию 3, размер ядра среднего слоя свертки на
                        основном пути
    :param filters: список integer, nb_filters 3 слоя свертки на главном пути
    :param stage: integer, метка текущей стадии, используемая для генерации
                  имен слоев
    :param block: 'a','b'..., текущая метка блока, используемая для генерации
                имен слоев
    :param use_bias: Boolean. Использовать или не использовать уклон в слоях
                     свертки.
    :param train_bn: Boolean. Обучить или заморозить слои нормализации пакетов
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = k_layers.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = k_layers.Activation('relu')(x)

    x = k_layers.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                        use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = k_layers.Activation('relu')(x)

    x = k_layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    x = k_layers.Add()([x, input_tensor])
    x = k_layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def convolution_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    """
    Блок, который имеет сверточный слой наt shortcut.
    Cо стадии 3 первый слой свертки на главном пути имеет подвыборку = (2,2).
    И shortcut  должен иметь подвыборку = (2,2).
    :param input_tensor: входной тензор
    :param kernel_size: по умолчанию 3, размер ядра среднего слоя свертки на
                        основном пути
    :param filters: список integer, nb_filters 3 слоя свертки на главном пути
    :param stage: integer, метка текущей стадии, используемая для генерации
                  имен слоев
    :param block: 'a','b'..., текущая метка блока, используемая для генерации
                имен слоев
    :param use_bias: Boolean. Использовать или не использовать уклон в слоях
                     свертки.
    :param train_bn: Boolean. Обучить или заморозить слои нормализации пакетов
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = k_layers.Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a',
                        use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = k_layers.Activation('relu')(x)

    x = k_layers.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                        use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = k_layers.Activation('relu')(x)

    x = k_layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                           '2c', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = k_layers.Conv2D(nb_filter3, (1, 1), strides=strides,
                               name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = k_layers.Add()([x, shortcut])
    x = k_layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet101_graph(input_image, train_bn=True):
    """
    Построить граф ResNet.
    :param train_bn: Boolean. Обучить или заморозить слои BN
    """
    x = k_layers.ZeroPadding2D((3, 3))(input_image)
    # Этaп 1
    x = k_layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNormalization(name='bn_conv1')(x, training=train_bn)
    x = k_layers.Activation('relu')(x)
    C1 = x = k_layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Этaп 2
    x = convolution_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Этaп 3
    x = convolution_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Этaп 4
    x = convolution_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = 22 # для сети resnet101
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    x = convolution_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
    C5 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    return [C1, C2, C3, C4, C5]


################################################################################
#  Region Proposal Слой
################################################################################

def apply_box_deltas_graph(boxes, deltas):
    """
    Применить данные дельты к указанным полям.
    :param boxes: [N, (y1, x1, y2, x2)] прямоугольники для обновления
    :param deltas: [N, (dy, dx, log(dh), log(dw))] уточнения, которые надо
                   применить
    """
    # Преобразование к y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Применение дельт
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Обратное преобразование к y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] в формате y1, x1, y2, x2
    """
    # Разделение
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Скрепление
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(k_engine.Layer):
    """
    Получает anchor показатели и выбирает подмножество для передачи в качестве
    предложений на второй этап. Фильтрация выполняется на основе anchor
    показателей и подавления немаксимумов для устранения наложений. Также
    применяет ограничивающие рамки уточнения anchor.
    """
    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Якоря
        anchors = inputs[2]

        # Улучшить производительность, обрезая до верхних якорей по счету и
        # делая остальное на меньшем подмножестве.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                            self.config.IMAGES_PER_GPU, names=["pre_nms_anchors"])

        # Примените дельты к якорям, чтобы получить уточненные якоря.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Прикрепить к границам изображения. Так как мы находимся в
        # нормализованных координатах, прикрепление в диапозоне 0..1.
        # [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Отфильтровать небольшие контейнеры
        # Согласно статье Xinlei Chen, это снижает точность обнаружения мелких
        # объектов, поэтому мы пропускаем это.

        # Подавление немаксимумов
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count,
                                                   self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Заполнение при необходимости
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([boxes, scores], nms, self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


################################################################################
#  ROIAlign Слой
################################################################################

def log2_graph(x):
    """Реализация Log2. TF не имеет встроенной реализации."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(k_engine.Layer):
    """
    Реализация ROI Pooling на нескольких уровнях пирамиды признаков.
    """

    def __init__(self, pool_shape, **kwargs):
        """
        :param pool_shape: [pool_height, pool_width] выходных объединенных
                           областей. Обычно [7, 7]
        """
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        """
        :param inputs:
            - boxes: [batch, num_boxes, (y1, x1, y2, x2)] в нормированных
                     координатах. Возможно заполнены нулями, если не достаточно
                     полей для заполнения массива.
            - image_meta: [batch, (meta data)] Детали изображения. Смотрите
                          compose_image_meta()
            - feature_maps: Список карт характеристик с разных уровней пирамиды.
                            Каждый [batch, height, width, channels]
        :return: Объединенные регионы в форме: [batch, num_boxes, pool_height,
                 pool_width, channels].
        """
        # Обрезать поля [batch, num_boxes, (y1, x1, y2, x2)] в нормализованных
        # координатах
        boxes = inputs[0]

        # Мета-данные изображения
        # Содержит детали об изображении. Смотрите compose_image_meta()
        image_meta = inputs[1]

        # Карты признаков. Список карт характеристик с разных уровней пирамиды.
        # Каждый [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Назначить каждый ROI на уровень в пирамиде на основе области ROI.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Использовать размерность первого изображения. Изображения в пакете
        # должны иметь одинаковый размер.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Уравнение 1 в документе Feature Pyramid Networks. Учтите тот факт,
        # что наши координаты здесь нормализованы. например ROI 224x224 (в
        # пикселях) отображается на P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Проход уровней и применение пула ROI к каждому. P2 до P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Индексы контейнеров для crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Следить за тем, какой блок сопоставлен с каким уровнем
            box_to_level.append(ix)

            # Прекратить градиентное распространение предложений ROI
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Обрезать и изменить размер
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Упаковать объединенные функции в один тензор
        pooled = tf.concat(pooled, axis=0)

        # Упаковать отображение box_to_level в один массив и добавьте другой
        # столбец, представляющий порядок объединенных блоков
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Переупорядочить объединенные функции в соответствии с порядком
        # оригинальных контейнеров
        # Сортировать box_to_level по пакету, а затем по индексу контейнера
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Повторно добавить размер партии
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


################################################################################
# Целевой слой обнаружения
################################################################################

def overlaps_graph(boxes1, boxes2):
    """
    Вычисляет IoU перекрытия между двумя наборами блоков.
    :param boxes1: [N, (y1, x1, y2, x2)].
    :param boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. Это позволяет нам сравнивать каждый
    # boxes1 с каждым boxes2 без петель.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Вычислить пересечения
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Вычислить объединения
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Вычислить IoU и изменить его на [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """
    Создает цели обнаружения для одного изображения. Подвыборки предлагаются и
    генерируются идентификаторы целевого класса, ограничивающие рамки и маски
    для каждого.
    :param proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] в нормированных
                      координатах. Может быть дополнен нулями, если предложений
                      недостаточно.
    :param gt_class_ids: [MAX_GT_INSTANCES] int класс IDs
    :param gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] в нормированных
                     координатах.
    :param gt_masks: [height, width, MAX_GT_INSTANCES] типа boolean.
    :return Целевые ROI и соответствующие идентификаторы классов, сдвиги
    ограничивающих рамок и маски. Возвращаемые массивы могут быть дополнены
    нулями, если не хватает целевых ROI.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] в нормированных
              координатах.
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer класс IDs. Дополнено нулями.
        deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
        masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Маски обрезаны до границ
               bbox и изменены до размера выходного сигнала нейронной сети.
    """
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Удалить нулевые дополнения
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Вычислить матрицу перекрытий [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Определить положительный и отрицательный ROI
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Положительные  ROIs это те, у которых >= 0.5 IoU с контейнером GT
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Отрицательные ROIs это те, у которых < 0.5 с каждым контейнером GT.
    # Пропустить множества.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # двыборка ROIs. Надо стремиться на 33% положительных
    # Положительные ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Отрицательные ROIs. Добавить достаточно, чтобы сохранить
    # положительное:отрицательное соотношение.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Собрать выбранные ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Назначьте положительные ROI для GT-контейнеров.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Вычислить уточнение bbox для положительных ROI
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Назначить положительные ROI для масок GT
    # Пермутировать маски в [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Выберать правильную маску для каждого ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Вычислить цели маски
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Преобразование координат ROI из нормализованного пространства
        # изображения в нормализованное пространство мини-маски.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Удалите лишнее измерение из масок.
    masks = tf.squeeze(masks, axis=3)

    # Округлить маска пикселей на 0,5, чтобы маски GT были 0 или 1 для
    # использования с двоичной кросс-энтропийной потерей.
    masks = tf.round(masks)

    # Добавьте отрицательные ROI, а также дельты и маски bbox, которые не
    # используются для отрицательных ROI с нулями..
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(k_engine.Layer):
    """
    Подвыбирает предложения и генерирует уточнение целевого контейнера,
    class_ids и маски для каждого. Возвращаемые массивы могут быть дополнены
    нулями, если не хватает целевых ROI.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        """
        :param inputs:
            proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] в нормированных
            координатах. Может быть дополнен нулями, если предложений
            недостаточно.
            gt_class_ids: [MAX_GT_INSTANCES] int класс IDs
            gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] в нормированных
            координатах.
            gt_masks: [height, width, MAX_GT_INSTANCES] типа boolean.
        :return: Целевые ROI и соответствующие идентификаторы классов, сдвиги
        ограничивающих рамок и маски. Возвращаемые массивы могут быть дополнены
        нулями, если не хватает целевых ROI.
            rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] в нормированных
                  координатах.
            target_class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer класс IDs.
            Дополнено нулями.
            target_deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
            target_masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Маски обрезаны
            до границ bbox и изменены до размера выходного сигнала нейронной сети.
        """
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Нарезать пакет и запустить график для каждого среза
        names = ["rois", "target_class_ids", "target_deltas", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


################################################################################
#  Слой распознавания
################################################################################

def refine_detections_graph(rois, probs, deltas, window, config):
    """
    Уточнить классифицированные предложения и отфильтровать перекрытия и вернуть
    окончательные результаты.
    :param rois: [N, (y1, x1, y2, x2)] в нормированных координатах
    :param probs: [N, num_classes]. Класс вероятностей.
    :param deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Специфичные для
                   класса ограничивающие рамки.
    :param window: (y1, x1, y2, x2) в нормированных координатах. Часть
                   изображения, которая содержит изображение, исключая отступы.
    :return обнаружения в форме: [num_detections, (y1, x1, y2, x2, class_id,
            score)], где координаты нормированы.
    """
    # ID классов на ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Класс вероятности высшего класса каждого ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Специфичные для класса ограничивающие рамки
    deltas_specific = tf.gather_nd(deltas, indices)
    # Применить ограничивающие рамки дельты
    # Формат: [boxes, (y1, x1, y2, x2)] в нормированных координатах
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Прикрепить контейнеры в окно изображения
    refined_rois = clip_boxes_graph(refined_rois, window)

    # Отфильтровать фоновые контейнеры
    keep = tf.where(class_ids > 0)[:, 0]
    # Отфильтровать контейнеры с низким уровнем доверия
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Применить NMS для каждого класса
    # 1. Подготоваить переменные
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Применить подавление немаксимумов на ROI данного класса."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Применить NMS
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Индексы карты
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Отступы с -1, поэтому возвращенные тензоры имеют одинаковую форму
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
        # Установить форму, чтобы map_fn () могла вывести форму результата
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Маппинг по идентификаторам классов
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Объединить результаты в один список и удалить -1 заполнение
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Вычислить пересечение между keep и nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Зафиксировать лучшие обнаружения
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Организовать вывод как [N, (y1, x1, y2, x2, class_id, score)]
    # Координаты нормализованы.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # Дополнение нулями, если обнаружение <DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(k_engine.Layer):
    """
    Принимает классифицированные предложения и их ограничивающие контейнеры и
    возвращает последние контейнеры обнаружения.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        """
        :return: [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
        координаты нормированы.
        """
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Получить окна изображений в нормализованных координатах. Окна - это
        # область изображения, исключающая отступы. Использовать форму первого
        # изображения в пакете, чтобы нормализовать окно, потому что мы знаем,
        # что все изображения изменяются до одинакового размера.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Запустить график уточнения обнаружения для каждого элемента в пакете
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Изменить форму вывода
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] в
        # нормированных координатах
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


################################################################################
#  Сеть для предложения регионов Region Proposal Network (RPN)
################################################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """
    Создает граф вычислений сети для предложения регионов.
    :param feature_map: основные функции [batch, height, width, depth]
    :param anchors_per_location: количество якорей на пиксель в карте признаков
    :param anchor_stride: Контролирует плотность якорей. Обычно 1 (привязки для
                          каждого пикселя на карте признаков) или 2 (для каждого
                          второго пикселя).
    :return
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Логиты
                          классификатора якоря (до softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Якорный
                   классификатор вероятностей.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh),
                  log(dw))] Дельты для применения к якорям.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map is not even.
    # Общая сверточная база RPN
    shared = k_layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                             strides=anchor_stride, name='rpn_conv_shared')(feature_map)

    # Показатели якорей. [batch, height, width, anchors per location * 2].
    x = k_layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                        activation='linear', name='rpn_class_raw')(shared)

    # Изменить форму [batch, anchors, 2]
    rpn_class_logits = k_layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax по последнему измерению BG / FG.
    rpn_probs = k_layers.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Уточнение ограничевающего контейнера.
    # [batch, H, W, anchors per location * depth] где глубина
    # [x, y, log(w), log(h)]
    x = k_layers.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                        activation='linear', name='rpn_bbox_pred')(shared)

    # Изменить форму  [batch, anchors, 4]
    rpn_bbox = k_layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """
    Создает модель Keras региональной сети предложений.
    Он оборачивает график RPN, поэтому ее можно использовать несколько раз с
    общими весами.
    :param anchors_per_location: количество якорей на пиксель в карте признаков
    :param anchor_stride: Контролирует плотность якорей. Обычно 1 (привязки для
                          каждого пикселя на карте признаков) или 2 (для каждого
                          второго пикселя).
    :param depth: Глубина магистрали карты признаков.
    :return Объект модели Keras. Выходные данные модели при вызове:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Логиты
                          классификатора якоря (до softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Якорный
                   классификатор вероятностей.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh),
                  log(dw))] Дельты для применения к якорям.
    """
    input_feature_map = k_layers.Input(shape=[None, None, depth], name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return k_models.Model([input_feature_map], outputs, name="rpn_model")


################################################################################
#  Сеть пирамиды признаков Heads
################################################################################

def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """
    Строит график вычислений пирамидального классификатора сети и головок
    регрессоров.
    :param rois: [batch, num_rois, (y1, x1, y2, x2)] Контейнеры для предложений
                 в нормализованных координатах.
    :param feature_maps: Список карт признаков из разных слоев пирамиды,
                         [P2, P3, P4, P5]. У каждого свое разрешение.
    :param image_meta: [batch, (meta data)] Детали изображения. Смотрите
                       compose_image_meta()
    :param pool_size: Ширина квадратной карты признаков, созданной из пула ROI.
    :param num_classes: количество классов, определяющее глубину результатов
    :param train_bn: Boolean. Обучить или заморозить слои Batch Normalization
    :param fc_layers_size: Размер 2 слоев FC
    :return:
        logits: [batch, num_rois, NUM_CLASSES] классификатор логи (перед softmax)
        probs: [batch, num_rois, NUM_CLASSES] классификатор вероятностей
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                     Дельты, чтобы применить к предлогаемым контейнерам
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = k_layers.TimeDistributed(k_layers.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                                 name="mrcnn_class_conv1")(x)
    x = k_layers.TimeDistributed(BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = k_layers.Activation('relu')(x)
    x = k_layers.TimeDistributed(k_layers.Conv2D(fc_layers_size, (1, 1)),
                                 name="mrcnn_class_conv2")(x)
    x = k_layers.TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = k_layers.Activation('relu')(x)

    shared = k_layers.Lambda(lambda x: k_backend.squeeze(k_backend.squeeze(x, 3), 2),
                             name="pool_squeeze")(x)

    # Головка классификатора
    mrcnn_class_logits = k_layers.TimeDistributed(k_layers.Dense(num_classes),
                                                  name='mrcnn_class_logits')(shared)
    mrcnn_probs = k_layers.TimeDistributed(k_layers.Activation("softmax"),
                                           name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = k_layers.TimeDistributed(k_layers.Dense(num_classes * 4, activation='linear'),
                                 name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = k_backend.int_shape(x)
    mrcnn_bbox = k_layers.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True):
    """
    Создает график вычислений головы маски в сети Feature Pyramid.
    :param rois: [batch, num_rois, (y1, x1, y2, x2)] Контейнеры для предложений
                 в нормализованных координатах.
    :param feature_maps: Список карт признаков из разных слоев пирамиды,
                         [P2, P3, P4, P5]. У каждого свое разрешение.
    :param image_meta: [batch, (meta data)] Детали изображения. Смотрите
                       compose_image_meta()
    :param pool_size: Ширина квадратной карты признаков, созданной из пула ROI.
    :param num_classes: количество классов, определяющее глубину результатов
    :param train_bn: Boolean. Обучить или заморозить слои Batch Normalization
    :param fc_layers_size: Размер 2 слоев FC
    :return: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = k_layers.TimeDistributed(k_layers.Conv2D(256, (3, 3), padding="same"),
                                 name="mrcnn_mask_conv1")(x)
    x = k_layers.TimeDistributed(BatchNormalization(),
                                 name='mrcnn_mask_bn1')(x, training=train_bn)
    x = k_layers.Activation('relu')(x)

    x = k_layers.TimeDistributed(k_layers.Conv2D(256, (3, 3), padding="same"),
                                 name="mrcnn_mask_conv2")(x)
    x = k_layers.TimeDistributed(BatchNormalization(),
                                 name='mrcnn_mask_bn2')(x, training=train_bn)
    x = k_layers.Activation('relu')(x)

    x = k_layers.TimeDistributed(k_layers.Conv2D(256, (3, 3), padding="same"),
                                 name="mrcnn_mask_conv3")(x)
    x = k_layers.TimeDistributed(BatchNormalization(),
                                 name='mrcnn_mask_bn3')(x, training=train_bn)
    x = k_layers.Activation('relu')(x)

    x = k_layers.TimeDistributed(k_layers.Conv2D(256, (3, 3), padding="same"),
                                 name="mrcnn_mask_conv4")(x)
    x = k_layers.TimeDistributed(BatchNormalization(),
                                 name='mrcnn_mask_bn4')(x, training=train_bn)
    x = k_layers.Activation('relu')(x)

    x = k_layers.TimeDistributed(k_layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                                 name="mrcnn_mask_deconv")(x)
    x = k_layers.TimeDistributed(k_layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                                 name="mrcnn_mask")(x)
    return x


################################################################################
# Функции Loss
################################################################################

def smooth_l1_loss(y_true, y_pred):
    """
    Реализует Smooth-L1 loss.
    y_true и y_pred обычно: [N, 4], но могут иметь любую форму.
    """
    diff = k_backend.abs(y_true - y_pred)
    less_than_one = k_backend.cast(k_backend.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """
    RPN anchor classifier loss.
    :param rpn_match: [batch, anchors, 1]. Тип соответствия якоря.
                      1=положительный, -1=отрицательный, 0= ейтральный якорь.
    :param rpn_class_logits: [batch, anchors, 2]. Логиты классификатора RPN
                             для BG / FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Получить якорные классы. Преобразовать соответствие -1/+1 в значения 0/1.
    anchor_class = k_backend.cast(k_backend.equal(rpn_match, 1), tf.int32)
    # Положительные и отрицательные якоря способствуют потере, но нейтральные
    # якоря (значение соответствуют = 0) нет.
    indices = tf.where(k_backend.not_equal(rpn_match, 0))
    # Выбрать строки, которые способствуют потере, и отфильтровать остальные.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Перекрестная потеря энтропии
    loss = k_backend.sparse_categorical_crossentropy(target=anchor_class,
                                                     output=rpn_class_logits,
                                                     from_logits=True)
    loss = k_backend.switch(tf.size(loss) > 0, k_backend.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """
    Вернуть граф loss ограничивающей рамки RPN.
    :param config: объект конфигурации модели.
    :param target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
                        Использует 0 отступов для заполнения unsed bbox deltas.
    :param rpn_match: [batch, anchors, 1]. Тип соответствия якоря.
                      1=положительный, -1=отрицательный, 0=нейтральный якорь.
    :param rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Положительные и отрицательные якоря способствуют потере, но нейтральные
    # якоря (значение соответствуют = 0) нет.
    rpn_match = k_backend.squeeze(rpn_match, -1)
    indices = tf.where(k_backend.equal(rpn_match, 1))

    # Выберите bbox дельты, которые способствуют потере
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Обрезать ограничивающие поля целевого поля до той же длины, что и rpn_bbox.
    batch_counts = k_backend.sum(k_backend.cast(k_backend.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = k_backend.switch(tf.size(loss) > 0, k_backend.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    """
    Loss для классификатора  head of Mask RCNN.
    :param target_class_ids: [batch, num_rois]. Integer класс IDs. Использует
           нулевые дополнения, чтобы заполнить массив.
    :param pred_class_logits: [batch, num_rois, num_classes]
    :param active_class_ids: [batch, num_classes]. Имеет значение 1 для классов,
           которые находятся в наборе данных изображения, и 0 для классов,
           которых нет в наборе данных.
    """
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Найти прогнозы классов, которых нет в наборе данных.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Стереть потери прогнозов классов, которые не входят в активные классы
    # изображения.
    loss = loss * pred_active

    # Вычислить МО loss. Использовать только прогнозы, которые способствуют
    # loss, чтобы получить правильное МО.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """
    Loss for Mask R-CNN bounding box refinement.
    :param target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    :param target_class_ids: [batch, num_rois]. Integer class IDs.
    :param pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = k_backend.reshape(target_class_ids, (-1,))
    target_bbox = k_backend.reshape(target_bbox, (-1, 4))
    pred_bbox = k_backend.reshape(pred_bbox, (-1, k_backend.int_shape(pred_bbox)[2], 4))

    # Только положительные ROI способствуют loss. И только правильный class_id
    # каждого ROI. Получите их индексы.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Соберать дельты (предсказанные и истинные), которые способствуют loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = k_backend.switch(tf.size(target_bbox) > 0,
                            smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                            tf.constant(0.0))
    loss = k_backend.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """
    Mask binary cross-entropy loss for the masks head.
    :param target_masks: [batch, num_rois, height, width]. Тензор типа float32
           со значениями 0 или 1. Использует заполнение нулями для заполнения
           массива.
    :param target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    :param pred_masks: [batch, proposals, height, width, num_classes] тензор
           float32 со значениями от 0 до 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = k_backend.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = k_backend.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = k_backend.reshape(pred_masks,
                                   (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Перестановка предсказанных масок [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Только положительные ROI способствуют loss. И только маска класса для
    # каждого ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Соберите маски (предсказанные и истинные), которые способствуют loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Вычислить двоичную перекрестную энтропию. Если нет положительных ROI,
    # вернуть 0.  shape: [batch, roi, num_classes]
    loss = k_backend.switch(tf.size(y_true) > 0,
                            k_backend.binary_crossentropy(target=y_true, output=y_pred),
                            tf.constant(0.0))
    loss = k_backend.mean(loss)
    return loss


################################################################################
# Генератор данных
################################################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None, use_mini_mask=False):
    """
    Загрузить и вернуть эталонные данные для изображения (изображение, маска,
    ограничивающие рамки).
    :param augment: (устарело. Вместо этого используйте augmentation). Если
           True, примените случайное увеличение изображения. В настоящее время
           предлагается только горизонтальное переключение.
    :param augmentation: Опционально. An imgaug (https://github.com/aleju/imgaug)
           augmentation. Например, передача imgaug.augmenters.Fliplr (0.5)
           переворачивает изображения вправо / влево в 50% случаев.
    :param use_mini_mask: Если False, возвращает полноразмерные маски, которые
           имеют ту же высоту и ширину, что и исходное изображение. Они могут быть
           большими, например 1024x1024x100 (для 100 экземпляров). Мини-маски
           меньше, обычно 224x224, и генерируются путем извлечения ограничивающего
           прямоугольника объекта и изменения его размера в MINI_MASK_SHAPE.
    :returns
        image: [height, width, 3]
        shape: исходная форма изображения до изменения размера и обрезки.
        class_ids: [instance_count] Integer класс IDs
        bbox: [instance_count, (y1, x1, y2, x2)]
        mask: [height, width, instance_count]. Высота и ширина соответствуют размеру
            изображения, если только для use_mini_mask не установлено значение True, в
            этом случае они определены в MINI_MASK_SHAPE.
    """
    # Загрузить изображение и маску
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Случайные горизонтальные сальто.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # Требуется библиотека imgaug (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Аргментеры, которые безопасно применять для масок. Некоторые, такие как
        # Affine, имеют настройки, которые делают их небезопасными, поэтому всегда
        # проверяйте свое увеличение на масках
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Определяет, какие augmenters применять к маскам."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Сохранить фигуры перед увеличением для сравнения
        image_shape = image.shape
        mask_shape = mask.shape
        # Сделать аугментаторы детерминированными, чтобы применять их к изображениям
        # и маскам
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Изменить маску на np.uint8, потому что imgaug не поддерживает np.bool
        mask = det.augment_image(mask.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))
        # Убедиться, что формы не изменились
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Сменить маску обратно на bool
        mask = mask.astype(np.bool)

    # Обратите внимание, что некоторые контейнеры могут иметь все нули, если
    # соответствующая маска была обрезана. И здесь необходимо отфильтровать их
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Ограничивающие контейнеры. Обратите внимание, что некоторые контейнеры
    # могут иметь все нули, если соответствующая маска была обрезана.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bounding_boxes(mask)

    # Активные классы
    # Разные наборы данных имеют разные классы, поэтому отслеживайте классы,
    # поддерживаемые в наборе данных этого изображения.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Изменение размера маски до меньшего размера, чтобы уменьшить использование
    # памяти
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Метаданные изображения
    image_meta = compose_image_meta(image_id, original_shape, image.shape, window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """
    Создайте цели для тренировки 2-го этапа и маскирующих голов. Это не
    используется в обычных тренировках. Это полезно для отладки или тренировки
    головок маски RCNN без использования головки RPN
    :param rpn_rois: [N, (y1, x1, y2, x2)] предлагаемые контейнеры.
    :param gt_class_ids: [instance count] Integer классы IDs
    :param gt_boxes: [instance count, (y1, x1, y2, x2)]
    :param gt_masks: [height, width, instance count] Маски эталонов. Могут быть
           в натуральную величину или мини-маски.
    :returns:
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer класс IDs.
        bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))].
            Специфичные для класса уточнения bbox.
        masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Маски,
            специфичные для класса, обрезаны до границ bbox и изменены до
            выходного размера нейронной сети.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(gt_masks.dtype)

    # Обычно добавляются GT Box в ROI, но мы здесь этого не делаем, потому что,
    # согласно статье XinLei Chen, это не помогает.

    # Обрезать пустые отступы в частях gt_boxes и gt_masks
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Вычислить области ROI и эталонные контейнеры.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Вычислить перекрытия [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Назначить ROI для GT-контейнеров
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT-контейнер назначен каждому ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Положительные ROI - это те, у которых> 0,5 IOU с GT-контейнером.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Отрицательные ROIs это те, которые имеют максимальный коэффициент IoU
    # 0,1-0,5
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Подвыборочные ROIs. Цель 33% переднего плана.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Объединить индексы ROI, чтобы сохранить
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Нужно больше?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Похоже, у нас недостаточно образцов для поддержания желаемого баланса.
        # Уменьшите требования и заполните остальные. Это, вероятно, отличается
        # от Маска RCNN документации.

        # Есть небольшой шанс, что у нас нет ни образцов fg, ни bg.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Заполнить остальное с повторным bg rois.
            keep_extra_ids = np.random.choice(keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Сбросить контейнеры gt, назначенные для BG ROI.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # Для каждого сохраняемого ROI присвойте class_id, а для FG ROI также
    # добавьте уточнение bbox.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Признаки класса bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Нормализовать уточнения bbox
    bboxes /= config.BBOX_STD_DEV

    # Генерация целевых масок класса
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # Создать маску-заполнитель, размер изображения
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Изменить размер мини-маски по размеру GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            # Поместите мини-пакет в заполнитель
            class_mask = placeholder

        # Выбрать часть маски и изменить ее размер
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(anchors, gt_class_ids, gt_boxes, config):
    """
    С учетом якорей и контейнеров GT вычислить перекрытия и определите
    положительные якоря и дельты, чтобы уточнить их, чтобы они соответствовали
    соответствующим контейнерам GT.
    :param anchors: [num_anchors, (y1, x1, y2, x2)]
    :param gt_class_ids: [num_gt_boxes] Integer класс IDs.
    :param gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
    :returns:
        rpn_match: [N] (int32) совпадения между якорями и контейнерами GT.
                   1 = положительная привязка, -1 = отрицательная привязка,
                   0 = нейтральная
        rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN соответствие: 1 = положительная привязка, -1 = отрицательная привязка,
    # 0 = нейтральная
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN ограничивающие контейнеры: [max anchors per image, (dy, dx, log(dh),
    # log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Обработать COCO множества
    # Множественные контейнеры в COCO - ограничивающая рамка вокруг нескольких
    # экземпляров. Исключите их из обучения. Множественным контейнерам
    # присваивается отрицательный идентификатор класса.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Отфильтровывать множества от эталонных идентификаторов класса и
        # контейнеров
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Вычислить перекрытия с множественными контейнерами [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # Все якоря не пересекаются с множественными контейнерами
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Вычислить перекрытия [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Сопоставляем якоря к GT Boxes
    # Если якорь перекрывает окно GT с IoU >= 0,7, то это положительно.
    # Если якорь перекрывает окно GT с IoU < 0.3, то это отрицательно.
    # Нейтральные якоря - это те, которые не соответствуют вышеуказанным
    # условиям, и они не влияют на функцию потерь. Тем не менее, не оставляйте
    # GT-бокс без матчинга (редко, но бывает). Вместо этого сопоставьте его с
    # ближайшим якорем (даже если его максимальная IoU <0,3).
    #
    # 1. Сначала установить отрицательные якоря. Они перезаписываются ниже, если
    # им соответствует GT-контейнеры. Опустить контейнеры в местах множеств.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Установите привязку для каждого контейнера GT (независимо от значения
    # IoU). Если несколько якорей имеют одинаковую IoU, все они совпадают
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Установить якоря с высоким перекрытием как положительные.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Подвыборка для баланса положительных и отрицательных якорей. Не позволить
    # позитивам быть больше половины якорей
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Сброс дополнительных на нейтральный
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # То же самое для негативных предложений
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        # Оставить лишние на нейтральной
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # Для положительных привязок вычислите смещение и масштаб, необходимые для
    # преобразования их в соответствующие контейнеры GT.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # индексировать в rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Ближайший контейнер GT (может быть IoU <0,7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Конвертировать координаты в центр плюс ширина / высота.
        # GT контейнер
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Якорь
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Вычислить уточнение bbox, которое должен предсказать RPN.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Нормализовать
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_boxes):
    """
    Генерирует предложения ROI, аналогичные тому, что генерирует сеть
    предложений региона.
    :param image_shape: [Height, Width, Depth]
    :param count: Количество ROI для генерации
    :param gt_boxes: [N, (y1, x1, y2, x2)] Эталонные контейнеры в пикселях
    :return [count, (y1, x1, y2, x2)] ROI контейнеры в пикселях.
    """
    # заполнитель
    rois = np.zeros((count, 4), dtype=np.int32)

    # Генерируются случайные ROI вокруг GT-контейнеров (90% от количества)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # случайные границы
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # Чтобы не создавать контейнеры с нулевой площадью, мы генерируем вдвое
        # больше, чем нужно, и отфильтровываем лишнее. Если мы получим меньше
        # допустимых контейнеров, чем нам нужно, мы зациклимся и попробуем снова.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Отфильтровать поля нулевой области
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Сортировать по оси 1, чтобы убедиться, что x1 <= x2 и y1 <= y2, а затем
        # изменить их в порядок x1, y1, x2, y2
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Генерация случайных областей интереса в любом месте изображения (10% от
    # количества)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # Чтобы не создавать контейнеры с нулевой площадью, мы генерируем вдвое
    # больше, чем нужно, и отфильтровываем лишнее. Если мы получим меньше
    # допустимых контейнеров, чем нам нужно, мы зациклимся и попробуем снова.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Отфильтровать контейнеры нулевой области
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Сортировать по оси 1, чтобы убедиться, что x1 <= x2 и y1 <= y2, а затем
    # изменить их в порядок x1, y1, x2, y2
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None, random_rois=0, batch_size=1,
                   detection_targets=False, no_augmentation_sources=None):
    """
    Генератор, который возвращает изображения и соответствующие идентификаторы
    целевого класса, ограничивающие рамки и маски.
    :param dataset: Объект набора данных для выбора данных
    :param config: Объект конфигурации модели
    :param shuffle: Если True, перетасовывает образцы перед каждой эпохой
    :param augment: (устарело. Вместо этого используйте augmentation). Если
           True, примените случайное увеличение изображения. В настоящее время
           предлагается только горизонтальное переключение.
    :param augmentation: Опционально. An imgaug (https://github.com/aleju/imgaug)
           augmentation. Например, передача imgaug.augmenters.Fliplr (0.5)
           переворачивает изображения вправо / влево в 50% случаев.
    :param random_rois: Если> 0, тогда сгенерируйте предложения, которые будут
           использоваться для обучения сетевого классификатора и заголовков
           маски. Полезно, если тренируется часть маски RCNN без RPN.
    :param batch_size: Сколько изображений нужно возвращать при каждом вызове
    :param detection_targets: Если True, генерировать цели обнаружения
           (идентификаторы классов, дельты bbox и маски). Обычно для отладки или
           визуализации, потому что в обучении цели обнаружения генерируются
           DetectionTargetLayer.
    :param no_augmentation_sources: Опционально. Список источников для исключения
           для дополнения. Источником является строка, которая идентифицирует
           набор данных и определена в классе набора данных.
    :return генератор Python. После вызова next() для него генератор возвращает
            два списка, входы и выходы. Содержимое списков различается в
            зависимости от полученных аргументов:
        inputs list:
        - images: [batch, H, W, C]
        - image_meta: [batch, (meta data)] Детали изображения. Смотрите
            compose_image_meta()
        - rpn_match: [batch, N] Integer (1 = положительный якорь,
            -1 = отрицательный, 0 = нейтральный)
        - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Якорь bbox deltas.
        - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer класс IDs
        - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. Высота и ширина
            соответствуют размеру изображения, если только для use_mini_mask не
            установлено значение True, в этом случае они определены в
            MINI_MASK_SHAPE.
        outputs list: Обычно пустые при регулярных тренировках. Но если для
            application_targets задано значение True, список выходных данных
            содержит целевые значения class_ids, bbox deltas и маски.
    """
    b = 0  # индекс пакета объекта
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # Якоря
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras требует, чтобы генератор работал бесконечно.
    while True:
        try:
            # Увеличение индекса для выбора следующего изображения. Перемешать,
            # если в начале эпохи.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Получить GT ограничивающие рамки и маски для изображения.
            image_id = image_ids[image_index]

            # Если источник изображения не должен быть расширен, передайте None
            # как увеличение
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(dataset, config, image_id, augment=augment, augmentation=None,
                                  use_mini_mask=config.USE_MINI_MASK)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(dataset, config, image_id, augment=augment, augmentation=augmentation,
                                  use_mini_mask=config.USE_MINI_MASK)

            # Пропустить изображения, которые не имеют экземпляров. Это может
            # случиться в тех случаях, когда мы тренируемся на подмножестве
            # классов, а на изображении нет классов, которые нам нужны.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN цели
            rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_class_ids, gt_boxes, config)

            # Mask R-CNN цели
            if random_rois:
                rpn_rois = generate_random_rois(image.shape, random_rois, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = \
                        build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

            # Инициализировать пакетные массивы
            if b == 0:
                batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros((batch_size, gt_masks.shape[0], gt_masks.shape[1],
                                           config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                if random_rois:
                    batch_rpn_rois = np.zeros((batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros((batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros((batch_size,) + mrcnn_class_ids.shape,
                                                         dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros((batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros( (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # Если экземпляров больше, чем умещается в массиве, выполнить выборку из них.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Добавить к пакету
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            # Пакет полон?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras требует, чтобы выходные данные и цели имели одинаковое количество измерений
                        batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)
                        outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # начать новый пакет
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Залогировать это и отбросить изображение
            logging.exception("Error processing image {}".format(dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


################################################################################
#  Класс MaskRCNN
################################################################################

class MaskRCNN:
    """
    Инкапсулирует функциональность модели Mask RCNN.
    Фактическая модель Keras находится в свойстве keras_model.
    """

    def __init__(self, mode, config, model_dir):
        """
        :param mode: "обучение" или "вывод"
        :param config: Подкласс класса Config
        :param model_dir: Каталог для сохранения тренировочных логов и
               тренировочных весов
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """
        Построить архитектуру Mask R-CNN архитектуры.
        :param mode: Либо "обучение", либо "вывод". Входы и выходы модели
                     отличаются соответственно.
        """
        assert mode in ['training', 'inference']

        # Размер изображения должен делиться на 2 6 раз
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Входные данные
        input_image = k_layers.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = k_layers.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = k_layers.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = k_layers.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Обнаружение GT (идентификаторы классов, ограничивающие рамки и маски)
            # 1. GT класс IDs (дополненный нулями)
            input_gt_class_ids = k_layers.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT контейнеры в пикселях (дополненные нулями)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] в координатах изображения
            input_gt_boxes = k_layers.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Нормализовать координаты
            gt_boxes = k_layers.Lambda(lambda x: norm_boxes_graph(x, k_backend.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT маски (дополненные нулями)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = k_layers.Input(shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                                                name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = k_layers.Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                                                name="input_gt_masks", dtype=bool)
        # elif mode == "inference"
        else:
            # Якоря в нормированных координатах
            input_anchors = k_layers.Input(shape=[None, 4], name="input_anchors")

        # Построить общие сверточные слои. Слои снизу вверх. Возвращает список
        # последних слоев каждого этапа, всего 5.
        _, C2, C3, C4, C5 = resnet101_graph(input_image, train_bn=config.TRAIN_BN)
        # Слои сверху вниз
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = k_layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = k_layers.Add(name="fpn_p4add")([
            k_layers.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            k_layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = k_layers.Add(name="fpn_p3add")([
            k_layers.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            k_layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = k_layers.Add(name="fpn_p2add")([
            k_layers.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            k_layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Прикрепите свертку 3х3 ко всем слоям P, чтобы получить окончательные
        # карты признаков.
        P2 = k_layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = k_layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = k_layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = k_layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 используется для 5-й якорной шкалы в RPN. Генерируется подвыборкой
        # из P5 с шагом 2.
        P6 = k_layers.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Обратите внимание, что P6 используется в RPN, но не в головках
        # классификатора.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Якоря
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Дублируйте размер по всему пакету, потому что это требуется для
            # Keras
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # Взлом, чтобы обойти плохую поддержку Keras для констант
            anchors = k_layers.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        # Цикл по слоям пирамиды
        layer_outputs = []  # список списков
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Конкатенация выходов слоя
        # Преобразование из списка списков выходов уровня в список списков
        # выходов по уровням.
        # [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [k_layers.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Генерация предложений
        # Предложения [batch, N, (y1, x1, y2, x2)] в нормализованных координатах
        # и дополнены нулями.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count, nms_threshold=config.RPN_NMS_THRESHOLD,
                                 name="ROI", config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Маска идентификатора класса для обозначения идентификаторов
            # классов, поддерживаемых набором данных, из которого получено
            # изображение.
            active_class_ids = k_layers.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])\
                (input_image_meta)

            if not config.USE_RPN_ROIS:
                # Игнорируйте прогнозируемые ROI и используйте ROI,
                # предоставленные в качестве входных данных.
                input_rois = k_layers.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=np.int32)
                # Нормализовать координаты
                target_rois = k_layers.Lambda(lambda x: norm_boxes_graph(x, k_backend.shape(input_image)[1:3]))\
                    (input_rois)
            else:
                target_rois = rpn_rois

            # Генерация целей обнаружения
            # Подвыборки предложений и создание целевых результатов для обучения
            # Обратите внимание, что идентификаторы классов предложений,
            # gt_boxes и gt_masks дополняются нулями. Точно так же, возвращенные
            # rois и цели дополнены нулями.
            rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(config, name="proposal_targets")([
                target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = k_layers.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = k_layers.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = k_layers.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = k_layers.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = k_layers.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = k_layers.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox, input_gt_class_ids,
                      input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = k_models.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Классификатор предложений и регуляторные головки BBox
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Распознавание
            # вывод [batch, num_detections, (y1, x1, y2, x2, class_id, score)] в
            # нормализированных координатах
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Создать маски для обнаружения
            detection_boxes = k_layers.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            model = k_models.Model([input_image, input_image_meta, input_anchors],
                                   [detections, mrcnn_class, mrcnn_bbox,
                              mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                                   name='mask_rcnn')

        return model

    def find_last(self):
        """
        Находит файл последней контрольной точки последней обученной модели в
        каталоге моделей.
        :return Путь к последнему файлу контрольных точек
        """
        # Получить имена каталогов. Каждый каталог соответствует модели
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(errno.ENOENT,
                                    "Could not find model directory under {}".format(self.model_dir))
        # Выбрать последний каталог
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Найти последнюю контрольную точку
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """
        Модифицированная версия соответствующей функции Keras с добавлением
        поддержки нескольких графических процессоров и возможностью исключения
        некоторых слоев из загрузки.
        :param exclude: список имен слоев для исключения
        """
        import h5py
        # Условный импорт для поддержки версий Keras до 2.2
        try:
            from keras.engine import saving
        except ImportError:
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        # Исключить несколько слоев
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Обновить каталог журнала
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """
        Загружает ImageNet обученные веса от Keras. Возвращает путь к файлу весов.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                                 'releases/download/v0.2/' \
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """
        Получить готовую модель для обучения. Добавить loss, регуляризацию и
        метрики. Затем вызывает функцию компиляции () Keras.
        """
        # Оптимизатор объекта
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Добавить Losses
        # Во-первых, очистить ранее установленные Losses, чтобы избежать дублирования
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss", "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Добавить регуляризацию L2
        # Пропустить гамма- и бета-весы слоев нормализации пакета.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Скомпилировать
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Добавить метрики к losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """
        Устанавливает обучаемые слои модели, если их имена соответствуют
        заданному регулярному выражению.
        """
        # Распечатать сообщение при первом вызове (но не при рекурсивных вызовах)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers

        for layer in layers:
            # Является ли слой моделью?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Это обучаемо?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Обновить слой. Если слой является контейнером, обновить внутренний слой.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Напечатать обучаемые названия слоев
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """
        Устанавливает директорию логов модели и счетчик эпох.
        :param model_path: Если None или формат отличается от того, который
               используется в этом коде, тогда установить новую директорию логов
               и запустить эпохи с 0. В противном случае извлечь директорию
               логов и счетчик эпох из имени файла.
        """
        # Установить дату и счетчик эпох, как будто модель -- новая
        self.epoch = 0
        now = datetime.datetime.now()

        # Если у нас есть модельный путь с датой и эпохами, использовать их
        if model_path:
            # Продолжить от того места, где оставились. Получить эпоху и дату из
            # имени файла.
            # Пример пути к модели может выглядеть следующим образом:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Номер эпохи в файле основан на 1, а в коде Keras - на 0. Итак,
                # отрегулируйте это, затем увеличьте на единицу, чтобы начать
                # со следующей эпохи.
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Директория для логов обучения
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))

        # Путь к сохранению после каждой эпохи. Включить заполнители, которые
        # заполняются Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """
        Тренировать модель.
        :param train_dataset: тренировочные объекты набора данных
        :param val_dataset: проверочные объекты набора данных
        :param learning_rate: Скорость обучения
        :param epochs: Количество тренировочных эпох. Обратите внимание, что
               предыдущие периоды обучения считаются выполненными уже, так что
               это фактически определяет эпохи для обучения в целом, а не в этом
               конкретном вызове.
        :param layers: Позволяет выбирать, какие слои тренировать. Может быть:
            - регулярное выражение для сопоставления имен слоев для обучения
            - Одно из этих предопределенных значений:
              головы: РПН, классификатор и маска головы сети
              все: все слои
              3+: Resnet этап 3 и выше
              4+: Resnet этап 4 и выше
              5+: Resnet этап 5 и выше
        :param augmentation: Опционально.
        :param custom_callbacks: Опционально. Добавить пользовательские обратные
               вызовы для вызова с помощью метода keras fit_generator. Должен
               быть список типа keras.callbacks.
        :param no_augmentation_sources: Опционально. Список источников для
               исключения для дополнения. Источником является строка, которая
               идентифицирует набор данных и определена в классе набора данных.
        """
        assert self.mode == "training", "Create model in training mode."

        # Предопределенные регулярные выражения слоя
        layer_regex = {
            # все слои, кроме основной
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # От определенного этапа Resnet и выше
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # Все слои
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Генераторы данных
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Создать log_dir, если он не существует
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Добавить пользовательские обратные вызовы в список
        if custom_callbacks:
            callbacks += custom_callbacks

        # Обучение
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """
        Принимает список изображений и изменяет их до ожидаемого формата в
        качестве входных данных для нейронной сети.
        :param images: Список матриц изображения  [height,width,depth].
               Изображения могут иметь разные размеры.
        :return 3 Матрицы Numpy:
            molded_images: [N, h, w, 3]. Изображения изменены и нормализованы.
            image_metas: [N, length of meta data]. Подробности о каждом
                изображении.
            windows: [N, (y1, x1, y2, x2)]. Часть изображения с исходным
                изображением (за исключением заполнения).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Изменить размер изображения
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Построить image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Добавить
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Упаковать в массив
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape, image_shape, window):
        """
        Переформатирует обнаружение одного изображения из формата вывода
        нейронной сети в формат, подходящий для использования в остальной части
        приложения.
        :param detections: [N, (y1, x1, y2, x2, class_id, score)] в
               нормированных координатах
        :param mrcnn_mask: [N, height, width, num_classes]
        :param original_image_shape: [H, W, C] Исходная форма изображения перед
               изменением размера
        :param image_shape: [H, W, C] Форма изображения после изменения размера
               и заполнения
        :param window: [y1, x1, y2, x2] Пиксельные координаты прямоугольника на
               изображении, где реальное изображение исключает отступы.
        :return:
            boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
            class_ids: [N] Integer class IDs for each bounding box
            scores: [N] Float probability scores of the class_id
            masks: [height, width, num_instances] Instance masks
        """
        # Сколько у нас обнаружений?
        # Массив обнаружений дополнен нулями. Найти первый class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Извлечь контейнеры, class_ids, показатели и специфичные для класса
        # маски
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Преобразуйте нормализованные координаты в измененном изображении в
        # пиксельные координаты в исходном изображении перед изменением размера
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # высота окна
        ww = wx2 - wx1  # ширина окна
        scale = np.array([wh, ww, wh, ww])
        # Конвертировать контейнеры в нормализованные координаты в окне
        boxes = np.divide(boxes - shift, scale)
        # Конвертировать контейнеры в пиксельные координаты на исходном
        # изображении
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Отфильтровать обнаружения с нулевой областью. Происходит на ранних
        # этапах обучения, когда веса сети все еще случайны
        exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Изменить размер маски до исходного размера изображения и установите
        # граничный порог
        full_masks = []
        for i in range(N):
            # Преобразовать маску нейронной сети в маску полного размера
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """
        Запустить конвейер обнаружения.
        :param images: Список изображений, потенциально разных размеров.
        :return список dict, по одному на изображение. dict содержит:
            rois: [N, (y1, x1, y2, x2)] ограничивающие рамки обнаружения
            class_ids: [N] int класс IDs
            scores: [N] float оценки вероятности для идентификаторов классов
            masks: [H, W, N] двоичные маски экземпляра
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Пресс-формы для форматирования ожидаемого нейронной сети
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Проверить размеры изображения. Все изображения в серии ДОЛЖНЫ быть
        # одинакового размера
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Якоря
        anchors = self.get_anchors(image_shape)
        # Дублировать размер по всему пакету, потому что это требуется Keras
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Запустить обнаружение объекта
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Обработать результат обнаружения
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """
        Запускает конвейер обнаружения, но ожидает входные данные, которые уже
        сформированы. Используется в основном для отладки и проверки модели.
        :param molded_images: Список изображений, загруженных с помощью
               load_image_gt()
        :param image_metas: метаданные изображения, также возвращаемые
               load_image_gt()
        :return список dict, по одному на изображение. dict содержит:
            rois: [N, (y1, x1, y2, x2)] ограничивающие рамки обнаружения
            class_ids: [N] int класс IDs
            scores: [N] float оценки вероятности для идентификаторов классов
            masks: [H, W, N] двоичные маски экземпляра
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE, "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Проверить размеры изображения
        # Все изображения в серии ДОЛЖНЫ быть одинакового размера
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Якоря
        anchors = self.get_anchors(image_shape)
        # Дублировать размер по всему пакету, потому что это требуется Keras
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Запустить обнаружение объекта
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Обработать результаты распознавания
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i], image.shape, molded_images[i].shape, window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def get_anchors(self, image_shape):
        """Возвращает якорную пирамиду для данного размера изображения."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Кэшировать якоря и используйте повторно, если форма изображения
        # одинакова
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Генерировать якоря
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Сохраняйте копии последних привязок в пиксельных координатах,
            # поскольку они используются в записных книжках inspect_model.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Нормализовать координаты
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """
        Находит предка тензора TF в графе вычислений.
        :param tensor: TensorFlow символический тензор.
        :param name: Имя тензора предка, чтобы найти
        :param checked: Для внутреннего использования. Список уже найденных
               тензоров, чтобы избежать циклов обхода графа.
        """
        checked = checked if checked is not None else []
        # Установить ограничение на глубину, чтобы избежать очень длинных петель
        if len(checked) > 500:
            return None
        # Преобразуйте имя в регулярное выражение и разрешите сопоставлять префикс
        # числа, потому что Keras добавляет их автоматически
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """
        Если уровень инкапсулирован другим слоем, эта функция просматривает
        инкапсуляцию и возвращает уровень, содержащий веса.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Возвращает список слоев с весами."""
        layers = []
        # Цикл по всем слоям
        for l in self.keras_model.layers:
            # Если слой является оберткой, найти внутренний обучаемый слой
            l = self.find_trainable_layer(l)
            # Включить слой, если он имеет вес
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """
        Запускает подмножество графа вычислений, которое вычисляет заданные
        выходные данные.
        :param image_metas: Если предусмотрено, предполагается, что изображения
               уже отлиты (то есть изменены, дополнены и нормализованы)
        :param outputs: Список кортежей (имя, тензор) для вычисления. Тензоры
               являются символическими тензорами TensorFlow, а их имена легко
               отслеживать.
        :return упорядоченный dict результатов. Ключи - это имена, полученные во
        входных данных, а значения - это массивы Numpy.
        """
        model = self.keras_model

        # Организовать желаемые результаты в упорядоченный dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Создать функцию Keras для запуска частей вычислительного графа
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(k_backend.learning_phase(), int):
            inputs += [k_backend.learning_phase()]
        kf = k_backend.function(model.inputs, list(outputs.values()))

        # Подготовить материалы
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Якоря
        anchors = self.get_anchors(image_shape)
        # Дублировать размер по всему пакету, потому что это требуется Keras
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        #
        # Выполнить вывод
        if model.uses_learning_phase and not isinstance(k_backend.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Упаковать сгенерированные массивы Numpy в dict и залогировать
        # результаты.
        outputs_np = OrderedDict([(k, v) for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


################################################################################
#  Форматирование данных
################################################################################

def compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids):
    """
    Берет атрибуты изображения и помещает их в один одномерный массив.
    :param image_id: Int ID изображения. Полезно для отладки.
    :param original_image_shape: [H, W, C] до изменения размера или заполнения.
    :param image_shape: [H, W, C] после изменения размера и заполнения
    :param window: (y1, x1, y2, x2) в пикселях. Область изображения, где
           находится реальное изображение (исключая отступы)
    :param scale: Коэффициент масштабирования, примененный к исходному
           изображению (float32)
    :param active_class_ids: Список class_ids, доступных в наборе данных, из
           которого пришло изображение. Полезно, если для обучения используются
           изображения из нескольких наборов данных, где не все классы
           присутствуют во всех наборах данных..
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) в координатах изображения
        [scale] +  # size=1
        list(active_class_ids)  # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """
    Анализирует массив, который содержит атрибуты изображения для его
    компонентов.
    Смотрите compose_image_meta () для более подробной информации.
    :param meta: [batch, meta length] где длина мета зависит от NUM_CLASSES
    :return dict проанализированных значений.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) окно изображения в пикселях
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """
    Анализирует тензор, который содержит атрибуты изображения для его
    компонентов.
    Смотрите compose_image_meta () для более подробной информации.
    :param meta: [batch, meta length] где длина мета зависит от NUM_CLASSES
    :return dict разобранных тензоров.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) окно изображения в пикселях
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """
    Ожидает RGB-изображение (или массив изображений), вычитает средний пиксель и
    преобразует его в плавающее. Ожидает цвета изображения в порядке RGB.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """
    Принимает изображение нормализованным с помощью mold () и возвращает оригинал.
    """
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


################################################################################
#  Разные функции графа
################################################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """
    Часто контейнеры представляются с матрицами формы [N, 4] и дополняются
    нулями. Данная функция удаляет нулевые контейнеры.
    :param boxes: [N, 4] матрица контейнеров.
    :param non_zeros: [N] 1D логическая маска, идентифицирующая строки
           для хранения
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """
    Выбирает различное количество значений из каждой строки в x в зависимости от
    значений в счетчиках.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """
    Преобразует контейнеры из координат пикселей в нормализованные координаты.
    В пиксельных координатах (y2, x2) находится за пределами контейнера. Но в
    нормализованных координатах -- внутри контейнерф.
    :param boxes: [..., (y1, x1, y2, x2)] в пиксельных координатах
    :param shape: [..., (height, width)] в пикселях
    :returns [..., (y1, x1, y2, x2)] в нормированных координатах
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """
    Преобразует контейнеры из нормализованных координат в пиксельные координаты.
    В пиксельных координатах (y2, x2) находится за пределами контейнера. Но в
    нормализованных координатах -- внутри контейнерф.
    :param boxes: [..., (y1, x1, y2, x2)] в нормированных координатах
    :param shape: [..., (height, width)] в пикселях
    :returns [..., (y1, x1, y2, x2)] в пиксельных координатах
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
