import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


############################################################
#  Играничиващие прямоугольники
############################################################

def extract_bounding_boxes(mask):
    """
    Вычисление ограничивающих рамок из масок.
    :param mask: [height, width, num_instances]. Пиксели маски имеют либо 1, либо 0.
    :return: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 и y2 не должны быть частью рамки. Увеличение на 1.
            x2 += 1
            y2 += 1
        else:
            # Нет маски для этого экземпляра. Может произойти из-за
            # изменения размера или обрезки. Установление bbox в нули
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """
    Вычисляет IoU данного прямоугольника с массивом данных прямоугольников.
    :param box: одномерный массив [y1, x1, y2, x2]
    :param boxes: [boxes_count, (y1, x1, y2, x2)]
    :param box_area: float. площадь 'рамки'
    :param boxes_area: массив длин boxes_count
    :return: значение метрики iou
    """
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """
    Вычисляет IoU перекрытия между двумя наборами блоков.
    Для лучшей производительности необходимо передавать первым
    параметром большой набор, а вторым - меньший.
    :param boxes1: [N, (y1, x1, y2, x2)]
    :param boxes2: [N, (y1, x1, y2, x2)]
    :return:
    """
    # Площади anchor'ов и GT блоков
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Вычислить перекрытия для генерации матрицы [boxes1 count, boxes2 count]
    # Каждая ячейка содержит значение IoU.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """
    Вычисляет перекрытия между двумя наборами масок.
    :param masks1: [Height, Width, instances]
    :param masks2: [Height, Width, instances]
    :return:
    """
    # Если какой-либо набор масок пуст, вернуть пустой результат
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # Уменьшить размерность маскок и вычислить их площади
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # пересечения и объединение
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """
    Выполнение алгоритма подавления немаксимумов.
    :param boxes: [N, (y1, x1, y2, x2)]. (y2, x2) лежат вне прямоугольника
    :param scores: одномерный массив показателей распознавания
    :param threshold: Float. IoU порог, используемый для фильтрации
    :return:
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Вычисление области прямоугольника
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Получение индексов прямоугольникав, отсортированных
    # по показателям (сначала самый высокий)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Индекс верхнего прямоугольник добавляется в список
        i = ixs[0]
        pick.append(i)
        # Вычисление IOU выбранного прямоугольника с остальными
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Идентифицировать прямоугольники с IoU выше порога threshold.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Удалите индексы выбранных и перекрытых прямоугольников.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """
    Применение указанных дельт к указанным полям.
    :param boxes: [N, (y1, x1, y2, x2)]. (y2, x2) лежат вне прямоугольника
    :param deltas: [N, (dy, dx, log(dh), log(dw))]
    :return:
    """
    boxes = boxes.astype(np.float32)
    # Преобразование к y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Применение дельт
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Обратное преобразование к y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """
    Вычисление уточнения, необходимого для преобразования
    прямоугольника в gt_box.
    :param box: [N, (y1, x1, y2, x2)]
    :param gt_box: [N, (y1, x1, y2, x2)]
    :return:
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """
    Вычисление уточнения, необходимого для преобразования
    прямоугольника в gt_box.
    :param box: [N, (y1, x1, y2, x2)]. (y2, x2) вне прямоугольника.
    :param gt_box: [N, (y1, x1, y2, x2)]. (y2, x2) вне прямоугольника.
    :return:
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Набор данных
############################################################

class Dataset:
    """
    Базовый класс для классов наборов данных.
    Чтобы использовать его, создайте новый класс, который добавляет функции,
    специфичные для набора данных, который вы хотите использовать.
    """

    def __init__(self):
        self._image_ids = []
        self.image_info = []
        # Фон всегда первый класс
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Класс уже существует?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # комбинация source.class_id уже доступна, пропустается
                return
        # Добавить класс
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """
        Возвращает ссылку на изображение на исходном веб-сайте
        или сведения об изображении, которые помогут найти его
        или отладить.
        Переопределите для своего набора данных, но перейдите
        к этой функции, если вы встретите изображения, отсутствующие
        в вашем наборе данных.
        """
        return ""

    def prepare(self):
        """
        Подготовка класса Dataset для использования.
        """

        def clean_name(name):
            """
            Возвращает более короткую версию имен объектов для более
            чистого отображения.
            """
            return ",".join(name.split(",")[:1])

        # Построить (или перестроить) все остальное из информационных диктов.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Преобразование из исходного класса и идентификаторов
        # изображений во внутренние идентификаторы
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Сопоставление источников с class_ids, которые они поддерживают
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Цикл по набору данных
        for source in self.sources:
            self.source_class_ids[source] = []
            # Нахождение классов, которые принадлежат этому набору данных
            for i, info in enumerate(self.class_info):
                # Включить класс BG во все наборы данных
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """
        Принимает идентификатор класса источника и возвращает
        присвоенный ему идентификатор класса int.
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """
        Сопоставление внутреннего идентификатора класса с
        соответствующим идентификатором класса в исходном наборе данных.
        """
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """
        Возвращает путь или URL к изображению.
        Переопределите этот метод, чтобы вернуть URL-адрес
        изображения, если оно доступно в Интернете для легкой отладки.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """
        Загрузка указанного изображения и возврат в виде
        массив [H, W, 3] Numpy.
        """
        # Загрузить изображение
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # Если в оттенках серого. Конвертировать в RGB для согласованности.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # Если есть альфа-канал, удалите его для согласованности
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """
        Загрузить маски экземпляра для данного изображения.
        Различные наборы данных используют разные способы
        хранения масок. Переопределите этот метод, чтобы
        загрузить маски экземпляров и вернуть их в виде
        массива двоичных масок формы [height, width, instances].
        :return:
            masks: Массив bool формы [height, width, instance count]
            с двоичной маской на экземпляр.
            class_ids: одномерный массив идентификаторов классов масок
            экземпляра.
        """
        # Переопределите эту функцию, чтобы загрузить маску из вашего набора данных.
        # В противном случае он возвращает пустую маску.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """
    Изменение размера изображения, сохраняя соотношение сторон
    без изменений.
    :param image: изображение
    :param min_dim: используется для изменения размера изображения так,
                    чтобы его меньшего сторона == min_dim
    :param max_dim: используется для гарантии того, что самая длинная
                    сторона изображения не превышает это значение.
    :param min_scale: используется для гарантии того, что изображение
                      увеличено как минимум на этот процент, даже если
                      min_dim этого не требует.
    :param mode: "square" -- изменение размера и дополнение нулями,
                 чтобы получить квадратное изображение размером
                 [max_dim, max_dim].
    :return: image: измененное изображение
             window: (y1, x1, y2, x2). Если указан max_dim, заполнение может
                     быть вставлено в возвращаемое изображение. Если это так,
                     это окно является координатами части изображения полного
                     изображения (исключая отступы). Пиксели x2, y2 не включены.
             scale: Коэффициент масштабирования, используемый для изменения
                    размера изображения
             padding: Дополненное добавлено к изображению
                      [(top, bottom), (left, right), (0, 0)]
    """
    # Запоминается dtype изображения и для возврата результата в том же dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Масштаб?
    if min_dim:
        # Увеличенин масштаба
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Изменение размера изображения с помощью билинейной интерполяции
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    if mode == "square":
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """
    Изменение размера маски с использованием заданного масштаба и
    отступов. Как правило, вы получаете масштаб и отступ от resize_image(),
    чтобы гарантировать, что размер изображения и маски будет изменяться
    согласованно.
    :param mask: маска
    :param scale: коэффициент масштабирования маски
    :param padding: отступ, чтобы добавить в маску в виде
                    [(top, bottom), (left, right), (0, 0)]
    :param crop:
    :return:
    """
    # Подавляется предупреждение от scipy 0.13.0, выходная форма zoom()
    # рассчитывается с помощью round () вместо int ()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """
    Изменение размера маски в меньшую версию, чтобы уменьшить
    нагрузку на память. Мини-маски могут быть изменены до
    масштаба изображения с помощью expand_masks()
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Выберите срез и приведите к bool в случае, если load_mask ()
        # вернул неправильный dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Изменение размера с билинейной интерполяцией
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """
    Изменение размера мини-масок обратно к размеру изображения.
    Отменяет изменение minimize_mask().
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Изменение размера с билинейной интерполяцией
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask


def unmold_mask(mask, bbox, image_shape):
    """
    Преобразует маску, сгенерированную нейронной сетью, в формат,
    аналогичный ее первоначальной форме.
    :param mask:  [height, width] типа float. Небольшая, обычно 28x28 маска.
    :param bbox: [y1, x1, y2, x2]. Рамка для маски.
    :param image_shape:
    :return: двоичная маска с тем же размером, что и исходное изображение.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Поместить маску в правильное место..
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Рамки скользящего окна
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    :param scales: 1D массив размеров рамок в пикселях.
    :param ratios: 1D массив соотношений рамок ширина / высота.
    :param shape: [height, width] пространственная форма карты
                  объектов, по которой создаются привязки.
    :param feature_stride: Шаг карты признаков относительно
                           изображения в пикселях.
    :param anchor_stride: Шаг рамок на карте признаков. Например,
                          если значение равно 2, генерируйте рамки
                          для каждого пикселя карты объектов.
    :return:
    """
    # Получить все комбинации масштабов и соотношений
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Перечислить высоты и ширины из масштабов и соотношений
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Перечислить сдвиги в пространсве признаков
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Перечислить комбинации сдвигов, широт и высот
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Изменить размерность, чтобы получить список (y, x) и список (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Преобразовать в угловые координаты (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """
    Создание рамок на разных уровнях пирамиды объектов. Каждый
    масштаб связан с уровнем пирамиды, но каждое соотношение
    используется на всех уровнях пирамиды.
    :param scales:
    :param ratios:
    :param feature_shapes:
    :param feature_strides:
    :param anchor_stride:
    :return: anchors: [N, (y1, x1, y2, x2)]. Все сгенерированные
                      рамки в одном массиве. Сортируется с тем же
                      порядком данных весов. Итак, рамки scale[0]
                      идут первыми, затем рамки scale[1] и так далее.
    """
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
# Разное
############################################################

def trim_zeros(x):
    """
    Обычно тензоры превышают доступные данные и дополняются нулями.
    Эта функция удаляет все строки, которые являются нулями.
    :param x: [rows, columns]
    :return:
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids,
                    pred_scores, pred_masks, iou_threshold=0.5, score_threshold=0.0):
    """
    Находит совпадения между предсказанными и эталонными объектами.
    :param gt_boxes:
    :param gt_class_ids:
    :param gt_masks:
    :param pred_boxes:
    :param pred_class_ids:
    :param pred_scores:
    :param pred_masks:
    :param iou_threshold:
    :param score_threshold:
    :return:
        gt_match: 1-D массив. Для каждого GT-прямоугольника он имеет индекс
                  совпадающего прогнозируемого прямоугольника.
        pred_match: 1-D массив. Для каждого прогнозируемого прямоугольника
                    у него есть индекс сопоставленного истинного прямоугольника.
        overlaps: [pred_boxes, gt_boxes] IoU перекрытия.
    """
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Сортировать прогнозы по показателям от высокого к низкому
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Вычислить IoU перекрытия [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Перебор прогнозов и нахождение соответствующих эталонных
    # прямоугольников
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Найти лучшее совпадение эталонному прямоугольнику
        # 1. Сортировка совпадений по показателям
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Удалить низкие показатели
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Найти совпадения
        for j in sorted_ixs:
            # Если эталонный прямоугольник уже найден, перейти к следующему
            if gt_match[j] > -1:
                continue
            # Если достигнут IoU меньше, чем порог threshold, закончить цикл
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Есть ли совпадения?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids,
               pred_scores, pred_masks, iou_threshold=0.5):
    """
    Вычислить среднюю точность при установленном пороге IoU (по умолчанию 0,5).
    :param gt_boxes:
    :param gt_class_ids:
    :param gt_masks:
    :param pred_boxes:
    :param pred_class_ids:
    :param pred_scores:
    :param pred_masks:
    :param iou_threshold:
    :return:
        mAP: медиана средней точности
        precisions: список точности при различных пороговых значениях класса.
        recalls: список значений отзыва при различных пороговых значениях класса.
        overlaps: [pred_boxes, gt_boxes] IoU перекрытий.
    """
    # Получить совпадения и перекрытия
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Вычислять точность и вызывать на каждом шаге предсказания
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad с начальными и конечными значениями для упрощения математики
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Убедитесь, что значения точности уменьшаются, но не увеличиваются.
    # Таким образом, значение точности при каждом пороге отзыва является
    # максимальным значением, которое может быть для всех следующих
    # порогов отзыва, как указано в документе VOC.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Вычислить среднее значение AP по диапазону отзыва
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask, pred_box, pred_class_id,
                     pred_score, pred_mask, iou_thresholds=None, verbose=1):
    """
    Вычислить AP в диапазоне или пороговых значениях IoU.
    Диапазон по умолчанию составляет 0,5-0,95.
    """
    # По умолчанию от 0,5 до 0,95 с шагом 0,05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Вычислить AP в диапазоне порогов IoU
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = \
            compute_ap(gt_box, gt_class_id, gt_mask,
                       pred_box, pred_class_id, pred_score, pred_mask,
                       iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """
    Вычислить отзыв при заданном пороге IoU. Это показатель
    того, сколько блоков GT было найдено с помощью данных блоков
    прогнозирования.
    :param pred_boxes: [N, (y1, x1, y2, x2)] в координатах изображения
    :param gt_boxes: [N, (y1, x1, y2, x2)] в координатах изображения
    :param iou:
    :return:
    """
    # Мера перекрытий
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Некоторые пользовательские слои поддерживают размер пакета
# только 1 и требуют много работы для поддержки пакетов,
# превышающих 1. Эта функция разбивает входной тензор по
# измерению пакета и подает пакеты размера 1. По сути,
# простой способ поддержки пакетов> 1 быстро с небольшим
# изменением кода. В конечном счете, более эффективно модифицировать
# код для поддержки больших пакетов и избавления от этой функции.
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """
    Разбивает входные данные на срезы и передает каждый срез в копию
    данного графа вычислений, а затем объединяет результаты. Это
    позволяет вам запускать график для группы входных данных, даже
    если график написан для поддержки только одного экземпляра.
    :param inputs: список тензоров. Все должны иметь одинаковую длину
                   первого измерения
    :param graph_fn: Функция, которая возвращает тензор TF, который
                     является частью графа.
    :param batch_size: количество срезов, чтобы разделить данные.
    :param names: При наличии присваивает имена полученным тензорам.
    :return:
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Измените выходные данные из списка срезов, где каждый
    # представляет собой список выходных данных, на список
    # выходных данных, и у каждого есть список срезов.
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """
    Загрузите обученные веса COCO из релизов.
    :param coco_model_path: локальный путь COCO обученных весов
    :param verbose:
    :return:
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def norm_boxes(boxes, shape):
    """
    Преобразует прямоугольники из координат пикселей в
    нормализованные координаты. В пиксельных координатах
    (y2, x2) находится за пределами прямоугольника. Но в
    нормализованных координатах они внутри прямоугольника.
    :param boxes: [N, (y1, x1, y2, x2)] в пиксельных координатах
    :param shape: [..., (height, width)] в пикселях
    :return: [N, (y1, x1, y2, x2)] в нормализированных координатах
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """
    Преобразует прямоугольники из нормализованных координат
    в координаты пикселей. В пиксельных координатах
    (y2, x2) находится за пределами прямоугольника. Но в
    нормализованных координатах они внутри прямоугольника.
    :param boxes: [N, (y1, x1, y2, x2)] в нормализированных координатах
    :param shape: [..., (height, width)] в пикселях
    :return: [N, (y1, x1, y2, x2)] в пиксельных координатах
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """
    Оболочка для Scikit-Image resize ().
    Scikit-Image генерирует предупреждения при каждом вызове resize(),
    если он не получает правильные параметры. Правильные параметры
    зависят от версии лыжного мага. Это решает проблему, используя
    разные параметры для каждой версии. И это обеспечивает центральное
    место для контроля изменения размеров по умолчанию.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # Новое в версии 0.14: anti_aliasing. По умолчанию
        # это False для обратной совместимости с Skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)