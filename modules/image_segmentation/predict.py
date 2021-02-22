from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import numpy as np
from detectron2 import model_zoo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = 'weights/model_segmentation.pth'  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

def get_segment_crop(img, tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def segment_single_images(image, save_img=False):
    error_ims = []
    segmen_info = []

#     image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_predictor = predictor(image)
    if output_predictor['instances'].pred_masks.shape[0] > 1:
        mask_check = output_predictor['instances'].pred_masks.cpu().numpy()
        masks = output_predictor['instances'].pred_masks.cpu().numpy()
        mask_binary = masks[np.argmax(np.sum(masks, axis=(1, 2))) ,:,:]

    else:
        mask_binary = np.squeeze(output_predictor['instances'].pred_masks.permute(1, 2, 0).cpu().numpy())

    try:
        crop_mask = get_segment_crop(img = image, mask = mask_binary)
        
    except ValueError:
        print("error")
    origin_mask = cv2.cvtColor(np.float32(mask_binary) * 255.0, cv2.COLOR_GRAY2RGB)

    for j in range(image.shape[2]):
        image[:,:,j] = image[:,:,j] * origin_mask[:,:,j] * 255

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
