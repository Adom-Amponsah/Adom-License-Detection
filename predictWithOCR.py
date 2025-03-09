import os
import torch
import cv2
import numpy as np
import hydra
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from google.cloud import vision
from spellchecker import SpellChecker

# Initialize Google Cloud Vision client
client = vision.ImageAnnotatorClient()

# Initialize spell checker
spell = SpellChecker()

def correct_text(text):
    words = text.split()
    corrected_words = [spell.correction(word) if word in spell.unknown([word]) else word for word in words]
    return ' '.join(corrected_words)

def getOCR(im, coors):
    x, y, w, h = map(int, coors)
    im = im[y:h, x:w]

    # Convert image to bytes for Google Cloud Vision API
    success, encoded_image = cv2.imencode('.jpg', im)
    if not success:
        return ""

    image = vision.Image(content=encoded_image.tobytes())
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        lines = [text.description for text in texts[1:]]
        combined_text = ' '.join(lines)
        return combined_text if combined_text else ""
    return ""

class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.as_tensor(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
                                        agnostic=self.args.agnostic_nms, max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = f"{idx}: " if self.webcam else ""
        im0 = im0.copy()
        self.data_path = p

        det = preds[idx]
        self.all_outputs.append(det)
        if not det:
            return log_string

        for *xyxy, conf, cls in reversed(det):
            label = f"{self.model.names[int(cls)]} {conf:.2f}" if not self.args.hide_conf else self.model.names[int(cls)]
            ocr_text = getOCR(im0, xyxy)
            corrected_ocr = correct_text(ocr_text) if ocr_text else None
            label = corrected_ocr if corrected_ocr else label
            self.annotator.box_label(xyxy, label, color=colors(int(cls), True))

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source or ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    predict()
