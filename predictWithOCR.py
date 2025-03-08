import hydra
import torch
import cv2
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from paddleocr import PaddleOCR
from spellchecker import SpellChecker

# Import Gemini API
import google.generativeai as genai
import os

# --------------------- API KEY WARNING ---------------------
# WARNING: Hardcoding your API key directly in the code is NOT recommended for production.
# It is insecure and can expose your key.
# For this demonstration, we are using it as requested, but for real applications:
# 1. Set GOOGLE_API_KEY environment variable (recommended - as shown previously).
# 2. Use a more secure method to manage secrets (like cloud secret management services).
# ------------------------------------------------------------

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyAV4XpiOKS2hXA9ood2-kObGhldgxeV7lc" # Directly using the provided API key - FOR DEMO ONLY
#  Uncomment below lines and comment the line above to use environment variable (Recommended)
# GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
# if GEMINI_API_KEY is None:
#     raise EnvironmentError("Please set the GOOGLE_API_KEY environment variable or hardcode it (DEV ONLY - INSECURE).")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Initialize PaddleOCR reader with the latest model
reader = PaddleOCR(model_type='PP-OCRv3', use_gpu=True, lang='en')

# Initialize spell checker (keeping it for potential fallback or comparison)
spell = SpellChecker()

def correct_text_spellchecker(text): # Renamed to differentiate
    words = text.split()
    corrected_words = []
    for word in words:
        if spell.unknown([word]):
            corrected_word = spell.correction(word)
            corrected_words.append(corrected_word if corrected_word else word)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

def correct_text_gemini(text):
    """Corrects text using Gemini Pro for improved OCR accuracy."""
    if not text.strip():  # Handle empty text to avoid Gemini API calls
        return ""

    prompt_parts = [
        "Please perform advanced text correction on the following OCR output to fix spelling, grammar, and improve readability, ensuring the meaning is preserved. Focus on making the text as accurate and natural as possible:\n",
        text,
        "\nEnsure the corrected text is highly accurate and reflects proper English." # Added stronger instruction for accuracy
    ]

    try:
        response = model.generate_content(prompt_parts)
        response.resolve() # Wait for response to be fully available
        corrected_text = response.text
        if corrected_text:
            return corrected_text
        else:
            print("Gemini returned empty correction, falling back to SpellChecker.")
            return correct_text_spellchecker(text) # Fallback to spellchecker if Gemini fails
    except Exception as e:
        print(f"Error during Gemini correction: {e}. Falling back to SpellChecker. Error: {e}")
        return correct_text_spellchecker(text) # Fallback to spellchecker on error


def getOCR(im, coors):
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im = im[y:h, x:w]
    conf = 0.2

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # --- Enhanced PaddleOCR Configuration (Optional, but can improve accuracy) ---
    # You can experiment with these parameters to potentially boost PaddleOCR's initial results.
    # For example, increasing `det_db_unclip_ratio` might help with detecting slightly overlapping text.
    # `rec_image_shape` can be adjusted if you know the typical aspect ratio of your text.
    # Experiment with these, or remove if default PaddleOCR is preferred.
    # results = reader.ocr(gray, cls=True, det_db_unclip_ratio=2.0, rec_image_shape='3x32x192')
    # -----------------------------------------------------------------------------

    results = reader.ocr(gray, cls=True) # Using default PaddleOCR settings for now

    if results:
        results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)
        if results_sorted and results_sorted[0]['confidence'] > conf: # Check if results_sorted is not empty
            ocr = results_sorted[0]['text']
        else:
            ocr = ""
    else:
        ocr = ""

    return str(ocr)

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.as_tensor(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh( torch.from_np(np.array(xyxy)).view(1, 4) / gn )).view(-1).to_list()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crops or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                ocr = getOCR(im0, xyxy)
                if ocr:
                    corrected_ocr = correct_text_gemini(ocr) # Use Gemini for enhanced text correction
                    label = corrected_ocr
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crops:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    predict()