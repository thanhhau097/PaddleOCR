import json
import os

from paddleocr import PaddleOCR

# need to run only once to load model into memory
ocr = PaddleOCR(use_angle_cls=True, lang='ch',
                det_model_dir='./weights/ch_ppocr_server_v2.0_det_infer',
                rec_model_dir='./weights/ch_ppocr_server_v2.0_rec_infer')


with open('../data/cellcuts.json', 'r') as f:
    data = json.load(f)

count = 0
for img_path, label in data.items():
    result = ocr.ocr(os.path.join('../data', img_path), det=True, cls=True)
    print(result, label)
    if result[0][1][0] == label:
        count += 1

print(count / len(data))


