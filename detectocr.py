import argparse
from pathlib import Path
import re
import torch
from utils.plots import save_one_box
from utils.general import increment_path, scale_boxes, check_img_size, non_max_suppression
from utils.dataloaders import LoadImages
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
import easyocr  # EasyOCR 추가

# EasyOCR 초기화
reader = easyocr.Reader(['en', 'ko'], gpu=torch.cuda.is_available())

def run(
    weights='best.pt',
    source='data/images',
    imgsz=640,
    conf_thres=0.4,
    iou_thres=0.45,
    max_det=1000,
    device='',
    save_crop=True,
    project='runs/detect',
    name='exp',
    exist_ok=False,
):
    # 결과 디렉토리 설정
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'crops').mkdir(parents=True, exist_ok=True)

    # 모델 로드
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # 데이터 로드
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # 결과 저장용 변수
    results = []

    for path, im, im0s, _, _ in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]

        # 추론
        pred = model(im)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        # NMS 적용
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
        if len(pred) == 0 or len(pred[0]) == 0:
            print(f"No predictions for {path}")
            continue

        # 탐지된 객체 처리
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in det:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    print(f"Detected {label} at {xyxy}")

                    # 특정 클래스(예: 번호판)만 처리
                    if save_crop and int(cls) == 0:
                        crop_path = save_dir / 'crops' / f"plate_{Path(path).stem}.jpg"
                        save_one_box(xyxy, im0s, file=crop_path, BGR=True)
                        print(f"Saved cropped plate to {crop_path}")

                        # OCR 처리
                        ocr_result = reader.readtext(str(crop_path))
                        for bbox, text, ocr_conf in ocr_result:
                            filtered_text = re.sub(r'[^가-힣0-9]', '', text)  # 텍스트 필터링
                            if filtered_text:
                                results.append({"image": path, "text": filtered_text})
                                print(f"Detected Plate Text: {filtered_text}")

    # OCR 결과 출력
    print("\n=== OCR 결과 ===")
    for result in results:
        print(f"Image: {result['image']} | Detected Text: {result['text']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='model path')
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    run(**vars(opt))
