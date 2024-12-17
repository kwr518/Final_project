from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, request
from werkzeug.utils import secure_filename
import os
import torch
import cv2  # OpenCV 추가
import easyocr
import re
import numpy as np
from pathlib import Path
from utils.plots import save_one_box
from utils.general import increment_path, scale_boxes, non_max_suppression
from utils.dataloaders import LoadImages
from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.torch_utils import select_device

app = Flask(__name__)

# 업로드 및 결과 경로 설정
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# YOLO 및 EasyOCR 초기화
weights = 'best.pt'  # YOLO 가중치 파일
device = select_device('')
model = DetectMultiBackend(weights, device=device)
reader = easyocr.Reader(['en', 'ko'], gpu=torch.cuda.is_available())  # EasyOCR

# YOLO + OCR 처리 함수 (이미지 및 동영상 공통)
def process_file(file_path, output_folder, is_video=False):
    imgsz = 640
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)

    first_crop_image = None  # 첫 번째 crop 이미지 경로

    if is_video:
        cap = cv2.VideoCapture(file_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        result_video_path = os.path.join(output_folder, f"result_{Path(file_path).stem}.mp4")
        result_video_path = result_video_path.replace('\\', '/')
        out = cv2.VideoWriter(result_video_path, fourcc, fps, (frame_width, frame_height))
        ocr_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_frame = frame.copy()
            frame = cv2.resize(frame, (imgsz, imgsz))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.transpose(2, 0, 1)
            frame = np.ascontiguousarray(frame)

            im = torch.from_numpy(frame).to(device)
            im = im.float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

            pred = model(im)
            pred = non_max_suppression(pred[0], 0.4, 0.45, max_det=1000)

            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], original_frame.shape).round()

                    for *xyxy, conf, cls in det:
                        crop_path = Path(output_folder) / f"crop_{Path(file_path).stem}_{int(cls)}.jpg"

                        # 첫 번째 crop 이미지만 저장
                        if not first_crop_image:
                            first_crop_image = crop_path

                        save_one_box(xyxy, original_frame, file=crop_path, BGR=True)
                        ocr_result = reader.readtext(str(crop_path))
                        for _, text, _ in ocr_result:
                            filtered_text = re.sub(r'[^가-힣0-9]', '', text)
                            if filtered_text:
                                ocr_results.append(filtered_text)

            out.write(original_frame)

        cap.release()
        out.release()
        print(f"Result video saved to: {result_video_path}")

        # 첫 번째 crop 이미지 반환
        if first_crop_image:
            return os.path.basename(first_crop_image), ocr_results
        return None, ocr_results



    else:
        # 이미지 처리
        dataset = LoadImages(file_path, img_size=imgsz, stride=stride)
        ocr_results = []
        result_image_path = None

        for path, im, im0s, _, _ in dataset:
            im = torch.from_numpy(im).to(device)
            im = im.float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

            pred = model(im)
            pred = non_max_suppression(pred[0], 0.4, 0.45, max_det=1000)

            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

                    for *xyxy, conf, cls in det:
                        crop_path = Path(output_folder) / f"crop_{Path(path).stem}_{int(cls)}.jpg"
                        save_one_box(xyxy, im0s, file=crop_path, BGR=True)

                        ocr_result = reader.readtext(str(crop_path))
                        for _, text, _ in ocr_result:
                            filtered_text = re.sub(r'[^가-힣0-9]', '', text)
                            if filtered_text:
                                ocr_results.append(filtered_text)

                result_image_path = os.path.join(output_folder, Path(path).name)
                cv2.imwrite(result_image_path, im0s)
        print(f"Image saved to {result_image_path}")

        return os.path.basename(result_image_path), ocr_results


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 파일 업로드 처리
        if 'file' in request.files:
            file = request.files['file']
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # 파일 유형에 따라 처리
                is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov'))
                result_path, ocr_results = process_file(file_path, app.config['RESULT_FOLDER'], is_video=is_video)

                # 결과 페이지로 리디렉션
                return render_template(
                    'result.html',
                    result_file=result_path,
                    ocr_results=", ".join(ocr_results),
                    is_video=is_video
                )

    return render_template('index.html')

@app.route('/download/<path:filename>')
def download_image(filename):
    # 결과 폴더에서 파일을 반환
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)

@app.route('/static/results/<path:filename>')
def serve_video(filename):
    return send_from_directory('static/results', filename, mimetype='video/mp4')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)