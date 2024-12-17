import cv2
import easyocr
import numpy as np

# 1. EasyOCR 리더 초기화
reader = easyocr.Reader(['en', 'ko'])  # 'en'은 영어, 'ko'는 한글 지원

# 2. 번호판 영역 검출 (OpenCV)
def detect_license_plate(image):
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 블러 적용 (노이즈 감소)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 엣지 검출 (Canny Edge Detection)
    edges = cv2.Canny(blurred, 50, 150)

    # 컨투어 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 번호판 후보 영역 선택
    for contour in contours:
        # 컨투어의 외곽 경계 박스 추출
        x, y, w, h = cv2.boundingRect(contour)

        # 번호판 크기 비율 필터링 (너무 작거나, 비율이 이상한 경우 제외)
        if w / h > 2 and w / h < 6 and w > 100 and h > 30:
            # 번호판 영역 추출
            plate = image[y:y+h, x:x+w]
            return plate, (x, y, w, h)

    return None, None

# 3. OCR 처리
def perform_ocr(image, plate_region):
    # 번호판 영역이 검출되지 않은 경우
    if plate_region is None:
        print("번호판을 찾을 수 없습니다.")
        return None

    # 번호판 영역 잘라내기
    plate, bbox = plate_region

    # OCR 수행
    ocr_result = reader.readtext(plate)

    # 결과 출력
    print("OCR 결과:")
    for (bbox, text, confidence) in ocr_result:
        print(f"텍스트: {text}, 신뢰도: {confidence}")

    return ocr_result

# 4. 이미지 로드 및 실행
def main(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)

    # 번호판 검출
    plate_region = detect_license_plate(image)

    # OCR 수행
    ocr_result = perform_ocr(image, plate_region)

    # 결과 표시 (원본 이미지에 경계 상자 그리기)
    if plate_region and plate_region[1]:
        x, y, w, h = plate_region[1]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "License Plate", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 이미지 보기
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 5. 실행
if __name__ == "__main__":
    image_path = "3.png"  # 차량 이미지 경로
    main(image_path)
