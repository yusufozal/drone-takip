import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import time
from ultralytics import YOLO


def main():
    # YOLOv11 modelini yükle
    model = YOLO("best (7).pt")

    # Video kaynağı
    video_source = 0
    cap = cv2.VideoCapture("handmade talon flight.mp4")

    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    # OpenCV sürümünü kontrol et ve uygun tracker oluşturucu seç
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    print(f"OpenCV Sürümü: {cv2.__version__}")

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]  # KCF seçildi

    # OpenCV sürümüne göre tracker oluşturma fonksiyonunu belirle
    if int(major_ver) < 4:
        tracker_creator = cv2.Tracker_create
    else:
        if tracker_type == 'BOOSTING':
            tracker_creator = cv2.legacy.TrackerBoosting_create
        elif tracker_type == 'MIL':
            tracker_creator = cv2.legacy.TrackerMIL_create
        elif tracker_type == 'KCF':
            tracker_creator = cv2.legacy.TrackerKCF_create
        elif tracker_type == 'TLD':
            tracker_creator = cv2.legacy.TrackerTLD_create
        elif tracker_type == 'MEDIANFLOW':
            tracker_creator = cv2.legacy.TrackerMedianFlow_create
        elif tracker_type == 'GOTURN':
            tracker_creator = cv2.legacy.TrackerGOTURN_create
        elif tracker_type == 'MOSSE':
            tracker_creator = cv2.legacy.TrackerMOSSE_create
        elif tracker_type == "CSRT":
            tracker_creator = cv2.legacy.TrackerCSRT_create

    print(f"Kullanılan tracker: {tracker_type}")

    # Tracker değişkenleri
    trackers = []
    bboxes = []
    colors = []

    # Tespit aralığı
    detection_interval = 30
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_display = frame.copy()

        # Belirli aralıklarla veya hiç tracker yoksa yeniden tespit yap
        if frame_count % detection_interval == 0 or len(trackers) == 0:
            # Eski trackerları temizle
            trackers = []
            bboxes = []
            colors = []

            # YOLOv11 tespiti
            results = model(frame, conf=0.4)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Kutu koordinatlarını al
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Tracker oluştur
                    tracker = tracker_creator()
                    bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, width, height)

                    # Tracker'ı başlat
                    try:
                        success = tracker.init(frame, bbox)
                        if success:
                            trackers.append(tracker)
                            bboxes.append(bbox)
                            # Rastgele renk
                            colors.append((np.random.randint(0, 255),
                                           np.random.randint(0, 255),
                                           np.random.randint(0, 255)))
                            print(f"Tracker başlatıldı: {bbox}")
                        else:
                            print("Tracker başlatma başarısız!")
                    except Exception as e:
                        print(f"Tracker hatası: {e}")

        # Her frame'de aktif trackerları güncelle
        if trackers:
            to_remove = []
            for i, tracker in enumerate(trackers):
                try:
                    success, bbox = tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        bboxes[i] = (x, y, w, h)

                        # İzlenen nesneyi çiz
                        cv2.rectangle(frame_display, (x, y), (x + w, y + h), colors[i], 2)
                        cv2.putText(frame_display, f"IHA #{i}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
                    else:
                        to_remove.append(i)
                        print(f"Tracker #{i} güncelleme başarısız")
                except Exception as e:
                    to_remove.append(i)
                    print(f"Tracker hatası: {e}")

            # Başarısız trackerları kaldır
            for idx in sorted(to_remove, reverse=True):
                trackers.pop(idx)
                bboxes.pop(idx)
                colors.pop(idx)

        # Güncel tracker sayısını göster
        cv2.putText(frame_display, f"Izlenen IHA: {len(trackers)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Görüntüyü göster
        cv2.imshow("IHA Tespiti ve Izleme", frame_display)

        # Frame sayacını güncelle
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()