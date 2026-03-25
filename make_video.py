import cv2
import subprocess
from pathlib import Path

IMAGE_DIR = Path("assets/images")
OUTPUT = Path("assets/scannetpp_99fa5c25e1.mp4")
FPS = 30          # 인코딩 FPS (표준값 유지)
SEC_PER_IMAGE = 0.5  # 이미지 한 장당 표시 시간 (초)
REPEAT = round(FPS * SEC_PER_IMAGE)  # 한 이미지를 반복할 프레임 수

# 이미지 파일 목록 (정렬)
images = sorted(IMAGE_DIR.glob("*.png"))
if not images:
    images = sorted(IMAGE_DIR.glob("*.jpg"))
if not images:
    images = sorted(IMAGE_DIR.glob("*.JPG"))

print(f"{len(images)}개 이미지 발견")

# 첫 이미지로 해상도 결정
first = cv2.imread(str(images[0]))
h, w = first.shape[:2]
print(f"해상도: {w}x{h}")

# 임시 파일에 mp4v로 먼저 저장
tmp = OUTPUT.with_suffix(".tmp.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(tmp), fourcc, FPS, (w, h))

for i, img_path in enumerate(images):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  건너뜀: {img_path.name}")
        continue
    if frame.shape[:2] != (h, w):
        frame = cv2.resize(frame, (w, h))
    for _ in range(REPEAT):
        writer.write(frame)
    print(f"  [{i+1}/{len(images)}] {img_path.name}")

writer.release()

# ffmpeg으로 H.264 재인코딩 (브라우저 호환)
print("\nH.264로 변환 중...")
subprocess.run([
    "ffmpeg", "-y", "-i", str(tmp),
    "-vcodec", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
    str(OUTPUT)
], check=True)
tmp.unlink()
print(f"\n완료: {OUTPUT}")
