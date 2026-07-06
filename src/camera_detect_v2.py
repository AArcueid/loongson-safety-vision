"""实时检测 v2 — 文件 I/O 版（已验证可用，稳）"""
import cv2, json, struct, subprocess, tempfile, os, time

MODEL_SIZE = 320
MODEL_PARAM = f"best{'_opt' if MODEL_SIZE == 640 else str(MODEL_SIZE) + '_opt'}.param"
MODEL_BIN = f"best{'_opt' if MODEL_SIZE == 640 else str(MODEL_SIZE) + '_opt'}.bin"
SKIP = 3

CLASSES = ["helmet","no-helmet","vest","no-vest","mask","gloves","no-gloves","person","boots"]
COLORS = [(0,255,0),(0,0,255),(255,200,0),(0,128,255),(255,0,255),(255,128,0),(128,0,255),(0,255,255),(128,128,0)]

cap = cv2.VideoCapture(0)
print(f"开始检测，{MODEL_SIZE}×{MODEL_SIZE}，按 q 退出...")
frame_count = 0
last_dets = []
fps_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    # 跳帧
    if frame_count % SKIP != 0:
        for d in last_dets:
            cv2.rectangle(frame, (int(d["x1"]),int(d["y1"])), (int(d["x2"]),int(d["y2"])),
                          COLORS[d["label"] % 9], 2)
        cv2.putText(frame, f"skip={SKIP}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    h, w = frame.shape[:2]
    scale = min(MODEL_SIZE / w, MODEL_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    padded = cv2.copyMakeBorder(resized, 0, MODEL_SIZE - nh, 0, MODEL_SIZE - nw,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    data = rgb.tobytes()

    with tempfile.NamedTemporaryFile(suffix='.rgb', delete=False) as fp:
        fp.write(struct.pack('iiiii', MODEL_SIZE, MODEL_SIZE, 3, w, h))
        fp.write(data)
        tmp = fp.name

    try:
        r = subprocess.run(['./yolo_detect', MODEL_PARAM, MODEL_BIN, tmp],
                           capture_output=True, text=True, timeout=30)
        dets = json.loads(r.stdout).get("detections", [])
    except:
        dets = []
    os.unlink(tmp)
    last_dets = dets

    # 画框
    for d in dets:
        cv2.rectangle(frame, (int(d["x1"]),int(d["y1"])), (int(d["x2"]),int(d["y2"])),
                      COLORS[d["label"] % 9], 2)
        cv2.putText(frame, f"{d['name']} {d['prob']:.2f}",
                    (int(d["x1"]), max(int(d["y1"]) - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[d["label"] % 9], 1)

    elapsed = time.time() - fps_start
    fps = frame_count / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"YOLOv5n-320 | FPS:{fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
