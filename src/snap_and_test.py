"""交互式测试——运行后按回车拍一张照并推理，输入q回车退出"""
import cv2, struct, subprocess, json, sys, os

MODEL_SIZE = 320
if os.path.exists(f"best_opt.param") and MODEL_SIZE == 640:
    param, bin = "best_opt.param", "best_opt.bin"
else:
    # 自动匹配已存在的模型
    for sz, pm in [(320, "best320_opt.param"), (640, "best_opt.param")]:
        if os.path.exists(pm):
            MODEL_SIZE = sz
            param = f"best{'' if sz==640 else str(sz)}_opt.param"
            bin = f"best{'' if sz==640 else str(sz)}_opt.bin"
            break

cap = cv2.VideoCapture(0)
print(f"摄像头已开，{MODEL_SIZE}x{MODEL_SIZE} 模型")
print("按 [回车] 拍照并推理，按 q+回车 退出\n")

while True:
    cmd = input("> ").strip()
    if cmd == 'q':
        break

    # 拍照
    ret, frame = cap.read()
    if not ret:
        print("读取失败")
        continue
    h, w = frame.shape[:2]
    cv2.imwrite("last_snap.jpg", frame)
    print(f"  已拍照 ({w}x{h})，推理中...")

    # 预处理
    scale = min(MODEL_SIZE / w, MODEL_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    padded = cv2.copyMakeBorder(resized, 0, MODEL_SIZE - nh, 0, MODEL_SIZE - nw,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    data = rgb.tobytes()

    with open('/tmp/test.rgb', 'wb') as fp:
        fp.write(struct.pack('iiiii', MODEL_SIZE, MODEL_SIZE, 3, w, h))
        fp.write(data)

    r = subprocess.run(['./yolo_detect', param, bin, '/tmp/test.rgb'],
                       capture_output=True, text=True, timeout=30)
    try:
        dets = json.loads(r.stdout).get("detections", [])
    except:
        print(f"  解析失败: {r.stdout[:200]}")
        dets = []

    print(f"  检测到 {len(dets)} 个目标:")
    for d in dets:
        print(f"    {d['name']}: {d['prob']:.3f} ({d['x1']:.0f},{d['y1']:.0f})-({d['x2']:.0f},{d['y2']:.0f})")

    # 画框保存
    frame_vis = frame.copy()
    for d in dets:
        x1, y1 = int(d["x1"]), int(d["y1"])
        x2, y2 = int(d["x2"]), int(d["y2"])
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_vis, f"{d['name']} {d['prob']:.2f}",
                    (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("last_detect.jpg", frame_vis)
    print(f"  已保存: last_snap.jpg (原图) + last_detect.jpg (检测结果)\n")

cap.release()
print("退出")
