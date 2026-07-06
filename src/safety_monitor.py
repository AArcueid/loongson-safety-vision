"""实时安全监控——检测 + 合规判断 + 画面标注"""
import cv2, time
from safety_check import detect, check

SKIP = 3
COLORS = [(0,255,0),(0,0,255),(255,200,0),(0,128,255)]

cap = cv2.VideoCapture(0)
print("安全监控启动，按 q 退出...")
fc = 0
last_result = None
fps_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    fc += 1

    if fc % SKIP != 0 and last_result:
        r = last_result
    else:
        try:
            r = check_frame(frame)
        except:
            r = {"status": "error", "missing": [], "details": [], "summary": "推理失败"}
        last_result = r

    dets = r.get("detections", [])
    # 画框
    for d in dets:
        x1, y1, x2, y2 = map(int, [d["x1"],d["y1"],d["x2"],d["y2"]])
        name = d["name"]
        color = COLORS[0] if name not in ["no-helmet","no-vest"] else COLORS[1]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {d['prob']:.2f}", (x1, max(y1-5,15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 合规状态横幅
    status = "PASS" if r["status"] == "pass" else "FAIL"
    missing_str = " | 缺:" + ",".join(r["missing"]) if r["missing"] else ""
    color = (0, 255, 0) if r["status"] == "pass" else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, -1)
    cv2.putText(frame, f"{status}{missing_str}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elapsed = time.time() - fps_start
    fps = fc / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Safety Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
