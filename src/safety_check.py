"""安全合规检测 API"""
import cv2, json, struct, subprocess, tempfile, os

MODEL_SIZE = 320
PARAM = "best320_opt.param"
BIN = "best320_opt.bin"
DETECT_BIN = "./yolo_detect"

# 合规规则：三元并列，全有才过
#   helmet 检测到 → 安全帽合规
#   mask   检测到 → 口罩合规
#   no-vest 未检测到 → 反光衣合规（vest 本身检测不准，用反例代替）
ITEMS = {
    "helmet":  "安全帽",      # 必须检测到
    "mask":    "口罩",        # 必须检测到
    "no-vest": "反光衣",      # 必须不出现（反例判定）
}

# 判定映射：positive=True 表示该类别必须出现，positive=False 表示必须不出现
RULES = [
    {"key": "helmet",  "label": "安全帽", "required": True},   # 必须有
    {"key": "mask",    "label": "口罩",   "required": True},   # 必须有
    {"key": "no-vest", "label": "反光衣", "required": False},  # 必须无
]

def detect(frame, conf_thresh=0.25, nms_thresh=0.45):
    """对一帧图像做目标检测，返回检测结果列表"""
    h, w = frame.shape[:2]
    scale = min(MODEL_SIZE / w, MODEL_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    padded = cv2.copyMakeBorder(resized, 0, MODEL_SIZE - nh, 0, MODEL_SIZE - nw,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    with tempfile.NamedTemporaryFile(suffix='.rgb', delete=False) as fp:
        fp.write(struct.pack('iiiii', MODEL_SIZE, MODEL_SIZE, 3, w, h))
        fp.write(rgb.tobytes())
        tmp = fp.name

    try:
        r = subprocess.run([DETECT_BIN, PARAM, BIN, tmp, str(conf_thresh), str(nms_thresh)],
                           capture_output=True, text=True, timeout=30)
        dets = json.loads(r.stdout).get("detections", [])
    except:
        dets = []
    os.unlink(tmp)
    return dets

def check(detections):
    """安全合规判定：helmet/mask/no-vest 三要素，全有才通过
    返回:
        status:   "pass" | "fail"
        missing:  缺失项列表，如 ["安全帽","口罩"]
        details:  逐项明细
        summary:  一句话总结
    """
    present = {}
    for d in detections:
        name = d["name"]
        if name not in present:
            present[name] = {"count": 0, "max_prob": 0}
        present[name]["count"] += 1
        present[name]["max_prob"] = max(present[name]["max_prob"], d["prob"])

    missing = []
    details = []

    for rule in RULES:
        key = rule["key"]
        label = rule["label"]
        required = rule["required"]
        found = key in present

        if required and not found:
            # 必须有但没检测到 → 缺失
            missing.append(label)
            details.append({"item": label, "status": "missing", "reason": "未检测到"})
        elif not required and found:
            # 不能有但检测到了 → 缺失（反例：检测到 no-vest = 缺反光衣）
            missing.append(label)
            details.append({"item": label, "status": "missing",
                            "reason": f"检测到 {key}，置信度 {present[key]['max_prob']:.2f}"})
        else:
            details.append({"item": label, "status": "ok",
                            "detail": f"置信度 {present.get(key,{}).get('max_prob',0):.2f}"})

    if missing:
        summary = "不合格：缺" + "、".join(missing)
    else:
        summary = "合格：安全帽、口罩、反光衣齐全"

    return {
        "status": "pass" if not missing else "fail",
        "missing": missing,
        "details": details,
        "summary": summary
    }

def check_frame(frame):
    """一步到位：对一帧图像检测并判断合规"""
    dets = detect(frame)
    result = check(dets)
    result["detections"] = dets
    return result

def check_frame_hybrid(frame):
    """混合推理：优先云端回退本地"""
    try:
        from cloud_api import check_hybrid
        return check_hybrid(frame)
    except ImportError:
        return check_frame(frame)
    return result

if __name__ == "__main__":
    # 测试：拍照 → 检测 → 合规判断
    cap = cv2.VideoCapture(0)
    input("按回车拍照并检查安全合规...")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("拍照失败")
    else:
        cv2.imwrite("safety_test.jpg", frame)
        result = check_frame(frame)
        print(f"\n{'='*40}")
        print(f"判定结果: {'✅ PASS' if result['status'] == 'pass' else '❌ FAIL'}")
        print(f"缺失项: {result['missing'] if result['missing'] else '无'}")
        print(f"\n明细:")
        for d in result['details']:
            print(f"  {'✅' if d['status'] == 'ok' else '❌'} {d['item']}: {d.get('reason', d.get('detail', ''))}")
        print(f"\n{result['summary']}")
        print(f"{'='*40}")
