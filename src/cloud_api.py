"""GLM-4V 云端视觉检测 API"""
import os, json, base64, cv2, time, requests

GLM_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

def _load_key():
    key = os.environ.get("GLM_API_KEY", "")
    if not key:
        key_file = os.path.join(os.path.dirname(__file__), ".glm_key")
        if os.path.exists(key_file):
            key = open(key_file).read().strip()
    return key

API_KEY = _load_key()
TIMEOUT = 8

PROMPT = '检查图中人员是否佩戴安全帽、佩戴口罩、穿着反光衣。只返回JSON：{"helmet":true/false,"mask":true/false,"vest":true/false,"person":true/false}'

def _encode_frame(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode()

def check_cloud(frame):
    """云端推理，返回与 safety_check.check() 相同格式"""
    if not API_KEY:
        print("[cloud] no API key")
        return None
    b64 = _encode_frame(frame)
    body = {
        "model": "glm-4v-plus",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        "temperature": 0.1, "max_tokens": 100
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    t0 = time.time()
    try:
        resp = requests.post(GLM_ENDPOINT, headers=headers, json=body, timeout=TIMEOUT)
    except Exception as e:
        print(f"[cloud] request error: {e}")
        return None
    elapsed = time.time() - t0
    if resp.status_code != 200:
        print(f"[cloud] HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    content = resp.json()["choices"][0]["message"]["content"]
    print(f"[cloud] {elapsed:.1f}s -> {content.strip()}")
    try:
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        r = json.loads(content.strip())
    except Exception:
        print(f"[cloud] parse failed: {content}")
        return None
    hp = r.get("person", False)
    hm = r.get("helmet", False)
    hw_mask = r.get("mask", False)
    hv = r.get("vest", False)
    missing = []
    if hp and not hm:
        missing.append("安全帽")
    if hp and not hw_mask:
        missing.append("口罩")
    if hp and not hv:
        missing.append("反光衣")
    return {
        "status": "pass" if not missing else "fail",
        "missing": missing,
        "summary": "合格" if not missing else "不合格：缺" + "、".join(missing),
        "backend": "cloud",
        "details": [
            {"item": "安全帽", "status": "ok" if hm else "missing"},
            {"item": "口罩", "status": "ok" if hw_mask else "missing"},
            {"item": "反光衣", "status": "ok" if hv else "missing"},
        ]
    }

def is_online():
    try:
        r = requests.head("https://open.bigmodel.cn", timeout=3)
        return r.status_code < 500
    except Exception:
        return False

def check_hybrid(frame):
    if is_online():
        result = check_cloud(frame)
        if result is not None:
            return result
        print("[hybrid] cloud failed, fallback to local")
    from safety_check import detect, check
    dets = detect(frame)
    result = check(dets)
    result["backend"] = "local"
    return result

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    input("按回车拍照并云端检测...")
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite("cloud_test.jpg", frame)
        result = check_cloud(frame) if is_online() else check_hybrid(frame)
        print(json.dumps(result, ensure_ascii=False, indent=2))
