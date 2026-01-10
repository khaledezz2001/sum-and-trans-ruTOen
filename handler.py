import base64
import requests
import time

# ===============================
# CONFIG
# ===============================
API_KEY = "YOUR_RUNPOD_API_KEY"
ENDPOINT_ID = "YOUR_ENDPOINT_ID"

RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


# ===============================
# HELPERS
# ===============================
def to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def submit_image(path: str) -> str:
    payload = {
        "input": {
            "image": to_base64(path)
        }
    }
    r = requests.post(RUN_URL, json=payload, headers=HEADERS)
    r.raise_for_status()
    return r.json()["id"]


def submit_pdf(path: str) -> str:
    payload = {
        "input": {
            "file": to_base64(path)
        }
    }
    r = requests.post(RUN_URL, json=payload, headers=HEADERS)
    r.raise_for_status()
    return r.json()["id"]


def wait_for_result(job_id: str, poll_seconds: int = 5):
    while True:
        r = requests.get(f"{STATUS_URL}/{job_id}", headers=HEADERS)
        r.raise_for_status()
        data = r.json()

        status = data.get("status")
        print(f"[STATUS] {status}")

        if status == "COMPLETED":
            return data["output"]

        if status == "FAILED":
            raise RuntimeError(data)

        time.sleep(poll_seconds)


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    # ---- CHOOSE ONE ----
    # job_id = submit_image("test.png")
    job_id = submit_pdf("document.pdf")

    print(f"[JOB ID] {job_id}")
    result = wait_for_result(job_id)

    print("\n===== FINAL RESULT =====\n")
    print(result)
