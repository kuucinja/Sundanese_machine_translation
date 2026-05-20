import os
import json
import shutil
import requests

API_KEY = "sk_ber_3p7oSMXGsls4fyOT1sG7ovRaec7oWbMEEenqt_06d0f90501dcadac"
API_URL = "https://api.berget.ai/v1/chat/completions"

# -------------------------
# FILE UTILITIES
# -------------------------

def list_files():
    files = []
    for root, dirs, filenames in os.walk("."):
        for f in filenames:
            full_path = os.path.join(root, f)
            files.append(full_path)
    return files

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def backup_file(path):
    backup_path = path + ".bak"
    shutil.copy(path, backup_path)

def write_file(path, content):
    backup_file(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Updated: {path} (backup created)")

# -------------------------
# AI CALL
# -------------------------

def ask_ai(files, task):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a code agent. "
                    "You must output ONLY valid JSON in this format:\n"
                    "{\n"
                    "  \"files\": {\n"
                    "    \"filename\": \"new full file content\"\n"
                    "  }\n"
                    "}\n"
                    "No explanations. No markdown."
                )
            },
            {
                "role": "user",
                "content": f"""
TASK:
{task}

FILES:
{json.dumps(files, indent=2)}
"""
            }
        ],
        "temperature": 0.2,
        "max_tokens": 4000
    }

    r = requests.post(API_URL, json=payload, headers=headers)
    return r.json()

def extract_ai_json(response):
    try:
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print("❌ Failed to parse AI output:", e)
        print(response)
        return None

# -------------------------
# MAIN LOOP
# -------------------------

def main():
    print("🤖 AUTONOMOUS FILE AGENT STARTED")
    print("Type /exit to stop\n")

    while True:
        task = input("\n🧠 Task: ")

        if task.strip() == "/exit":
            break

        print("\n📁 Scanning folder...")

        files = list_files()

        file_data = {}
        for f in files:
            try:
                file_data[f] = read_file(f)
            except:
                pass

        print("\n🤖 Sending to AI...")

        response = ask_ai(file_data, task)
        result = extract_ai_json(response)

        if not result:
            print("❌ No valid result")
            continue

        if "files" not in result:
            print("❌ Invalid format from AI")
            continue

        print("\n⚡ Applying changes...\n")

        for filename, content in result["files"].items():
            if os.path.exists(filename):
                write_file(filename, content)
            else:
                print(f"⚠️ Skipped missing file: {filename}")

        print("\n✅ Done.")

if __name__ == "__main__":
    main()