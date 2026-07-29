"""Setup script for OculiX 3.0.3 Java Visual Automation Engine in Project Grace.

Downloads oculixapi-3.0.3.jar and openpnp opencv-4.7.0-0.jar from Maven Central and verifies Java 17 installation.
"""

import os
import shutil
import subprocess
import sys
import urllib.request

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

OCULIX_VERSION = "3.0.3"
OPENCV_JAR_VERSION = "4.7.0-0"

OCULIX_MAVEN_URL = (
    f"https://repo1.maven.org/maven2/io/github/oculix-org/"
    f"oculixapi/{OCULIX_VERSION}/oculixapi-{OCULIX_VERSION}.jar"
)

OPENCV_MAVEN_URL = (
    f"https://repo1.maven.org/maven2/org/openpnp/opencv/"
    f"{OPENCV_JAR_VERSION}/opencv-{OPENCV_JAR_VERSION}.jar"
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIBS_DIR = os.path.join(PROJECT_ROOT, "libs")
OCULIX_JAR_PATH = os.path.join(LIBS_DIR, f"oculixapi-{OCULIX_VERSION}.jar")
OPENCV_JAR_PATH = os.path.join(LIBS_DIR, f"opencv-{OPENCV_JAR_VERSION}.jar")


def check_java() -> bool:
    """Check if Java 11+ is installed and on PATH."""
    java_cmd = shutil.which("java")
    if not java_cmd:
        possible_paths = [
            r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot\bin\java.exe",
            r"C:\Program Files\Java\jdk-17\bin\java.exe",
        ]
        import glob
        for pattern in possible_paths:
            matches = glob.glob(pattern)
            if matches:
                java_cmd = matches[0]
                break

    if not java_cmd:
        print("[X] Java executable not found on system PATH.")
        print("    Please install Java 17 (e.g. winget install EclipseAdoptium.Temurin.17.JDK)")
        return False

    try:
        res = subprocess.run([java_cmd, "-version"], capture_output=True, text=True)
        ver_str = res.stderr.splitlines()[0] if res.stderr else res.stdout
        print(f"[OK] Java detected: {ver_str}")
        return True
    except Exception as e:
        print(f"[X] Failed to run java: {e}")
        return False


def download_jars() -> bool:
    """Download OculiX API and OpenCV JARs from Maven Central."""
    os.makedirs(LIBS_DIR, exist_ok=True)

    # 1. Download OculiX JAR
    if not os.path.isfile(OCULIX_JAR_PATH):
        print(f"[-->] Downloading OculiX {OCULIX_VERSION} JAR from Maven Central...")
        try:
            urllib.request.urlretrieve(OCULIX_MAVEN_URL, OCULIX_JAR_PATH)
            size_mb = os.path.getsize(OCULIX_JAR_PATH) / (1024 * 1024)
            print(f"[OK] Downloaded OculiX JAR ({size_mb:.1f} MB) -> {OCULIX_JAR_PATH}")
        except Exception as e:
            print(f"[X] Failed to download OculiX JAR: {e}")
            return False
    else:
        size_mb = os.path.getsize(OCULIX_JAR_PATH) / (1024 * 1024)
        print(f"[OK] OculiX JAR exists ({size_mb:.1f} MB): {OCULIX_JAR_PATH}")

    # 2. Download OpenCV JAR
    if not os.path.isfile(OPENCV_JAR_PATH):
        print(f"[-->] Downloading OpenCV {OPENCV_JAR_VERSION} JAR from Maven Central...")
        try:
            urllib.request.urlretrieve(OPENCV_MAVEN_URL, OPENCV_JAR_PATH)
            size_mb = os.path.getsize(OPENCV_JAR_PATH) / (1024 * 1024)
            print(f"[OK] Downloaded OpenCV JAR ({size_mb:.1f} MB) -> {OPENCV_JAR_PATH}")
        except Exception as e:
            print(f"[X] Failed to download OpenCV JAR: {e}")
            return False
    else:
        size_mb = os.path.getsize(OPENCV_JAR_PATH) / (1024 * 1024)
        print(f"[OK] OpenCV JAR exists ({size_mb:.1f} MB): {OPENCV_JAR_PATH}")

    return True


def test_jpype_bridge() -> bool:
    """Verify JPype bridge initialization."""
    print("[...] Testing OculiX JPype Java bridge...")
    try:
        src_dir = os.path.join(PROJECT_ROOT, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from grace.automation.oculix_bridge import OculixBridge
        success = OculixBridge.initialize()
        if success:
            print("[OK] OculiX 3.0.3 Java Bridge is fully operational!")
            return True
        else:
            print("[!] OculiX Bridge initialization returned False. Will use OpenCV fallback.")
            return False
    except Exception as e:
        print(f"[X] JPype Bridge test error: {e}")
        return False


def main():
    print("=" * 65)
    print(f"  Project Grace - OculiX {OCULIX_VERSION} Setup")
    print("=" * 65)
    print()

    java_ok = check_java()
    jars_ok = download_jars()
    bridge_ok = test_jpype_bridge() if (java_ok and jars_ok) else False

    print()
    print("-" * 65)
    if bridge_ok:
        print("[SUCCESS] Setup Complete! OculiX is ready to provide visual precision clicks.")
    else:
        print("[INFO] OculiX setup incomplete. Project Grace will seamlessly use pure OpenCV fallback.")
    print("-" * 65)


if __name__ == "__main__":
    main()
