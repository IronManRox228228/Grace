"""Setup script for OculiX 3.0.3 Java Visual Automation Engine in Project Grace.

Downloads oculixapi-3.0.3.jar and openpnp opencv-4.7.0-0.jar from Maven Central and verifies Java 17 installation.
"""

import os
import shutil
import subprocess
import sys

from verified_download import HashMismatch, download_and_verify, verify_sha256

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

# Pinned SHA-256 for each JAR (R32/theme 3: these used to be pulled from Maven
# Central with no integrity check at all, into the same process that issues
# real clicks). Computed from the copies already present in this repo's
# libs/ directory, matching the exact versions pinned above - not invented.
OCULIX_JAR_SHA256 = "d1e8c2e9290550abb1c6c96ded791eac95f7bcce911cc7f839b70cab75f83cea"
OPENCV_JAR_SHA256 = "f022c042faad7e2fc1d4fd5fb181929f410b9a5f9da910da158e0199fd213f3b"


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


def _ensure_jar(label: str, url: str, path: str, expected_sha256: str) -> bool:
    """Verify a JAR already on disk, or download and verify a fresh one.

    Every run re-checks a JAR that already exists, not just a freshly
    downloaded one - a compromised JAR left over from an earlier, unverified
    run must not keep being trusted just because it is already present (R32).
    """
    if os.path.isfile(path):
        try:
            verify_sha256(path, expected_sha256)
        except HashMismatch as e:
            print(f"[X] {label} JAR on disk failed verification: {e}")
            return False
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[OK] {label} JAR verified ({size_mb:.1f} MB): {path}")
        return True

    print(f"[-->] Downloading {label} JAR from Maven Central...")
    try:
        download_and_verify(url, path, expected_sha256, progress=False)
    except HashMismatch as e:
        print(f"[X] {label} JAR failed verification after download: {e}")
        return False
    except Exception as e:
        print(f"[X] Failed to download {label} JAR: {e}")
        return False
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[OK] Downloaded {label} JAR ({size_mb:.1f} MB) -> {path}")
    return True


def download_jars() -> bool:
    """Download (or verify already-present) OculiX and OpenCV JARs."""
    os.makedirs(LIBS_DIR, exist_ok=True)

    oculix_ok = _ensure_jar("OculiX", OCULIX_MAVEN_URL, OCULIX_JAR_PATH, OCULIX_JAR_SHA256)
    opencv_ok = _ensure_jar("OpenCV", OPENCV_MAVEN_URL, OPENCV_JAR_PATH, OPENCV_JAR_SHA256)
    return oculix_ok and opencv_ok


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
