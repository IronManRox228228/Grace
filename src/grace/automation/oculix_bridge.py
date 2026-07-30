"""OculiX (SikuliX successor) Java Bridge for Grace.

Embeds OculiX 3.0.3's JVM-based visual automation engine inside Grace's
Python process via JPype. Provides pixel-precise template matching with
5 cascaded matching strategies, DPI awareness, and embedded Tesseract OCR.

Falls back to pure-Python OpenCV if Java/OculiX is unavailable.
"""

import atexit
import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

logger = logging.getLogger("grace.automation.oculix_bridge")


@dataclass
class VisualMatch:
    """Result of a visual search operation."""
    x: int             # Center X coordinate (screen absolute)
    y: int             # Center Y coordinate (screen absolute)
    width: int         # Matched region width
    height: int        # Matched region height
    confidence: float  # Match score 0.0 – 1.0
    method: str        # "oculix" | "oculix_ocr" | "opencv" | "ocr"
    label: str         # Human-readable label / image name / text


class OculixBridge:
    """Bridge to OculiX Java engine via JPype."""

    _jvm_started = False
    _Screen = None
    _Region = None
    _Pattern = None
    _Match = None
    _available = False

    LIBS_DIR = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "libs"
    ))

    @classmethod
    def get_classpath(cls) -> List[str]:
        """Collect all JAR paths in libs/ directory."""
        if not os.path.isdir(cls.LIBS_DIR):
            return []
        import glob
        jars = glob.glob(os.path.join(cls.LIBS_DIR, "*.jar"))
        return jars

    @classmethod
    def initialize(cls) -> bool:
        """Start the JVM and load OculiX classes. Safe to call multiple times."""
        if cls._available:
            return True
        if cls._jvm_started:
            return cls._available

        classpath = cls.get_classpath()
        if not classpath:
            logger.warning(
                f"No JAR files found in {cls.LIBS_DIR}. "
                f"Run 'python scripts/setup_oculix.py' to download them."
            )
            return False

        try:
            import glob
            import jpype
            import jpype.imports

            if not jpype.isJVMStarted():
                jvm_path = None
                try:
                    jvm_path = jpype.getDefaultJVMPath()
                except Exception:
                    pass

                if not jvm_path or not os.path.exists(jvm_path):
                    patterns = [
                        r"C:\Program Files\Eclipse Adoptium\**\server\jvm.dll",
                        r"C:\Program Files\Java\**\server\jvm.dll",
                        r"C:\Program Files (x86)\Java\**\server\jvm.dll",
                    ]
                    for pat in patterns:
                        matches = glob.glob(pat, recursive=True)
                        if matches:
                            jvm_path = matches[0]
                            jdk_root = os.path.dirname(os.path.dirname(os.path.dirname(jvm_path)))
                            os.environ["JAVA_HOME"] = jdk_root
                            break

                logger.info(f"Starting JVM ({jvm_path}) with classpath: {classpath}")
                if jvm_path:
                    jpype.startJVM(jvm_path, classpath=classpath, convertStrings=True)
                else:
                    jpype.startJVM(classpath=classpath, convertStrings=True)

                cls._jvm_started = True
                atexit.register(cls.shutdown)

            try:
                if not jpype.isThreadAttachedToJVM():
                    jpype.attachThreadToJVM()
            except Exception:
                pass

            # Import OculiX / SikuliX Java classes
            from org.sikuli.script import Screen, Region, Pattern, Match
            cls._Screen = Screen
            cls._Region = Region
            cls._Pattern = Pattern
            cls._Match = Match
            cls._available = True
            logger.info("OculiX 3.0.3 Java Bridge initialized successfully")
            return True

        except ImportError:
            logger.warning("JPype1 not installed. Run: pip install JPype1")
            return False
        except Exception as e:
            logger.error(f"OculiX initialization failed: {e}")
            cls._jvm_started = True  # Prevent repeated crash attempts
            return False

    @classmethod
    def _ensure_thread(cls):
        """Ensure current thread is attached to JVM."""
        try:
            import jpype
            if jpype.isJVMStarted() and not jpype.isThreadAttachedToJVM():
                jpype.attachThreadToJVM()
        except Exception:
            pass

    @classmethod
    def is_enabled(cls) -> bool:
        """Whether the OculiX visual fallback is switched on.

        Off by default: it starts a JVM inside the Python process, and the
        resolver tries three rendered font templates at a 2 s timeout each, so
        an unresolved click can block for seconds. With the UIA/DOM element
        graph doing the work, this path should almost never be needed.
        """
        import os
        return os.getenv("USE_OCULIX", "false").lower() in ("true", "1", "yes")

    @classmethod
    def is_available(cls) -> bool:
        """Check if OculiX is enabled, initialized and ready."""
        if not cls.is_enabled():
            return False
        if not cls._available and not cls._jvm_started:
            cls.initialize()
        return cls._available

    @classmethod
    def find(
        cls,
        image_path: str,
        region: Optional[Tuple[int, int, int, int]] = None,
        similarity: float = 0.7,
    ) -> Optional[VisualMatch]:
        """Find an image template on screen using OculiX's matching engine.

        Args:
            image_path: Absolute path to PNG template image.
            region: Optional (x, y, w, h) bounding box to restrict search.
            similarity: Minimum match confidence threshold (0.0 to 1.0).
        """
        if not cls.is_available():
            return None

        cls._ensure_thread()
        try:
            pattern = cls._Pattern(image_path).similar(float(similarity))

            if region:
                search_region = cls._Region(int(region[0]), int(region[1]), int(region[2]), int(region[3]))
            else:
                search_region = cls._Screen(0)

            match = search_region.exists(pattern, 2.0)  # 2 second timeout
            if match:
                cx = int(match.getCenter().getX())
                cy = int(match.getCenter().getY())
                w = int(match.getW())
                h = int(match.getH())
                score = float(match.getScore())
                logger.info(f"OculiX found template '{os.path.basename(image_path)}' at ({cx}, {cy}) score={score:.2f}")
                return VisualMatch(
                    x=cx, y=cy, width=w, height=h,
                    confidence=score, method="oculix",
                    label=os.path.basename(image_path)
                )
        except Exception as e:
            logger.debug(f"OculiX find exception for '{image_path}': {e}")
        return None

    @classmethod
    def find_text(
        cls,
        text: str,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[VisualMatch]:
        """Find text on screen using OculiX's embedded Tesseract OCR."""
        if not cls.is_available():
            return None

        cls._ensure_thread()
        try:
            if region:
                search_region = cls._Region(int(region[0]), int(region[1]), int(region[2]), int(region[3]))
            else:
                search_region = cls._Screen(0)

            match = search_region.existsText(text, 2.0)
            if match:
                cx = int(match.getCenter().getX())
                cy = int(match.getCenter().getY())
                w = int(match.getW())
                h = int(match.getH())
                score = float(match.getScore())
                logger.info(f"OculiX OCR found text '{text}' at ({cx}, {cy})")
                return VisualMatch(
                    x=cx, y=cy, width=w, height=h,
                    confidence=score, method="oculix_ocr",
                    label=text
                )
        except Exception as e:
            logger.debug(f"OculiX text search exception for '{text}': {e}")
        return None

    @classmethod
    def shutdown(cls):
        """Cleanly shutdown the JVM."""
        try:
            import jpype
            if jpype.isJVMStarted():
                cls._available = False
                jpype.shutdownJVM()
                logger.info("OculiX JVM shutdown complete")
        except Exception:
            pass
