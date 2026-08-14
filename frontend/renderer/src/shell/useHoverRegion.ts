import { useEffect, useRef } from 'react';
import { reportHoverRegion } from './index';

/**
 * Keeps the shell told about which element should be touchable.
 *
 * Attach the returned ref to the one element the pointer is allowed to reach -
 * the pill. Everything else in the overlay window stays click-through, so the
 * desktop behind it keeps working.
 *
 * The region is re-reported whenever the element is laid out at a new size,
 * which covers the pill growing into the expanded card. Framer drives the
 * in-between frames of that change with a transform rather than by resizing, so
 * for the few hundred milliseconds the spring is settling the shell is aiming
 * at the destination rectangle rather than the animating one. Pointer input
 * during the animation is not a case worth more machinery than that.
 */
export function useHoverRegion<T extends HTMLElement>() {
  const ref = useRef<T>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const report = () => reportHoverRegion(element.getBoundingClientRect());
    report();

    const observer = new ResizeObserver(report);
    observer.observe(element);
    window.addEventListener('resize', report);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', report);
      // Nothing is touchable once the pill is gone. Without this the shell
      // would keep a stale rectangle solid against the desktop.
      reportHoverRegion(null);
    };
  }, []);

  return ref;
}
