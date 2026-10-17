import { motion, AnimatePresence } from 'framer-motion';

interface ResponseRendererProps {
  text: string;
  speaking: boolean;
}

/**
 * Large live caption of what Grace is saying.
 * Streams in step with Kokoro TTS audio with bold, high-contrast typography.
 */
export function ResponseRenderer({ text, speaking }: ResponseRendererProps) {
  return (
    <AnimatePresence>
      {text && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.3 }}
          className="flex w-full flex-col gap-1 rounded-xl border border-stone-300/80 bg-stone-50/90 px-3.5 py-2.5 shadow-xs"
          role="region"
          aria-label="Grace assistant reply"
        >
          <div className="flex items-center justify-between">
            <span className="text-xs font-black uppercase tracking-wider text-emerald-800">
              Grace:
            </span>
            {speaking && (
              <span className="flex items-center gap-1.5 text-xs font-semibold text-emerald-700">
                <motion.span
                  className="h-2 w-2 rounded-full bg-emerald-600"
                  animate={{ scale: [1, 1.35, 1], opacity: [0.7, 1, 0.7] }}
                  transition={{ duration: 0.8, repeat: Infinity, ease: 'easeInOut' }}
                />
                Speaking
              </span>
            )}
          </div>
          <p className="text-base font-semibold leading-relaxed text-stone-950">
            {text}
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
