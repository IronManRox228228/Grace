import { motion, AnimatePresence } from 'framer-motion';

interface TranscriptPanelProps {
  userTranscript: string;
  visible: boolean;
}

/**
 * Large live caption of what Grace heard.
 * Displays user speech in high-contrast, bold, readable typography.
 */
export function TranscriptPanel({ userTranscript, visible }: TranscriptPanelProps) {
  return (
    <AnimatePresence>
      {visible && userTranscript && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.3 }}
          className="w-full rounded-xl border border-stone-300/80 bg-white/90 px-3.5 py-2.5 shadow-xs"
          role="region"
          aria-label="User spoken transcript"
        >
          <div className="flex items-start gap-2">
            <span className="shrink-0 text-xs font-black uppercase tracking-wider text-blue-700">
              You:
            </span>
            <p className="text-base font-bold leading-snug text-stone-950">
              {userTranscript}
            </p>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
