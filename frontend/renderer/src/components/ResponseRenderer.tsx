import { motion, AnimatePresence } from 'framer-motion';

interface ResponseRendererProps {
  text: string;
  speaking: boolean;
}

// Grace's spoken response, streamed in step with Kokoro's audio.
// A soft dot stands in for a "speaking" cue — no waveform, no
// equalizer bars, nothing that reads as a media player.
export function ResponseRenderer({ text, speaking }: ResponseRendererProps) {
  return (
    <AnimatePresence>
      {text && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          className="flex w-full items-start gap-2 px-1"
        >
          {speaking && (
            <motion.span
              className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-sage"
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
            />
          )}
          <p className="text-[15px] leading-relaxed text-olive">{text}</p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
