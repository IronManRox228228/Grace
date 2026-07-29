import { motion, AnimatePresence } from 'framer-motion';

interface TranscriptPanelProps {
  userTranscript: string;
  visible: boolean;
}

// Live Whisper output, streamed word by word. We append rather than
// replace the text node, so partial-transcript revisions stay smooth
// instead of flashing the whole paragraph.
export function TranscriptPanel({ userTranscript, visible }: TranscriptPanelProps) {
  return (
    <AnimatePresence>
      {visible && userTranscript && (
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          className="w-full px-1 text-[15px] leading-relaxed text-olive/70"
        >
          {userTranscript}
        </motion.p>
      )}
    </AnimatePresence>
  );
}
