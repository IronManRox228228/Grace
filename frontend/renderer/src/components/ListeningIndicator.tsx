import { motion } from 'framer-motion';
import { Mic } from 'lucide-react';

interface ListeningIndicatorProps {
  active: boolean;
}

// A softly glowing mic with a single gentle ripple — not a pulsing
// rainbow ring. Communicates "I'm hearing you" without demanding focus.
export function ListeningIndicator({ active }: ListeningIndicatorProps) {
  return (
    <div className="relative flex h-9 w-9 items-center justify-center shrink-0">
      {active && (
        <motion.span
          className="absolute inset-0 rounded-full bg-periwinkle/40"
          initial={{ scale: 0.8, opacity: 0.5 }}
          animate={{ scale: 1.6, opacity: 0 }}
          transition={{ duration: 1.8, repeat: Infinity, ease: 'easeOut' }}
        />
      )}
      <motion.div
        className="relative flex h-9 w-9 items-center justify-center rounded-full"
        animate={{
          backgroundColor: active ? 'rgba(204,204,255,0.55)' : 'rgba(59,68,48,0.06)',
        }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        <Mic size={16} strokeWidth={1.75} className="text-olive" />
      </motion.div>
    </div>
  );
}
