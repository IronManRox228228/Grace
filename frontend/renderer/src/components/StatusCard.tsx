import { motion, AnimatePresence } from 'framer-motion';

interface StatusCardProps {
  label: string;
}

// "Understanding request…", "Closing Microsoft Edge…" — never
// "Thinking..." and never token counts. The label itself is the
// entire affordance; motion just keeps it from popping in and out.
export function StatusCard({ label }: StatusCardProps) {
  return (
    <AnimatePresence mode="wait">
      {label && (
        <motion.div
          key={label}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
          className="flex items-center gap-2 px-1"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-plum/70" />
          <span className="text-xs font-medium tracking-wide text-plum/80">{label}</span>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
