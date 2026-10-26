import { motion, AnimatePresence } from 'framer-motion';

interface StatusCardProps {
  label: string;
}

/**
 * The one status line: exactly the text the model or backend sent
 * ("Opening Word…"). No state badge, no hard-coded labels beside it.
 */
export function StatusCard({ label }: StatusCardProps) {
  return (
    <div className="flex w-full items-center gap-2 px-1" role="status" aria-live="polite">
      <AnimatePresence mode="wait">
        {label && (
          <motion.span
            key={label}
            initial={{ opacity: 0, x: -4 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 4 }}
            transition={{ duration: 0.2 }}
            className="truncate text-sm font-semibold tracking-normal text-stone-900"
          >
            {label}
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  );
}
