import { motion } from 'framer-motion';

interface ActionIndicatorProps {
  active: boolean;
}

// Three soft dots, gently and slowly breathing in sequence — enough
// to say "something is happening" without a spinner or progress bar
// pretending to know something it doesn't.
export function ActionIndicator({ active }: ActionIndicatorProps) {
  if (!active) return null;

  return (
    <div className="flex items-center gap-1 px-1">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="h-1 w-1 rounded-full bg-olive/40"
          animate={{ opacity: [0.25, 0.9, 0.25] }}
          transition={{
            duration: 1.4,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: i * 0.2,
          }}
        />
      ))}
    </div>
  );
}
