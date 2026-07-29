import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface GracePillProps {
  expanded: boolean;
  onClick: () => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
  children?: ReactNode;
}

// Idle: a small frosted glyph, barely there.
// Expanded: the same surface grows upward into a conversation card.
// Framer's `layout` animation handles the shape change with one
// physically believable spring — no separate "modal" ever mounts.
export function GracePill({ expanded, onClick, onMouseEnter, onMouseLeave, children }: GracePillProps) {
  return (
    <motion.div
      layout
      onClick={!expanded ? onClick : undefined}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      transition={{ type: 'spring', stiffness: 260, damping: 30, mass: 0.9 }}
      className={[
        'mx-auto flex flex-col justify-end overflow-hidden',
        'border border-white/60 bg-cream/70 backdrop-blur-xs',
        'shadow-pill',
        expanded ? 'w-[540px] rounded-[28px] shadow-panel' : 'w-[132px] cursor-pointer rounded-full',
      ].join(' ')}
      style={{
        // Frosted ceramic, not holographic glass: a faint warm gradient
        // instead of heavy blur or saturation.
        backgroundImage:
          'linear-gradient(180deg, rgba(255,255,255,0.55) 0%, rgba(253,251,247,0.65) 100%)',
      }}
    >
      <motion.div layout className={expanded ? 'flex flex-col gap-3 px-5 pb-5 pt-4' : 'flex h-11 items-center justify-center'}>
        {expanded ? (
          children
        ) : (
          <motion.span
            className="h-2 w-2 rounded-full"
            style={{
              background: 'radial-gradient(circle at 35% 30%, #E3E3FF, #CCCCFF 60%, #705553)',
            }}
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          />
        )}
      </motion.div>
    </motion.div>
  );
}
