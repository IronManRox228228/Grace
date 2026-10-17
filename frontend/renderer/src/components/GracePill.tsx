import { motion } from 'framer-motion';
import { ReactNode, Ref } from 'react';

interface GracePillProps {
  expanded: boolean;
  onClick: () => void;
  hoverRef?: Ref<HTMLDivElement>;
  children?: ReactNode;
}

export function GracePill({ expanded, onClick, hoverRef, children }: GracePillProps) {
  return (
    <motion.div
      layout
      ref={hoverRef}
      tabIndex={0}
      role={!expanded ? 'button' : 'region'}
      aria-label={!expanded ? 'Activate Grace Voice Assistant (Press Enter or Space)' : 'Grace Voice Assistant'}
      onClick={!expanded ? onClick : undefined}
      onKeyDown={!expanded ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
      transition={{ type: 'spring', stiffness: 260, damping: 30, mass: 0.9 }}
      className={[
        'mx-auto flex flex-col justify-end overflow-hidden',
        'border border-stone-300/80 bg-white/95 backdrop-blur-md',
        'shadow-pill transition-shadow focus:outline-none focus-visible:ring-4 focus-visible:ring-blue-600',
        expanded ? 'w-[560px] rounded-[28px] shadow-panel' : 'w-[168px] cursor-pointer rounded-full hover:shadow-lg',
      ].join(' ')}
      style={{
        backgroundImage:
          'linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(250,248,245,0.90) 100%)',
      }}
    >
      <motion.div layout className={expanded ? 'flex flex-col gap-3 px-5 pb-5 pt-4' : 'flex h-11 items-center justify-center'}>
        {expanded ? (
          children
        ) : (
          <div className="flex items-center gap-2 px-2 text-xs font-bold text-stone-800">
            <motion.span
              className="h-2 w-2 rounded-full bg-emerald-600 shadow-xs"
              animate={{ opacity: [0.6, 1, 0.6] }}
              transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
            />
            <span>Grace • Ready</span>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}
