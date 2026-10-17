import { motion, AnimatePresence } from 'framer-motion';
import { GraceState } from '../state/types';

interface StatusCardProps {
  label: string;
  state?: GraceState;
  isWaitingForAnswer?: boolean;
  isOffline?: boolean;
}

export function StatusCard({ label, state = 'idle', isWaitingForAnswer = false, isOffline = true }: StatusCardProps) {
  // Determine user-friendly state descriptor
  let stateTitle = 'Ready';
  let badgeClasses = 'bg-emerald-100 text-emerald-900 border-emerald-300';
  let dotClasses = 'bg-emerald-500';

  if (state === 'listening') {
    stateTitle = 'Listening';
    badgeClasses = 'bg-blue-100 text-blue-900 border-blue-300';
    dotClasses = 'bg-blue-600 animate-pulse';
  } else if (state === 'understanding') {
    stateTitle = 'Thinking';
    badgeClasses = 'bg-amber-100 text-amber-900 border-amber-300';
    dotClasses = 'bg-amber-600 animate-pulse';
  } else if (state === 'executing') {
    stateTitle = 'Acting';
    badgeClasses = 'bg-purple-100 text-purple-900 border-purple-300';
    dotClasses = 'bg-purple-600 animate-pulse';
  } else if (isWaitingForAnswer) {
    stateTitle = 'Waiting for your answer';
    badgeClasses = 'bg-orange-100 text-orange-950 border-orange-300';
    dotClasses = 'bg-orange-500 animate-pulse';
  } else if (state === 'error') {
    stateTitle = 'Error';
    badgeClasses = 'bg-rose-100 text-rose-950 border-rose-300';
    dotClasses = 'bg-rose-600';
  }

  const plainLabel = label || (state === 'idle' ? 'Ready for your command' : '');

  return (
    <div className="flex w-full items-center justify-between gap-2 px-1">
      <div className="flex min-w-0 items-center gap-2">
        {/* Always-visible high-contrast state badge */}
        <span
          className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-bold tracking-wide shadow-xs ${badgeClasses}`}
          role="status"
          aria-live="polite"
        >
          <span className={`h-2 w-2 rounded-full ${dotClasses}`} />
          {stateTitle}
        </span>

        {/* Current step in plain words ("Opening Word…") */}
        <AnimatePresence mode="wait">
          {plainLabel && (
            <motion.span
              key={plainLabel}
              initial={{ opacity: 0, x: -4 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 4 }}
              transition={{ duration: 0.2 }}
              className="truncate text-sm font-semibold tracking-normal text-stone-900"
            >
              {plainLabel}
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* Verified True Offline Indicator (R19) */}
      {isOffline && (
        <span
          className="inline-flex shrink-0 items-center gap-1 rounded-md border border-stone-300 bg-white/80 px-2 py-0.5 text-[11px] font-bold text-stone-700 shadow-xs"
          title="Strict Offline Mode: Zero non-loopback network connections"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-600" />
          Offline
        </span>
      )}
    </div>
  );
}
