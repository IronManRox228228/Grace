import { motion, AnimatePresence } from 'framer-motion';
import { ChatMessage } from '../state/types';

interface TranscriptPanelProps {
  messages: ChatMessage[];
  speaking: boolean;
  visible: boolean;
}

/**
 * The conversation as separate blocks, in order: user, assistant, user,
 * assistant. Each message is its own block and is never merged with the one
 * before it. High-contrast, bold, readable typography.
 */
export function TranscriptPanel({ messages, speaking, visible }: TranscriptPanelProps) {
  if (!visible || messages.length === 0) return null;
  const lastIndex = messages.length - 1;
  return (
    <div className="flex w-full flex-col gap-2" role="log" aria-label="Conversation">
      <AnimatePresence initial={false}>
        {messages.map((message, index) => {
          const isUser = message.role === 'user';
          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.3 }}
              data-role={message.role}
              className={
                isUser
                  ? 'w-full rounded-xl border border-stone-300/80 bg-white/90 px-3.5 py-2.5 shadow-xs'
                  : 'flex w-full flex-col gap-1 rounded-xl border border-stone-300/80 bg-stone-50/90 px-3.5 py-2.5 shadow-xs'
              }
              role="region"
              aria-label={isUser ? 'User spoken transcript' : 'Grace assistant reply'}
            >
              <div className="flex items-start gap-2">
                <span
                  className={
                    isUser
                      ? 'shrink-0 text-xs font-black uppercase tracking-wider text-blue-700'
                      : 'shrink-0 text-xs font-black uppercase tracking-wider text-emerald-800'
                  }
                >
                  {isUser ? 'You:' : 'Grace:'}
                </span>
                <p
                  className={
                    isUser
                      ? 'text-base font-bold leading-snug text-stone-950'
                      : 'text-base font-semibold leading-relaxed text-stone-950'
                  }
                >
                  {message.text}
                </p>
                {!isUser && speaking && index === lastIndex && (
                  <motion.span
                    aria-hidden="true"
                    className="ml-auto mt-1.5 h-2 w-2 shrink-0 rounded-full bg-emerald-600"
                    animate={{ scale: [1, 1.35, 1], opacity: [0.7, 1, 0.7] }}
                    transition={{ duration: 0.8, repeat: Infinity, ease: 'easeInOut' }}
                  />
                )}
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
