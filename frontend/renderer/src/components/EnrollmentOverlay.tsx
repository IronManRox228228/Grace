import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface EnrollmentPhraseData {
  phraseId: string;
  text: string;
  category: string;
  currentIndex: number;
  totalPhrases: number;
  currentRound: number;
}

interface EnrollmentOverlayProps {
  active?: boolean;
  phrase?: EnrollmentPhraseData | null;
  responseText?: string;
  accuracyReport?: string;
  onHearAgain?: () => void;
  onSkip?: () => void;
  onRedo?: () => void;
  onClose?: () => void;
}

export function EnrollmentOverlay({
  active = true,
  phrase,
  responseText,
  accuracyReport,
  onHearAgain,
  onSkip,
  onRedo,
  onClose,
}: EnrollmentOverlayProps) {
  const activePhrase = React.useMemo(() => {
    if (phrase) return phrase;
    if (!responseText) return null;

    // Pattern: 'Please repeat: "..."' or 'Phrase 12 of 150: "..."' or '[Enrollment: "..."]'
    const match = responseText.match(/(?:repeat|say|phrase|enrollment|listen and repeat)(?:\s+(\d+)\s+of\s+(\d+))?[:\s]+["“']([^"”']+)["”']/i);
    if (match) {
      return {
        phraseId: 'detected',
        text: match[3],
        category: 'commands',
        currentIndex: match[1] ? parseInt(match[1], 10) : 1,
        totalPhrases: match[2] ? parseInt(match[2], 10) : 100,
        currentRound: 1,
      } as EnrollmentPhraseData;
    }
    return null;
  }, [phrase, responseText]);

  if (!active || !activePhrase) {
    return null;
  }

  const categoryLabels: Record<string, string> = {
    wake_word: 'Wake Word ("Grace")',
    safety_confirmation: 'Safety & Confirmations',
    commands: 'Computer Commands',
    apps_folders: 'Apps & Folders',
    alphabet_digits: 'Letters & Numbers',
    phonetic_sentences: 'Phonetically Balanced Sentences',
    personal_words: 'Personal Words & Places',
  };

  const categoryName = categoryLabels[activePhrase.category] || activePhrase.category.replace('_', ' ');

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 8 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ duration: 0.25 }}
        className="mt-3 flex w-full flex-col gap-3 rounded-2xl border-2 border-indigo-700/60 bg-stone-900/98 p-4 text-white shadow-2xl backdrop-blur-lg"
        role="region"
        aria-label="Speech Enrollment Listen and Repeat"
      >
        {/* Header / Meta */}
        <div className="flex items-center justify-between border-b border-stone-800 pb-2">
          <div className="flex items-center gap-2">
            <span className="flex h-2.5 w-2.5 rounded-full bg-indigo-500 animate-pulse" />
            <span className="text-xs font-bold uppercase tracking-wider text-indigo-400">
              Personal Voice Enrollment
            </span>
          </div>
          <div className="flex items-center gap-3 text-xs text-stone-400">
            <span className="font-semibold text-stone-300">
              Phrase {activePhrase.currentIndex} of {activePhrase.totalPhrases}
            </span>
            <span className="rounded bg-stone-800 px-2 py-0.5 text-[11px] font-medium text-stone-300">
              Round {activePhrase.currentRound}
            </span>
            {onClose && (
              <button
                onClick={onClose}
                className="ml-1 text-stone-400 hover:text-white"
                aria-label="Pause enrollment"
              >
                ✕
              </button>
            )}
          </div>
        </div>

        {/* Category banner */}
        <div className="text-[12px] font-semibold text-indigo-300">
          Category: <span className="text-white">{categoryName}</span>
        </div>

        {/* Big On-Screen Phrase Text (Listen and Repeat) */}
        <div className="rounded-xl border border-indigo-900/50 bg-stone-950/80 p-4 text-center shadow-inner">
          <p className="text-xs uppercase tracking-wider text-stone-400">Please repeat aloud:</p>
          <p className="mt-1.5 text-2xl font-bold tracking-tight text-white select-none sm:text-3xl">
            "{activePhrase.text}"
          </p>
        </div>

        {/* Concrete Accuracy Report when available */}
        {accuracyReport && (
          <div className="rounded-lg bg-emerald-950/60 border border-emerald-800/60 px-3 py-1.5 text-xs text-emerald-300">
            <span className="font-bold">Live Accuracy: </span>
            {accuracyReport}
          </div>
        )}

        {/* Big Accessible Controls (Switch Friendly) */}
        <div className="grid grid-cols-3 gap-2 pt-1">
          <button
            type="button"
            onClick={onHearAgain}
            className="flex flex-col items-center justify-center rounded-xl border border-stone-700 bg-stone-800/90 py-2.5 px-2 text-center text-xs font-semibold text-stone-200 transition hover:bg-stone-700 active:scale-95"
          >
            <span>Hear Again</span>
            <span className="text-[10px] text-stone-400 font-normal mt-0.5">Switch: Tap</span>
          </button>

          <button
            type="button"
            onClick={onRedo}
            className="flex flex-col items-center justify-center rounded-xl border border-amber-800/70 bg-amber-950/40 py-2.5 px-2 text-center text-xs font-semibold text-amber-300 transition hover:bg-amber-900/50 active:scale-95"
          >
            <span>Redo Last</span>
            <span className="text-[10px] text-amber-400/80 font-normal mt-0.5">Switch: Long Press</span>
          </button>

          <button
            type="button"
            onClick={onSkip}
            className="flex flex-col items-center justify-center rounded-xl border border-stone-700 bg-stone-800/90 py-2.5 px-2 text-center text-xs font-semibold text-stone-200 transition hover:bg-stone-700 active:scale-95"
          >
            <span>Skip Phrase</span>
            <span className="text-[10px] text-stone-400 font-normal mt-0.5">Switch: Double Tap</span>
          </button>
        </div>

        {/* Fatigue footnote */}
        <p className="text-center text-[11px] text-stone-400">
          Sessions are 10–15 min to prevent vocal fatigue. Resumes automatically where you stopped.
        </p>
      </motion.div>
    </AnimatePresence>
  );
}
