import { motion } from 'framer-motion';
import { useMemo } from 'react';

interface ChoiceOverlayProps {
  responseText: string;
  onSelectChoice?: (choice: string) => void;
}

interface ParsedChoice {
  number: string;
  label: string;
}

export function ChoiceOverlay({ responseText, onSelectChoice }: ChoiceOverlayProps) {
  const choices: ParsedChoice[] = useMemo(() => {
    if (!responseText) return [];

    // Match patterns like: "button 4 or button 7", "button 4 or 7", "4 or 7", "option 1 or option 2"
    const match = responseText.match(/(?:button|option|choice|number|#)?\s*(\d+)\s+or\s+(?:button|option|choice|number|#)?\s*(\d+)/i);
    if (match) {
      const num1 = match[1];
      const num2 = match[2];
      return [
        { number: num1, label: `Button ${num1}` },
        { number: num2, label: `Button ${num2}` },
      ];
    }

    // Match generic lists of button numbers: "button 1, button 2"
    const allButtons = Array.from(responseText.matchAll(/(?:button|option|choice)\s+(\d+)/gi));
    if (allButtons.length >= 2) {
      const unique = Array.from(new Set(allButtons.map((m) => m[1])));
      return unique.map((num) => ({
        number: num,
        label: `Button ${num}`,
      }));
    }

    return [];
  }, [responseText]);

  if (choices.length === 0) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.3 }}
      className="mt-2 flex w-full flex-col gap-2 rounded-2xl border-2 border-stone-800 bg-stone-900/95 p-3.5 text-white shadow-xl backdrop-blur-md"
      role="region"
      aria-label="Numbered marks disambiguation"
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold uppercase tracking-wider text-amber-400">
          Target Ambiguity — Select an Option
        </span>
        <span className="text-[11px] font-medium text-stone-300">
          Answer by voice or tap/key
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-3 pt-1">
        {choices.map((c) => (
          <button
            key={c.number}
            type="button"
            onClick={() => onSelectChoice?.(c.label.toLowerCase())}
            className="group flex flex-1 items-center justify-center gap-3 rounded-xl border-2 border-red-500 bg-stone-950 px-4 py-2.5 text-left transition-all hover:bg-stone-800 focus:outline-none focus-visible:ring-4 focus-visible:ring-amber-400"
            tabIndex={0}
            aria-label={`Select ${c.label}`}
          >
            {/* High-contrast Red Badge matching Set-of-Marks Screenshot Overlay */}
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-red-600 text-base font-black text-white shadow-md group-hover:scale-105">
              {c.number}
            </span>
            <span className="text-base font-bold text-white group-hover:text-amber-300">
              {c.label}
            </span>
          </button>
        ))}
      </div>
    </motion.div>
  );
}
