import { useState, useEffect } from 'react';
import { useGraceState } from './state/useGraceState';
import { GracePill } from './components/GracePill';
import { ListeningIndicator } from './components/ListeningIndicator';
import { TranscriptPanel } from './components/TranscriptPanel';
import { StatusCard } from './components/StatusCard';
import { ActionIndicator } from './components/ActionIndicator';
import { ResponseRenderer } from './components/ResponseRenderer';
import { ChoiceOverlay } from './components/ChoiceOverlay';
import { useHoverRegion } from './shell/useHoverRegion';
import { getOfflineMode } from './shell';

export default function App() {
  const { snapshot, wake } = useGraceState();
  const expanded = snapshot.state !== 'idle';
  const hoverRef = useHoverRegion<HTMLDivElement>();
  const [isOffline, setIsOffline] = useState(true);

  useEffect(() => {
    let active = true;
    void getOfflineMode().then((offline) => {
      if (active) setIsOffline(offline);
    });
    return () => {
      active = false;
    };
  }, []);

  // Determine if Grace is waiting for user confirmation / choice
  const isWaitingForAnswer =
    Boolean(snapshot.responseText && snapshot.responseText.includes('?')) &&
    (snapshot.state === 'speaking' || snapshot.state === 'completed' || snapshot.state === 'listening');

  return (
    <div className="flex h-full w-full items-end justify-center pb-2">
      <GracePill expanded={expanded} onClick={wake} hoverRef={hoverRef}>
        <div className="flex items-center gap-3">
          <ListeningIndicator active={snapshot.state === 'listening'} />
          <div className="flex-1">
            <StatusCard
              label={snapshot.statusLabel}
              state={snapshot.state}
              isWaitingForAnswer={isWaitingForAnswer}
              isOffline={isOffline}
            />
          </div>
          <ActionIndicator active={snapshot.state === 'executing'} />
        </div>

        <TranscriptPanel userTranscript={snapshot.userTranscript} visible={expanded} />
        <ResponseRenderer text={snapshot.responseText} speaking={snapshot.state === 'speaking'} />

        {/* Numbered-marks / Choice disambiguation overlay */}
        <ChoiceOverlay
          responseText={snapshot.responseText}
          onSelectChoice={() => {
            wake();
          }}
        />

        {snapshot.state === 'error' && (
          <p className="px-1 text-sm font-semibold text-rose-700">{snapshot.errorMessage}</p>
        )}
      </GracePill>
    </div>
  );
}
