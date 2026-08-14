import { useGraceState } from './state/useGraceState';
import { GracePill } from './components/GracePill';
import { ListeningIndicator } from './components/ListeningIndicator';
import { TranscriptPanel } from './components/TranscriptPanel';
import { StatusCard } from './components/StatusCard';
import { ActionIndicator } from './components/ActionIndicator';
import { ResponseRenderer } from './components/ResponseRenderer';
import { useHoverRegion } from './shell/useHoverRegion';

export default function App() {
  const { snapshot, wake } = useGraceState();
  const expanded = snapshot.state !== 'idle';
  const hoverRef = useHoverRegion<HTMLDivElement>();

  return (
    <div className="flex h-full w-full items-end justify-center pb-2">
      <GracePill expanded={expanded} onClick={wake} hoverRef={hoverRef}>
        <div className="flex items-center gap-3">
          <ListeningIndicator active={snapshot.state === 'listening'} />
          <div className="flex-1">
            <StatusCard label={snapshot.statusLabel} />
          </div>
          <ActionIndicator active={snapshot.state === 'executing'} />
        </div>

        <TranscriptPanel userTranscript={snapshot.userTranscript} visible={expanded} />
        <ResponseRenderer text={snapshot.responseText} speaking={snapshot.state === 'speaking'} />

        {snapshot.state === 'error' && (
          <p className="px-1 text-sm text-plum">{snapshot.errorMessage}</p>
        )}
      </GracePill>
    </div>
  );
}
