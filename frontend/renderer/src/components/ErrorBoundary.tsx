import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[Grace] Uncaught Error caught by ErrorBoundary:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[60px] p-2 bg-red-950/80 text-red-200 border border-red-500/50 rounded-lg text-xs font-sans">
          <p className="font-semibold">Something went wrong in the interface.</p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="mt-1 px-2 py-0.5 bg-red-800 hover:bg-red-700 text-white rounded text-[11px]"
          >
            Recover
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
