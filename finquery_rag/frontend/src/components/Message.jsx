import React, { useState } from 'react';
import { getQueryTrace } from '../api';

const formatPercent = (value) => {
  if (typeof value !== 'number') return null;
  return `${Math.round(value * 100)}%`;
};

const shortText = (value, limit = 220) => {
  if (!value) return '—';
  return value.length > limit ? `${value.slice(0, limit)}...` : value;
};

const Message = ({ message }) => {
  const isUser = message.role === 'user';
  const diagnostics = message.diagnostics;
  const confidence = diagnostics ? formatPercent(diagnostics.intentConfidence) : null;
  const [copiedTraceId, setCopiedTraceId] = useState(false);
  const [traceDetails, setTraceDetails] = useState(null);
  const [isTraceOpen, setIsTraceOpen] = useState(false);
  const [isTraceLoading, setIsTraceLoading] = useState(false);
  const [traceError, setTraceError] = useState(null);

  const handleCopyTraceId = async () => {
    if (!diagnostics?.traceId) return;

    try {
      await navigator.clipboard.writeText(diagnostics.traceId);
      setCopiedTraceId(true);
      window.setTimeout(() => setCopiedTraceId(false), 1600);
    } catch (error) {
      console.error('Failed to copy trace ID:', error);
    }
  };

  const handleToggleTraceDetails = async () => {
    if (!diagnostics?.traceId) return;
    if (traceDetails || traceError) {
      setIsTraceOpen(!isTraceOpen);
      return;
    }

    setIsTraceLoading(true);
    setTraceError(null);
    setIsTraceOpen(true);

    try {
      const data = await getQueryTrace(diagnostics.traceId);
      setTraceDetails(data.trace);
    } catch (error) {
      console.error('Failed to load trace details:', error);
      setTraceError('Trace details unavailable');
    } finally {
      setIsTraceLoading(false);
    }
  };

  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-content">
        {!isUser && (
          <div className="message-sources">
            FinQuery
          </div>
        )}
        <div style={{ whiteSpace: 'pre-wrap' }}>{message.content}</div>
        {!isUser && diagnostics && (
          <>
            <div className="message-diagnostics" aria-label="Answer diagnostics">
              {diagnostics.traceId && (
                <>
                  <button
                    type="button"
                    className="diagnostic-chip trace-chip"
                    title={`Copy full trace ID: ${diagnostics.traceId}`}
                    onClick={handleCopyTraceId}
                  >
                    {copiedTraceId ? 'Trace copied' : `Trace ${diagnostics.traceId.slice(0, 8)}`}
                  </button>
                  <button
                    type="button"
                    className="diagnostic-chip trace-chip"
                    onClick={handleToggleTraceDetails}
                  >
                    {isTraceOpen ? 'Hide details' : 'Details'}
                  </button>
                </>
              )}
              {typeof diagnostics.contextSufficient === 'boolean' && (
                <span className={`diagnostic-chip ${diagnostics.contextSufficient ? 'ok' : 'warn'}`}>
                  Context {diagnostics.contextSufficient ? 'sufficient' : 'weak'}
                </span>
              )}
              {diagnostics.intent && (
                <span className="diagnostic-chip">
                  Intent {diagnostics.intent}{confidence ? ` · ${confidence}` : ''}
                </span>
              )}
            </div>
            {isTraceOpen && (
              <div className="trace-details">
                {isTraceLoading && <div className="trace-muted">Loading trace details...</div>}
                {traceError && <div className="trace-error">{traceError}</div>}
                {traceDetails && (
                  <>
                    <div className="trace-row">
                      <span>Question</span>
                      <p>{shortText(traceDetails.query_original)}</p>
                    </div>
                    {traceDetails.query_rewritten && (
                      <div className="trace-row">
                        <span>Rewritten</span>
                        <p>{shortText(traceDetails.query_rewritten)}</p>
                      </div>
                    )}
                    <div className="trace-row trace-grid">
                      <div>
                        <span>Intent</span>
                        <p>{traceDetails.intent || '—'}</p>
                      </div>
                      <div>
                        <span>Latency</span>
                        <p>{typeof traceDetails.latency_ms === 'number' ? `${Math.round(traceDetails.latency_ms)} ms` : '—'}</p>
                      </div>
                    </div>
                    {traceDetails.sources?.length > 0 && (
                      <div className="trace-row">
                        <span>Sources</span>
                        <p>{traceDetails.sources.map((source) => source.filename || source.doc_name || 'source').join(', ')}</p>
                      </div>
                    )}
                    {traceDetails.error_message && (
                      <div className="trace-row trace-error">
                        <span>Error</span>
                        <p>{traceDetails.error_message}</p>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default Message;
