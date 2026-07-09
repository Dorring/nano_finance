import React, { useState } from 'react';

const formatPercent = (value) => {
  if (typeof value !== 'number') return null;
  return `${Math.round(value * 100)}%`;
};

const Message = ({ message }) => {
  const isUser = message.role === 'user';
  const diagnostics = message.diagnostics;
  const confidence = diagnostics ? formatPercent(diagnostics.intentConfidence) : null;
  const [copiedTraceId, setCopiedTraceId] = useState(false);

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
          <div className="message-diagnostics" aria-label="Answer diagnostics">
            {diagnostics.traceId && (
              <button
                type="button"
                className="diagnostic-chip trace-chip"
                title={`Copy full trace ID: ${diagnostics.traceId}`}
                onClick={handleCopyTraceId}
              >
                {copiedTraceId ? 'Trace copied' : `Trace ${diagnostics.traceId.slice(0, 8)}`}
              </button>
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
        )}
      </div>
    </div>
  );
};

export default Message;
