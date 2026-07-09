import React from 'react';

const formatPercent = (value) => {
  if (typeof value !== 'number') return null;
  return `${Math.round(value * 100)}%`;
};

const Message = ({ message }) => {
  const isUser = message.role === 'user';
  const diagnostics = message.diagnostics;
  const confidence = diagnostics ? formatPercent(diagnostics.intentConfidence) : null;

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
              <span className="diagnostic-chip" title={diagnostics.traceId}>
                Trace {diagnostics.traceId.slice(0, 8)}
              </span>
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
