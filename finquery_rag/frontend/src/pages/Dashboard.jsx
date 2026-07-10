import React, { useState, useEffect, useCallback } from 'react';
import toast from 'react-hot-toast';
import Sidebar from '../components/Sidebar';
import ChatArea from '../components/ChatArea';
import InputBar from '../components/InputBar';
import { uploadDocument, listDocuments, listDocumentRegistry, queryDocumentsStream, deleteDocument, getSessionHistory, clearSession, getApiErrorMessage } from '../api';
import { useAuth } from '../context/useAuth';
import '../App.css';

const createSessionId = () => {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }
  return `session_${Date.now()}_${Math.random().toString(36).slice(2)}`;
};

const sessionStorageKey = (email) => `finquery_session_id:${email || 'anonymous'}`;
const RETRIEVAL_K_STORAGE_KEY = 'finquery_retrieval_k';
const RETRIEVAL_K_OPTIONS = [3, 5, 8, 12, 20];

const loadRetrievalK = () => {
  const stored = Number(localStorage.getItem(RETRIEVAL_K_STORAGE_KEY));
  return RETRIEVAL_K_OPTIONS.includes(stored) ? stored : 5;
};

function Dashboard() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [retrievalK, setRetrievalK] = useState(loadRetrievalK);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [hasProcessingDocuments, setHasProcessingDocuments] = useState(false);
  const { user, logout } = useAuth();
  const readyDocumentCount = documents.filter((doc) => (doc.status || 'ready') === 'ready').length;
  const queryDisabledReason = readyDocumentCount === 0
    ? (hasProcessingDocuments ? 'Documents are still processing. Query will be available when indexing finishes.' : 'Upload a ready PDF before asking a question.')
    : null;
  const isQueryDisabled = isLoading || readyDocumentCount === 0;


  const fetchDocuments = useCallback(async ({ silent = false } = {}) => {
    try {
      let nextDocuments;
      try {
        const registryData = await listDocumentRegistry();
        nextDocuments = registryData.documents.map((doc) => ({
          ...doc,
          name: doc.filename,
          count: doc.chunk_count,
          pages: doc.page_count,
        }));
      } catch (registryError) {
        console.warn('Document registry unavailable, falling back to /documents:', registryError);
        const data = await listDocuments();
        nextDocuments = data.documents.map((doc) => ({
          ...doc,
          status: doc.status || 'ready',
        }));
      }

      const readyNames = new Set(
        nextDocuments
          .filter((doc) => (doc.status || 'ready') === 'ready')
          .map((doc) => doc.name)
      );
      setDocuments(nextDocuments);
      setSelectedDocs((current) => current.filter((name) => readyNames.has(name)));
      setHasProcessingDocuments(
        nextDocuments.some((doc) => ['pending', 'parsing', 'indexing'].includes(doc.status))
      );
    } catch (error) {
      console.error('Error fetching documents:', error);
      if (!silent) {
        toast.error(getApiErrorMessage(error, 'Failed to load documents'));
      }
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  useEffect(() => {
    if (!hasProcessingDocuments) return undefined;

    const intervalId = window.setInterval(() => {
      fetchDocuments({ silent: true });
    }, 5000);

    return () => window.clearInterval(intervalId);
  }, [fetchDocuments, hasProcessingDocuments]);

  useEffect(() => {
    const key = sessionStorageKey(user?.email);
    const existing = localStorage.getItem(key);
    if (existing) {
      setSessionId(existing);
      return;
    }

    const next = createSessionId();
    localStorage.setItem(key, next);
    setSessionId(next);
  }, [user?.email]);

  useEffect(() => {
    if (!sessionId) return;

    let cancelled = false;

    const restoreSession = async () => {
      try {
        const data = await getSessionHistory(sessionId);
        if (cancelled) return;

        setMessages((currentMessages) => {
          if (currentMessages.length > 0) return currentMessages;
          return (data.messages || []).map((message) => {
            const metadata = message.metadata || {};
            return {
              role: message.role,
              content: message.content,
              sources: metadata.sources || message.sources || [],
              diagnostics: metadata.diagnostics || message.diagnostics || null,
            };
          });
        });
      } catch (error) {
        console.warn('Failed to restore session history:', error);
      }
    };

    restoreSession();

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const ensureSessionId = () => {
    if (sessionId) return sessionId;

    const next = createSessionId();
    localStorage.setItem(sessionStorageKey(user?.email), next);
    setSessionId(next);
    return next;
  };

  const handleUpload = async (file) => {
    if (!file.name.endsWith('.pdf')) {
      toast.error('Please upload a PDF file');
      return;
    }

    setIsUploading(true);
    const uploadToast = toast.loading(`Uploading ${file.name}...`);

    try {
      await uploadDocument(file);
      await fetchDocuments();
      toast.success(`Successfully uploaded ${file.name}`, { id: uploadToast });
    } catch (error) {
      console.error('Error uploading document:', error);
      toast.error(getApiErrorMessage(error, `Failed to upload ${file.name}`), { id: uploadToast });
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (docName) => {
    try {
      await deleteDocument(docName);
      setSelectedDocs((current) => current.filter((name) => name !== docName));
      await fetchDocuments();
      toast.success(`Deleted ${docName}`);
    } catch (error) {
      console.error('Error deleting document:', error);
      toast.error(getApiErrorMessage(error, `Failed to delete ${docName}`));
    }
  };

  const handleSelectDoc = (docName) => {
    const doc = documents.find((item) => item.name === docName);
    if (doc && (doc.status || 'ready') !== 'ready') {
      toast.error(`${docName} is still ${doc.status}; wait until it is ready`);
      return;
    }

    setSelectedDocs((current) => {
      if (current.includes(docName)) {
        return current.filter((name) => name !== docName);
      }
      toast.success(`Selected ${docName}`);
      return [...current, docName];
    });
  };

  const handleSelectAllReadyDocs = () => {
    const readyDocs = documents
      .filter((doc) => (doc.status || 'ready') === 'ready')
      .map((doc) => doc.name);

    setSelectedDocs(readyDocs);
    toast.success(`Selected ${readyDocs.length} ready document${readyDocs.length === 1 ? '' : 's'}`);
  };

  const handleClearSelectedDocs = () => {
    setSelectedDocs([]);
  };

  const handleRemoveDoc = (docName) => {
    setSelectedDocs((current) => current.filter((name) => name !== docName));
  };

  const handleRetrievalKChange = (nextValue) => {
    const nextK = Number(nextValue);
    if (!RETRIEVAL_K_OPTIONS.includes(nextK)) return;
    localStorage.setItem(RETRIEVAL_K_STORAGE_KEY, String(nextK));
    setRetrievalK(nextK);
  };

  const handleSendMessage = async (question) => {
    if (readyDocumentCount === 0) {
      toast.error(queryDisabledReason || 'No ready documents are available');
      return;
    }

    const userMessage = {
      role: 'user',
      content: question,
    };
    setMessages((prev) => [...prev, userMessage]);

    // Add empty assistant message that will be streamed into
    const assistantMessage = {
      role: 'assistant',
      content: '',
      sources: [],
      diagnostics: null,
    };
    setMessages((prev) => [...prev, assistantMessage]);
    setIsLoading(true);

    try {
      const documentNames = selectedDocs.length > 0 ? selectedDocs : null;
      const activeSessionId = ensureSessionId();

      await queryDocumentsStream(
        question,
        documentNames,
        activeSessionId,
        retrievalK,
        // onToken - append each token to the message
        (token) => {
          setMessages((prev) => {
            const lastMsg = prev[prev.length - 1];
            return [
              ...prev.slice(0, -1),
              { ...lastMsg, content: lastMsg.content + token }
            ];
          });
        },
        // onDone - add sources and diagnostic metadata when complete
        (donePayload) => {
          setMessages((prev) => {
            const lastMsg = prev[prev.length - 1];
            return [
              ...prev.slice(0, -1),
              {
                ...lastMsg,
                sources: donePayload.sources || [],
                diagnostics: {
                  traceId: donePayload.trace_id || null,
                  contextSufficient: donePayload.context_sufficient,
                  retrievalConfidence: donePayload.confidence,
                  intent: donePayload.intent || null,
                  intentConfidence: donePayload.intent_confidence,
                },
              }
            ];
          });
        }
      );
    } catch (error) {
      console.error('Error querying documents:', error);
      setMessages((prev) => {
        const updated = [...prev];
        const lastMsg = updated[updated.length - 1];
        if (!lastMsg.content) {
          lastMsg.content = error.message || 'Sorry, an error occurred while processing your question. Please try again.';
        }
        return [...updated];
      });
      toast.error(getApiErrorMessage(error, 'Failed to get response'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewSession = async () => {
    const previousSessionId = sessionId;
    const nextSessionId = createSessionId();
    localStorage.setItem(sessionStorageKey(user?.email), nextSessionId);
    setSessionId(nextSessionId);
    setMessages([]);

    if (previousSessionId) {
      try {
        await clearSession(previousSessionId);
      } catch (error) {
        console.warn('Failed to clear previous session:', error);
      }
    }

    toast.success('Started a new chat');
  };

  const handleLogout = () => {
    logout();
    toast.success('Logged out successfully');
  };

  return (
    <div className="app-container">
      <Sidebar
        documents={documents}
        selectedDocs={selectedDocs}
        onSelectDoc={handleSelectDoc}
        onUpload={handleUpload}
        onDelete={handleDelete}
        onSelectAllReadyDocs={handleSelectAllReadyDocs}
        onClearSelectedDocs={handleClearSelectedDocs}
        isUploading={isUploading}
        user={user}
        onLogout={handleLogout}
      />
      <div className="main-content">
        <ChatArea
          messages={messages}
          isLoading={isLoading}
          onExampleClick={handleSendMessage}
          sessionId={sessionId}
          retrievalK={retrievalK}
          retrievalKOptions={RETRIEVAL_K_OPTIONS}
          onRetrievalKChange={handleRetrievalKChange}
          onNewSession={handleNewSession}
          queryDisabled={isQueryDisabled}
          queryDisabledReason={queryDisabledReason}
        />
        <InputBar
          selectedDocs={selectedDocs}
          onRemoveDoc={handleRemoveDoc}
          onSendMessage={handleSendMessage}
          disabled={isQueryDisabled}
          disabledReason={queryDisabledReason}
        />
      </div>
    </div>
  );
}

export default Dashboard;
