import axios from 'axios';

// Use environment variable for API URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getApiErrorMessage = (errorOrPayload, fallback = 'Request failed') => {
  const payload = errorOrPayload?.response?.data || errorOrPayload;
  const detail = payload?.detail;

  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    return detail.message || detail.error_code || fallback;
  }
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || item.message || String(item)).join('; ');
  }
  if (typeof payload?.message === 'string') return payload.message;
  if (typeof errorOrPayload?.message === 'string') return errorOrPayload.message;
  return fallback;
};

// Add token to requests if it exists
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 errors (token expired)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    error.userMessage = getApiErrorMessage(error);
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth endpoints
export const register = async (email, password) => {
  const response = await api.post('/register', { email, password });
  return response.data;
};

export const login = async (email, password) => {
  const response = await api.post('/login', { email, password });
  return response.data;
};

export const getCurrentUser = async () => {
  const response = await api.get('/me');
  return response.data;
};

// Upload document
export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// List all documents
export const listDocuments = async () => {
  const response = await api.get('/documents');
  return response.data;
};

// List document lifecycle registry entries
export const listDocumentRegistry = async (status = null) => {
  const params = status ? { status } : undefined;
  const response = await api.get('/document-registry', { params });
  return response.data;
};

// Get one query trace
export const getQueryTrace = async (traceId) => {
  const response = await api.get(`/traces/${traceId}`);
  return response.data;
};

// Submit answer feedback for a traced response
export const submitAnswerFeedback = async (traceId, rating, comment = null) => {
  const response = await api.post('/feedback', {
    trace_id: traceId,
    rating,
    comment,
  });
  return response.data;
};

// Query documents (non-streaming)
export const queryDocuments = async (question, documentNames = null, nResults = 5) => {
  const response = await api.post('/query', {
    question,
    document_names: documentNames,
    n_results: nResults,
  });
  return response.data;
};

// Get server-side conversation memory for a session
export const getSessionHistory = async (sessionId) => {
  const response = await api.get(`/sessions/${sessionId}`);
  return response.data;
};

// Clear server-side conversation memory for a session
export const clearSession = async (sessionId) => {
  const response = await api.post('/sessions/clear', {
    question: 'Clear session',
    session_id: sessionId,
    n_results: 1,
  });
  return response.data;
};

const readErrorDetail = async (response) => {
  let text = '';
  try {
    text = await response.text();
  } catch (error) {
    console.error('Failed to read error response:', error);
  }

  if (!text) return `HTTP error: ${response.status}`;

  try {
    const payload = JSON.parse(text);
    return getApiErrorMessage(payload, `HTTP error: ${response.status}`);
  } catch {
    return text;
  }
};

// Query documents with streaming
export const queryDocumentsStream = async (question, documentNames, sessionId, nResults, onToken, onDone, onError) => {
  const token = localStorage.getItem('token');

  const response = await fetch(`${API_BASE_URL}/query/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify({
      question,
      document_names: documentNames,
      n_results: nResults,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    if (response.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    throw new Error(await readErrorDetail(response));
  }

  if (!response.body) {
    throw new Error('Streaming response body is not available');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let sawDoneEvent = false;

  const processEvent = (eventText) => {
    const dataLines = eventText
      .split('\n')
      .filter(line => line.startsWith('data: '))
      .map(line => line.slice(6));

    if (dataLines.length === 0) return;

    let data;
    try {
      data = JSON.parse(dataLines.join('\n'));
    } catch (parseError) {
      console.error('Error parsing SSE data:', parseError);
      throw new Error('Malformed streaming response');
    }

    if (data.type === 'token') {
      onToken(data.content || '');
    } else if (data.type === 'done') {
      sawDoneEvent = true;
      onDone(data);
    } else if (data.type === 'error') {
      throw new Error(getApiErrorMessage(data, data.message || 'Streaming query failed'));
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() || '';

      for (const eventText of events) {
        processEvent(eventText);
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      processEvent(buffer);
    }
    if (!sawDoneEvent) {
      throw new Error('Streaming response ended before completion');
    }
  } catch (error) {
    if (onError) onError(error);
    throw error;
  }
};

// Delete document
export const deleteDocument = async (docName) => {
  const response = await api.delete(`/documents/${docName}`);
  return response.data;
};

export default api;
