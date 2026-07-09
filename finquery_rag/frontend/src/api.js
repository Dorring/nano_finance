import axios from 'axios';

// Use environment variable for API URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

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

// Query documents (non-streaming)
export const queryDocuments = async (question, documentNames = null) => {
  const response = await api.post('/query', {
    question,
    document_names: documentNames,
    n_results: 5,
  });
  return response.data;
};

// Query documents with streaming
export const queryDocumentsStream = async (question, documentNames, onToken, onDone, onError) => {
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
      n_results: 5,
    }),
  });

  if (!response.ok) {
    if (response.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    throw new Error(`HTTP error: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const processEvent = (eventText) => {
    const dataLines = eventText
      .split('\n')
      .filter(line => line.startsWith('data: '))
      .map(line => line.slice(6));

    if (dataLines.length === 0) return;

    try {
      const data = JSON.parse(dataLines.join('\n'));
      if (data.type === 'token') {
        onToken(data.content);
      } else if (data.type === 'done') {
        onDone(data);
      }
    } catch (parseError) {
      console.error('Error parsing SSE data:', parseError);
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
