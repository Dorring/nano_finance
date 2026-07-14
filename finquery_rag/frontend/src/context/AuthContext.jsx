import React, { useCallback, useEffect, useState } from 'react';
import { getCurrentUser } from '../api';
import { AuthContext } from './authContextValue';

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const logout = useCallback(() => {
    localStorage.removeItem('token');
    setUser(null);
  }, []);

  const loadUser = useCallback(async () => {
    try {
      const userData = await getCurrentUser();
      setUser(userData);
    } catch (error) {
      console.error('Failed to load user:', error);
      logout();
    } finally {
      setLoading(false);
    }
  }, [logout]);

  useEffect(() => {
    // Check if user is logged in on mount
    const token = localStorage.getItem('token');
    if (token) {
      loadUser();
    } else {
      setLoading(false);
    }
  }, [loadUser]);

  const loginUser = (token, email) => {
    localStorage.setItem('token', token);
    setUser({ email });
  };

  return (
    <AuthContext.Provider value={{ user, loading, loginUser, logout, loadUser }}>
      {children}
    </AuthContext.Provider>
  );
};
