import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { compareEvaluationReports, getApiErrorMessage, scoreEvaluationReport } from '../api';

const readTextFile = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(String(reader.result || ''));
  reader.onerror = () => reject(reader.error || new Error('Failed to read file'));
  reader.readAsText(file);
});

const parseJsonOrJsonl = (text, label) => {
  const trimmed = text.trim();
  if (!trimmed) throw new Error(`${label} file is empty`);
  if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
    return JSON.parse(trimmed);
  }
  return trimmed
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`${label} JSONL line ${index + 1} is invalid: ${error.message}`);
      }
    });
};

const normalizeRows = (payload, label) => {
  if (Array.isArray(payload)) return payload;
  if (payload && Array.isArray(payload.cases)) return payload.cases;
  if (payload && Array.isArray(payload.predictions)) return payload.predictions;
  throw new Error(`${label} must be a JSON array, JSONL file, or object with cases/predictions`);
};

const normalizeReport = (payload, label) => {
  if (payload && typeof payload === 'object' && !Array.isArray(payload) && payload.summary) {
    return payload;
  }
  throw new Error(`${label} must be an evaluation report JSON object`);
};

const downloadJson = (payload, filename) => {
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
};

const formatPercent = (value) => {
  if (typeof value !== 'number') return '-';
  return `${Math.round(value * 100)}%`;
};

const EvalWorkbench = () => {
  const [caseFile, setCaseFile] = useState(null);
  const [predictionFile, setPredictionFile] = useState(null);
  const [baselineFile, setBaselineFile] = useState(null);
  const [candidateFile, setCandidateFile] = useState(null);
  const [tolerance, setTolerance] = useState('0');
  const [scoreReport, setScoreReport] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [isScoring, setIsScoring] = useState(false);
  const [isComparing, setIsComparing] = useState(false);

  const handleScore = async () => {
    if (!caseFile || !predictionFile) {
      toast.error('Choose cases and predictions files first');
      return;
    }
    setIsScoring(true);
    try {
      const [casesText, predictionsText] = await Promise.all([
        readTextFile(caseFile),
        readTextFile(predictionFile),
      ]);
      const cases = normalizeRows(parseJsonOrJsonl(casesText, 'Cases'), 'Cases');
      const predictions = normalizeRows(parseJsonOrJsonl(predictionsText, 'Predictions'), 'Predictions');
      const report = await scoreEvaluationReport({ cases, predictions });
      setScoreReport(report);
      toast.success(`Scored ${report.summary?.scored_cases || 0} evaluation case${report.summary?.scored_cases === 1 ? '' : 's'}`);
    } catch (error) {
      console.error('Failed to score evaluation report:', error);
      toast.error(getApiErrorMessage(error, error.message || 'Failed to score evaluation report'));
    } finally {
      setIsScoring(false);
    }
  };

  const handleCompare = async () => {
    if (!baselineFile || !candidateFile) {
      toast.error('Choose baseline and candidate reports first');
      return;
    }
    setIsComparing(true);
    try {
      const [baselineText, candidateText] = await Promise.all([
        readTextFile(baselineFile),
        readTextFile(candidateFile),
      ]);
      const baseline = normalizeReport(parseJsonOrJsonl(baselineText, 'Baseline'), 'Baseline');
      const candidate = normalizeReport(parseJsonOrJsonl(candidateText, 'Candidate'), 'Candidate');
      const result = await compareEvaluationReports({
        baseline,
        candidate,
        regressionTolerance: Number(tolerance || 0),
      });
      setComparison(result);
      toast.success(result.passed ? 'Evaluation comparison passed' : 'Evaluation comparison found regressions');
    } catch (error) {
      console.error('Failed to compare evaluation reports:', error);
      toast.error(getApiErrorMessage(error, error.message || 'Failed to compare evaluation reports'));
    } finally {
      setIsComparing(false);
    }
  };

  return (
    <div className="eval-workbench" aria-label="Evaluation workbench">
      <div className="eval-workbench-header">
        <div>
          <div className="eval-workbench-title">Evaluation workbench</div>
          <div className="eval-workbench-subtitle">Score JSON/JSONL fixtures and compare reports locally through authenticated APIs.</div>
        </div>
      </div>

      <div className="eval-workbench-grid">
        <div className="eval-card">
          <div className="eval-card-title">Score cases</div>
          <label>
            Cases JSON/JSONL
            <input type="file" accept=".json,.jsonl,application/json" onChange={(event) => setCaseFile(event.target.files?.[0] || null)} />
          </label>
          <label>
            Predictions JSON/JSONL
            <input type="file" accept=".json,.jsonl,application/json" onChange={(event) => setPredictionFile(event.target.files?.[0] || null)} />
          </label>
          <div className="eval-actions">
            <button type="button" onClick={handleScore} disabled={isScoring}>
              {isScoring ? 'Scoring...' : 'Score'}
            </button>
            <button type="button" onClick={() => downloadJson(scoreReport, 'finquery-eval-report.json')} disabled={!scoreReport}>
              Download report
            </button>
          </div>
          {scoreReport && (
            <div className="eval-result">
              <span>Pass rate {formatPercent(scoreReport.summary?.pass_rate)}</span>
              <span>{scoreReport.summary?.scored_cases || 0}/{scoreReport.summary?.total_cases || 0} scored</span>
              <span>{scoreReport.summary?.missing_predictions || 0} missing</span>
            </div>
          )}
        </div>

        <div className="eval-card">
          <div className="eval-card-title">Compare reports</div>
          <label>
            Baseline report JSON
            <input type="file" accept=".json,application/json" onChange={(event) => setBaselineFile(event.target.files?.[0] || null)} />
          </label>
          <label>
            Candidate report JSON
            <input type="file" accept=".json,application/json" onChange={(event) => setCandidateFile(event.target.files?.[0] || null)} />
          </label>
          <label>
            Regression tolerance
            <input type="number" min="0" max="1" step="0.01" value={tolerance} onChange={(event) => setTolerance(event.target.value)} />
          </label>
          <div className="eval-actions">
            <button type="button" onClick={handleCompare} disabled={isComparing}>
              {isComparing ? 'Comparing...' : 'Compare'}
            </button>
            <button type="button" onClick={() => downloadJson(comparison, 'finquery-eval-comparison.json')} disabled={!comparison}>
              Download comparison
            </button>
          </div>
          {comparison && (
            <div className={`eval-result ${comparison.passed ? 'ok' : 'warn'}`}>
              <span>{comparison.passed ? 'Passed' : 'Regressions found'}</span>
              <span>{comparison.regressions?.length || 0} metric regressions</span>
              <span>{comparison.newly_failed?.length || 0} newly failed</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EvalWorkbench;