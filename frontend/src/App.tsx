import { ChangeEvent, FormEvent, useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  AlgorithmRecommendation,
  api,
  AuditLog,
  Dataset,
  DatasetDetail,
  DatasetProfile,
  DeploymentRecord,
  DriftReport,
  ExplanationResponse,
  Job,
  Model,
  ModelApproval,
  ModelRankingItem,
  PerformanceReport,
  RecommendationResponse,
  RetrainingJob,
  RetrainingRule,
  TrainingBody,
} from './api'

type View =
  | 'overview'
  | 'datasets'
  | 'recommendations'
  | 'train'
  | 'comparisons'
  | 'explainability'
  | 'monitoring'
  | 'retraining'
  | 'governance'
  | 'deployments'
  | 'automation'

const nav: [View, string, string][] = [
  ['overview', 'Overview', '◈'],
  ['datasets', 'Dataset Intelligence', '📊'],
  ['recommendations', 'Model Recommendations', '💡'],
  ['train', 'Train & Experiments', '⚡'],
  ['comparisons', 'Model Comparison', '⚖️'],
  ['explainability', 'Model Explainability', '🔍'],
  ['monitoring', 'Drift & Performance', '🛡️'],
  ['retraining', 'Automated Retraining', '🔄'],
  ['governance', 'Governance & Approvals', '📋'],
  ['deployments', 'Deployment & Rollback', '🚀'],
  ['automation', 'Automation Center', '⚙️'],
]

export default function App() {
  const [view, setView] = useState<View>('overview')
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [approvals, setApprovals] = useState<ModelApproval[]>([])
  const [deployments, setDeployments] = useState<DeploymentRecord[]>([])
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)

  const refresh = async () => {
    try {
      const [d, j, m, a, dep] = await Promise.all([
        api.datasets().catch(() => []),
        api.jobs().catch(() => []),
        api.models().catch(() => []),
        api.approvals().catch(() => []),
        api.deployments().catch(() => []),
      ])
      setDatasets(d)
      setJobs(j)
      setModels(m)
      setApprovals(a)
      setDeployments(dep)
    } catch (e) {
      setMessage((e as Error).message)
    }
  }

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 5000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="shell">
      <aside>
        <div className="brand">
          <span>◈</span> SentinelML
        </div>
        <p className="tag">Intelligent MLOps & Governance</p>
        {nav.map(([id, label, icon]) => (
          <button
            className={view === id ? 'nav active' : 'nav'}
            key={id}
            onClick={() => setView(id)}
          >
            <span>{icon}</span> {label}
          </button>
        ))}
        <div className="aside-bottom">
          SentinelML 2.0 (Phase 3)
          <br />
          MLflow & Celery Engine
        </div>
      </aside>

      <main>
        <header>
          <div>
            <p className="eyebrow">MLOPS WORKSPACE</p>
            <h1>{nav.find(([id]) => id === view)?.[1]}</h1>
          </div>
          <button className="refresh" onClick={refresh}>
            ↻ Refresh
          </button>
        </header>

        {message && (
          <div className="notice">
            <span>{message}</span>
            <button onClick={() => setMessage('')}>×</button>
          </div>
        )}

        {view === 'overview' && (
          <Overview
            datasets={datasets}
            jobs={jobs}
            models={models}
            approvals={approvals}
            deployments={deployments}
            setView={setView}
          />
        )}
        {view === 'datasets' && (
          <Datasets
            datasets={datasets}
            onChanged={refresh}
            announce={setMessage}
            busy={busy}
            setBusy={setBusy}
          />
        )}
        {view === 'recommendations' && (
          <Recommendations
            datasets={datasets}
            onTrainApplied={() => setView('train')}
            announce={setMessage}
          />
        )}
        {view === 'train' && (
          <Train
            datasets={datasets}
            jobs={jobs}
            onStarted={() => {
              refresh()
              setView('train')
            }}
            announce={setMessage}
          />
        )}
        {view === 'comparisons' && <Comparisons />}
        {view === 'explainability' && (
          <Explainability models={models} datasets={datasets} announce={setMessage} />
        )}
        {view === 'monitoring' && <Monitoring models={models} datasets={datasets} announce={setMessage} />}
        {view === 'retraining' && <Retraining models={models} datasets={datasets} announce={setMessage} />}
        {view === 'governance' && (
          <Governance
            approvals={approvals}
            models={models}
            onRefresh={refresh}
            announce={setMessage}
          />
        )}
        {view === 'deployments' && (
          <Deployments
            deployments={deployments}
            models={models}
            onRefresh={refresh}
            announce={setMessage}
          />
        )}
        {view === 'automation' && <Automation announce={setMessage} />}
      </main>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const s = status.toLowerCase()
  return <span className={`status ${s}`}>{s}</span>
}

function Empty({ text }: { text: string }) {
  return <p className="empty">{text}</p>
}

// ----------------- 1. OVERVIEW -----------------
function Overview({
  datasets,
  jobs,
  models,
  approvals,
  deployments,
  setView,
}: {
  datasets: Dataset[]
  jobs: Job[]
  models: Model[]
  approvals: ModelApproval[]
  deployments: DeploymentRecord[]
  setView: (v: View) => void
}) {
  const completed = jobs.filter((x) => x.status === 'COMPLETED').length
  const pendingApprovals = approvals.filter((x) => x.status === 'PENDING').length

  return (
    <>
      <section className="stats">
        <div className="stat">
          <p>Datasets</p>
          <strong>{datasets.length}</strong>
          <small>Profiling & Intelligence ready</small>
        </div>
        <div className="stat">
          <p>Training Runs</p>
          <strong>{jobs.length}</strong>
          <small>{completed} completed runs</small>
        </div>
        <div className="stat">
          <p>Registered Models</p>
          <strong>{models.length}</strong>
          <small>MLflow tracking registry</small>
        </div>
        <div className="stat">
          <p>Pending Approvals</p>
          <strong style={{ color: pendingApprovals > 0 ? '#b54708' : 'inherit' }}>
            {pendingApprovals}
          </strong>
          <small>Human-in-the-loop governance</small>
        </div>
      </section>

      <div className="twocol">
        <section className="panel">
          <div className="panel-head">
            <h2>Recent Training Jobs</h2>
            <button className="btn-sm" onClick={() => setView('train')}>
              View All
            </button>
          </div>
          {jobs.length ? (
            <table>
              <thead>
                <tr>
                  <th>Job</th>
                  <th>Target</th>
                  <th>Status</th>
                  <th>Best Algorithm</th>
                </tr>
              </thead>
              <tbody>
                {jobs.slice(0, 5).map((job) => (
                  <tr key={job.id}>
                    <td>#{job.id}</td>
                    <td>{job.target_column}</td>
                    <td>
                      <StatusBadge status={job.status} />
                    </td>
                    <td>{job.best_algorithm ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty text="Upload a CSV dataset to initiate training." />
          )}
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Pending Governance Approvals</h2>
            <button className="btn-sm" onClick={() => setView('governance')}>
              Review
            </button>
          </div>
          {approvals.filter((a) => a.status === 'PENDING').length ? (
            <table>
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Version</th>
                  <th>Current Prod</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {approvals
                  .filter((a) => a.status === 'PENDING')
                  .slice(0, 5)
                  .map((a) => (
                    <tr key={a.id}>
                      <td>{a.candidate_model_name}</td>
                      <td>v{a.candidate_version}</td>
                      <td>{a.current_production_version ? `v${a.current_production_version}` : 'None'}</td>
                      <td>
                        <StatusBadge status={a.status} />
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          ) : (
            <Empty text="No candidates currently pending approval." />
          )}
        </section>
      </div>
    </>
  )
}

// ----------------- 2. DATASETS & INTELLIGENCE -----------------
function Datasets({
  datasets,
  onChanged,
  announce,
  busy,
  setBusy,
}: {
  datasets: Dataset[]
  onChanged: () => Promise<void>
  announce: (x: string) => void
  busy: boolean
  setBusy: (x: boolean) => void
}) {
  const [selected, setSelected] = useState<DatasetDetail | null>(null)
  const [profile, setProfile] = useState<DatasetProfile | null>(null)
  const [loadingProfile, setLoadingProfile] = useState(false)

  const loadDatasetDetails = async (id: number) => {
    setLoadingProfile(true)
    try {
      const [d, p] = await Promise.all([api.dataset(id), api.datasetProfile(id)])
      setSelected(d)
      setProfile(p)
    } catch (err) {
      announce((err as Error).message)
    } finally {
      setLoadingProfile(false)
    }
  }

  const upload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true)
    try {
      const uploaded = await api.upload(file)
      announce(`Uploaded and automatically profiled ${uploaded.name}`)
      await onChanged()
      await loadDatasetDetails(uploaded.id)
    } catch (err) {
      announce((err as Error).message)
    } finally {
      setBusy(false)
      e.target.value = ''
    }
  }

  return (
    <div className="split">
      <section className="panel">
        <div className="panel-head">
          <h2>Datasets</h2>
          <label className="primary">
            {busy ? 'Processing…' : 'Upload CSV'}
            <input type="file" accept=".csv,text/csv" onChange={upload} disabled={busy} />
          </label>
        </div>
        {datasets.length ? (
          <div className="dataset-list">
            {datasets.map((d) => (
              <button
                key={d.id}
                className={`dataset-item ${selected?.id === d.id ? 'selected' : ''}`}
                onClick={() => loadDatasetDetails(d.id)}
              >
                <b>{d.name}</b>
                <span>
                  {d.row_count.toLocaleString()} rows · {d.column_count} columns
                </span>
              </button>
            ))}
          </div>
        ) : (
          <Empty text="No uploaded datasets yet." />
        )}
      </section>

      <section className="panel detail">
        {loadingProfile ? (
          <p>Generating dataset profiling and intelligence report…</p>
        ) : selected && profile ? (
          <>
            <div className="panel-head">
              <h2>{selected.name}</h2>
              <div
                className={`score-badge ${
                  profile.quality_score >= 80
                    ? 'score-good'
                    : profile.quality_score >= 60
                    ? 'score-warn'
                    : 'score-bad'
                }`}
              >
                <span>Quality Score:</span>
                <strong>{profile.quality_score} / 100</strong>
              </div>
            </div>

            <div className="mini-stats">
              <span><b>{selected.row_count.toLocaleString()}</b> rows</span>
              <span><b>{selected.column_count}</b> columns</span>
              <span><b>{selected.duplicate_rows}</b> duplicates</span>
              <span><b>{profile.warnings.length}</b> quality warnings</span>
            </div>

            {profile.warnings.length > 0 && (
              <>
                <h3>Data Quality Warnings</h3>
                <div className="warnings-box">
                  {profile.warnings.map((w, i) => (
                    <div key={i} className={`warning-item sev-${w.severity}`}>
                      <span className="badge-pill">{w.severity}</span>
                      <span>
                        <b>{w.column}:</b> {w.message}
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}

            {profile.suggested_operations.length > 0 && (
              <>
                <h3>Suggested Preprocessing</h3>
                <ul>
                  {profile.suggested_operations.map((op, i) => (
                    <li key={i} style={{ fontSize: '13px', marginBottom: '4px' }}>
                      {op}
                    </li>
                  ))}
                </ul>
              </>
            )}

            {profile.risks.length > 0 && (
              <>
                <h3>Identified Training Risks</h3>
                <ul>
                  {profile.risks.map((risk, i) => (
                    <li key={i} style={{ fontSize: '13px', color: '#b42318', marginBottom: '4px' }}>
                      {risk}
                    </li>
                  ))}
                </ul>
              </>
            )}

            <h3>Descriptive Statistics & Feature Analysis</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Type</th>
                    <th>Missing</th>
                    <th>Mean / Top</th>
                    <th>Std / Freq</th>
                    <th>Min / Unique</th>
                    <th>Max</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(profile.statistics).map(([feat, st]) => (
                    <tr key={feat}>
                      <td><b>{feat}</b></td>
                      <td>{st.type}</td>
                      <td>{profile.outliers[feat]?.outlier_count ? `${profile.outliers[feat].outlier_count} outliers` : '0'}</td>
                      <td>{st.mean ?? st.top ?? '—'}</td>
                      <td>{st.std ?? st.top_freq ?? '—'}</td>
                      <td>{st.min ?? st.unique ?? '—'}</td>
                      <td>{st.max ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <h3>Dataset Preview (First 15 Rows)</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {selected.preview.length > 0 &&
                      Object.keys(selected.preview[0]).map((c) => <th key={c}>{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {selected.preview.slice(0, 15).map((row, i) => (
                    <tr key={i}>
                      {Object.keys(row).map((k) => (
                        <td key={k}>{String(row[k] ?? '—')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <Empty text="Select a dataset to view its automated profiling intelligence report." />
        )}
      </section>
    </div>
  )
}

// ----------------- 3. RECOMMENDATIONS -----------------
function Recommendations({
  datasets,
  onTrainApplied,
  announce,
}: {
  datasets: Dataset[]
  onTrainApplied: () => void
  announce: (x: string) => void
}) {
  const [datasetId, setDatasetId] = useState('')
  const [target, setTarget] = useState('')
  const [probType, setProbType] = useState<string>('classification')
  const [recResult, setRecResult] = useState<RecommendationResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const selectedDs = datasets.find((d) => d.id === Number(datasetId))

  useEffect(() => {
    if (selectedDs && (!target || !selectedDs.columns[target])) {
      setTarget(Object.keys(selectedDs.columns)[0] ?? '')
    }
  }, [datasetId, selectedDs])

  const analyze = async (e: FormEvent) => {
    e.preventDefault()
    if (!datasetId || !target) return announce('Select a dataset and target column.')
    setLoading(true)
    try {
      const res = await api.recommend({
        dataset_id: Number(datasetId),
        target_column: target,
        problem_type: probType,
      })
      setRecResult(res)
      announce('Generated intelligent model recommendations.')
    } catch (err) {
      announce((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <form className="panel form" onSubmit={analyze}>
        <h2>Intelligent Algorithm Recommendation Engine</h2>
        <p style={{ fontSize: '13px', color: '#475467', marginBottom: '16px' }}>
          SentinelML analyzes dataset dimensions, class imbalance, missingness, and feature types to
          recommend suitable machine learning algorithms with transparent reasoning.
        </p>

        <div className="twocol">
          <label>
            Dataset
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
              <option value="">Select an uploaded dataset</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.row_count} rows)
                </option>
              ))}
            </select>
          </label>

          <label>
            Target Column
            <select
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              disabled={!selectedDs}
            >
              <option value="">Select target</option>
              {selectedDs &&
                Object.keys(selectedDs.columns).map((c) => (
                  <option key={c} value={c}>
                    {c} ({selectedDs.columns[c]})
                  </option>
                ))}
            </select>
          </label>
        </div>

        <label>
          Problem Type
          <select value={probType} onChange={(e) => setProbType(e.target.value)}>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
        </label>

        <button className="primary submit" disabled={loading || !datasetId || !target}>
          {loading ? 'Evaluating Algorithms…' : 'Generate Recommendations'}
        </button>
      </form>

      {recResult && (
        <section className="panel">
          <h2>Recommended Algorithms for {recResult.target_column}</h2>
          <p style={{ fontSize: '13px', color: '#344054', fontStyle: 'italic', marginBottom: '14px' }}>
            {recResult.explanation}
          </p>

          <div className="rec-grid">
            {recResult.recommended_algorithms.map((rec) => (
              <div
                key={rec.algorithm}
                className={`rec-card ${rec.recommended ? 'highlight' : ''}`}
              >
                <h4>
                  <span>{rec.display_name}</span>
                  <span className={`status ${rec.recommended ? 'completed' : 'failed'}`}>
                    Score: {rec.score}
                  </span>
                </h4>
                <p>
                  <b>Suitability:</b> {rec.recommended ? 'Recommended' : 'Secondary candidate'}
                </p>
                <ul>
                  {rec.reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
                <div className="rec-meta">
                  <span>Speed: {rec.estimated_speed}</span> · 
                  <span>Compute: {rec.computational_requirement}</span>
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: '20px' }}>
            <button className="primary" onClick={onTrainApplied}>
              Train Recommended Models →
            </button>
          </div>
        </section>
      )}
    </div>
  )
}

// ----------------- 4. TRAIN & EXPERIMENTS -----------------
function Train({
  datasets,
  jobs,
  onStarted,
  announce,
}: {
  datasets: Dataset[]
  jobs: Job[]
  onStarted: () => void
  announce: (x: string) => void
}) {
  const [datasetId, setDatasetId] = useState('')
  const [target, setTarget] = useState('')
  const [type, setType] = useState<'classification' | 'regression'>('classification')
  const [algorithms, setAlgorithms] = useState<string[]>([
    'logistic_regression',
    'random_forest',
    'gradient_boosting',
  ])
  const [metric, setMetric] = useState('f1')
  const [saving, setSaving] = useState(false)

  const selected = datasets.find((x) => x.id === Number(datasetId))

  useEffect(() => {
    if (selected && !selected.columns[target]) {
      setTarget(Object.keys(selected.columns)[0] ?? '')
    }
  }, [datasetId, datasets])

  useEffect(() => {
    setAlgorithms(
      type === 'classification'
        ? ['logistic_regression', 'random_forest', 'gradient_boosting']
        : ['linear_regression', 'random_forest', 'gradient_boosting']
    )
    setMetric(type === 'classification' ? 'f1' : 'rmse')
  }, [type])

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!datasetId || !target || !algorithms.length)
      return announce('Choose a dataset, target, and at least one algorithm.')
    setSaving(true)
    try {
      const job = await api.train({
        dataset_id: Number(datasetId),
        target_column: target,
        problem_type: type,
        algorithms,
        primary_metric: metric,
      } as TrainingBody)
      announce(`Training job #${job.id} dispatched to Celery background worker.`)
      onStarted()
    } catch (err) {
      announce((err as Error).message)
    } finally {
      setSaving(false)
    }
  }

  const options =
    type === 'classification'
      ? [
          ['logistic_regression', 'Logistic Regression'],
          ['random_forest', 'Random Forest'],
          ['gradient_boosting', 'Gradient Boosting'],
        ]
      : [
          ['linear_regression', 'Linear Regression'],
          ['random_forest', 'Random Forest'],
          ['gradient_boosting', 'Gradient Boosting'],
        ]

  return (
    <>
      <form className="form panel" onSubmit={submit}>
        <h2>Configure & Train Models</h2>
        <div className="twocol">
          <label>
            Dataset
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
              <option value="">Select an uploaded dataset</option>
              {datasets.map((d) => (
                <option value={d.id} key={d.id}>
                  {d.name} ({d.row_count} rows)
                </option>
              ))}
            </select>
          </label>
          <label>
            Target column
            <select
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              disabled={!selected}
            >
              <option value="">Select target</option>
              {Object.keys(selected?.columns ?? {}).map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="twocol">
          <label>
            Problem type
            <select value={type} onChange={(e) => setType(e.target.value as typeof type)}>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </label>
          <label>
            Primary metric
            <select value={metric} onChange={(e) => setMetric(e.target.value)}>
              {(type === 'classification'
                ? ['f1', 'accuracy', 'precision', 'recall']
                : ['rmse', 'mae', 'r2']
              ).map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </label>
        </div>

        <fieldset>
          <legend>Candidate Algorithms</legend>
          {options.map(([value, label]) => (
            <label className="check" key={value}>
              <input
                type="checkbox"
                checked={algorithms.includes(value)}
                onChange={(e) =>
                  setAlgorithms(
                    e.target.checked
                      ? [...algorithms, value]
                      : algorithms.filter((x) => x !== value)
                  )
                }
              />
              {label}
            </label>
          ))}
        </fieldset>

        <button className="primary submit" disabled={saving || !datasetId || !target}>
          {saving ? 'Queueing Job…' : 'Start Training in Celery Worker'}
        </button>
      </form>

      <section className="panel">
        <h2>Training Jobs & MLflow Runs</h2>
        {jobs.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Dataset</th>
                  <th>Target</th>
                  <th>Status & Progress</th>
                  <th>Winner</th>
                  <th>MLflow Run</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.id}>
                    <td>#{j.id}</td>
                    <td>Dataset #{j.dataset_id}</td>
                    <td>{j.target_column}</td>
                    <td>
                      <StatusBadge status={j.status} />
                      <small style={{ marginLeft: '6px' }}>{j.progress}%</small>
                      {j.error && <small className="err">{j.error}</small>}
                    </td>
                    <td><b>{j.best_algorithm ?? '—'}</b></td>
                    <td className="mono">{j.mlflow_run_id ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty text="Training jobs will appear here." />
        )}
      </section>
    </>
  )
}

// ----------------- 5. MODEL COMPARISONS & MULTI-OBJECTIVE RANKING -----------------
function Comparisons() {
  const [probType, setProbType] = useState('classification')
  const [metric, setMetric] = useState('f1')
  const [ranked, setRanked] = useState<ModelRankingItem[]>([])
  const [loading, setLoading] = useState(false)

  const fetchRankings = async () => {
    setLoading(true)
    try {
      const res = await api.compare(probType, metric)
      setRanked(res)
    } catch {
      setRanked([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchRankings()
  }, [probType, metric])

  const chartData = ranked.map((r) => ({
    name: r.algorithm,
    Score: r.overall_score,
    LatencyMs: r.inference_latency_ms,
  }))

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>Multi-Objective Model Comparison & Ranking</h2>
          <div style={{ display: 'flex', gap: '10px' }}>
            <select value={probType} onChange={(e) => setProbType(e.target.value)}>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
            <select value={metric} onChange={(e) => setMetric(e.target.value)}>
              {(probType === 'classification'
                ? ['f1', 'accuracy', 'precision', 'recall']
                : ['rmse', 'mae', 'r2']
              ).map((m) => (
                <option key={m} value={m}>
                  Metric: {m.toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        </div>

        <p style={{ fontSize: '13px', color: '#475467', marginBottom: '18px' }}>
          Models are evaluated across predictive performance, inference latency, training time, and
          artifact size. No single model is claimed as universally best.
        </p>

        {chartData.length > 0 && (
          <div style={{ marginBottom: '22px' }}>
            <h3>Multi-Objective Score Comparison</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="Score" fill="#4f46e5" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {loading ? (
          <p>Computing rankings…</p>
        ) : ranked.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Model / Algorithm</th>
                  <th>Primary Metric ({metric})</th>
                  <th>Latency</th>
                  <th>Train Time</th>
                  <th>Artifact Size</th>
                  <th>Overall Score</th>
                  <th>Selection Reasoning</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((m) => (
                  <tr key={m.model_name}>
                    <td><b>#{m.rank}</b></td>
                    <td><b>{m.algorithm}</b></td>
                    <td>{m.metrics[metric] !== undefined ? Number(m.metrics[metric]).toFixed(4) : '—'}</td>
                    <td>{m.inference_latency_ms} ms</td>
                    <td>{m.training_time_seconds} s</td>
                    <td>{m.model_size_kb} KB</td>
                    <td>
                      <span className="status completed">{m.overall_score}/100</span>
                    </td>
                    <td style={{ fontSize: '11px', whiteSpace: 'normal', minWidth: '280px' }}>
                      {m.selection_reasoning}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty text="No model evaluations recorded yet. Train models to see comparisons." />
        )}
      </section>
    </>
  )
}

// ----------------- 6. EXPLAINABILITY -----------------
function Explainability({
  models,
  datasets,
  announce,
}: {
  models: Model[]
  datasets: Dataset[]
  announce: (x: string) => void
}) {
  const [modelName, setModelName] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const loadExplanation = async () => {
    if (!modelName) return announce('Select a model to explain.')
    setLoading(true)
    try {
      const res = await api.explanations(modelName, 'latest', datasetId ? Number(datasetId) : undefined)
      setExplanation(res)
    } catch (err) {
      announce((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const chartData = (explanation?.global_importance ?? []).slice(0, 10).map((f) => ({
    name: f.feature,
    Impact: f.percentage,
  }))

  return (
    <>
      <section className="panel form">
        <h2>Model Explainability & Feature Attribution</h2>
        <div className="twocol">
          <label>
            Model
            <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
              <option value="">Select registered model</option>
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Reference Dataset
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
              <option value="">Use latest dataset</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </label>
        </div>
        <button className="primary submit" disabled={loading || !modelName} onClick={loadExplanation}>
          {loading ? 'Computing SHAP Attributions…' : 'Generate Model Explanations'}
        </button>
      </section>

      {explanation && (
        <section className="panel">
          <div className="panel-head">
            <h2>Explainability Report for {explanation.model_name}</h2>
            <span className="badge-pill" style={{ background: '#eef2ff', color: '#4f46e5' }}>
              Method: {explanation.method}
            </span>
          </div>

          <p style={{ fontSize: '13px', color: '#344054', fontStyle: 'italic', marginBottom: '18px' }}>
            {explanation.summary}
          </p>

          <h3>Top Feature Contributions (%)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" unit="%" />
              <YAxis dataKey="name" type="category" width={110} />
              <Tooltip />
              <Bar dataKey="Impact" fill="#6366f1" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>

          <h3>Feature Importance Breakdown</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Relative Contribution</th>
                  <th>Importance Score</th>
                </tr>
              </thead>
              <tbody>
                {explanation.global_importance.map((item) => (
                  <tr key={item.feature}>
                    <td><b>{item.feature}</b></td>
                    <td>{item.percentage}%</td>
                    <td>{item.importance}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  )
}

// ----------------- 7. DRIFT & MONITORING -----------------
function Monitoring({
  models,
  datasets,
  announce,
}: {
  models: Model[]
  datasets: Dataset[]
  announce: (x: string) => void
}) {
  const [modelName, setModelName] = useState(models[0]?.name ?? '')
  const [reports, setReports] = useState<DriftReport[]>([])
  const [perfReport, setPerfReport] = useState<PerformanceReport | null>(null)
  const [testingDrift, setTestingDrift] = useState(false)

  const fetchReports = async () => {
    try {
      const [d, p] = await Promise.all([
        api.driftReports(modelName || undefined),
        modelName ? api.performance(modelName).catch(() => null) : Promise.resolve(null),
      ])
      setReports(d)
      setPerfReport(p)
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    fetchReports()
  }, [modelName])

  const triggerDriftSimulation = async () => {
    if (!modelName) return announce('Select a model.')
    setTestingDrift(true)
    try {
      // Create synthetic sample simulating distribution drift
      const dummyProd = Array.from({ length: 30 }, (_, i) => ({
        feature: i * 10,
        age: 60 + (i % 25),
        income: 120000 + i * 1500,
      }))
      const rep = await api.detectDrift({
        model_name: modelName,
        production_records: dummyProd,
      })
      announce(`Drift analysis completed: Severity is ${rep.drift_severity}`)
      await fetchReports()
    } catch (err) {
      announce((err as Error).message)
    } finally {
      setTestingDrift(false)
    }
  }

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>Production Data Drift & Performance Degradation</h2>
          <div style={{ display: 'flex', gap: '10px' }}>
            <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
              <option value="">Select registered model</option>
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </select>
            <button className="primary" onClick={triggerDriftSimulation} disabled={testingDrift}>
              {testingDrift ? 'Analyzing Drift…' : 'Run Drift Analysis'}
            </button>
          </div>
        </div>

        {perfReport && (
          <div style={{ marginBottom: '20px', padding: '14px', borderRadius: '8px', background: perfReport.degradation_detected ? '#fef3f2' : '#f0fdf4', border: `1px solid ${perfReport.degradation_detected ? '#fee4e2' : '#bbf7d0'}` }}>
            <h4 style={{ margin: '0 0 6px', color: perfReport.degradation_detected ? '#b42318' : '#166534' }}>
              Performance Status: {perfReport.status}
            </h4>
            <p style={{ margin: 0, fontSize: '13px' }}>{perfReport.recommended_action}</p>
          </div>
        )}

        <h3>Historical Drift Reports</h3>
        {reports.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Model</th>
                  <th>Version</th>
                  <th>Samples</th>
                  <th>Severity</th>
                  <th>Overall Score</th>
                  <th>Recommended Next Action</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((r) => (
                  <tr key={r.id}>
                    <td>{new Date(r.created_at).toLocaleTimeString()}</td>
                    <td><b>{r.model_name}</b></td>
                    <td>v{r.model_version}</td>
                    <td>{r.sample_count}</td>
                    <td>
                      <span className={`status ${r.drift_severity === 'NONE' ? 'completed' : r.drift_severity === 'CRITICAL' ? 'failed' : 'queued'}`}>
                        {r.drift_severity}
                      </span>
                    </td>
                    <td>{r.overall_drift_score}</td>
                    <td style={{ whiteSpace: 'normal', minWidth: '260px', fontSize: '12px' }}>
                      {r.recommended_action}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty text="No drift analyses recorded yet. Click 'Run Drift Analysis' above." />
        )}
      </section>
    </>
  )
}

// ----------------- 8. RETRAINING -----------------
function Retraining({
  models,
  datasets,
  announce,
}: {
  models: Model[]
  datasets: Dataset[]
  announce: (x: string) => void
}) {
  const [modelName, setModelName] = useState(models[0]?.name ?? '')
  const [reason, setReason] = useState('Manual trigger for testing retraining pipeline')
  const [history, setHistory] = useState<RetrainingJob[]>([])
  const [triggering, setTriggering] = useState(false)

  const fetchHistory = async () => {
    try {
      const h = await api.retrainingHistory()
      setHistory(h)
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [])

  const handleRetrain = async (e: FormEvent) => {
    e.preventDefault()
    if (!modelName) return announce('Select a model to retrain.')
    setTriggering(true)
    try {
      const job = await api.triggerRetraining({
        model_name: modelName,
        trigger_reason: reason,
      })
      announce(`Retraining initiated: Job #${job.id} created candidate v${job.candidate_version}`)
      await fetchHistory()
    } catch (err) {
      announce((err as Error).message)
    } finally {
      setTriggering(false)
    }
  }

  return (
    <>
      <form className="panel form" onSubmit={handleRetrain}>
        <h2>Intelligent Retraining Orchestrator</h2>
        <p style={{ fontSize: '13px', color: '#475467', marginBottom: '16px' }}>
          Retraining workflows evaluate candidate models, compare against production baselines, and
          submit approval requests before production deployment.
        </p>
        <div className="twocol">
          <label>
            Target Model
            <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
              <option value="">Select model</option>
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Trigger Reason
            <input
              type="text"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g. Feature drift exceeded 0.3 threshold"
            />
          </label>
        </div>
        <button className="primary submit" disabled={triggering || !modelName}>
          {triggering ? 'Executing Retraining Workflow…' : 'Trigger Retraining Job'}
        </button>
      </form>

      <section className="panel">
        <h2>Retraining Workflow Executions</h2>
        {history.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Model</th>
                  <th>Trigger Reason</th>
                  <th>Status</th>
                  <th>Candidate Version</th>
                  <th>Approval Request</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {history.map((j) => (
                  <tr key={j.id}>
                    <td>#{j.id}</td>
                    <td><b>{j.model_name}</b></td>
                    <td style={{ whiteSpace: 'normal', minWidth: '220px', fontSize: '12px' }}>
                      {j.trigger_reason}
                    </td>
                    <td>
                      <StatusBadge status={j.status} />
                    </td>
                    <td>{j.candidate_version ? `v${j.candidate_version}` : '—'}</td>
                    <td>{j.approval_id ? `#${j.approval_id} (Pending Review)` : 'None'}</td>
                    <td>{new Date(j.created_at).toLocaleTimeString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty text="No automated retraining workflows executed yet." />
        )}
      </section>
    </>
  )
}

// ----------------- 9. GOVERNANCE & APPROVALS -----------------
function Governance({
  approvals,
  models,
  onRefresh,
  announce,
}: {
  approvals: ModelApproval[]
  models: Model[]
  onRefresh: () => Promise<void>
  announce: (x: string) => void
}) {
  const [comment, setComment] = useState('')

  const handleApprove = async (id: number) => {
    try {
      await api.approveCandidate(id, { comments: comment || 'Approved by MLOps reviewer', reviewer: 'Lead Engineer' })
      announce(`Candidate #${id} approved and promoted to PRODUCTION.`)
      setComment('')
      await onRefresh()
    } catch (err) {
      announce((err as Error).message)
    }
  }

  const handleReject = async (id: number) => {
    try {
      await api.rejectCandidate(id, { comments: comment || 'Rejected during audit', reviewer: 'Lead Engineer' })
      announce(`Candidate #${id} rejected.`)
      setComment('')
      await onRefresh()
    } catch (err) {
      announce((err as Error).message)
    }
  }

  return (
    <>
      <section className="panel">
        <h2>Model Governance & Promotion Workflow</h2>
        <p style={{ fontSize: '13px', color: '#475467', marginBottom: '16px' }}>
          Lifecycle management follows strict state transitions: DEVELOPMENT → CANDIDATE → APPROVED
          → PRODUCTION → ARCHIVED. Human approval is enforced by default.
        </p>

        <h3>Pending Candidate Approvals</h3>
        {approvals.filter((a) => a.status === 'PENDING').length ? (
          approvals
            .filter((a) => a.status === 'PENDING')
            .map((appr) => (
              <div key={appr.id} className="approval-card">
                <div className="panel-head">
                  <h4>
                    Candidate: <b>{appr.candidate_model_name}</b> (v{appr.candidate_version})
                  </h4>
                  <span className="status queued">Pending Review</span>
                </div>
                <p style={{ fontSize: '12px', color: '#475467', margin: '4px 0 10px' }}>
                  Current Production: {appr.current_production_version ? `v${appr.current_production_version}` : 'None'}
                </p>
                <div style={{ marginBottom: '12px' }}>
                  <input
                    type="text"
                    placeholder="Enter review comments or reasoning…"
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                  />
                </div>
                <div className="actions-row">
                  <button className="btn-sm btn-approve" onClick={() => handleApprove(appr.id)}>
                    ✓ Approve & Promote to Production
                  </button>
                  <button className="btn-sm btn-reject" onClick={() => handleReject(appr.id)}>
                    ✕ Reject Candidate
                  </button>
                </div>
              </div>
            ))
        ) : (
          <Empty text="No candidate models pending approval." />
        )}

        <h3>Approval History & Decisions</h3>
        {approvals.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Request ID</th>
                  <th>Model</th>
                  <th>Candidate Version</th>
                  <th>Decision</th>
                  <th>Reviewer</th>
                  <th>Comments</th>
                  <th>Resolved At</th>
                </tr>
              </thead>
              <tbody>
                {approvals.map((a) => (
                  <tr key={a.id}>
                    <td>#{a.id}</td>
                    <td>{a.candidate_model_name}</td>
                    <td>v{a.candidate_version}</td>
                    <td>
                      <StatusBadge status={a.status} />
                    </td>
                    <td>{a.reviewed_by ?? '—'}</td>
                    <td style={{ fontSize: '11px' }}>{a.reviewer_comments ?? '—'}</td>
                    <td>{a.resolved_at ? new Date(a.resolved_at).toLocaleTimeString() : 'Pending'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty text="No approvals recorded yet." />
        )}
      </section>
    </>
  )
}

// ----------------- 10. DEPLOYMENTS & ROLLBACK -----------------
function Deployments({
  deployments,
  models,
  onRefresh,
  announce,
}: {
  deployments: DeploymentRecord[]
  models: Model[]
  onRefresh: () => Promise<void>
  announce: (x: string) => void
}) {
  const [modelToRollback, setModelToRollback] = useState(models[0]?.name ?? '')
  const [reason, setReason] = useState('Degradation detected in production traffic')
  const [busy, setBusy] = useState(false)

  const handleRollback = async () => {
    if (!modelToRollback) return announce('Select a model to roll back.')
    setBusy(true)
    try {
      const rec = await api.rollback(modelToRollback, { reason, initiated_by: 'MLOps Lead' })
      announce(`Successfully rolled back ${modelToRollback} to version v${rec.version}!`)
      await onRefresh()
    } catch (err) {
      announce((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <section className="panel">
        <h2>Production Deployments & Safe Rollback</h2>
        <p style={{ fontSize: '13px', color: '#475467', marginBottom: '16px' }}>
          SentinelML preserves full model lineage in MLflow. When a model underperforms, authorized
          users can instantly restore the previous verified production version.
        </p>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '22px' }}>
          <select value={modelToRollback} onChange={(e) => setModelToRollback(e.target.value)}>
            <option value="">Select model for rollback</option>
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
          <input
            type="text"
            placeholder="Rollback justification…"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            style={{ maxWidth: '350px' }}
          />
          <button className="btn-sm btn-rollback" onClick={handleRollback} disabled={busy || !modelToRollback}>
            {busy ? 'Restoring Version…' : '⟲ Rollback Model'}
          </button>
        </div>

        <h3>Deployment Audit Log</h3>
        {deployments.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Deployment ID</th>
                  <th>Model</th>
                  <th>Version</th>
                  <th>Previous Version</th>
                  <th>Action</th>
                  <th>Initiated By</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {deployments.map((d) => (
                  <tr key={d.id}>
                    <td>#{d.id}</td>
                    <td><b>{d.model_name}</b></td>
                    <td>v{d.version}</td>
                    <td>{d.previous_version ? `v${d.previous_version}` : 'None'}</td>
                    <td>
                      <span className={`status ${d.action === 'PROMOTE' ? 'production' : 'failed'}`}>
                        {d.action}
                      </span>
                    </td>
                    <td>{d.initiated_by}</td>
                    <td>{new Date(d.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty text="No deployments recorded yet." />
        )}
      </section>
    </>
  )
}

// ----------------- 11. AUTOMATION & AUDIT LOGS -----------------
function Automation({ announce }: { announce: (x: string) => void }) {
  const [rules, setRules] = useState<RetrainingRule[]>([])
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [ruleName, setRuleName] = useState('')
  const [modelTarget, setModelTarget] = useState('')
  const [triggerType, setTriggerType] = useState('DRIFT')

  const fetchData = async () => {
    try {
      const [r, l] = await Promise.all([api.retrainingRules(), api.auditLogs()])
      setRules(r)
      setLogs(l)
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const addRule = async (e: FormEvent) => {
    e.preventDefault()
    if (!ruleName || !modelTarget) return announce('Specify rule name and target model.')
    try {
      await api.createRetrainingRule({
        name: ruleName,
        model_name: modelTarget,
        trigger_type: triggerType,
        threshold: 0.25,
        cooldown_minutes: 60,
      })
      announce(`Rule '${ruleName}' configured.`)
      setRuleName('')
      setModelTarget('')
      await fetchData()
    } catch (err) {
      announce((err as Error).message)
    }
  }

  return (
    <>
      <form className="panel form" onSubmit={addRule}>
        <h2>Configure Automation Rule</h2>
        <div className="twocol">
          <label>
            Rule Name
            <input
              type="text"
              value={ruleName}
              onChange={(e) => setRuleName(e.target.value)}
              placeholder="e.g. Automated Drift Watcher"
            />
          </label>
          <label>
            Model Name
            <input
              type="text"
              value={modelTarget}
              onChange={(e) => setModelTarget(e.target.value)}
              placeholder="e.g. sentinelml_job_1_best"
            />
          </label>
        </div>
        <label>
          Trigger Type
          <select value={triggerType} onChange={(e) => setTriggerType(e.target.value)}>
            <option value="DRIFT">Feature Drift Threshold</option>
            <option value="DEGRADATION">Accuracy Degradation</option>
            <option value="SCHEDULED">Scheduled Interval</option>
          </select>
        </label>
        <button className="primary submit">Register Automation Rule</button>
      </form>

      <section className="panel">
        <h2>Active Automation Rules</h2>
        {rules.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Rule</th>
                  <th>Model</th>
                  <th>Trigger</th>
                  <th>Cooldown</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {rules.map((r) => (
                  <tr key={r.id}>
                    <td><b>{r.name}</b></td>
                    <td>{r.model_name}</td>
                    <td>{r.trigger_type}</td>
                    <td>{r.cooldown_minutes} min</td>
                    <td>
                      <StatusBadge status={r.is_active ? 'running' : 'failed'} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty text="No active automation rules configured." />
        )}
      </section>

      <section className="panel">
        <h2>Platform Governance Audit Trail</h2>
        {logs.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Action</th>
                  <th>Entity Type</th>
                  <th>Entity ID</th>
                  <th>Actor</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log) => (
                  <tr key={log.id}>
                    <td>{new Date(log.created_at).toLocaleTimeString()}</td>
                    <td>
                      <span className="badge-pill" style={{ background: '#f2f4f7', color: '#344054' }}>
                        {log.action}
                      </span>
                    </td>
                    <td>{log.entity_type}</td>
                    <td>{log.entity_id ?? '—'}</td>
                    <td><b>{log.actor}</b></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty text="Audit logs will appear as actions occur." />
        )}
      </section>
    </>
  )
}
