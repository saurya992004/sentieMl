import { ChangeEvent, FormEvent, useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Cpu,
  Database,
  FileCheck,
  Flame,
  Layers,
  Play,
  RefreshCw,
  Rocket,
  RotateCcw,
  Scale,
  Search,
  Settings,
  ShieldAlert,
  Sparkles,
  Terminal,
  TrendingUp,
  Upload,
  XCircle,
  Zap,
} from 'lucide-react'
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

  const pendingApprovals = approvals.filter((a) => a.status === 'PENDING').length

  const navItems: { id: View; label: string; icon: any; badge?: number }[] = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'datasets', label: 'Dataset Intelligence', icon: Database },
    { id: 'recommendations', label: 'Model Recommendations', icon: Sparkles },
    { id: 'train', label: 'Train & Experiments', icon: Cpu },
    { id: 'comparisons', label: 'Model Comparison', icon: Scale },
    { id: 'explainability', label: 'Model Explainability', icon: Search },
    { id: 'monitoring', label: 'Drift & Performance', icon: ShieldAlert },
    { id: 'retraining', label: 'Automated Retraining', icon: RefreshCw },
    { id: 'governance', label: 'Governance & Approvals', icon: FileCheck, badge: pendingApprovals },
    { id: 'deployments', label: 'Deploy & Rollback', icon: Rocket },
    { id: 'automation', label: 'Automation Center', icon: Settings },
  ]

  return (
    <div className="shell">
      <aside>
        <div className="brand">
          <div className="brand-icon">
            <Zap size={18} color="#fff" />
          </div>
          <span className="brand-gradient">SentinelML</span>
        </div>
        <p className="tag">Autonomous MLOps 2.0</p>

        <div className="nav-group">
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <button
                className={`nav ${view === item.id ? 'active' : ''}`}
                key={item.id}
                onClick={() => setView(item.id)}
              >
                <Icon size={16} />
                <span>{item.label}</span>
                {item.badge !== undefined && item.badge > 0 && (
                  <span className="nav-badge">{item.badge}</span>
                )}
              </button>
            )
          })}
        </div>

        <div className="aside-bottom">
          <div className="status-pill">
            <span style={{ color: '#94a3b8' }}>MLflow Server</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div className="status-indicator"></div>
              <span style={{ color: '#34d399', fontWeight: 700 }}>v3.18</span>
            </div>
          </div>
          <div className="status-pill">
            <span style={{ color: '#94a3b8' }}>Celery Workers</span>
            <span style={{ color: '#818cf8', fontWeight: 700 }}>Redis Queue</span>
          </div>
        </div>
      </aside>

      <main>
        <header>
          <div>
            <p className="eyebrow">
              <Zap size={12} /> ENTERPRISE ML LIFECYCLE
            </p>
            <h1>{navItems.find((n) => n.id === view)?.label}</h1>
          </div>
          <div className="header-actions">
            <button className="refresh" onClick={refresh}>
              <RefreshCw size={14} /> Refresh
            </button>
          </div>
        </header>

        {message && (
          <div className="notice">
            <span>{message}</span>
            <button onClick={() => setMessage('')}>×</button>
          </div>
        )}

        {/* Global Pipeline Visualizer */}
        <div className="pipeline-stepper">
          <div className={`step-chip ${view === 'datasets' ? 'active' : ''}`}>
            <Database size={13} /> 1. Ingestion & Quality
          </div>
          <ArrowRight size={13} color="#475467" />
          <div className={`step-chip ${view === 'recommendations' ? 'active' : ''}`}>
            <Sparkles size={13} /> 2. Algorithm Fit
          </div>
          <ArrowRight size={13} color="#475467" />
          <div className={`step-chip ${view === 'train' ? 'active' : ''}`}>
            <Cpu size={13} /> 3. Celery + MLflow
          </div>
          <ArrowRight size={13} color="#475467" />
          <div className={`step-chip ${view === 'comparisons' ? 'active' : ''}`}>
            <Scale size={13} /> 4. Multi-Objective
          </div>
          <ArrowRight size={13} color="#475467" />
          <div className={`step-chip ${view === 'explainability' ? 'active' : ''}`}>
            <Search size={13} /> 5. SHAP Attribution
          </div>
          <ArrowRight size={13} color="#475467" />
          <div className={`step-chip ${view === 'monitoring' ? 'active' : ''}`}>
            <ShieldAlert size={13} /> 6. Drift Monitor
          </div>
          <ArrowRight size={13} color="#475467" />
          <div className={`step-chip ${view === 'governance' ? 'active' : ''}`}>
            <FileCheck size={13} /> 7. Human Governance
          </div>
          <ArrowRight size={13} color="#475467" />
          <div className={`step-chip ${view === 'deployments' ? 'active' : ''}`}>
            <RotateCcw size={13} /> 8. Rollback
          </div>
        </div>

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
        {view === 'monitoring' && (
          <Monitoring models={models} datasets={datasets} announce={setMessage} />
        )}
        {view === 'retraining' && (
          <Retraining models={models} datasets={datasets} announce={setMessage} />
        )}
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
  return (
    <span className={`status ${s}`}>
      <span
        style={{
          width: '5px',
          height: '5px',
          borderRadius: '50%',
          backgroundColor: 'currentColor',
        }}
      ></span>
      {s}
    </span>
  )
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
          <div className="stat-icon-bg">
            <Database size={18} />
          </div>
          <p>Datasets Ingested</p>
          <strong>{datasets.length}</strong>
          <small>Profiling & Quality validated</small>
        </div>
        <div className="stat">
          <div className="stat-icon-bg">
            <Cpu size={18} />
          </div>
          <p>Training Runs</p>
          <strong>{jobs.length}</strong>
          <small>{completed} completed via Celery</small>
        </div>
        <div className="stat">
          <div className="stat-icon-bg">
            <Layers size={18} />
          </div>
          <p>Registered in MLflow</p>
          <strong>{models.length}</strong>
          <small>Versioned CloudPickle pipelines</small>
        </div>
        <div className="stat">
          <div className="stat-icon-bg">
            <FileCheck size={18} />
          </div>
          <p>Pending Approvals</p>
          <strong style={{ color: pendingApprovals > 0 ? '#fbbf24' : '#fff' }}>
            {pendingApprovals}
          </strong>
          <small>Human governance gate</small>
        </div>
      </section>

      <div className="twocol">
        <section className="panel">
          <div className="panel-head">
            <h2>
              <Activity size={18} color="#818cf8" /> Recent Training Jobs
            </h2>
            <button className="btn-sm" onClick={() => setView('train')}>
              View All
            </button>
          </div>
          {jobs.length ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Job ID</th>
                    <th>Target</th>
                    <th>Status</th>
                    <th>Winner</th>
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
                      <td><b>{job.best_algorithm ?? '—'}</b></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty text="Upload a CSV dataset to initiate training." />
          )}
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>
              <ShieldAlert size={18} color="#f59e0b" /> Pending Governance Reviews
            </h2>
            <button className="btn-sm" onClick={() => setView('governance')}>
              Review
            </button>
          </div>
          {approvals.filter((a) => a.status === 'PENDING').length ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Candidate</th>
                    <th>Version</th>
                    <th>Current Prod</th>
                    <th>Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {approvals
                    .filter((a) => a.status === 'PENDING')
                    .slice(0, 5)
                    .map((a) => (
                      <tr key={a.id}>
                        <td><b>{a.candidate_model_name}</b></td>
                        <td>v{a.candidate_version}</td>
                        <td>{a.current_production_version ? `v${a.current_production_version}` : 'None'}</td>
                        <td>
                          <StatusBadge status={a.status} />
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty text="All models are verified. Zero pending governance reviews." />
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
      announce(`Uploaded & auto-profiled ${uploaded.name}`)
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
          <h2>Datasets Ingested</h2>
          <label className="primary">
            <Upload size={14} />
            {busy ? 'Profiling…' : 'Upload CSV'}
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
                  {d.row_count.toLocaleString()} rows · {d.column_count} features
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
          <p className="empty">Generating dataset profiling and quality score…</p>
        ) : selected && profile ? (
          <>
            <div className="panel-head">
              <div>
                <h2>{selected.name}</h2>
                <span style={{ fontSize: '12px', color: '#94a3b8' }}>
                  Ingested {new Date(selected.created_at).toLocaleString()}
                </span>
              </div>
              <div
                className={`score-badge ${
                  profile.quality_score >= 80
                    ? 'score-good'
                    : profile.quality_score >= 60
                    ? 'score-warn'
                    : 'score-bad'
                }`}
              >
                <Sparkles size={16} />
                <span>Quality Score: {profile.quality_score} / 100</span>
              </div>
            </div>

            <div className="stats" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: '16px' }}>
              <div className="stat" style={{ padding: '14px' }}>
                <p>Rows</p>
                <strong style={{ fontSize: '22px' }}>{selected.row_count.toLocaleString()}</strong>
              </div>
              <div className="stat" style={{ padding: '14px' }}>
                <p>Features</p>
                <strong style={{ fontSize: '22px' }}>{selected.column_count}</strong>
              </div>
              <div className="stat" style={{ padding: '14px' }}>
                <p>Duplicates</p>
                <strong style={{ fontSize: '22px' }}>{selected.duplicate_rows}</strong>
              </div>
              <div className="stat" style={{ padding: '14px' }}>
                <p>Warnings</p>
                <strong style={{ fontSize: '22px', color: profile.warnings.length ? '#f59e0b' : '#10b981' }}>
                  {profile.warnings.length}
                </strong>
              </div>
            </div>

            {profile.warnings.length > 0 && (
              <>
                <h3>Data Quality Warnings</h3>
                <div className="warnings-box">
                  {profile.warnings.map((w, i) => (
                    <div key={i} className={`warning-item sev-${w.severity}`}>
                      <AlertTriangle size={15} />
                      <span className="badge-pill">{w.severity}</span>
                      <span>
                        <b>{w.column}:</b> {w.message}
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}

            <h3>Suggested Preprocessing Operations</h3>
            <ul style={{ paddingLeft: '20px', color: '#94a3b8', fontSize: '13px', marginBottom: '16px' }}>
              {profile.suggested_operations.map((op, i) => (
                <li key={i} style={{ marginBottom: '6px' }}>
                  {op}
                </li>
              ))}
            </ul>

            <h3>Feature Statistics & Outlier Counts (IQR)</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Type</th>
                    <th>Outliers</th>
                    <th>Mean / Mode</th>
                    <th>Std / Freq</th>
                    <th>Min / Uniques</th>
                    <th>Max</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(profile.statistics).map(([feat, st]) => (
                    <tr key={feat}>
                      <td><b>{feat}</b></td>
                      <td>{st.type}</td>
                      <td>
                        {profile.outliers[feat]?.outlier_count ? (
                          <span style={{ color: '#f59e0b' }}>
                            {profile.outliers[feat].outlier_count} ({profile.outliers[feat].outlier_percentage}%)
                          </span>
                        ) : (
                          '0'
                        )}
                      </td>
                      <td>{st.mean ?? st.top ?? '—'}</td>
                      <td>{st.std ?? st.top_freq ?? '—'}</td>
                      <td>{st.min ?? st.unique ?? '—'}</td>
                      <td>{st.max ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <h3>Sample Records (First 10)</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {selected.preview.length > 0 &&
                      Object.keys(selected.preview[0]).map((c) => <th key={c}>{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {selected.preview.slice(0, 10).map((row, i) => (
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
          <Empty text="Select a dataset on the left to inspect its profiling report." />
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
      announce('Generated algorithmic recommendations.')
    } catch (err) {
      announce((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <form className="panel form" onSubmit={analyze}>
        <h2>
          <Sparkles size={18} color="#818cf8" /> Algorithm Recommendation Engine
        </h2>
        <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '18px' }}>
          SentinelML evaluates dataset scale, dimensional cardinality, feature interactions, and
          class distributions to recommend candidate algorithms with transparent rationale.
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
          <Sparkles size={14} />
          {loading ? 'Evaluating Algorithms…' : 'Generate Model Recommendations'}
        </button>
      </form>

      {recResult && (
        <section className="panel">
          <h2>Recommendations for Target: {recResult.target_column}</h2>
          <p style={{ fontSize: '13px', color: '#a5b4fc', fontStyle: 'italic', marginBottom: '16px' }}>
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
                    Fit: {rec.score}%
                  </span>
                </h4>
                <p>
                  <b>Recommendation Status:</b> {rec.recommended ? 'Recommended Candidate' : 'Secondary Alternative'}
                </p>
                <ul>
                  {rec.reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
                <div className="rec-meta">
                  <span>Speed: {rec.estimated_speed}</span>
                  <span>Compute: {rec.computational_requirement}</span>
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: '24px' }}>
            <button className="primary" onClick={onTrainApplied}>
              <Play size={14} /> Proceed to Train Recommended Models →
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
        <h2>
          <Cpu size={18} color="#818cf8" /> Configure Distributed Training Run
        </h2>
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
            Target Column
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
            Problem Type
            <select value={type} onChange={(e) => setType(e.target.value as typeof type)}>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </label>
          <label>
            Primary Metric
            <select value={metric} onChange={(e) => setMetric(e.target.value)}>
              {(type === 'classification'
                ? ['f1', 'accuracy', 'precision', 'recall']
                : ['rmse', 'mae', 'r2']
              ).map((x) => (
                <option key={x}>{x.toUpperCase()}</option>
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
          <Play size={14} />
          {saving ? 'Queueing in Redis…' : 'Start Asynchronous Training Run'}
        </button>
      </form>

      <section className="panel">
        <h2>
          <Activity size={18} color="#818cf8" /> Training Jobs & MLflow Registration
        </h2>
        {jobs.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Dataset</th>
                  <th>Target</th>
                  <th>Status & Progress</th>
                  <th>Winning Algorithm</th>
                  <th>MLflow Run ID</th>
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
                      <span style={{ marginLeft: '8px', fontSize: '11px', color: '#94a3b8' }}>
                        {j.progress}%
                      </span>
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

// ----------------- 5. MODEL COMPARISON -----------------
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
          <h2>
            <Scale size={18} color="#818cf8" /> Multi-Objective Model Comparison & Ranking
          </h2>
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

        <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '20px' }}>
          In production MLOps, raw accuracy is insufficient. SentinelML normalizes predictive
          power, inference latency, training time, and artifact size into a balanced suitability index.
        </p>

        {chartData.length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            <h3>Multi-Objective Suitability Score</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: 'rgba(255,255,255,0.1)' }}
                />
                <Bar dataKey="Score" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {loading ? (
          <p className="empty">Computing multi-objective rankings…</p>
        ) : ranked.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Algorithm</th>
                  <th>{metric.toUpperCase()}</th>
                  <th>Latency</th>
                  <th>Train Duration</th>
                  <th>Size (KB)</th>
                  <th>Composite Score</th>
                  <th>Selection Rationale</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((m) => (
                  <tr key={m.model_name}>
                    <td>
                      <span
                        className="badge-pill"
                        style={{
                          background: m.rank === 1 ? 'rgba(16,185,129,0.15)' : 'rgba(255,255,255,0.05)',
                          color: m.rank === 1 ? '#34d399' : '#94a3b8',
                        }}
                      >
                        #{m.rank}
                      </span>
                    </td>
                    <td><b>{m.algorithm}</b></td>
                    <td>{m.metrics[metric] !== undefined ? Number(m.metrics[metric]).toFixed(4) : '—'}</td>
                    <td>{m.inference_latency_ms} ms</td>
                    <td>{m.training_time_seconds} s</td>
                    <td>{m.model_size_kb} KB</td>
                    <td>
                      <span className="status completed">{m.overall_score} / 100</span>
                    </td>
                    <td style={{ whiteSpace: 'normal', minWidth: '300px', fontSize: '12px' }}>
                      {m.selection_reasoning}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty text="No model evaluations found. Train models to view ranking trade-offs." />
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
      const res = await api.explanations(
        modelName,
        'latest',
        datasetId ? Number(datasetId) : undefined
      )
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
        <h2>
          <Search size={18} color="#818cf8" /> Model Explainability (SHAP Values)
        </h2>
        <div className="twocol">
          <label>
            Registered Model
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
              <option value="">Use latest uploaded dataset</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </label>
        </div>
        <button
          className="primary submit"
          disabled={loading || !modelName}
          onClick={loadExplanation}
        >
          <Search size={14} />
          {loading ? 'Computing Shapley Attributions…' : 'Generate Model Explainability Report'}
        </button>
      </section>

      {explanation && (
        <section className="panel">
          <div className="panel-head">
            <h2>Explainability: {explanation.model_name}</h2>
            <span className="badge-pill" style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8' }}>
              Attribution Method: {explanation.method}
            </span>
          </div>

          <p style={{ fontSize: '13px', color: '#a5b4fc', fontStyle: 'italic', marginBottom: '20px' }}>
            {explanation.summary}
          </p>

          <h3>Global Feature Importance (%)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis type="number" unit="%" stroke="#94a3b8" />
              <YAxis dataKey="name" type="category" width={110} stroke="#94a3b8" />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: 'rgba(255,255,255,0.1)' }}
              />
              <Bar dataKey="Impact" fill="#818cf8" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>

          <h3>Feature Importance Table</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Relative Contribution</th>
                  <th>Attribution Weight</th>
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
          <h2>
            <ShieldAlert size={18} color="#818cf8" /> Data Drift & Ground Truth Degradation
          </h2>
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
              <ShieldAlert size={14} />
              {testingDrift ? 'Analyzing Distributions…' : 'Simulate & Detect Drift'}
            </button>
          </div>
        </div>

        {perfReport && (
          <div
            style={{
              marginBottom: '20px',
              padding: '14px',
              borderRadius: '10px',
              background: perfReport.degradation_detected ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
              border: `1px solid ${perfReport.degradation_detected ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'}`,
            }}
          >
            <h4 style={{ margin: '0 0 6px', color: perfReport.degradation_detected ? '#f87171' : '#34d399' }}>
              Model Performance Status: {perfReport.status}
            </h4>
            <p style={{ margin: 0, fontSize: '13px', color: '#cbd5e1' }}>
              {perfReport.recommended_action}
            </p>
          </div>
        )}

        <h3>Historical Drift Reports (KS-Test & PSI)</h3>
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
                  <th>Score</th>
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
                      <span
                        className={`status ${
                          r.drift_severity === 'NONE'
                            ? 'completed'
                            : r.drift_severity === 'CRITICAL'
                            ? 'failed'
                            : 'queued'
                        }`}
                      >
                        {r.drift_severity}
                      </span>
                    </td>
                    <td>{r.overall_drift_score}</td>
                    <td style={{ whiteSpace: 'normal', minWidth: '280px', fontSize: '12px' }}>
                      {r.recommended_action}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty text="No drift analyses recorded yet. Click 'Simulate & Detect Drift' above." />
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
  const [reason, setReason] = useState('Feature drift exceeded 0.3 threshold')
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
      announce(`Retraining completed! Created candidate model v${job.candidate_version}`)
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
        <h2>
          <RefreshCw size={18} color="#818cf8" /> Trigger Intelligent Retraining Workflow
        </h2>
        <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '16px' }}>
          Retraining workflows safely train candidate models in background workers, evaluate against
          the production baseline, and generate an approval ticket without directly overwriting production.
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
            Trigger Rationale
            <input
              type="text"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g. KS-test detected drift on age/income"
            />
          </label>
        </div>
        <button className="primary submit" disabled={triggering || !modelName}>
          <RefreshCw size={14} />
          {triggering ? 'Executing Retraining Workflow…' : 'Trigger Automated Retraining'}
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
                  <th>Governance Gate</th>
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
                    <td>
                      {j.approval_id ? (
                        <span className="status queued">Approval Pending (#{j.approval_id})</span>
                      ) : (
                        '—'
                      )}
                    </td>
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
      await api.approveCandidate(id, {
        comments: comment || 'Verified evaluation metrics and promoted to production.',
        reviewer: 'Lead MLOps Engineer',
      })
      announce(`Candidate #${id} approved and promoted to PRODUCTION.`)
      setComment('')
      await onRefresh()
    } catch (err) {
      announce((err as Error).message)
    }
  }

  const handleReject = async (id: number) => {
    try {
      await api.rejectCandidate(id, {
        comments: comment || 'Did not meet latency threshold in audit.',
        reviewer: 'Lead MLOps Engineer',
      })
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
        <h2>
          <FileCheck size={18} color="#818cf8" /> Model Governance & Promotion Gate
        </h2>
        <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '18px' }}>
          Lifecycle management follows strict state transitions: DEVELOPMENT → CANDIDATE → APPROVED
          → PRODUCTION → ARCHIVED. Human review is enforced by default.
        </p>

        <h3>Pending Candidate Approvals</h3>
        {approvals.filter((a) => a.status === 'PENDING').length ? (
          approvals
            .filter((a) => a.status === 'PENDING')
            .map((appr) => (
              <div key={appr.id} className="approval-card">
                <div className="panel-head">
                  <h4>
                    Candidate Model: <b>{appr.candidate_model_name}</b> (v{appr.candidate_version})
                  </h4>
                  <span className="status queued">Review Pending</span>
                </div>
                <p style={{ fontSize: '12px', color: '#94a3b8', margin: '4px 0 12px' }}>
                  Current Active Production:{' '}
                  {appr.current_production_version ? `v${appr.current_production_version}` : 'None'}
                </p>
                <div style={{ marginBottom: '14px' }}>
                  <input
                    type="text"
                    placeholder="Enter review comments or reasoning…"
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                  />
                </div>
                <div className="actions-row">
                  <button className="btn-sm btn-approve" onClick={() => handleApprove(appr.id)}>
                    <CheckCircle2 size={13} /> Approve & Promote to Production
                  </button>
                  <button className="btn-sm btn-reject" onClick={() => handleReject(appr.id)}>
                    <XCircle size={13} /> Reject Candidate
                  </button>
                </div>
              </div>
            ))
        ) : (
          <Empty text="No candidate models pending approval." />
        )}

        <h3>Approval History</h3>
        {approvals.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Request ID</th>
                  <th>Model</th>
                  <th>Version</th>
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
                    <td style={{ fontSize: '12px' }}>{a.reviewer_comments ?? '—'}</td>
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
  const [reason, setReason] = useState('Observed latency spike in production traffic')
  const [busy, setBusy] = useState(false)

  const handleRollback = async () => {
    if (!modelToRollback) return announce('Select a model to roll back.')
    setBusy(true)
    try {
      const rec = await api.rollback(modelToRollback, { reason, initiated_by: 'MLOps Lead' })
      announce(`Successfully restored ${modelToRollback} to version v${rec.version}!`)
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
        <h2>
          <Rocket size={18} color="#818cf8" /> Production Deployments & Safe Rollback
        </h2>
        <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '18px' }}>
          SentinelML maintains immutable model lineage in MLflow. If an active model degrades in production,
          authorized users can instantly restore the previous verified version with zero downtime.
        </p>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '24px' }}>
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
            style={{ maxWidth: '360px' }}
          />
          <button
            className="btn-sm btn-rollback"
            onClick={handleRollback}
            disabled={busy || !modelToRollback}
          >
            <RotateCcw size={13} />
            {busy ? 'Restoring…' : 'Rollback Model'}
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
        <h2>
          <Settings size={18} color="#818cf8" /> Register Automation Rule
        </h2>
        <div className="twocol">
          <label>
            Rule Name
            <input
              type="text"
              value={ruleName}
              onChange={(e) => setRuleName(e.target.value)}
              placeholder="e.g. Continuous Drift Watcher"
            />
          </label>
          <label>
            Model Target
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
        <button className="primary submit">
          <Settings size={14} /> Register Automation Rule
        </button>
      </form>

      <section className="panel">
        <h2>Active Automation Rules</h2>
        {rules.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Rule</th>
                  <th>Target Model</th>
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
        <h2>
          <Terminal size={18} color="#818cf8" /> Governance Audit Trail
        </h2>
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
                      <span className="badge-pill" style={{ background: 'rgba(255,255,255,0.06)', color: '#e2e8f0' }}>
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
