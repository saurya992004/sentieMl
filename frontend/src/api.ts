const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, options)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed (${response.status})`)
  }
  return response.json()
}

export const api = {
  // Datasets & Profiling
  datasets: () => request<Dataset[]>('/datasets'),
  dataset: (id: number) => request<DatasetDetail>(`/datasets/${id}`),
  upload: (file: File) => {
    const data = new FormData()
    data.append('file', file)
    return request<Dataset>('/datasets/upload', { method: 'POST', body: data })
  },
  datasetProfile: (id: number, target?: string) =>
    request<DatasetProfile>(`/datasets/${id}/profile${target ? `?target_column=${encodeURIComponent(target)}` : ''}`),
  runProfile: (id: number, target?: string) =>
    request<DatasetProfile>(`/datasets/${id}/profile${target ? `?target_column=${encodeURIComponent(target)}` : ''}`, { method: 'POST' }),

  // Training & Jobs
  jobs: () => request<Job[]>('/training'),
  job: (id: number) => request<Job>(`/training/${id}`),
  train: (body: TrainingBody) =>
    request<Job>('/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  // Recommendations & Multi-objective Comparisons
  recommend: (body: { dataset_id: number; target_column: string; problem_type?: string }) =>
    request<RecommendationResponse>('/models/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  compare: (problem_type = 'classification', primary_metric = 'f1') =>
    request<ModelRankingItem[]>(`/models/compare?problem_type=${problem_type}&primary_metric=${primary_metric}`),

  // Explainability
  explanations: (name: string, version = 'latest', datasetId?: number) =>
    request<ExplanationResponse>(
      `/models/${encodeURIComponent(name)}/explanations?version=${version}${datasetId ? `&dataset_id=${datasetId}` : ''}`
    ),

  // Drift & Monitoring
  driftReports: (modelName?: string) =>
    request<DriftReport[]>(`/monitoring/drift${modelName ? `?model_name=${encodeURIComponent(modelName)}` : ''}`),
  detectDrift: (body: { model_name: string; version?: string; production_records: Record<string, unknown>[] }) =>
    request<DriftReport>('/monitoring/drift', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  performance: (modelName: string, version = 'latest') =>
    request<PerformanceReport>(`/monitoring/performance?model_name=${encodeURIComponent(modelName)}&version=${version}`),
  submitGroundTruth: (body: { model_name: string; version?: string; predictions_with_actuals: Record<string, unknown>[] }) =>
    request<{ status: string; records_count: number }>('/monitoring/ground-truth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  // Retraining & Automation
  retrainingRules: () => request<RetrainingRule[]>('/retraining/rules'),
  createRetrainingRule: (body: Partial<RetrainingRule>) =>
    request<RetrainingRule>('/retraining/configure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  triggerRetraining: (body: { model_name: string; trigger_reason?: string; dataset_id?: number }) =>
    request<RetrainingJob>('/retraining/trigger', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  retrainingHistory: () => request<RetrainingJob[]>('/retraining/history'),

  // Approvals & Governance
  approvals: () => request<ModelApproval[]>('/approvals'),
  approveCandidate: (id: number, body: { comments?: string; reviewer?: string }) =>
    request<ModelApproval>(`/approvals/${id}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  rejectCandidate: (id: number, body: { comments?: string; reviewer?: string }) =>
    request<ModelApproval>(`/approvals/${id}/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  // Deployments & Rollback
  promote: (name: string, body: { version: string; promoter?: string; comments?: string }) =>
    request<{ status: string; model_name: string; version: string; stage: string }>(
      `/models/${encodeURIComponent(name)}/promote`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }
    ),
  rollback: (name: string, body: { target_version?: string; reason?: string; initiated_by?: string }) =>
    request<DeploymentRecord>(`/models/${encodeURIComponent(name)}/rollback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  deployments: () => request<DeploymentRecord[]>('/deployments/history'),

  // Governance & Audit
  auditLogs: () => request<AuditLog[]>('/audit-logs'),

  // Core MLflow & Predictions
  experiments: () => request<Experiment[]>('/experiments'),
  models: () => request<Model[]>('/models'),
  model: (name: string) => request<ModelDetail>(`/models/${encodeURIComponent(name)}`),
  predict: (body: { model_name: string; version: string; records: Record<string, unknown>[] }) =>
    request<{ predictions: unknown[] }>('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
}

export type Dataset = {
  id: number
  name: string
  row_count: number
  column_count: number
  columns: Record<string, string>
  missing_values: Record<string, number>
  duplicate_rows: number
  created_at: string
}

export type DatasetDetail = Dataset & { preview: Record<string, unknown>[] }

export type DatasetProfile = {
  dataset_id: number
  quality_score: number
  warnings: { column: string; severity: string; message: string }[]
  statistics: Record<string, any>
  outliers: Record<string, { lower_bound: number | null; upper_bound: number | null; outlier_count: number; outlier_percentage: number }>
  correlations: Record<string, Record<string, number>>
  target_analysis: { target_column: string; type: string; unique_count: number; distribution: Record<string, number>; imbalance_ratio: number; is_imbalanced: boolean } | null
  suggested_operations: string[]
  risks: string[]
}

export type AlgorithmRecommendation = {
  algorithm: string
  display_name: string
  score: number
  recommended: boolean
  reasons: string[]
  estimated_speed: string
  computational_requirement: string
}

export type RecommendationResponse = {
  dataset_id: number
  target_column: string
  problem_type: string
  recommended_algorithms: AlgorithmRecommendation[]
  explanation: string
  dataset_summary: Record<string, any>
}

export type ModelRankingItem = {
  model_name: string
  version: string
  algorithm: string
  problem_type: string
  metrics: Record<string, number>
  training_time_seconds: number
  inference_latency_ms: number
  model_size_kb: number
  overall_score: number
  rank: number
  selection_reasoning: string
}

export type ExplanationResponse = {
  model_name: string
  version: string
  algorithm: string
  feature_names: string[]
  global_importance: { feature: string; importance: number; percentage: number }[]
  local_explanation?: { feature: string; value: any; relative_impact: number; importance_pct: number }[]
  method: string
  summary: string
}

export type DriftReport = {
  id: number
  model_name: string
  model_version: string
  reference_dataset_id: number | null
  sample_count: number
  feature_drift: Record<string, any>
  overall_drift_score: number
  drift_severity: 'NONE' | 'LOW' | 'MODERATE' | 'CRITICAL'
  drift_detected: boolean
  recommended_action: string
  created_at: string
}

export type PerformanceReport = {
  sample_count: number
  current_metrics: Record<string, number>
  baseline_metrics: Record<string, number>
  degradation_detected: boolean
  status: string
  degradation_details: string[]
  recommended_action: string
}

export type RetrainingRule = {
  id: number
  name: string
  model_name: string
  trigger_type: string
  threshold: number | null
  metric_name: string | null
  is_active: boolean
  cooldown_minutes: number
  last_triggered_at: string | null
  created_at: string
}

export type RetrainingJob = {
  id: number
  rule_id: number | null
  model_name: string
  trigger_reason: string
  status: string
  candidate_model_name: string | null
  candidate_version: string | null
  evaluation_summary: Record<string, any> | null
  approval_id: number | null
  error: string | null
  created_at: string
  completed_at: string | null
}

export type ModelApproval = {
  id: number
  candidate_model_name: string
  candidate_version: string
  current_production_version: string | null
  comparison_summary: Record<string, any>
  status: 'PENDING' | 'APPROVED' | 'REJECTED'
  reviewer_comments: string | null
  reviewed_by: string | null
  created_at: string
  resolved_at: string | null
}

export type DeploymentRecord = {
  id: number
  model_name: string
  version: string
  previous_version: string | null
  action: 'PROMOTE' | 'ROLLBACK'
  deployment_config: Record<string, any> | null
  initiated_by: string
  created_at: string
}

export type AuditLog = {
  id: number
  action: string
  entity_type: string
  entity_id: string | null
  actor: string
  details: Record<string, any> | null
  created_at: string
}

export type Job = {
  id: number
  dataset_id: number
  target_column: string
  problem_type: 'classification' | 'regression'
  algorithms: string[]
  primary_metric: string
  status: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED'
  progress: number
  logs: string[]
  celery_task_id: string | null
  attempt_count: number
  cancel_requested: boolean
  metrics: Record<string, Record<string, number | number[]>> | null
  best_algorithm: string | null
  mlflow_experiment_id: string | null
  mlflow_run_id: string | null
  registered_model_name: string | null
  error: string | null
  created_at: string
}

export type TrainingBody = {
  dataset_id: number
  target_column: string
  problem_type: 'classification' | 'regression'
  algorithms: string[]
  primary_metric: string
}

type Experiment = { experiment_id: string; name: string; lifecycle_stage: string }
export type Model = { name: string; description: string | null; latest_versions: { version: string; run_id: string; status: string }[] }
export type ModelDetail = { name: string; description: string | null; versions: { version: string; run_id: string; status: string; creation_timestamp: number }[] }
