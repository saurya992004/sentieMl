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
  datasets: () => request<Dataset[]>('/datasets'),
  dataset: (id: number) => request<DatasetDetail>(`/datasets/${id}`),
  upload: (file: File) => { const data = new FormData(); data.append('file', file); return request<Dataset>('/datasets/upload', { method: 'POST', body: data }) },
  jobs: () => request<Job[]>('/training'),
  job: (id: number) => request<Job>(`/training/${id}`),
  train: (body: TrainingBody) => request<Job>('/training/start', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) }),
  experiments: () => request<Experiment[]>('/experiments'),
  models: () => request<Model[]>('/models'),
  model: (name: string) => request<ModelDetail>(`/models/${encodeURIComponent(name)}`),
  predict: (body: {model_name:string, version:string, records:Record<string, unknown>[]}) => request<{predictions: unknown[]}>('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})
}
export type Dataset = {id:number; name:string; row_count:number; column_count:number; columns:Record<string,string>; missing_values:Record<string,number>; duplicate_rows:number; created_at:string}
export type DatasetDetail = Dataset & {preview: Record<string, unknown>[]}
export type Job = {id:number; dataset_id:number; target_column:string; problem_type:'classification'|'regression'; algorithms:string[]; primary_metric:string; status:'QUEUED'|'RUNNING'|'COMPLETED'|'FAILED'|'CANCELLED'; progress:number; logs:string[]; celery_task_id:string|null; attempt_count:number; cancel_requested:boolean; metrics:Record<string, Record<string, number|number[]>>|null; best_algorithm:string|null; mlflow_experiment_id:string|null; mlflow_run_id:string|null; registered_model_name:string|null; error:string|null; created_at:string}
export type TrainingBody = {dataset_id:number; target_column:string; problem_type:'classification'|'regression'; algorithms:string[]; primary_metric:string}
type Experiment = {experiment_id:string; name:string; lifecycle_stage:string}
export type Model = {name:string; description:string|null; latest_versions:{version:string;run_id:string;status:string}[]}
export type ModelDetail = {name:string; description:string|null; versions:{version:string;run_id:string;status:string;creation_timestamp:number}[]}

