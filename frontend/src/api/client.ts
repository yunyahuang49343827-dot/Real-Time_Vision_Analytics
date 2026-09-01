export type AnalysisMode = "standard" | "aerial"
export type JobLifecycle = "CREATED" | "PROCESSING" | "COMPLETED" | "FAILED"

export interface HealthResponse {
  status: string
  service: string
  runtime_model: string
}

export interface JobCreateResponse {
  job_id: string
  status: "CREATED"
}

export interface JobStatusResponse {
  job_id: string
  status: JobLifecycle
  progress: number
  analysis_mode: AnalysisMode
  processed_frames: number
  total_frames: number | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  error: { code: string; message: string } | null
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "")

export class ApiError extends Error {
  constructor(public readonly code: string, message: string) {
    super(message)
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init)
  } catch {
    throw new ApiError("BACKEND_UNAVAILABLE", "無法連線至分析服務，請確認 FastAPI 已啟動。")
  }
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = payload?.error
    throw new ApiError(detail?.code ?? "API_ERROR", detail?.message ?? "分析服務發生錯誤。")
  }
  return payload as T
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson("/health")
}

export function createAnalysisJob(file: File, analysisMode: AnalysisMode): Promise<JobCreateResponse> {
  const body = new FormData()
  body.append("video", file)
  body.append("analysis_mode", analysisMode)
  return requestJson("/jobs", { method: "POST", body })
}

export function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return requestJson(`/jobs/${encodeURIComponent(jobId)}`)
}
