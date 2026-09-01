import Papa from "papaparse"

export type AnalysisMode = "standard" | "aerial"
export type JobLifecycle = "CREATED" | "PROCESSING" | "COMPLETED" | "FAILED"

export interface HealthResponse {
  status: string
  service: string
  runtime_model: string
  runtime_model_sha256: string
  device: string
  runtime_profiles: Record<string, { imgsz: number; confidence_threshold: number }>
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

export interface TrafficAnalytics {
  total_line_crossing_count: number
  person_crossing_count: number
  motorized_vehicle_crossing_count: number
  bicycle_crossing_count: number
  peak_interval_start_seconds: number | null
  peak_interval_end_seconds: number | null
  peak_interval_count: number
  zone_peak_occupancy: number
  density: string | null
  reconciliation_status: string
}

export interface EventSummary {
  event_type: string
  severity: string
  status: string
  count: number
}

export interface ArtifactReferences {
  tracking_browser_video?: string | null
  heatmap_browser_video?: string | null
  class_distribution_csv?: string | null
  direction_distribution_csv?: string | null
  traffic_over_time_csv?: string | null
  [key: string]: string | null | undefined
}

export interface JobResultResponse {
  job_id: string
  video_metadata: {
    filename: string; width: number; height: number; fps: number; frame_count: number
    duration_seconds: number; codec: string; validation_status: string
  }
  traffic_analytics: TrafficAnalytics
  event_summary: EventSummary[]
  artifacts: ArtifactReferences
  warnings: Array<{ code: string; message: string }>
}

export interface EventRecord {
  event_id: string
  event_type: string
  timestamp_seconds: number
  track_id: number | null
  secondary_track_id: number | null
  class_name: string | null
  severity: string
  status: string
  evidence_path: string | null
  rule_source?: string
  rule_value?: string
  threshold?: string
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "")
const API_TIMEOUT_MS = 15_000

export class ApiError extends Error {
  constructor(public readonly code: string, message: string) {
    super(message)
  }
}

async function fetchWithTimeout(url: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController()
  let timedOut = false
  const timeout = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, API_TIMEOUT_MS)
  try {
    return await fetch(url, { ...init, signal: controller.signal })
  } catch (error) {
    if (timedOut) throw new ApiError("API_TIMEOUT", "分析服務回應逾時，請稍後再試。")
    if (error instanceof ApiError) throw error
    throw new ApiError("BACKEND_UNAVAILABLE", "無法連線至分析服務，請確認 FastAPI 已啟動。")
  } finally {
    window.clearTimeout(timeout)
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetchWithTimeout(`${API_BASE_URL}${path}`, init)
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

export function getJobResults(jobId: string): Promise<JobResultResponse> {
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/results`)
}

export function getEvents(jobId: string): Promise<EventRecord[]> {
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/events`)
}

export function artifactUrl(jobId: string, artifactKey: string): string {
  return `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(artifactKey)}`
}

export function evidenceUrl(jobId: string, eventId: string): string {
  return `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/evidence/${encodeURIComponent(eventId)}`
}

export async function getArtifactRows<T extends Record<string, string>>(
  jobId: string,
  artifactKey: string,
): Promise<T[]> {
  const response = await fetchWithTimeout(artifactUrl(jobId, artifactKey))
  if (!response.ok) throw new ApiError("ARTIFACT_UNAVAILABLE", "分析資料目前無法取得。")
  const parsed = Papa.parse<T>(await response.text(), { header: true, skipEmptyLines: true })
  if (parsed.errors.length) throw new ApiError("ARTIFACT_INVALID", "分析資料格式無法解析。")
  return parsed.data
}
