import { Activity, Cpu, FileVideo2, Gauge, Hash, Timer } from "lucide-react"

import type { HealthResponse, JobResultResponse, JobStatusResponse } from "../../api/client"
import { Card, CardContent } from "../../components/ui/card"

function elapsedSeconds(status: JobStatusResponse): number | null {
  if (!status.started_at || !status.completed_at) return null
  const elapsed = (Date.parse(status.completed_at) - Date.parse(status.started_at)) / 1000
  return Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : null
}

export function EngineeringPage({ result, status, health }: { result: JobResultResponse; status: JobStatusResponse; health: HealthResponse }) {
  const elapsed = elapsedSeconds(status)
  const processingFps = elapsed && elapsed > 0 ? status.processed_frames / elapsed : null
  const profile = health.runtime_profiles?.[status.analysis_mode]
  const profileLabel = status.analysis_mode === "aerial" ? "Aerial" : "Standard"
  const details = [
    ["Job ID", status.job_id, Activity],
    ["Runtime model", health.runtime_model.split("/").at(-1) ?? health.runtime_model, Cpu],
    ["模型 SHA256", health.runtime_model_sha256 ?? "無資料", Hash],
    ["Runtime profile", profile ? `${profileLabel} · ${profile.imgsz} / ${profile.confidence_threshold}` : profileLabel, Gauge],
    ["Device", health.device?.toUpperCase() ?? "無資料", Cpu],
    ["處理 FPS", processingFps == null ? "無資料" : `${processingFps.toFixed(2)} FPS`, Gauge],
    ["解析度", `${result.video_metadata.width} × ${result.video_metadata.height}`, FileVideo2],
    ["Frame count", result.video_metadata.frame_count.toLocaleString(), FileVideo2],
    ["Codec", result.video_metadata.codec, FileVideo2],
    ["處理時間", elapsed == null ? "無資料" : `${elapsed.toFixed(2)} 秒`, Timer],
    ["API status", health.status === "ok" ? "API 正常" : "API 異常", Activity],
  ] as const
  return <div className="space-y-7">
    <div><p className="text-sm font-semibold text-teal-700">工程資訊</p><h1 className="mt-1 text-3xl font-bold tracking-tight">分析執行資訊</h1><p className="mt-2 text-sm text-slate-500">技術細節集中於此，不影響一般分析與事件判讀。</p></div>
    <Card><CardContent className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-3">{details.map(([label, value, Icon]) => <div key={label} className="min-w-0 rounded-xl border border-slate-100 bg-slate-50/80 p-4"><div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500"><Icon size={16} />{label}</div><p className="mt-2 break-all font-mono text-sm font-semibold text-slate-900">{value}</p></div>)}</CardContent></Card>
  </div>
}
