import { AlertTriangle, ArrowRight, CarFront, Clock3, Eye, Flag, Route } from "lucide-react"
import { useMemo, useState } from "react"

import { artifactUrl, type EventRecord, type JobResultResponse } from "../../api/client"
import { Button } from "../../components/ui/button"
import { Card, CardContent } from "../../components/ui/card"
import { cn } from "../../lib/utils"
import { buildOverviewKpis, eventInterpretation, eventLabels, formatClock, mapClassRows, mapDirectionRows, mapTrafficRows } from "./presentation"

function severityClass(severity: string) {
  if (severity === "CRITICAL") return "border-rose-200 bg-rose-50 text-rose-700"
  if (severity === "WARNING") return "border-orange-200 bg-orange-50 text-orange-700"
  return "border-slate-200 bg-slate-50 text-slate-600"
}

export function OverviewPage({ jobId, result, events, classRows, directionRows, trafficRows, analyticsLoading, onReviewEvent }: {
  jobId: string; result: JobResultResponse; events: EventRecord[]
  classRows: Array<Record<string, string>>; directionRows: Array<Record<string, string>>
  trafficRows: Array<Record<string, string>>; analyticsLoading: boolean
  onReviewEvent: (eventId: string) => void
}) {
  const [mode, setMode] = useState<"tracking" | "heatmap">("tracking")
  const kpis = buildOverviewKpis(result, classRows)
  const primaryDirection = [...mapDirectionRows(directionRows)].sort((a, b) => b.count - a.count)[0]?.name ?? "無資料"
  const reviewCount = result.event_summary.filter((item) => item.status === "REVIEW_REQUIRED").reduce((sum, item) => sum + item.count, 0)
  const videoKey = mode === "tracking" ? "tracking_browser_video" : "heatmap_browser_video"
  const timeline = useMemo(() => [...events].sort((a, b) => a.timestamp_seconds - b.timestamp_seconds), [events])
  const recent = [...timeline].reverse().slice(0, 4)
  const classData = mapClassRows(classRows).slice(0, 5)
  const trafficData = mapTrafficRows(trafficRows).slice(-8)
  const classMax = Math.max(...classData.map((item) => item.count), 1)
  const trafficMax = Math.max(...trafficData.map((item) => item.count), 1)
  const kpiItems = [
    { label: "通過計數線", value: kpis.crossings, icon: Route },
    { label: "需關注事件", value: kpis.attentionEvents, icon: AlertTriangle },
    { label: "主要車種", value: kpis.primaryVehicle, icon: CarFront },
    { label: "車流高峰區間", value: kpis.peakInterval, icon: Clock3 },
  ]
  return <div className="min-w-0 space-y-7">
    <div><p className="text-sm font-semibold text-teal-700">分析總覽</p><h1 className="mt-1 text-2xl font-bold tracking-tight sm:text-3xl">交通場景分析結果</h1></div>
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{kpiItems.map(({ label, value, icon: Icon }) => <Card key={label}><CardContent className="flex items-start justify-between gap-3 p-5"><div className="min-w-0"><p className="text-sm text-slate-500">{label}</p><p className="mt-2 truncate text-2xl font-bold">{value}</p></div><span className="shrink-0 rounded-xl bg-teal-50 p-2.5 text-teal-700"><Icon size={20} /></span></CardContent></Card>)}</div>

    <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_320px]">
      <Card className="min-w-0 overflow-hidden">
        <div className="flex flex-col gap-3 border-b border-slate-100 p-4 sm:flex-row sm:items-center sm:justify-between sm:px-5">
          <div><h2 className="font-bold">分析影片</h2><p className="text-xs text-slate-500">切換已產生的 artifact，不會重新分析。</p></div>
          <div role="group" aria-label="影片視覺化模式" className="grid w-full grid-cols-2 rounded-xl border border-slate-200 bg-slate-100 p-1 sm:w-auto">
            <button type="button" aria-pressed={mode === "tracking"} onClick={() => setMode("tracking")} className={cn("rounded-lg px-3 py-2 text-xs font-semibold transition-all", mode === "tracking" ? "bg-white text-teal-800 shadow-sm" : "text-slate-500 hover:text-slate-800")}>追蹤／移動軌跡</button>
            <button type="button" aria-pressed={mode === "heatmap"} onClick={() => setMode("heatmap")} className={cn("rounded-lg px-3 py-2 text-xs font-semibold transition-all", mode === "heatmap" ? "bg-white text-teal-800 shadow-sm" : "text-slate-500 hover:text-slate-800")}>交通活動熱圖</button>
          </div>
        </div>
        <div className="bg-slate-950">{result.artifacts[videoKey] ? <video key={videoKey} data-testid="analysis-video" src={artifactUrl(jobId, videoKey)} controls className="block aspect-video h-auto w-full max-w-full object-contain" /> : <div className="grid aspect-video place-items-center px-6 text-center text-sm text-white">瀏覽器相容的分析影片目前無法使用。</div>}</div>
        <div className="border-t border-slate-100 px-5 py-3 text-xs leading-5 text-slate-500">{mode === "heatmap" ? "Heatmap 只代表影像座標中的交通活動分布，不代表實際交通密度或事故風險。" : "Tracking View 沿用 bbox、Track ID、trajectory、line 與 event overlay；Zone / ROI 已在視覺化層弱化。"}</div>
      </Card>
      <Card><CardContent className="space-y-5 p-5"><h2 className="font-bold">分析摘要</h2>{[["待人工確認", `${reviewCount} 件`], ["主要方向", primaryDirection], ["主要車種", kpis.primaryVehicle], ["高峰區間", kpis.peakInterval]].map(([label, value]) => <div key={label} className="flex items-center justify-between gap-4 border-b border-slate-100 pb-3 text-sm"><span className="text-slate-500">{label}</span><strong className="text-right">{value}</strong></div>)}<div className="rounded-xl bg-orange-50 p-4 text-xs leading-5 text-orange-900"><strong className="mb-1 block">判讀提醒</strong>規則事件與影像座標分析均需搭配人工情境判讀。</div></CardContent></Card>
    </div>

    <div className="grid min-w-0 gap-5 lg:grid-cols-2">
      <Card><CardContent className="p-5"><div className="mb-4 flex items-center justify-between"><h2 className="font-bold">車種概況</h2><span className="text-xs text-slate-400">Crossing count</span></div>{analyticsLoading ? <p className="py-10 text-center text-sm text-slate-500">正在載入交通資料…</p> : classData.length ? <div className="space-y-3">{classData.map((item) => <div key={item.name} className="grid grid-cols-[64px_minmax(0,1fr)_36px] items-center gap-3 text-xs"><span className="font-semibold text-slate-700">{item.name}</span><div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-teal-600" style={{ width: `${Math.max(4, item.count / classMax * 100)}%` }} /></div><strong className="text-right">{item.count}</strong></div>)}</div> : <p className="py-10 text-center text-sm text-slate-500">尚無交通資料</p>}</CardContent></Card>
      <Card><CardContent className="p-5"><div className="mb-4 flex items-center justify-between"><h2 className="font-bold">車流趨勢</h2><span className="text-xs text-slate-400">時間區間</span></div>{analyticsLoading ? <p className="py-10 text-center text-sm text-slate-500">正在載入交通資料…</p> : trafficData.length ? <div className="flex h-32 min-w-0 items-end gap-2" aria-label="車流趨勢圖">{trafficData.map((item) => <div key={`${item.interval}-${item.end}`} className="flex min-w-0 flex-1 flex-col items-center gap-2"><span className="text-[10px] font-semibold text-slate-500">{item.count}</span><div className="w-full rounded-t-md bg-cyan-700" style={{ height: `${Math.max(8, item.count / trafficMax * 80)}px` }} /><span className="w-full truncate text-center text-[9px] text-slate-400">{item.interval}</span></div>)}</div> : <p className="py-10 text-center text-sm text-slate-500">尚無交通資料</p>}</CardContent></Card>
    </div>

    <Card><CardContent className="p-5"><div className="mb-5 flex items-center gap-2"><Flag className="text-teal-700" size={18} /><h2 className="font-bold">Event Timeline</h2></div>{timeline.length ? <div className="overflow-x-auto"><div className="flex min-w-max gap-3 pb-2">{timeline.map((event) => <div key={event.event_id} className="w-64 rounded-xl border border-slate-200 p-4"><p className="text-xs font-semibold text-teal-700">{formatClock(event.timestamp_seconds)}</p><p className="mt-2 text-sm font-bold">{eventLabels[event.event_type] ?? event.event_type}</p><div className="mt-3 flex flex-wrap gap-2"><span className={cn("rounded-full border px-2 py-1 text-[10px] font-bold", severityClass(event.severity))}>{event.severity}</span><span className="rounded-full bg-violet-50 px-2 py-1 text-[10px] font-bold text-violet-700">{event.status}</span></div></div>)}</div></div> : <p className="py-6 text-center text-sm text-slate-500">無事件</p>}</CardContent></Card>
    <div><div className="mb-4 flex items-center justify-between"><h2 className="text-lg font-bold">近期事件</h2><span className="text-xs text-slate-500">最多 4 筆</span></div>{recent.length ? <div className="grid gap-4 md:grid-cols-2">{recent.map((event) => <Card key={event.event_id} data-testid="recent-event"><CardContent className="p-5"><div className="flex items-start justify-between gap-4"><div><p className="font-bold">{eventLabels[event.event_type] ?? event.event_type}</p><p className="mt-1 text-sm text-slate-500">{formatClock(event.timestamp_seconds)} · Track ID {event.track_id ?? "—"}</p></div><span className={cn("rounded-full border px-2 py-1 text-[10px] font-bold", severityClass(event.severity))}>{event.status}</span></div><p className="mt-4 text-xs leading-5 text-slate-500">{eventInterpretation(event.event_type)}</p><Button variant="ghost" className="mt-3 px-0 text-teal-700" onClick={() => onReviewEvent(event.event_id)}><Eye size={16} />查看證據<ArrowRight size={14} /></Button></CardContent></Card>)}</div> : <p className="rounded-xl border border-dashed border-slate-200 bg-white py-10 text-center text-sm text-slate-500">無事件</p>}</div>
  </div>
}
