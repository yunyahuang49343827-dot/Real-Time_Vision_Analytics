import { ImageOff, Search, ShieldAlert } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

import { evidenceUrl, type EventRecord } from "../../api/client"
import { Card, CardContent } from "../../components/ui/card"
import { cn } from "../../lib/utils"
import { eventInterpretation, eventLabels, filterEvents, formatClock, type EventFilter } from "./presentation"

const filters: Array<[EventFilter, string]> = [["all", "全部"], ["warning", "警示"], ["info", "資訊"], ["review", "需人工確認"]]

export function EventReviewPage({ jobId, events, initialEventId }: { jobId: string; events: EventRecord[]; initialEventId?: string | null }) {
  const [filter, setFilter] = useState<EventFilter>("all")
  const [selectedId, setSelectedId] = useState(initialEventId ?? events[0]?.event_id ?? null)
  const [imageFailed, setImageFailed] = useState(false)
  const filtered = useMemo(() => filterEvents(events, filter), [events, filter])
  const selected = filtered.find((event) => event.event_id === selectedId) ?? filtered[0] ?? null
  useEffect(() => setImageFailed(false), [selected?.event_id])
  return <div className="space-y-7"><div><p className="text-sm font-semibold text-teal-700">事件檢視</p><h1 className="mt-1 text-3xl font-bold">事件與 Evidence Workspace</h1><p className="mt-2 text-sm text-slate-500">規則判定候選需要人工情境確認。</p></div>
    <div className="flex flex-wrap gap-2">{filters.map(([value, label]) => <button key={value} type="button" onClick={() => setFilter(value)} className={cn("rounded-full border px-4 py-2 text-sm font-semibold", filter === value ? "border-teal-700 bg-teal-700 text-white" : "border-slate-200 bg-white text-slate-600")}>{label}</button>)}</div>
    <div className="grid min-h-[620px] gap-6 xl:grid-cols-[380px_minmax(0,1fr)]">
      <div className="space-y-3 overflow-y-auto pr-1 xl:max-h-[720px]">{filtered.length ? filtered.map((event) => <button key={event.event_id} type="button" onClick={() => setSelectedId(event.event_id)} className={cn("w-full rounded-xl border bg-white p-4 text-left transition-colors", selected?.event_id === event.event_id ? "border-teal-600 ring-1 ring-teal-600" : "border-slate-200 hover:border-slate-300")}><div className="flex items-start justify-between gap-3"><p className="text-sm font-bold">{eventLabels[event.event_type] ?? event.event_type}</p><span className={cn("rounded-full px-2 py-1 text-[10px] font-bold", event.severity === "CRITICAL" ? "bg-rose-50 text-rose-700" : event.severity === "WARNING" ? "bg-orange-50 text-orange-700" : "bg-slate-100 text-slate-600")}>{event.severity}</span></div><p className="mt-2 text-xs text-slate-500">{formatClock(event.timestamp_seconds)} · Track ID {event.track_id ?? "—"}</p><p className="mt-3 text-xs font-semibold text-slate-700">{event.status}</p></button>) : <Card><CardContent className="py-12 text-center text-sm text-slate-500"><Search className="mx-auto mb-3" />此篩選條件沒有事件。</CardContent></Card>}</div>
      <Card className="h-fit overflow-hidden">{selected ? <><div className="bg-slate-100">{selected.evidence_path && !imageFailed ? <img src={evidenceUrl(jobId, selected.event_id)} onError={() => setImageFailed(true)} alt={`${(eventLabels[selected.event_type] ?? selected.event_type).split("｜").at(-1)}事件證據`} className="aspect-video w-full object-contain" /> : <div className="grid aspect-video place-items-center px-8 text-center text-sm text-slate-500"><div><ImageOff className="mx-auto mb-3" /><p>此事件沒有可用的 Evidence Snapshot。</p></div></div>}</div><CardContent className="space-y-5 py-6"><div className="flex items-start justify-between gap-4"><div><p className="text-sm text-slate-500">事件類型</p><h2 className="mt-1 text-xl font-bold">{eventLabels[selected.event_type] ?? selected.event_type}</h2></div><ShieldAlert className="text-orange-600" /></div><div className="grid gap-4 text-sm sm:grid-cols-2">{[["時間", formatClock(selected.timestamp_seconds)], ["Track ID", selected.track_id ?? "—"], ["Severity", selected.severity], ["Status", selected.status]].map(([label, value]) => <div key={label} className="rounded-lg bg-slate-50 p-3"><p className="text-xs text-slate-500">{label}</p><p className="mt-1 font-semibold">{value}</p></div>)}</div><div className="rounded-xl border border-orange-200 bg-orange-50 p-4 text-sm leading-6 text-orange-900"><strong className="block">判讀說明</strong>{eventInterpretation(selected.event_type)}</div></CardContent></> : <CardContent className="py-20 text-center text-sm text-slate-500">選擇左側事件以查看內容。</CardContent>}</Card>
    </div>
  </div>
}
