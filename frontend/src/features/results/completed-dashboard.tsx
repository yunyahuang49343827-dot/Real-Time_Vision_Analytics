import { lazy, Suspense, useState } from "react"

import type { EventRecord, JobResultResponse } from "../../api/client"
import { AppShell, type DashboardSection } from "../../components/app-shell"
import { Card, CardContent } from "../../components/ui/card"
import { EventReviewPage } from "./event-review-page"
import { OverviewPage } from "./overview-page"
import { useAnalyticsArtifacts } from "./result-hooks"

const AnalyticsPage = lazy(() => import("./analytics-page").then((module) => ({ default: module.AnalyticsPage })))

export function CompletedDashboard({ jobId, result, events, onNewAnalysis }: { jobId: string; result: JobResultResponse; events: EventRecord[]; onNewAnalysis?: () => void }) {
  const [section, setSection] = useState<DashboardSection>("overview")
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null)
  const analytics = useAnalyticsArtifacts(jobId, result.artifacts)
  function navigate(next: DashboardSection) {
    if (next === "new" && onNewAnalysis) return onNewAnalysis()
    setSection(next)
  }
  function reviewEvent(eventId: string) {
    setSelectedEventId(eventId)
    setSection("events")
  }
  return <AppShell completed activeSection={section} onNavigate={navigate}>
    {section === "overview" && <OverviewPage jobId={jobId} result={result} events={events} classRows={analytics.classRows} directionRows={analytics.directionRows} onReviewEvent={reviewEvent} />}
    {section === "analytics" && <Suspense fallback={<p className="text-sm text-slate-500">正在載入交通分析…</p>}><AnalyticsPage result={result} classRows={analytics.classRows} directionRows={analytics.directionRows} trafficRows={analytics.trafficRows} /></Suspense>}
    {section === "events" && <EventReviewPage jobId={jobId} events={events} initialEventId={selectedEventId} />}
    {section === "engineering" && <div><p className="text-sm font-semibold text-teal-700">工程資訊</p><h1 className="mt-1 text-3xl font-bold">分析執行資訊</h1><Card className="mt-6"><CardContent className="grid gap-4 py-6 text-sm md:grid-cols-3"><div><p className="text-slate-500">Runtime model</p><strong>yolo26n.pt</strong></div><div><p className="text-slate-500">影片解析度</p><strong>{result.video_metadata.width} × {result.video_metadata.height}</strong></div><div><p className="text-slate-500">Reconciliation</p><strong>{result.traffic_analytics.reconciliation_status}</strong></div></CardContent></Card></div>}
  </AppShell>
}
