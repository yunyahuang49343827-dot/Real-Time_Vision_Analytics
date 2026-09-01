import { lazy, Suspense, useState } from "react"

import type { EventRecord, HealthResponse, JobResultResponse, JobStatusResponse } from "../../api/client"
import { AppShell, type DashboardSection } from "../../components/app-shell"
import { EventReviewPage } from "./event-review-page-final"
import { EngineeringPage } from "./engineering-page"
import { OverviewPage } from "./overview-page-final"
import { useAnalyticsArtifacts } from "./result-hooks"

const AnalyticsPage = lazy(() => import("./analytics-page-final").then((module) => ({ default: module.AnalyticsPage })))

export function CompletedDashboard({ jobId, result, events, status, health, onNewAnalysis }: { jobId: string; result: JobResultResponse; events: EventRecord[]; status: JobStatusResponse; health: HealthResponse; onNewAnalysis?: () => void }) {
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
    {section === "overview" && <OverviewPage jobId={jobId} result={result} events={events} classRows={analytics.classRows} directionRows={analytics.directionRows} trafficRows={analytics.trafficRows} analyticsLoading={analytics.loading} onReviewEvent={reviewEvent} />}
    {section === "analytics" && <Suspense fallback={<p className="text-sm text-slate-500">正在載入交通分析…</p>}><AnalyticsPage result={result} classRows={analytics.classRows} directionRows={analytics.directionRows} trafficRows={analytics.trafficRows} loading={analytics.loading} error={analytics.error} /></Suspense>}
    {section === "events" && <EventReviewPage jobId={jobId} events={events} initialEventId={selectedEventId} />}
    {section === "engineering" && <EngineeringPage result={result} status={status} health={health} />}
  </AppShell>
}
