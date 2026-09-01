import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { EventRecord, JobResultResponse } from "../../api/client"
import { CompletedDashboard } from "./completed-dashboard"

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

const result: JobResultResponse = {
  job_id: "job-1",
  video_metadata: { filename: "traffic.mp4", width: 1920, height: 1080, fps: 30, frame_count: 300, duration_seconds: 10, codec: "h264", validation_status: "PASS" },
  traffic_analytics: { total_line_crossing_count: 18, person_crossing_count: 2, motorized_vehicle_crossing_count: 15, bicycle_crossing_count: 1, peak_interval_start_seconds: 20, peak_interval_end_seconds: 30, peak_interval_count: 7, zone_peak_occupancy: 12, density: "HIGH", reconciliation_status: "PASS" },
  event_summary: [
    { event_type: "LINE_CROSSING", severity: "INFO", status: "DETECTED", count: 18 },
    { event_type: "PROXIMITY_WARNING", severity: "WARNING", status: "REVIEW_REQUIRED", count: 3 },
    { event_type: "WRONG_WAY", severity: "CRITICAL", status: "REVIEW_REQUIRED", count: 2 },
  ],
  artifacts: {
    tracking_browser_video: "tracking_browser.mp4",
    heatmap_browser_video: "heatmap_browser.mp4",
    class_distribution_csv: "class_distribution.csv",
    direction_distribution_csv: "direction_distribution.csv",
    traffic_over_time_csv: "traffic_over_time.csv",
  },
  warnings: [],
}

const events: EventRecord[] = [
  { event_id: "evt-info", event_type: "LINE_CROSSING", timestamp_seconds: 1, track_id: 2, secondary_track_id: null, class_name: "car", severity: "INFO", status: "DETECTED", evidence_path: null },
  { event_id: "evt-proximity", event_type: "PROXIMITY_WARNING", timestamp_seconds: 3.2, track_id: 7, secondary_track_id: 9, class_name: "person", severity: "WARNING", status: "REVIEW_REQUIRED", evidence_path: "evidence/evt-proximity.jpg" },
  { event_id: "evt-wrong", event_type: "WRONG_WAY", timestamp_seconds: 5.4, track_id: 12, secondary_track_id: null, class_name: "car", severity: "CRITICAL", status: "REVIEW_REQUIRED", evidence_path: null },
]

function csvResponse(url: string) {
  if (url.includes("class_distribution_csv")) return "class_name,crossing_count,percentage\ncar,12,66.7\nmotorcycle,4,22.2\nperson,2,11.1\n"
  if (url.includes("direction_distribution_csv")) return "crossing_direction,crossing_count,percentage\nA_TO_B,11,61.1\nB_TO_A,7,38.9\n"
  return "interval_start_seconds,interval_end_seconds,total_crossing_count\n0,10,3\n10,20,8\n"
}

function renderDashboard() {
  const fetchMock = vi.fn().mockImplementation((input: string) => Promise.resolve(new Response(csvResponse(input), { status: 200 })))
  vi.stubGlobal("fetch", fetchMock)
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={client}><CompletedDashboard jobId="job-1" result={result} events={events} /></QueryClientProvider>)
  return fetchMock
}

describe("React B completed analysis workspace", () => {
  it("renders governed KPIs and switches existing video artifacts without creating a job", async () => {
    const fetchMock = renderDashboard()
    const user = userEvent.setup()
    expect(await screen.findByText("18")).toBeInTheDocument()
    expect(screen.getByText("5")).toBeInTheDocument()
    expect((await screen.findAllByText("汽車")).length).toBeGreaterThan(0)
    expect(screen.getAllByText("00:20–00:30").length).toBeGreaterThan(0)
    const video = screen.getByTestId("analysis-video")
    expect(video).toHaveAttribute("src", expect.stringContaining("tracking_browser_video"))
    await user.click(screen.getByRole("button", { name: "交通活動熱圖" }))
    expect(screen.getByTestId("analysis-video")).toHaveAttribute("src", expect.stringContaining("heatmap_browser_video"))
    expect(screen.getByText(/Heatmap 只代表影像座標中的交通活動分布/)).toBeInTheDocument()
    expect(fetchMock.mock.calls.every(([, options]) => options?.method !== "POST")).toBe(true)
  })

  it("renders timeline, mapped analytics charts, and zone peak occupancy", async () => {
    renderDashboard()
    const user = userEvent.setup()
    expect(screen.getAllByText("PROXIMITY_WARNING｜接近警示").length).toBeGreaterThan(0)
    expect(screen.queryByText("evt-proximity")).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "交通分析" }))
    expect(await screen.findByText("車種分布")).toBeInTheDocument()
    expect(screen.getByText("Traffic Over Time")).toBeInTheDocument()
    expect(screen.getByText("Direction Distribution")).toBeInTheDocument()
    expect(screen.getByText("12 輛次觀測峰值")).toBeInTheDocument()
    expect(screen.getByText("汽車：12")).toBeInTheDocument()
  })

  it("filters events, loads evidence, and handles missing evidence safely", async () => {
    renderDashboard()
    const user = userEvent.setup()
    await user.click(screen.getByRole("button", { name: "事件檢視" }))
    await user.click(screen.getByRole("button", { name: "需人工確認" }))
    expect(screen.queryByText("LINE_CROSSING｜通過計數線")).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: /PROXIMITY_WARNING｜接近警示/ }))
    const evidence = screen.getByAltText("接近警示事件證據")
    expect(evidence).toHaveAttribute("src", expect.stringContaining("/evidence/evt-proximity"))
    await user.click(screen.getByRole("button", { name: /WRONG_WAY｜逆向候選/ }))
    expect(screen.getByText("此事件沒有可用的 Evidence Snapshot。")).toBeInTheDocument()
    fireEvent.error(evidence)
    await waitFor(() => expect(screen.queryByAltText("接近警示事件證據")).not.toBeInTheDocument())
  })
})
