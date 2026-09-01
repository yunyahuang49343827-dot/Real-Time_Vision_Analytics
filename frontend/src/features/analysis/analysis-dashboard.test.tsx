import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AnalysisDashboard } from "./analysis-dashboard"

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function renderDashboard() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <AnalysisDashboard pollIntervalMs={10} />
    </QueryClientProvider>,
  )
}

describe("新增分析流程", () => {
  it("uploads an aerial video and polls through processing to completion", async () => {
    let statusCalls = 0
    const fetchMock = vi.fn().mockImplementation((input: string, init?: RequestInit) => {
      if (input.endsWith("/health")) return Promise.resolve(new Response(JSON.stringify({
        status: "ok", service: "vision-analytics", runtime_model: "models/pretrained/yolo26n.pt",
        runtime_model_sha256: "a".repeat(64), device: "mps",
        runtime_profiles: { standard: { imgsz: 640, confidence_threshold: 0.25 }, aerial: { imgsz: 960, confidence_threshold: 0.15 } },
      }), { status: 200 }))
      if (input.endsWith("/jobs") && init?.method === "POST") return Promise.resolve(new Response(JSON.stringify({ job_id: "job-1", status: "CREATED" }), { status: 202 }))
      if (input.endsWith("/jobs/job-1/results")) return Promise.resolve(new Response(JSON.stringify({
        job_id: "job-1",
        video_metadata: { filename: "aerial.mp4", width: 1280, height: 720, fps: 30, frame_count: 100, duration_seconds: 3.33, codec: "h264", validation_status: "PASS" },
        traffic_analytics: { total_line_crossing_count: 0, person_crossing_count: 0, motorized_vehicle_crossing_count: 0, bicycle_crossing_count: 0, peak_interval_start_seconds: null, peak_interval_end_seconds: null, peak_interval_count: 0, zone_peak_occupancy: 0, density: "LOW", reconciliation_status: "PASS" },
        event_summary: [], artifacts: {}, warnings: [],
      }), { status: 200 }))
      if (input.endsWith("/jobs/job-1/events")) return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      statusCalls += 1
      const processing = statusCalls === 1
      return Promise.resolve(new Response(JSON.stringify({ job_id: "job-1", status: processing ? "PROCESSING" : "COMPLETED", progress: processing ? 0.5 : 1, processed_frames: processing ? 50 : 100, total_frames: 100, analysis_mode: "aerial", created_at: "2026-09-01T00:00:00Z", started_at: null, completed_at: processing ? null : "2026-09-01T00:00:10Z", error: null }), { status: 200 }))
    })
    vi.stubGlobal("fetch", fetchMock)
    renderDashboard()
    const user = userEvent.setup()
    expect(screen.getByRole("button", { name: /分析總覽/ })).toBeDisabled()

    await user.upload(screen.getByLabelText("選擇交通影片"), new File(["video"], "aerial.mp4", { type: "video/mp4" }))
    await user.click(screen.getByLabelText("空拍 / Aerial"))
    await user.click(screen.getByRole("button", { name: "開始分析" }))

    expect(await screen.findByText("PROCESSING｜分析中")).toBeInTheDocument()
    expect(screen.getByText("50 / 100 frames")).toBeInTheDocument()
    expect(await screen.findByRole("button", { name: "分析總覽" })).toBeEnabled()
    await waitFor(() => expect(fetchMock.mock.calls.filter(([, options]) => options?.method === "POST")).toHaveLength(1))
  })

  it("shows a safe failed state", async () => {
    const responses = [
      new Response(JSON.stringify({
        status: "ok", service: "vision-analytics", runtime_model: "models/pretrained/yolo26n.pt",
        runtime_model_sha256: "a".repeat(64), device: "mps",
        runtime_profiles: { standard: { imgsz: 640, confidence_threshold: 0.25 }, aerial: { imgsz: 960, confidence_threshold: 0.15 } },
      }), { status: 200 }),
      new Response(JSON.stringify({ job_id: "job-2", status: "CREATED" }), { status: 202 }),
      new Response(JSON.stringify({ job_id: "job-2", status: "FAILED", progress: 0.2, processed_frames: 20, total_frames: 100, analysis_mode: "standard", created_at: "2026-09-01T00:00:00Z", started_at: null, completed_at: "2026-09-01T00:00:03Z", error: { code: "PIPELINE_FAILED", message: "影片分析失敗" } }), { status: 200 }),
    ]
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => Promise.resolve(responses.shift())))
    renderDashboard()
    const user = userEvent.setup()
    await user.upload(screen.getByLabelText("選擇交通影片"), new File(["video"], "road.mp4", { type: "video/mp4" }))
    await user.click(screen.getByRole("button", { name: "開始分析" }))

    expect(await screen.findByText("FAILED｜失敗")).toBeInTheDocument()
    expect(screen.getByText(/^影片分析失敗/)).toBeInTheDocument()
    expect(screen.queryByText(/Traceback/)).not.toBeInTheDocument()
  })

  it("shows a safe backend unavailable state", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("connection refused with internal details")))
    renderDashboard()
    expect(await screen.findByRole("alert")).toHaveTextContent("無法連線至分析服務")
    expect(screen.queryByText(/internal details|Traceback/)).not.toBeInTheDocument()
  })
})
