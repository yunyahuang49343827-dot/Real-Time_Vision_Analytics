import { describe, expect, it, vi } from "vitest"

import { createAnalysisJob, getJobStatus } from "./client"

describe("FastAPI client contract", () => {
  it("uploads the selected file and governed analysis mode", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ job_id: "job-1", status: "CREATED" }),
      { status: 202, headers: { "Content-Type": "application/json" } },
    ))
    vi.stubGlobal("fetch", fetchMock)
    const file = new File(["video"], "traffic.mp4", { type: "video/mp4" })

    await expect(createAnalysisJob(file, "aerial")).resolves.toEqual({
      job_id: "job-1", status: "CREATED",
    })
    const [, options] = fetchMock.mock.calls[0]
    const body = options.body as FormData
    expect(body.get("video")).toBe(file)
    expect(body.get("analysis_mode")).toBe("aerial")
  })

  it("reads lifecycle and frame progress without exposing raw pipeline data", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      job_id: "job-1",
      status: "PROCESSING",
      progress: 0.42,
      processed_frames: 42,
      total_frames: 100,
      analysis_mode: "standard",
      created_at: "2026-09-01T00:00:00Z",
      started_at: "2026-09-01T00:00:01Z",
      completed_at: null,
      error: null,
    }), { status: 200, headers: { "Content-Type": "application/json" } })))

    const status = await getJobStatus("job-1")
    expect(status).toMatchObject({
      status: "PROCESSING", progress: 0.42, processed_frames: 42, total_frames: 100,
    })
  })
})
