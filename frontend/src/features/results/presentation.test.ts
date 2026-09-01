import { describe, expect, it } from "vitest"

import {
  buildOverviewKpis,
  filterEvents,
  mapClassRows,
  mapDirectionRows,
  mapTrafficRows,
} from "./presentation"

const result = {
  job_id: "job-1",
  video_metadata: { filename: "traffic.mp4", width: 1920, height: 1080, fps: 30, frame_count: 300, duration_seconds: 10, codec: "h264", validation_status: "PASS" },
  traffic_analytics: {
    total_line_crossing_count: 18,
    person_crossing_count: 2,
    motorized_vehicle_crossing_count: 15,
    bicycle_crossing_count: 1,
    peak_interval_start_seconds: 20,
    peak_interval_end_seconds: 30,
    peak_interval_count: 7,
    zone_peak_occupancy: 12,
    density: "HIGH",
    reconciliation_status: "PASS",
  },
  event_summary: [
    { event_type: "LINE_CROSSING", severity: "INFO", status: "DETECTED", count: 18 },
    { event_type: "ZONE_ENTRY", severity: "INFO", status: "DETECTED", count: 6 },
    { event_type: "PROXIMITY_WARNING", severity: "WARNING", status: "REVIEW_REQUIRED", count: 3 },
    { event_type: "WRONG_WAY", severity: "CRITICAL", status: "REVIEW_REQUIRED", count: 2 },
    { event_type: "LONG_DWELL", severity: "WARNING", status: "DETECTED", count: 1 },
  ],
  artifacts: {}, warnings: [],
}

describe("結果呈現 mapping", () => {
  it("maps overview KPIs without counting ordinary INFO events as attention", () => {
    const kpis = buildOverviewKpis(result, [
      { class_name: "car", crossing_count: "12", percentage: "66.7" },
      { class_name: "motorcycle", crossing_count: "4", percentage: "22.2" },
      { class_name: "person", crossing_count: "2", percentage: "11.1" },
    ])
    expect(kpis).toEqual({
      crossings: 18,
      attentionEvents: 6,
      primaryVehicle: "汽車",
      peakInterval: "00:20–00:30",
    })
  })

  it("maps chart CSV values to typed chart data", () => {
    expect(mapClassRows([{ class_name: "motorcycle", crossing_count: "7", percentage: "35.0" }]))
      .toEqual([{ name: "機車", count: 7, percentage: 35 }])
    expect(mapDirectionRows([{ crossing_direction: "A_TO_B", crossing_count: "9", percentage: "60" }]))
      .toEqual([{ name: "A → B", count: 9, percentage: 60 }])
    expect(mapTrafficRows([{ interval_start_seconds: "10", interval_end_seconds: "20", total_crossing_count: "5" }]))
      .toEqual([{ interval: "00:10", end: "00:20", count: 5 }])
  })

  it("filters event cards by warning, info, and human-review meaning", () => {
    const events = [
      { event_id: "1", event_type: "LINE_CROSSING", severity: "INFO", status: "DETECTED" },
      { event_id: "2", event_type: "LONG_DWELL", severity: "WARNING", status: "DETECTED" },
      { event_id: "3", event_type: "WRONG_WAY", severity: "CRITICAL", status: "REVIEW_REQUIRED" },
    ]
    expect(filterEvents(events, "warning").map((event) => event.event_id)).toEqual(["2", "3"])
    expect(filterEvents(events, "info").map((event) => event.event_id)).toEqual(["1"])
    expect(filterEvents(events, "review").map((event) => event.event_id)).toEqual(["3"])
  })
})
