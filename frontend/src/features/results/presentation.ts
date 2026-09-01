import type { EventRecord, JobResultResponse } from "../../api/client"

export type EventFilter = "all" | "warning" | "info" | "review"

const classLabels: Record<string, string> = {
  person: "行人", bicycle: "自行車", car: "汽車", motorcycle: "機車", bus: "公車", truck: "卡車",
}

export const eventLabels: Record<string, string> = {
  LINE_CROSSING: "LINE_CROSSING｜通過計數線",
  ZONE_ENTRY: "ZONE_ENTRY｜進入區域",
  ZONE_EXIT: "ZONE_EXIT｜離開區域",
  WRONG_WAY: "WRONG_WAY｜逆向候選",
  LONG_DWELL: "LONG_DWELL｜長時間停留",
  STATIONARY_VEHICLE: "STATIONARY_VEHICLE｜靜止車輛",
  PEDESTRIAN_INTRUSION: "PEDESTRIAN_INTRUSION｜行人進入監控區",
  PROXIMITY_WARNING: "PROXIMITY_WARNING｜接近警示",
}

export function formatClock(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—"
  const rounded = Math.floor(seconds)
  const minutes = Math.floor(rounded / 60)
  return `${String(minutes).padStart(2, "0")}:${String(rounded % 60).padStart(2, "0")}`
}

export function buildOverviewKpis(
  result: Pick<JobResultResponse, "traffic_analytics" | "event_summary">,
  classRows: Array<Record<string, string>>,
) {
  const attentionEvents = result.event_summary
    .filter((item) => item.severity === "WARNING" || item.severity === "CRITICAL" || item.status === "REVIEW_REQUIRED")
    .reduce((sum, item) => sum + item.count, 0)
  const main = classRows
    .filter((row) => row.class_name !== "person")
    .sort((left, right) => Number(right.crossing_count) - Number(left.crossing_count))[0]
  const start = result.traffic_analytics.peak_interval_start_seconds
  const end = result.traffic_analytics.peak_interval_end_seconds
  return {
    crossings: result.traffic_analytics.total_line_crossing_count,
    attentionEvents,
    primaryVehicle: main ? (classLabels[main.class_name] ?? main.class_name) : "無資料",
    peakInterval: start == null || end == null ? "無資料" : `${formatClock(start)}–${formatClock(end)}`,
  }
}

export function mapClassRows(rows: Array<Record<string, string>>) {
  return rows.map((row) => ({
    name: classLabels[row.class_name] ?? row.class_name,
    count: Number(row.crossing_count),
    percentage: Number(row.percentage),
  }))
}

export function mapDirectionRows(rows: Array<Record<string, string>>) {
  const names: Record<string, string> = { A_TO_B: "A → B", B_TO_A: "B → A" }
  return rows.map((row) => ({
    name: names[row.crossing_direction] ?? row.crossing_direction,
    count: Number(row.crossing_count),
    percentage: Number(row.percentage),
  }))
}

export function mapTrafficRows(rows: Array<Record<string, string>>) {
  return rows.map((row) => ({
    interval: formatClock(Number(row.interval_start_seconds)),
    end: formatClock(Number(row.interval_end_seconds)),
    count: Number(row.total_crossing_count),
  }))
}

export function filterEvents<T extends Pick<EventRecord, "severity" | "status">>(events: T[], filter: EventFilter): T[] {
  if (filter === "warning") return events.filter((event) => event.severity === "WARNING" || event.severity === "CRITICAL")
  if (filter === "info") return events.filter((event) => event.severity === "INFO")
  if (filter === "review") return events.filter((event) => event.status === "REVIEW_REQUIRED")
  return events
}

export function eventInterpretation(eventType: string): string {
  if (eventType === "PROXIMITY_WARNING") return "接近警示僅代表影像座標中的接近關係，不代表實際距離或碰撞風險。"
  if (eventType === "WRONG_WAY") return "逆向事件是規則判定候選，需要人工確認，不代表已確認交通違規。"
  if (eventType === "LINE_CROSSING") return "通過計數線是虛擬線 crossing count，不代表完整交通流量普查。"
  return "規則產生的系統事件，投入作業判讀前請先人工檢視。"
}
