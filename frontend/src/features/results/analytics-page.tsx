import { Activity, BarChart3, Gauge } from "lucide-react"
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { JobResultResponse } from "../../api/client"
import { Card, CardContent, CardHeader } from "../../components/ui/card"
import { mapClassRows, mapDirectionRows, mapTrafficRows } from "./presentation"

const colors = ["#0f766e", "#155e75", "#1e3a5f", "#d97706", "#64748b", "#be123c"]

function RawData({ title, rows }: { title: string; rows: Array<Record<string, string>> }) {
  return <details className="rounded-xl border border-slate-200 bg-white p-4"><summary className="cursor-pointer text-sm font-semibold text-slate-700">查看原始資料 · {title}</summary><div className="mt-4 max-h-64 overflow-auto"><pre className="text-xs text-slate-500">{JSON.stringify(rows, null, 2)}</pre></div></details>
}

export function AnalyticsPage({ result, classRows, directionRows, trafficRows }: { result: JobResultResponse; classRows: Array<Record<string, string>>; directionRows: Array<Record<string, string>>; trafficRows: Array<Record<string, string>> }) {
  const classes = mapClassRows(classRows)
  const directions = mapDirectionRows(directionRows)
  const traffic = mapTrafficRows(trafficRows)
  return <div className="space-y-7"><div><p className="text-sm font-semibold text-teal-700">交通分析</p><h1 className="mt-1 text-3xl font-bold">結構化交通指標</h1><p className="mt-2 text-sm text-slate-500">Traffic count 來源為 Track-based line crossing，不使用 detection occurrences。</p></div>
    <div className="grid gap-6 xl:grid-cols-2">
      <Card><CardHeader><h2 className="font-bold">車種分布</h2></CardHeader><CardContent><div className="h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={classes}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="name" /><YAxis allowDecimals={false} /><Tooltip /><Bar dataKey="count" radius={[6, 6, 0, 0]}>{classes.map((item, index) => <Cell key={item.name} fill={colors[index % colors.length]} />)}</Bar></BarChart></ResponsiveContainer></div><div className="mt-4 flex flex-wrap gap-2">{classes.map((item) => <span key={item.name} className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold">{item.name}：{item.count}</span>)}</div></CardContent></Card>
      <Card><CardHeader><h2 className="font-bold">Traffic Over Time</h2></CardHeader><CardContent><div className="h-72"><ResponsiveContainer width="100%" height="100%"><LineChart data={traffic}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="interval" /><YAxis allowDecimals={false} /><Tooltip /><Line type="monotone" dataKey="count" stroke="#0f766e" strokeWidth={3} dot={{ fill: "#0f766e" }} /></LineChart></ResponsiveContainer></div></CardContent></Card>
      <Card><CardHeader><h2 className="font-bold">Direction Distribution</h2></CardHeader><CardContent><div className="h-64"><ResponsiveContainer width="100%" height="100%"><BarChart data={directions} layout="vertical"><CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" allowDecimals={false} /><YAxis type="category" dataKey="name" width={60} /><Tooltip /><Bar dataKey="count" fill="#155e75" radius={[0, 6, 6, 0]} /></BarChart></ResponsiveContainer></div></CardContent></Card>
      <Card><CardHeader><h2 className="font-bold">Zone Activity / Peak Occupancy</h2></CardHeader><CardContent className="space-y-5"><div className="flex items-center gap-4"><span className="rounded-xl bg-teal-50 p-3 text-teal-700"><Gauge /></span><div><p className="text-3xl font-bold">{result.traffic_analytics.zone_peak_occupancy} 輛次觀測峰值</p><p className="mt-1 text-sm text-slate-500">Density heuristic：{result.traffic_analytics.density ?? "無資料"}</p></div></div><div className="rounded-xl bg-slate-50 p-4 text-xs leading-5 text-slate-600">目前 backend 提供可靠的 peak occupancy summary，未提供完整 occupancy time series，因此不重新執行 CV pipeline 補算。</div></CardContent></Card>
    </div>
    <div className="space-y-3"><div className="flex items-center gap-2 text-sm font-semibold"><Activity size={17} />原始資料</div><RawData title="車種分布" rows={classRows} /><RawData title="方向分布" rows={directionRows} /><RawData title="時間序列" rows={trafficRows} /></div>
    <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900"><BarChart3 className="mr-2 inline" size={17} />通過計數線不代表完整交通流量普查；Zone density 為專案透明 heuristic。</div>
  </div>
}
