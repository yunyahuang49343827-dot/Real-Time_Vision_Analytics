import { useMutation, useQuery } from "@tanstack/react-query"
import { CheckCircle2, CloudUpload, Film, LoaderCircle, TriangleAlert } from "lucide-react"
import { useState } from "react"

import { createAnalysisJob, getHealth, getJobStatus, type AnalysisMode } from "../../api/client"
import { AppShell } from "../../components/app-shell"
import { Button } from "../../components/ui/button"
import { Card, CardContent, CardHeader } from "../../components/ui/card"
import { Progress } from "../../components/ui/progress"
import { cn } from "../../lib/utils"

const lifecycleLabels = {
  CREATED: "CREATED｜已建立",
  PROCESSING: "PROCESSING｜分析中",
  COMPLETED: "COMPLETED｜已完成",
  FAILED: "FAILED｜失敗",
} as const

export function AnalysisDashboard({ pollIntervalMs = 1200 }: { pollIntervalMs?: number }) {
  const [file, setFile] = useState<File | null>(null)
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("standard")
  const [jobId, setJobId] = useState<string | null>(null)
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, retry: false })
  const creation = useMutation({
    mutationFn: () => {
      if (!file) throw new Error("請先選擇影片")
      return createAnalysisJob(file, analysisMode)
    },
    onSuccess: (job) => setJobId(job.job_id),
  })
  const statusQuery = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === "COMPLETED" || status === "FAILED" ? false : pollIntervalMs
    },
    retry: false,
  })
  const status = statusQuery.data?.status ?? creation.data?.status
  const completed = status === "COMPLETED"
  const progress = Math.round((statusQuery.data?.progress ?? 0) * 100)
  const busy = creation.isPending || status === "CREATED" || status === "PROCESSING"
  const displayError = statusQuery.data?.error?.message
    ?? (creation.error instanceof Error ? creation.error.message : null)
    ?? (statusQuery.error instanceof Error ? statusQuery.error.message : null)

  return (
    <AppShell completed={completed}>
      <div className="mb-8 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="mb-2 text-sm font-semibold text-teal-700">新增分析</p>
          <h1 className="text-3xl font-bold tracking-tight text-slate-950">上傳交通影片</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">選擇分析模式並提交影片。偵測、追蹤與事件分析均由 FastAPI 後端執行。</p>
        </div>
        <div className={cn(
          "inline-flex w-fit items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold",
          health.isSuccess ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-slate-200 bg-white text-slate-500",
        )}>
          <span className={cn("size-2 rounded-full", health.isSuccess ? "bg-emerald-500" : "bg-slate-300")} />
          {health.isSuccess ? "FastAPI 已連線" : health.isError ? "FastAPI 無法連線" : "正在連線 FastAPI"}
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.6fr)]">
        <Card>
          <CardHeader>
            <h2 className="text-lg font-bold">分析設定</h2>
            <p className="mt-1 text-sm text-slate-500">支援 MP4、MOV、AVI；影片只會送往本機分析服務。</p>
          </CardHeader>
          <CardContent className="space-y-7">
            <div>
              <label htmlFor="traffic-video" className="mb-2 block text-sm font-semibold">選擇交通影片</label>
              <label htmlFor="traffic-video" className="flex min-h-44 cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50 px-6 text-center hover:border-teal-500 hover:bg-teal-50/30">
                <CloudUpload className="mb-3 text-teal-700" size={30} />
                <span className="text-sm font-semibold text-slate-800">{file?.name ?? "點擊選擇影片"}</span>
                <span className="mt-1 text-xs text-slate-500">單一交通影片，最大容量依後端設定</span>
              </label>
              <input id="traffic-video" className="sr-only" type="file" accept="video/mp4,video/quicktime,video/x-msvideo,.mp4,.mov,.avi" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            </div>

            <fieldset>
              <legend className="mb-3 text-sm font-semibold">分析模式</legend>
              <div className="grid gap-3 sm:grid-cols-2">
                {([
                  ["standard", "一般道路", "標準道路與路口場景 · 640 / 0.25"],
                  ["aerial", "空拍 / Aerial", "小型物件空拍場景 · 960 / 0.15"],
                ] as const).map(([value, title, description]) => (
                  <label key={value} className={cn(
                    "cursor-pointer rounded-xl border p-4 transition-colors",
                    analysisMode === value ? "border-teal-600 bg-teal-50 ring-1 ring-teal-600" : "border-slate-200 hover:border-slate-300",
                  )}>
                    <input className="sr-only" type="radio" name="analysis-mode" value={value} aria-label={title} checked={analysisMode === value} onChange={() => setAnalysisMode(value)} />
                    <span className="block text-sm font-bold text-slate-900">{title}</span>
                    <span className="mt-1 block text-xs leading-5 text-slate-500">{description}</span>
                  </label>
                ))}
              </div>
            </fieldset>

            <Button variant="accent" size="lg" className="w-full" disabled={!file || busy || !health.isSuccess} onClick={() => creation.mutate()}>
              {busy ? <LoaderCircle className="animate-spin" size={18} /> : <Film size={18} />}
              {busy ? "分析進行中" : "開始分析"}
            </Button>
          </CardContent>
        </Card>

        <Card className="h-fit">
          <CardHeader><h2 className="text-lg font-bold">分析狀態</h2></CardHeader>
          <CardContent>
            {!status && !displayError && <div className="py-10 text-center text-sm text-slate-500">提交影片後，這裡會顯示即時進度。</div>}
            {status && (
              <div className="space-y-5">
                <div className="flex items-center gap-3">
                  <div className={cn("grid size-11 place-items-center rounded-full", completed ? "bg-emerald-50 text-emerald-700" : status === "FAILED" ? "bg-rose-50 text-rose-700" : "bg-teal-50 text-teal-700")}>
                    {completed ? <CheckCircle2 /> : status === "FAILED" ? <TriangleAlert /> : <LoaderCircle className="animate-spin" />}
                  </div>
                  <div><p className="font-bold">{lifecycleLabels[status]}</p><p className="text-xs text-slate-500">Job {jobId}</p></div>
                </div>
                <Progress value={progress} />
                <div className="flex items-center justify-between text-sm"><span className="font-semibold text-slate-800">Progress {progress}%</span><span className="text-slate-500">{statusQuery.data?.processed_frames ?? 0} / {statusQuery.data?.total_frames ?? "—"} frames</span></div>
                {completed && <div className="rounded-lg bg-emerald-50 p-3 text-sm text-emerald-800">分析已完成，可前往其他功能區查看結果。</div>}
              </div>
            )}
            {displayError && <div role="alert" className="mt-5 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">{displayError}</div>}
          </CardContent>
        </Card>
      </div>
    </AppShell>
  )
}
