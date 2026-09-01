import { Activity, BarChart3, CirclePlus, Info, ShieldAlert } from "lucide-react"
import type { ReactNode } from "react"

import { cn } from "../lib/utils"

const navigation = [
  { id: "new", label: "新增分析", icon: CirclePlus, enabledBeforeCompletion: true },
  { id: "overview", label: "分析總覽", icon: Activity },
  { id: "analytics", label: "交通分析", icon: BarChart3 },
  { id: "events", label: "事件檢視", icon: ShieldAlert },
  { id: "engineering", label: "工程資訊", icon: Info },
]

export type DashboardSection = "new" | "overview" | "analytics" | "events" | "engineering"

export function AppShell({
  children,
  completed = false,
  activeSection = "new",
  onNavigate,
}: {
  children: ReactNode
  completed?: boolean
  activeSection?: DashboardSection
  onNavigate?: (section: DashboardSection) => void
}) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-950">
      <aside className="fixed inset-y-0 left-0 hidden w-64 border-r border-slate-200 bg-white lg:flex lg:flex-col">
        <div className="border-b border-slate-100 px-6 py-7">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-xl bg-slate-900 text-white"><Activity size={20} /></div>
            <div><p className="font-bold">Vision Analytics</p><p className="text-xs text-slate-500">交通事件分析平台</p></div>
          </div>
        </div>
        <nav className="space-y-1 p-4" aria-label="主要導覽">
          {navigation.map(({ id, label, icon: Icon, enabledBeforeCompletion }) => {
            const disabled = !enabledBeforeCompletion && !completed
            return (
              <button
                key={label}
                type="button"
                disabled={disabled}
                onClick={() => onNavigate?.(id as DashboardSection)}
                className={cn(
                  "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm font-medium",
                  activeSection === id ? "bg-teal-50 text-teal-800" : "text-slate-600 hover:bg-slate-50",
                  disabled && "cursor-not-allowed text-slate-300",
                )}
              >
                <Icon size={18} />{label}
                {disabled && <span className="ml-auto text-[10px] uppercase">Unavailable</span>}
              </button>
            )
          })}
        </nav>
        <div className="mt-auto border-t border-slate-100 p-5 text-xs leading-5 text-slate-500">
          Runtime model<br /><span className="font-mono text-slate-700">yolo26n.pt</span>
        </div>
      </aside>
      <main className="min-h-screen lg:pl-64">
        <header className="border-b border-slate-200 bg-white px-5 py-4 lg:px-10">
          <div className="mx-auto flex max-w-6xl items-center justify-between">
            <div><p className="text-sm font-semibold text-slate-900">即時視覺分析</p><p className="text-xs text-slate-500">React Dashboard</p></div>
            <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">本機分析環境</span>
          </div>
        </header>
        <div className="mx-auto max-w-6xl px-5 py-8 lg:px-10 lg:py-10">{children}</div>
      </main>
    </div>
  )
}
