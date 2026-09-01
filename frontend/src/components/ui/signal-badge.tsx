import type { HTMLAttributes } from "react"

import { cn } from "../../lib/utils"

const signalStyles: Record<string, string> = {
  INFO: "border-slate-200 bg-slate-50 text-slate-600",
  WARNING: "border-orange-200 bg-orange-50 text-orange-700",
  CRITICAL: "border-rose-200 bg-rose-50 text-rose-700",
  REVIEW_REQUIRED: "border-violet-200 bg-violet-50 text-violet-700",
  DETECTED: "border-cyan-200 bg-cyan-50 text-cyan-700",
  CONFIRMED: "border-emerald-200 bg-emerald-50 text-emerald-700",
}

export function SignalBadge({ value, className, ...props }: { value: string } & HTMLAttributes<HTMLSpanElement>) {
  return <span className={cn("inline-flex rounded-full border px-2.5 py-1 text-[10px] font-bold tracking-wide", signalStyles[value] ?? signalStyles.INFO, className)} {...props}>{value}</span>
}
