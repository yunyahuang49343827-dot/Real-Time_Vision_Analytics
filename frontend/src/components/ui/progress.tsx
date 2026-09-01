import { cn } from "../../lib/utils"

export function Progress({ value, className }: { value: number; className?: string }) {
  const bounded = Math.max(0, Math.min(100, value))
  return (
    <div
      className={cn("h-2.5 w-full overflow-hidden rounded-full bg-slate-100", className)}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={bounded}
    >
      <div className="h-full rounded-full bg-teal-600 transition-all duration-500" style={{ width: `${bounded}%` }} />
    </div>
  )
}
