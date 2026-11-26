/** Stands in for a chart with nothing to draw, at the chart's size. */
export function NoData({ note, className = "h-64 w-full" }: { note: string; className?: string }) {
  return (
    <div className={`grid place-items-center rounded-xl border border-dashed border-[var(--line)] text-center ${className}`}>
      <span className="label px-4">{note}</span>
    </div>
  );
}
