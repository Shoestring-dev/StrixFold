import { useEffect, useRef, useState } from "react";
import { animate, useReducedMotion } from "motion/react";

/** Animates from the previous value to the new one. */
export function CountUp({ value, format }: { value: number | null; format: (n: number | null) => string }) {
  const [shown, setShown] = useState<number | null>(value);
  const from = useRef(0);
  const reduced = useReducedMotion();
  useEffect(() => {
    if (value == null || reduced) {
      setShown(value);
      from.current = value ?? 0;
      return;
    }
    const c = animate(from.current, value, {
      duration: 0.9,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => setShown(v),
    });
    from.current = value;
    return () => c.stop();
  }, [value, reduced]);
  return <>{format(shown)}</>;
}
