import type { ReactNode } from "react";
import { motion } from "motion/react";

interface Props {
  title?: ReactNode;
  subtitle?: ReactNode;
  right?: ReactNode;
  className?: string;
  children: ReactNode;
  delay?: number;
}

export function Card({ title, subtitle, right, className = "", children, delay = 0 }: Props) {
  return (
    <motion.section
      className={`card flex flex-col p-5 ${className}`}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay, ease: [0.2, 0.8, 0.2, 1] }}
    >
      {(title || right) && (
        <header className="mb-3 flex items-start justify-between gap-4">
          <div>
            {title && <h2 className="text-[15px] font-semibold tracking-tight">{title}</h2>}
            {subtitle && <p className="label mt-0.5">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      <div className="relative min-h-0 flex-1">{children}</div>
    </motion.section>
  );
}
