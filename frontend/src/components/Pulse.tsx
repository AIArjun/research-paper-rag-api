/** Loading indicator: an orbiting point on a ring. No progress is implied. */
export function Pulse({ label }: { label: string }) {
  return (
    <span className="pulse" role="img" aria-label={label}>
      <svg viewBox="0 0 48 48" aria-hidden="true" focusable="false">
        <circle className="pulse__ring" cx="24" cy="24" r="18" />
        <circle className="pulse__ring pulse__ring--dash" cx="24" cy="24" r="18" />
        <g className="pulse__orbit">
          <circle className="pulse__dot" cx="24" cy="6" r="3.2" />
        </g>
        <circle className="pulse__core" cx="24" cy="24" r="3" />
      </svg>
    </span>
  );
}
