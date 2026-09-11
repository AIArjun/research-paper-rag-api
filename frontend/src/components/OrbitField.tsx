/**
 * Decorative orbital geometry (pure SVG, aria-hidden). Rings and points are
 * fixed composition, not data.
 */
export function OrbitField({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 800 800" aria-hidden="true" focusable="false">
      <defs>
        <radialGradient id="orbit-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(127, 211, 196, 0.22)" />
          <stop offset="55%" stopColor="rgba(127, 211, 196, 0.05)" />
          <stop offset="100%" stopColor="rgba(127, 211, 196, 0)" />
        </radialGradient>
      </defs>
      <circle cx="400" cy="400" r="380" fill="url(#orbit-glow)" />
      <g fill="none" stroke="currentColor" strokeOpacity="0.16">
        <circle cx="400" cy="400" r="120" />
        <circle cx="400" cy="400" r="210" strokeDasharray="2 10" />
        <circle cx="400" cy="400" r="300" strokeOpacity="0.1" />
        <ellipse cx="400" cy="400" rx="360" ry="150" transform="rotate(-28 400 400)" strokeOpacity="0.12" />
      </g>
      <g fill="currentColor">
        <circle cx="400" cy="280" r="3" fillOpacity="0.8" />
        <circle cx="552" cy="486" r="2.5" fillOpacity="0.6" />
        <circle cx="186" cy="372" r="2" fillOpacity="0.5" />
        <circle cx="640" cy="240" r="1.6" fillOpacity="0.45" />
        <circle cx="300" cy="640" r="2.2" fillOpacity="0.5" />
      </g>
    </svg>
  );
}
