interface BrandMarkProps {
  className?: string;
  title?: string;
}

/** StockReason's mark: an S-shaped price path ending in an upward signal. */
export const BrandMark = ({ className, title = 'StockReason' }: BrandMarkProps) => (
  <svg className={className} viewBox="0 0 32 32" fill="none" role="img" aria-label={title}>
    <path d="M7 10.5C7 7.9 9.4 6 12.9 6c3 0 5.3 1.1 7.1 2.8" stroke="currentColor" strokeWidth="3.1" strokeLinecap="round" />
    <path d="M24.8 7.3 20.1 8.8l2.2 4.2" stroke="currentColor" strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M24.5 21.5c0 2.7-2.5 4.5-6 4.5-3.3 0-6-1.2-7.8-3.1" stroke="currentColor" strokeWidth="3.1" strokeLinecap="round" />
    <path d="m7.2 24.8 4.8-1.2-1.9-4.4" stroke="currentColor" strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M8.3 19.7c0-2.6 2.3-4.1 7.2-4.1 5.1 0 8.4-1.7 8.4-5.1" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" opacity=".72" />
  </svg>
);
