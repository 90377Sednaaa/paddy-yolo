import React from 'react';

interface BrandMarkProps {
  size?: number;
  className?: string;
}

/** Reticle + rice-leaf monogram. Linework follows the current text color. */
export const BrandMark: React.FC<BrandMarkProps> = ({ size = 38, className }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 40 40"
    fill="none"
    className={className}
    aria-hidden="true"
    focusable="false"
  >
    <path
      d="M2.5 12V2.5H12M28 2.5h9.5V12M37.5 28v9.5H28M12 37.5H2.5V28"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="square"
      opacity="0.55"
    />
    <path d="M11 30c0-9.5 7-16 19-18-1 12-7.5 18.5-19 18Z" fill="var(--c-leaf)" />
    <path
      d="M11 30c5-7.5 11-13 19-18"
      stroke="#18201A"
      strokeWidth="1.3"
      strokeLinecap="round"
      opacity="0.72"
    />
    <circle cx="29.5" cy="29.5" r="4.2" stroke="var(--c-ember)" strokeWidth="1.5" />
    <circle cx="29.5" cy="29.5" r="1.2" fill="var(--c-ember)" />
  </svg>
);
