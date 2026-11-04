// Shivir wordmark. Kept under the upstream component name so callers (and
// upstream merges) don't change; a designed logo replaces this text later.
const BRAND = "shivir";

const HandyTextLogo = ({
  width,
  height,
  className,
}: {
  width?: number;
  height?: number;
  className?: string;
}) => {
  return (
    <svg
      width={width}
      height={height}
      className={className}
      viewBox="0 0 930 328"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label={BRAND}
    >
      <text
        x="465"
        y="236"
        textAnchor="middle"
        fontSize="250"
        fontWeight="700"
        letterSpacing="-8"
        fill="currentColor"
        className="logo-primary"
        style={{ fontFamily: "var(--font-ui, system-ui)" }}
      >
        {BRAND}
      </text>
    </svg>
  );
};

export default HandyTextLogo;
