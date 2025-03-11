export function Switch({ checked, onCheckedChange }) {
  return (
    <div
      className={`toggle-switch ${checked ? "active" : ""}`}
      onClick={() => onCheckedChange(!checked)}
    >
      <div className="dot"></div>
    </div>
  );
}
