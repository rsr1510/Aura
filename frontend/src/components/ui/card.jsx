export function Card({ children }) {
  return <div className="card">{children}</div>;
}

export function CardContent({ children }) {
  return (
    <div className="mt-2 text-gray-300">
      {children}
    </div>
  );
}
