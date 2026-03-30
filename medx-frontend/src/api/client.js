export function getApiBase() {
  const configured = import.meta.env.VITE_API_BASE?.trim();
  if (configured) {
    return configured.replace(/\/+$/, "");
  }

  return "/api";
}

export const API_BASE = getApiBase();
