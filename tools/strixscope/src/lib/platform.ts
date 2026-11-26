/** True under WebKitGTK (Tauri on Linux); the updater is Linux-only here. */
export const IS_LINUX: boolean =
  typeof navigator !== "undefined" && /Linux/i.test(navigator.userAgent) && !/Android/i.test(navigator.userAgent);
