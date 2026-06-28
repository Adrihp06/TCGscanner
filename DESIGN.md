# Design

## System

Task-focused product UI using system fonts, restrained color, and compact controls. The interface should prioritize camera input, scan state, and price result visibility.

## Color

- Background: `oklch(0.985 0.006 210)`
- Surface: `oklch(1 0 0)`
- Surface muted: `oklch(0.955 0.012 210)`
- Ink: `oklch(0.19 0.018 245)`
- Muted ink: `oklch(0.42 0.018 245)`
- Primary: `oklch(0.52 0.14 210)`
- Accent: `oklch(0.62 0.16 150)`
- Warning: `oklch(0.67 0.15 65)`
- Error: `oklch(0.56 0.18 28)`

## Typography

Use `system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`. Product scale: 12, 14, 16, 20, 24, 32. No display fonts in labels or data.

## Components

- Primary action: filled button with icon slot where available.
- Secondary action: quiet bordered button.
- Form controls: native select/input styling with visible focus ring.
- Scan result: single focused panel with identity, confidence, price, and timestamp.
- Empty states: actionable, short, and tied to the scan workflow.

## Layout

Mobile-first one-column shell. On wider screens, use a two-column layout: scanner/input controls left, resolved card and diagnostics right. Avoid nested cards.
