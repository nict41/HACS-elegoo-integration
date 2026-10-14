# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Canvas (AMS) support for the Centauri Carbon (CC1): per-slot filament sensors, active tray, and filament colors, matching the existing CC2 support. Canvas presence is auto-detected during setup and stored per printer.
- Per-slot filament usage sensors for the CC1 via the gcode capture proxy. The proxy URL is now configured in the WebSocket printer options, and works with or without a Canvas installed.
- New FDM print states reported verbatim from the printer — auto leveling, resonance testing, preheating/homing/leveling completed, auto feeding, and filament unload states — with an explicit `unrecognized` fallback so unknown codes can no longer freeze the status sensor.
- New `update_ip` service: change a printer's IP address from the Developer Tools UI or an automation (e.g. an external IP-detection script) without deleting and re-adding the integration; the config entry reloads with the new address and the service reports whether the printer is reachable at it.

### Changed

- FDM print status codes now map 1:1 from the printer's own status table instead of being approximated through resin states; mid-print milestones no longer surface as misleading states like "leveling".
- Config entries migrate automatically (v4 → v5) to record the per-printer Canvas flag; no action is needed.

### Fixed

- CC2 chamber camera: the printer now only ever sees one camera connection. A new frame hub holds a single MJPEG connection and serves every snapshot and every live viewer from it, instead of each consumer opening its own. A live view running next to a snapshot poll had crashed the printer's camera server, which only a printer restart recovered. The hub starts on demand, stops after 60 s without viewers or snapshots, never serves a frame older than 5 s, and reconnects with exponential backoff (2 s doubling to 120 s, one video enable per step). Live viewers can cap their frame rate with `?fps=` (1-20). The CC2 camera no longer offers a stream source, so Home Assistant's stream worker and go2rtc cannot open a second connection.
- WebSocket discovery now recognizes hostnames by their resolved IPv4 address, and options updates validate the newly submitted address.
- CC1 print status no longer sticks on a stale state for the remainder of a job when the printer holds an unmapped milestone code.
- The Canvas auto-detection step in setup is bounded by a timeout, so a device that accepts connections but never answers can no longer hang the config flow.

### Breaking Changes

- The unique ID format for entities has changed to include the machine ID. This may cause Home Assistant to create new entities. The integration will attempt to migrate existing entities to the new format, but this may not be successful in all cases. If you experience issues, you may need to remove and re-add the integration.
