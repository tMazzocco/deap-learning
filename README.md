# Photo Classification PoC (Expo SDK 54)

Expo Go app that auto-captures a photo every _X_ seconds and sends it to a
backend over one of two routes — for building an AI image-classification PoC.

- **Sample** → `POST /sample` — upload photos with an optional text label
  (dataset building).
- **Analyse** → `POST /analyse` — upload a photo and display the label the
  backend returns (live classification).
- **Settings** → backend base URL + capture interval, persisted on device.

## Run (Expo Go)

```bash
npm install
npx expo start
```

Scan the QR with **Expo Go** (SDK 54). Grant camera permission on first launch.

> Set the backend URL in **Settings** to your machine's LAN IP
> (e.g. `http://192.168.1.10:8000`) — `localhost` on the phone points to the
> phone, not your dev machine.

## App flow

```
Home ─┬─ Sample   → live camera, optional label field, uploads every Xs
      ├─ Analyse  → live camera, shows predicted label every Xs
      └─ Settings → base URL + interval (AsyncStorage)
```

Captures never overlap: the next shot waits until the current upload resolves.

## Backend contract

Both routes receive `multipart/form-data`.

### `POST /sample`
| field   | type   | notes                      |
|---------|--------|----------------------------|
| `image` | file   | JPEG                       |
| `label` | string | optional, omitted if empty |

Response body is ignored (any 2xx = success).

### `POST /analyse`
| field   | type | notes |
|---------|------|-------|
| `image` | file | JPEG  |

Expected JSON response:

```json
{ "label": "cat", "confidence": 0.93 }
```

`label` falls back to `prediction`; `confidence` (0..1) is optional.

## Structure

```
app/
  _layout.js     Stack navigator (expo-router)
  index.js       Home — 2 buttons + settings
  sample.js      /sample screen
  analyse.js     /analyse screen
  settings.js    base URL + interval
components/
  CaptureCamera.js  reusable camera + capture loop
lib/
  api.js         multipart upload helpers
  settings.js    AsyncStorage-backed settings
```

## Stack

Expo SDK 54 · expo-router 6 · expo-camera 17 · React Native 0.81 · React 19.
