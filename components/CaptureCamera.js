import { CameraView, useCameraPermissions } from 'expo-camera';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';

// Reusable auto-capture camera.
//
// Renders a live preview and drives a capture loop: every `intervalSec`
// seconds it takes a picture and calls `onCapture(uri)`. Captures never
// overlap — the next one waits until the previous `onCapture` resolves.
//
// Props:
//   intervalSec : number  seconds between captures
//   onCapture   : (uri) => Promise<void>
//   accent      : string  color for the start button
//   children    : extra UI rendered under the controls (label input, result...)
export default function CaptureCamera({ intervalSec, onCapture, accent = '#2563eb', children }) {
  const [permission, requestPermission] = useCameraPermissions();
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [count, setCount] = useState(0);
  const [error, setError] = useState(null);

  const cameraRef = useRef(null);
  const runningRef = useRef(false);
  const timerRef = useRef(null);
  const onCaptureRef = useRef(onCapture);
  onCaptureRef.current = onCapture;

  const captureOnce = useCallback(async () => {
    if (!cameraRef.current) return;
    setBusy(true);
    setError(null);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.6,
        skipProcessing: true,
      });
      if (photo?.uri) {
        await onCaptureRef.current(photo.uri);
        setCount((c) => c + 1);
      }
    } catch (e) {
      setError(e?.message || 'capture failed');
    } finally {
      setBusy(false);
    }
  }, []);

  // Recursive timeout loop (avoids overlap even when uploads are slow).
  const scheduleNext = useCallback(() => {
    timerRef.current = setTimeout(async () => {
      if (!runningRef.current) return;
      await captureOnce();
      if (runningRef.current) scheduleNext();
    }, Math.max(1, intervalSec) * 1000);
  }, [captureOnce, intervalSec]);

  const start = useCallback(async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    setRunning(true);
    setCount(0);
    await captureOnce(); // fire first shot immediately
    if (runningRef.current) scheduleNext();
  }, [captureOnce, scheduleNext]);

  const stop = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
  }, []);

  // Cleanup on unmount.
  useEffect(() => () => stop(), [stop]);

  if (!permission) {
    return (
      <View style={styles.center}>
        <ActivityIndicator />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.msg}>Camera access is required.</Text>
        <Pressable style={[styles.btn, { backgroundColor: accent }]} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant permission</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.preview}>
        <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />
        <View style={styles.badge}>
          <Text style={styles.badgeText}>
            {running ? `● live · every ${intervalSec}s` : 'idle'} · sent {count}
          </Text>
        </View>
        {busy && (
          <View style={styles.busy}>
            <ActivityIndicator color="#fff" />
          </View>
        )}
      </View>

      {error ? <Text style={styles.error}>⚠ {error}</Text> : null}

      <View style={styles.controls}>
        {!running ? (
          <Pressable style={[styles.btn, { backgroundColor: accent }]} onPress={start}>
            <Text style={styles.btnText}>▶ Start</Text>
          </Pressable>
        ) : (
          <Pressable style={[styles.btn, styles.stop]} onPress={stop}>
            <Text style={styles.btnText}>■ Stop</Text>
          </Pressable>
        )}
      </View>

      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24 },
  msg: { fontSize: 16, color: '#333' },
  preview: {
    aspectRatio: 3 / 4,
    margin: 12,
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  badge: {
    position: 'absolute',
    top: 10,
    left: 10,
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
  },
  badgeText: { color: '#fff', fontSize: 12, fontWeight: '600' },
  busy: {
    position: 'absolute',
    bottom: 10,
    right: 10,
    backgroundColor: 'rgba(0,0,0,0.55)',
    padding: 8,
    borderRadius: 20,
  },
  controls: { paddingHorizontal: 12 },
  btn: { paddingVertical: 16, borderRadius: 12, alignItems: 'center' },
  btnText: { color: '#fff', fontSize: 18, fontWeight: '700' },
  stop: { backgroundColor: '#dc2626' },
  error: { color: '#dc2626', textAlign: 'center', paddingHorizontal: 12, paddingBottom: 4 },
});
