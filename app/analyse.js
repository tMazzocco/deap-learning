import { useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import CaptureCamera from '../components/CaptureCamera';
import { uploadAnalyse } from '../lib/api';
import { loadSettings } from '../lib/settings';

export default function AnalyseScreen() {
  const [settings, setSettings] = useState(null);
  const [result, setResult] = useState(null);

  useFocusEffect(
    useCallback(() => {
      loadSettings().then(setSettings);
    }, [])
  );

  const onCapture = useCallback(
    async (uri) => {
      const r = await uploadAnalyse({ baseUrl: settings.baseUrl, uri });
      setResult(r);
    },
    [settings]
  );

  if (!settings) return null;

  return (
    <View style={styles.container}>
      <CaptureCamera intervalSec={settings.intervalSec} onCapture={onCapture} accent="#7c3aed">
        <View style={styles.resultBox}>
          <Text style={styles.resultLabel}>Prediction</Text>
          <Text style={styles.result}>{result ? result.label : '—'}</Text>
          {result?.confidence != null && (
            <Text style={styles.conf}>{Math.round(result.confidence * 100)}% confidence</Text>
          )}
          <Text style={styles.target}>POST {settings.baseUrl}/analyse</Text>
        </View>
      </CaptureCamera>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  resultBox: { padding: 16, alignItems: 'center', gap: 4 },
  resultLabel: { fontSize: 13, color: '#555' },
  result: { fontSize: 32, fontWeight: '800', color: '#7c3aed' },
  conf: { fontSize: 14, color: '#777' },
  target: { fontSize: 12, color: '#999', marginTop: 6 },
});
