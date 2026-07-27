import { useFocusEffect } from 'expo-router';
import { useCallback, useRef, useState } from 'react';
import { StyleSheet, Text, TextInput, View } from 'react-native';
import CaptureCamera from '../components/CaptureCamera';
import { uploadSample } from '../lib/api';
import { loadSettings } from '../lib/settings';

export default function SampleScreen() {
  const [settings, setSettings] = useState(null);
  const [label, setLabel] = useState('');
  const labelRef = useRef('');
  labelRef.current = label;

  useFocusEffect(
    useCallback(() => {
      loadSettings().then(setSettings);
    }, [])
  );

  const onCapture = useCallback(
    async (uri) => {
      await uploadSample({ baseUrl: settings.baseUrl, uri, label: labelRef.current });
    },
    [settings]
  );

  if (!settings) return null;

  return (
    <View style={styles.container}>
      <CaptureCamera
        intervalSec={settings.intervalSec}
        onCapture={onCapture}
        maxSize={1024}
        quality={0.85}
        accent="#2563eb"
      >
        <View style={styles.field}>
          <Text style={styles.label}>Label (optional, sent with each photo)</Text>
          <TextInput
            style={styles.input}
            value={label}
            onChangeText={setLabel}
            placeholder="e.g. cat, defect, empty..."
            autoCapitalize="none"
          />
          <Text style={styles.target}>POST {settings.baseUrl}/sample</Text>
        </View>
      </CaptureCamera>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  field: { padding: 12, gap: 6 },
  label: { fontSize: 13, color: '#555' },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
  },
  target: { fontSize: 12, color: '#999', marginTop: 4 },
});
