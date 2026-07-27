import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { DEFAULT_SETTINGS, loadSettings, saveSettings } from '../lib/settings';

export default function SettingsScreen() {
  const router = useRouter();
  const [baseUrl, setBaseUrl] = useState('');
  const [intervalSec, setIntervalSec] = useState('');

  useEffect(() => {
    loadSettings().then((s) => {
      setBaseUrl(s.baseUrl);
      setIntervalSec(String(s.intervalSec));
    });
  }, []);

  const onSave = async () => {
    const n = parseFloat(intervalSec);
    await saveSettings({
      baseUrl: baseUrl.trim(),
      intervalSec: Number.isFinite(n) && n > 0 ? n : DEFAULT_SETTINGS.intervalSec,
    });
    router.back();
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.field}>
        <Text style={styles.label}>Backend base URL</Text>
        <TextInput
          style={styles.input}
          value={baseUrl}
          onChangeText={setBaseUrl}
          placeholder="http://192.168.1.10:8000"
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
        />
        <Text style={styles.hint}>Routes /sample and /analyse are appended automatically.</Text>
      </View>

      <View style={styles.field}>
        <Text style={styles.label}>Capture interval (seconds)</Text>
        <TextInput
          style={styles.input}
          value={intervalSec}
          onChangeText={setIntervalSec}
          placeholder="5"
          keyboardType="numeric"
        />
      </View>

      <Pressable style={styles.save} onPress={onSave}>
        <Text style={styles.saveText}>Save</Text>
      </Pressable>

      <Text style={styles.note}>
        Tip: on Expo Go use your machine's LAN IP (not localhost) so the phone can reach the backend.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, gap: 20, backgroundColor: '#fff', flexGrow: 1 },
  field: { gap: 8 },
  label: { fontSize: 15, fontWeight: '600', color: '#333' },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 12,
    fontSize: 16,
  },
  hint: { fontSize: 12, color: '#888' },
  save: { backgroundColor: '#16a34a', paddingVertical: 16, borderRadius: 12, alignItems: 'center' },
  saveText: { color: '#fff', fontSize: 18, fontWeight: '700' },
  note: { fontSize: 12, color: '#999', marginTop: 8 },
});
